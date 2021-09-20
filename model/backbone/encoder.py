"""
Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""

import torch
import logging
from torch import nn
logger = logging.getLogger(__name__)

try:
  import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
  BertLayerNorm = torch.nn.LayerNorm

from model.transformer.bert import BertEncoder
from model.layers import (NetVLAD, LinearLayer)
from model.transformer.bert_embed import (BertEmbeddings)
from utils.model_utils import mask_logits
import torch.nn.functional as F



class TransformerBaseModel(nn.Module):
    """
    Base Transformer model
    """
    def __init__(self, config):
        super(TransformerBaseModel, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)


    def forward(self,features,position_ids,token_type_ids,attention_mask):
        # embedding layer
        embedding_output = self.embeddings(token_type_ids=token_type_ids,
                                           inputs_embeds=features,
                                           position_ids=position_ids)

        encoder_outputs = self.encoder(embedding_output, attention_mask)

        sequence_output = encoder_outputs[0]

        return sequence_output

class TwoModalEncoder(nn.Module):
    """
        Two modality Transformer Encoder model
    """

    def __init__(self, config,img_dim,text_dim,hidden_dim,split_num,output_split=True):
        super(TwoModalEncoder, self).__init__()
        self.img_linear = LinearLayer(
            in_hsz=img_dim, out_hsz=hidden_dim)
        self.text_linear = LinearLayer(
            in_hsz=text_dim, out_hsz=hidden_dim)

        self.transformer = TransformerBaseModel(config)
        self.output_split = output_split
        if self.output_split:
            self.split_num = split_num


    def forward(self, visual_features, visual_position_ids, visual_token_type_ids, visual_attention_mask,
                text_features,text_position_ids,text_token_type_ids,text_attention_mask):

        transformed_im = self.img_linear(visual_features)
        transformed_text = self.text_linear(text_features)

        transformer_input_feat = torch.cat((transformed_im,transformed_text),dim=1)
        transformer_input_feat_pos_id = torch.cat((visual_position_ids,text_position_ids),dim=1)
        transformer_input_feat_token_id = torch.cat((visual_token_type_ids,text_token_type_ids),dim=1)
        transformer_input_feat_mask = torch.cat((visual_attention_mask,text_attention_mask),dim=1)

        output = self.transformer(features=transformer_input_feat,
                                  position_ids=transformer_input_feat_pos_id,
                                  token_type_ids=transformer_input_feat_token_id,
                                  attention_mask=transformer_input_feat_mask)

        if self.output_split:
            return torch.split(output,self.split_num,dim=1)
        else:
            return output


class OneModalEncoder(nn.Module):
    """
        One modality  Transformer Encoder model
    """

    def __init__(self, config,input_dim,hidden_dim):
        super(OneModalEncoder, self).__init__()
        self.linear = LinearLayer(
            in_hsz=input_dim, out_hsz=hidden_dim)
        self.transformer = TransformerBaseModel(config)

    def forward(self, features, position_ids, token_type_ids, attention_mask):

        transformed_features = self.linear(features)

        output = self.transformer(features=transformed_features,
                                  position_ids=position_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        return output


class VideoQueryEncoder(nn.Module):
    def __init__(self, config, video_modality,
                 visual_dim=4352, text_dim= 768,
                 query_dim=768, hidden_dim = 768,split_num=100,):
        super(VideoQueryEncoder, self).__init__()
        self.use_sub = len(video_modality) > 1
        if self.use_sub:
            self.videoEncoder = TwoModalEncoder(config=config.bert_config,
                                                img_dim = visual_dim,
                                                text_dim = text_dim ,
                                                hidden_dim = hidden_dim,
                                                split_num = split_num
                                                )
        else:
            self.videoEncoder = OneModalEncoder(config=config.bert_config,
                                                input_dim = visual_dim,
                                                hidden_dim = hidden_dim,
                                                )

        self.queryEncoder = OneModalEncoder(config=config.query_bert_config,
                                            input_dim= query_dim,
                                            hidden_dim=hidden_dim,
                                            )

    def forward_repr_query(self, batch):

        query_output = self.queryEncoder(
            features=batch["query"]["feat"],
            position_ids=batch["query"]["feat_pos_id"],
            token_type_ids=batch["query"]["feat_token_id"],
            attention_mask=batch["query"]["feat_mask"]
        )

        return query_output

    def forward_repr_video(self,batch):
        video_output = dict()

        if len(batch["visual"]["feat"].size()) == 4:
            bsz, num_video = batch["visual"]["feat"].size()[:2]
            for key in batch.keys():
                if key in ["visual", "sub"]:
                    for key_2 in batch[key]:
                        if key_2 in ["feat", "feat_mask", "feat_pos_id", "feat_token_id"]:
                            shape_list = batch[key][key_2].size()[2:]
                            batch[key][key_2] = batch[key][key_2].view((bsz * num_video,) + shape_list)


        if self.use_sub:
            video_output["visual"], video_output["sub"] = self.videoEncoder(
                visual_features=batch["visual"]["feat"],
                visual_position_ids=batch["visual"]["feat_pos_id"],
                visual_token_type_ids=batch["visual"]["feat_token_id"],
                visual_attention_mask=batch["visual"]["feat_mask"],
                text_features=batch["sub"]["feat"],
                text_position_ids=batch["sub"]["feat_pos_id"],
                text_token_type_ids=batch["sub"]["feat_token_id"],
                text_attention_mask=batch["sub"]["feat_mask"]
            )
        else:
            video_output["visual"] = self.videoEncoder(
                features=batch["visual"]["feat"],
                position_ids=batch["visual"]["feat_pos_id"],
                token_type_ids=batch["visual"]["feat_token_id"],
                attention_mask=batch["visual"]["feat_mask"]
            )

        return video_output


    def forward_repr_both(self, batch):
        video_output = self.forward_repr_video(batch)
        query_output = self.forward_repr_query(batch)

        return {"video_feat": video_output,
                "query_feat": query_output}

    def forward(self,batch,task="repr_both"):

        if task == "repr_both":
            return self.forward_repr_both(batch)
        elif task == "repr_video":
            return self.forward_repr_video(batch)
        elif task == "repr_query":
            return self.forward_repr_query(batch)


class QueryWeightEncoder(nn.Module):
    """
        Query Weight Encoder
        Using NetVLAD to aggreate contextual query features
        Using FC + Softmax to get fusion weights for each modality
    """
    def __init__(self, config, video_modality):
        super(QueryWeightEncoder, self).__init__()

        ##NetVLAD
        self.text_pooling = NetVLAD(feature_size=config.hidden_size,cluster_size=config.text_cluster)
        self.moe_txt_dropout = nn.Dropout(config.moe_dropout_prob)

        ##FC
        self.moe_fc_txt = nn.Linear(
            in_features=self.text_pooling.out_dim,
            out_features=len(video_modality),
            bias=False)

        self.video_modality = video_modality

    def forward(self, query_feat):
        ##NetVLAD
        pooled_text = self.text_pooling(query_feat)
        pooled_text = self.moe_txt_dropout(pooled_text)

        ##FC + Softmax
        moe_weights = self.moe_fc_txt(pooled_text)
        softmax_moe_weights = F.softmax(moe_weights, dim=1)


        moe_weights_dict = dict()
        for modality, moe_weight in zip(self.video_modality, torch.split(softmax_moe_weights, 1, dim=1)):
            moe_weights_dict[modality] = moe_weight.squeeze(1)

        return  moe_weights_dict




