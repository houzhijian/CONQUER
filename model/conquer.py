import torch
import torch.nn as nn
from model.backbone.encoder import VideoQueryEncoder, QueryWeightEncoder
from model.qal.query_aware_learning_module import BiDirectionalAttention
from model.layers import FCPlusTransformer#,MomentLocalizationHead
from model.head.ml_head import MomentLocalizationHead
from model.head.vs_head import VideoScoringHead

import logging
logger = logging.getLogger(__name__)


class CONQUER(nn.Module):
    def __init__(self, config,
                 visual_dim = 4352,
                 text_dim = 768,
                 query_dim = 768,
                 hidden_dim = 768,
                 video_len = 100,
                 ctx_mode = "visual_sub",
                 lw_st_ed = 0.01,
                 lw_video_ce = 0.05,
                 similarity_measure="general",
                 use_debug=False,
                 no_output_moe_weight=False):

        super(CONQUER, self).__init__()
        self.config = config

        #  related configs
        self.lw_st_ed = lw_st_ed
        self.lw_video_ce = lw_video_ce
        self.similarity_measure = similarity_measure

        self.video_modality = ctx_mode.split("_")
        logger.info("video modality : %s" % self.video_modality)
        self.output_moe_weight = not no_output_moe_weight

        hidden_dim = hidden_dim
        base_bert_layer_config = config.bert_config

        ## Backbone encoder
        self.encoder = VideoQueryEncoder(config,video_modality=self.video_modality,
                                         visual_dim=visual_dim,text_dim=text_dim,query_dim=query_dim,
                                         hidden_dim=hidden_dim,split_num=video_len)

        if self.output_moe_weight and len(self.video_modality) > 1:
            self.query_weight = QueryWeightEncoder(config.netvlad_config,video_modality=self.video_modality)

        ## Query_aware_feature_learning Module
        self.query_aware_feature_learning_layer = BiDirectionalAttention(hidden_dim)

        ## Shared transformer for both moment localization and video scoring heads
        self.contextual_QAL_feature_learning = FCPlusTransformer(base_bert_layer_config,hidden_dim * 4)

        ## Moment_localization_head
        self.moment_localization_head = MomentLocalizationHead(config.moment_localization_config,base_bert_layer_config,hidden_dim)
        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")

        ## Optional video_scoring_head
        if self.similarity_measure == "exclusive":
            self.video_scoring_head = VideoScoringHead(config.video_scoring_config,base_bert_layer_config,hidden_dim)
            self.score_ce = nn.CrossEntropyLoss(reduction="mean")

        self.debug_model = use_debug
        if self.debug_model:
            logger.setLevel(level=logging.DEBUG)

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                #print("nn.Linear, nn.Embedding: ", module)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)


    def compute_final_score(self,score_dict,moe_weights=None):

        sample_key = list(score_dict.keys())[0]
        final_query_context_scores = torch.zeros_like(score_dict[sample_key])
        shape_size = len(score_dict[sample_key].shape)
        if moe_weights is not None:
            for mod in self.video_modality:
                if shape_size == 2:
                    final_query_context_scores += torch.einsum("nm,n->nm", score_dict[mod], moe_weights[mod])
                elif shape_size == 3:
                    final_query_context_scores += torch.einsum("nlm,n->nlm", score_dict[mod], moe_weights[mod])
        else:
            for mod in self.video_modality:
                final_query_context_scores += torch.div(score_dict[mod], len(self.video_modality))

        return final_query_context_scores


    def get_pred_from_raw_query(self, batch):

        ## Extract query and video feature through MMT backbone
        _query_feature = self.encoder(batch, task="repr_query") #Widehat_Q

        _video_feature_dict = self.encoder(batch, task="repr_video") #Widehat_V and #Widehat_S

        ## Shared normalization technique
        ## Use the same query feature for shared_video_num times
        sample_key = list(_video_feature_dict.keys())[0]
        query_batch = _query_feature.size()[0]
        video_batch, video_len = _video_feature_dict[sample_key].size()[:2]
        shared_video_num = int(video_batch / query_batch)

        query_feature = torch.repeat_interleave(_query_feature, shared_video_num, dim=0)
        query_mask = torch.repeat_interleave(batch["query"]["feat_mask"], shared_video_num, dim=0)


        ## Compute Query Dependent Fusion video feature
        if self.output_moe_weight and len(self.video_modality) > 1:
            moe_weights_dict = self.query_weight(query_feature)
            QDF_feature = self.compute_final_score(_video_feature_dict, moe_weights_dict)
        else:
            QDF_feature = self.compute_final_score(_video_feature_dict,None)

        video_mask = batch["visual"]["feat_mask"]


        ## Compute Query Aware Learning video feature
        QAL_feature = self.query_aware_feature_learning_layer(QDF_feature, query_feature,
                                video_mask,query_mask)

        ## Contextualize QAL features
        Contextual_QAL  = self.contextual_QAL_feature_learning(
            features=QAL_feature,
            feat_mask=video_mask)

        G = torch.cat([QAL_feature,Contextual_QAL], dim=2)

        ## Moment localization head
        begin_score_distribution , end_score_distribution = self.moment_localization_head(G,Contextual_QAL,video_mask)
        begin_score_distribution = begin_score_distribution.view(query_batch, shared_video_num, video_len)
        end_score_distribution = end_score_distribution.view(query_batch, shared_video_num, video_len)

        ## Optional video scoring head
        video_similarity_score = None
        if self.similarity_measure == "exclusive":
            video_similarity_score = self.video_scoring_head(G,video_mask)
            video_similarity_score = video_similarity_score.view(query_batch, shared_video_num)

        return video_similarity_score, begin_score_distribution , end_score_distribution


    def get_moment_loss_share_norm(self, begin_score_distribution, end_score_distribution ,st_ed_indices):

        bs , shared_video_num , video_len = begin_score_distribution.size()

        begin_score_distribution = begin_score_distribution.view(bs,-1)
        end_score_distribution = end_score_distribution.view(bs,-1)

        loss_st = self.temporal_criterion(begin_score_distribution, st_ed_indices[:, 0])
        loss_ed = self.temporal_criterion(end_score_distribution, st_ed_indices[:, 1])
        moment_ce_loss = loss_st + loss_ed

        return moment_ce_loss


    def forward(self,batch):

        video_similarity_score, begin_score_distribution , end_score_distribution = \
            self.get_pred_from_raw_query(batch)

        moment_ce_loss, video_ce_loss = 0, 0

        # moment cross-entropy loss
        # if neg_video_num = 0, we do not sample negative videos
        # the softmax operator is performed only for the ground-truth video
        # which mean to not use shared normalization training objective
        moment_ce_loss = self.get_moment_loss_share_norm(
            begin_score_distribution, end_score_distribution, batch["st_ed_indices"])
        moment_ce_loss = self.lw_st_ed * moment_ce_loss

        if self.similarity_measure == "exclusive":
            ce_label = batch["st_ed_indices"].new_zeros(video_similarity_score.size()[0])
            video_ce_loss = self.score_ce(video_similarity_score, ce_label)
            video_ce_loss = self.lw_video_ce*video_ce_loss


        loss = moment_ce_loss + video_ce_loss
        return loss, {"moment_ce_loss": float(moment_ce_loss),
                      "video_ce_loss": float(video_ce_loss),
                      "loss_overall": float(loss)}




