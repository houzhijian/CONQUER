import torch
from torch import nn
import logging
logger = logging.getLogger(__name__)


from model.layers import FCPlusTransformer, ConvSE


class MomentLocalizationHead(nn.Module):
    """
        Moment localization head model
    """

    def __init__(self, config,base_bert_layer_config,hidden_dim):
        super(MomentLocalizationHead, self).__init__()

        base_bert_layer_config = base_bert_layer_config
        hidden_dim = hidden_dim

        self.begin_feature_modeling = FCPlusTransformer(base_bert_layer_config, hidden_dim * 5)

        self.end_feature_modeling = FCPlusTransformer(base_bert_layer_config, hidden_dim * 2)

        self.begin_score_modeling = ConvSE(config)
        self.end_score_modeling = ConvSE(config)

    def forward(self, G, Contextual_QAL, video_mask):
        """
        Inputs:
            :param contextual_qal_features: (batch, feat_size, L_v)
            :param video_mask: (batch, L_v)
        Return:
             score: (begin or end) score distribution
        """
        ## OUTPUT LAYER
        begin_features = self.begin_feature_modeling(
            features=G,
            feat_mask=video_mask)

        end_features = self.end_feature_modeling(
            features=torch.cat([Contextual_QAL, begin_features], dim=2),
            feat_mask=video_mask)

        ## Un-normalized
        begin_input_feature = torch.transpose(begin_features, 1, 2)
        end_input_feature = torch.transpose(end_features, 1, 2)

        begin_score_distribution = self.begin_score_modeling(
            contextual_qal_features=begin_input_feature,
            video_mask=video_mask,
        )

        end_score_distribution = self.end_score_modeling(
            contextual_qal_features=end_input_feature,
            video_mask=video_mask,
        )

        return begin_score_distribution , end_score_distribution


