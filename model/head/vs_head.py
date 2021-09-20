import torch
from torch import nn

import logging
logger = logging.getLogger(__name__)

from model.layers import FCPlusTransformer

class VideoScoringHead(nn.Module):
    """
         Video Scoring Head
    """

    def __init__(self, config,base_bert_layer_config,hidden_dim):
        super(VideoScoringHead, self).__init__()

        base_bert_layer_config = base_bert_layer_config
        hidden_dim = hidden_dim


        self.video_feature_modeling = FCPlusTransformer(base_bert_layer_config, hidden_dim * 5)

        self.video_score_predictor = nn.Sequential(
            nn.Linear(**config.linear_1_cfg),
            nn.ReLU(),
            nn.Linear(**config.linear_2_cfg)
        )


    def forward(self, G, video_mask):


        ## Contexual_QAL_feature for video scoring
        R = self.video_feature_modeling(
            features=G,
            feat_mask=video_mask)

        holistic_video_feature, _ = torch.max(R, dim=1)

        video_similarity_score = self.video_score_predictor(holistic_video_feature.squeeze(1)) # r

        return video_similarity_score