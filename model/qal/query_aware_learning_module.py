import torch
from torch import nn

import logging
logger = logging.getLogger(__name__)

try:
  import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
  BertLayerNorm = torch.nn.LayerNorm

from utils.model_utils import mask_logits
import torch.nn.functional as F


class BiDirectionalAttention(nn.Module):
    """
         Bi-directional attention flow
         Perform query-to-video attention (Q2V) and video-to-query attention (V2Q)
         Append QDF features with a set of query-aware features to form QAL feature
    """

    def __init__(self, video_dim):
        super(BiDirectionalAttention, self).__init__()
        ## Core Attention for query-aware feature learining
        self.similarity_weight = nn.Linear(video_dim * 3, 1, bias=False)


    def forward(self, QDF_emb, query_emb,video_mask, query_mask):
        """
        Inputs:
        :param QDF_emb: (batch, L_v, feat_size)
        :param query_emb: (batch, L_q, feat_size)
        :param video_mask: (batch, L_v)
        :param query_mask: (batch, L_q)
        Return:
        QAL: (batch, L_v, feat_size*4)
        """

        ## CREATE SIMILARITY MATRIX
        video_len = QDF_emb.size()[1]
        query_len = query_emb.size()[1]

        _QDF_emb = QDF_emb.unsqueeze(2).repeat(1, 1, query_len, 1)
        # [bs, video_len, 1, feat_size] => [bs, video_len, query_len, feat_size]

        _query_emb = query_emb.unsqueeze(1).repeat(1, video_len, 1, 1)
        # [bs, 1, query_len, feat_size] => [bs, video_len, query_len, feat_size]

        elementwise_prod = torch.mul(_QDF_emb, _query_emb)
        # [bs, video_len, query_len, feat_size]

        alpha = torch.cat([_QDF_emb, _query_emb, elementwise_prod], dim=3)
        # [bs, video_len, query_len, feat_size*3]

        similarity_matrix = self.similarity_weight(alpha).view(-1, video_len, query_len)

        similarity_matrix_mask = torch.einsum("bn,bm->bnm", video_mask, query_mask)
        # [bs, video_len, query_len]

        ## CALCULATE Video2Query ATTENTION

        a = F.softmax(mask_logits(similarity_matrix,
                                  similarity_matrix_mask), dim=-1)
        # [bs, video_len, query_len]

        V2Q = torch.bmm(a, query_emb)
        # [bs] ([video_len, query_len] X [query_len, feat_size]) => [bs, video_len, feat_size]

        ## CALCULATE Query2Video ATTENTION

        b = F.softmax(torch.max(mask_logits(similarity_matrix, similarity_matrix_mask), 2)[0], dim=-1)
        # [bs, video_len]

        b = b.unsqueeze(1)
        # [bs, 1, video_len]

        Q2V = torch.bmm(b, QDF_emb)
        # [bs] ([bs, 1, video_len] X [bs, video_len, feat_size]) => [bs, 1, feat_size]

        Q2V = Q2V.repeat(1, video_len, 1)
        # [bs, video_len, feat_size]

        ## Append QDF_emb with three query-aware features

        QAL = torch.cat([QDF_emb, V2Q,
                         torch.mul(QDF_emb, V2Q),
                         torch.mul(QDF_emb, Q2V)], dim=2)

        # [bs, video_len, feat_size*4]

        return QAL