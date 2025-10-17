import pdb

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from multi_head_attention import MultiheadAttention, MSDeformAttn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, image_posembed=None, use_image_feature=False, image_nhead=6, return_offset=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.use_image_feature = use_image_feature
        self.image_posembed = None
        self.return_offset = False
        if use_image_feature:
            self.multihead_attn_image = MSDeformAttn(d_model, n_levels=1, n_heads=image_nhead, n_points=4, return_offset=return_offset)
            self.norm_image = nn.LayerNorm(d_model)
            self.dropout_image = nn.Dropout(dropout)
            self.image_posembed = image_posembed
            self.return_offset = return_offset
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, reference_points=None, image_key=None, input_spatial_shapes=None,
                input_level_start_index=None, input_padding_mask=None, query_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]

        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None
        if self.image_posembed is not None:
            image_pos_embed = self.image_posembed(image_key)
        else:
            image_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        q = k = v = self.with_pos_embed(query, query_pos_embed)
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed))[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        if self.use_image_feature:
            query2, sampling_offset = self.multihead_attn_image(query=self.with_pos_embed(query, query_pos_embed),
                                               reference_points=reference_points,
                                               input=self.with_pos_embed(image_key, image_pos_embed),
                                               input_spatial_shapes=input_spatial_shapes,
                                               input_level_start_index=input_level_start_index,
                                               query_mask=query_mask,
                                               input_padding_mask=input_padding_mask)
            query = query + self.dropout_image(query2)
            query = self.norm_image(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)

        if self.return_offset:
            return query, sampling_offset
        else:
            return query, None


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


