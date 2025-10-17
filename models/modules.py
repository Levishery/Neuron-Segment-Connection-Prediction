import pdb

import torch
import torch.nn as nn
import numpy as np
import sys
import os
import math

import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
# import pointnet2_utils
from growvector.model_util_vector import VectorDatasetConfig as DC


class PointsObjClsModule(nn.Module):
    def __init__(self, seed_feature_dim):
        """ object candidate point prediction from seed point features.

        Args:
            seed_feature_dim: int
                number of channels of seed point features
        """
        super().__init__()
        self.in_dim = seed_feature_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        self.conv3 = torch.nn.Conv1d(self.in_dim, 1, 1)

    def forward(self, seed_features):
        """ Forward pass.

        Arguments:
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            logits: (batch_size, 1, num_seed)
        """
        net = F.relu(self.bn1(self.conv1(seed_features)))
        net = F.relu(self.bn2(self.conv2(net)))
        logits = self.conv3(net)  # (batch_size, 1, num_seed)

        return logits


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


# class FPSModule(nn.Module):
#     def __init__(self, num_proposal):
#         super().__init__()
#         self.num_proposal = num_proposal

#     def forward(self, xyz, features):
#         """
#         Args:
#             xyz: (B,K,3)
#             features: (B,C,K)
#         """
#         # Farthest point sampling (FPS)
#         sample_inds = pointnet2_utils.furthest_point_sample(xyz, self.num_proposal)
#         xyz_flipped = xyz.transpose(1, 2).contiguous()
#         new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
#         new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

#         return new_xyz, new_features, sample_inds

def furthest_point_sample(xyz, npoint):
    """
    Farthest Point Sampling (FPS) with PyTorch.
    Args:
        xyz: (B, N, 3) input points
        npoint: int, number of points to sample
    Returns:
        centroids: (B, npoint) sampled point indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)  # (B, npoint)
    distance = torch.full((B, N), 1e10, device=device)  # 初始化距离
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)  # 随机初始点
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid_xyz = xyz[batch_indices, farthest, :].view(B, 1, 3)  # (B,1,3)
        dist = torch.sum((xyz - centroid_xyz) ** 2, -1)  # (B,N)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # (B,)

    return centroids


class FPSModule(nn.Module):
    def __init__(self, num_proposal):
        super().__init__()
        self.num_proposal = num_proposal

    def forward(self, xyz, features):
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
        Returns:
            new_xyz: (B, num_proposal, 3)
            new_features: (B, C, num_proposal)
            sample_inds: (B, num_proposal)
        """
        # ---- Farthest Point Sampling ----
        sample_inds = furthest_point_sample(xyz, self.num_proposal)  # (B, num_proposal)
        sample_inds = sample_inds.long()  # 确保索引是 int64

        # ---- gather xyz ----
        new_xyz = torch.gather(
            xyz,
            1,
            sample_inds.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous()

        # ---- gather features ----
        new_features = torch.gather(
            features,
            2,
            sample_inds.unsqueeze(1).expand(-1, features.shape[1], -1)
        ).contiguous()

        return new_xyz, new_features, sample_inds



# class GeneralSamplingModule(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, xyz, features, sample_inds):
#         """
#         Args:
#             xyz: (B,K,3)
#             features: (B,C,K)
#         """
#         xyz_flipped = xyz.transpose(1, 2).contiguous()
#         new_xyz = pointnet2_utils.gather_operation(xyz_flipped, sample_inds).transpose(1, 2).contiguous()
#         new_features = pointnet2_utils.gather_operation(features, sample_inds).contiguous()

#         return new_xyz, new_features, sample_inds

class GeneralSamplingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, features, sample_inds):
        """
        Args:
            xyz: (B, K, 3)
            features: (B, C, K)
            sample_inds: (B, npoint)
        Returns:
            new_xyz: (B, npoint, 3)
            new_features: (B, C, npoint)
            sample_inds: (B, npoint)
        """
        sample_inds = sample_inds.long()  # 确保索引是 int64

        # new_xyz
        new_xyz = torch.gather(
            xyz,
            1,
            sample_inds.unsqueeze(-1).expand(-1, -1, 3)
        ).contiguous()

        # new_features
        new_features = torch.gather(
            features,
            2,
            sample_inds.unsqueeze(1).expand(-1, features.shape[1], -1)
        ).contiguous()

        return new_xyz, new_features, sample_inds


class PredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster,
                 mean_size_arr, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_class_head = torch.nn.Conv1d(seed_feat_dim, num_size_cluster, 1)
        self.size_residual_head = torch.nn.Conv1d(seed_feat_dim, num_size_cluster * 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(seed_feat_dim, self.num_class, 1)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        mean_size_arr = torch.from_numpy(self.mean_size_arr.astype(np.float32)).cuda()  # (num_size_cluster, 3)
        mean_size_arr = mean_size_arr.unsqueeze(0).unsqueeze(0)  # (1, 1, num_size_cluster, 3)
        size_scores = self.size_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_size_cluster)
        size_residuals_normalized = self.size_residual_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, self.num_size_cluster, 3])  # (batch_size, num_proposal, num_size_cluster, 3)
        size_residuals = size_residuals_normalized * mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        size_recover = size_residuals + mean_size_arr  # (batch_size, num_proposal, num_size_cluster, 3)
        pred_size_class = torch.argmax(size_scores, -1)  # batch_size, num_proposal
        pred_size_class = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        pred_size = torch.gather(size_recover, 2, pred_size_class)  # batch_size, num_proposal, 1, 3
        pred_size = pred_size.squeeze_(2)  # batch_size, num_proposal, 3

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}size_scores'] = size_scores
        end_points[f'{prefix}size_residuals_normalized'] = size_residuals_normalized
        end_points[f'{prefix}size_residuals'] = size_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        # # used to check bbox size
        # l = pred_size[:, :, 0]
        # h = pred_size[:, :, 1]
        # w = pred_size[:, :, 2]
        # x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], -1)  # N Pq 8
        # y_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], -1)  # N Pq 8
        # z_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], -1)  # N Pq 8
        # corners = torch.stack([x_corners, y_corners, z_corners], -1)  # N Pq 8 3
        # bbox = center.unsqueeze(2) + corners
        # end_points[f'{prefix}bbox_check'] = bbox
        return center, pred_size


class ClsAgnosticPredictHead(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_proposal, seed_feat_dim=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.heading_class_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.heading_residual_head = torch.nn.Conv1d(seed_feat_dim, num_heading_bin, 1)
        self.size_pred_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.sem_cls_scores_head = torch.nn.Conv1d(seed_feat_dim, self.num_class, 1)

    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        heading_scores = self.heading_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        heading_residuals_normalized = self.heading_residual_head(net).transpose(2, 1)
        heading_residuals = heading_residuals_normalized * (np.pi / self.num_heading_bin)

        # size
        pred_size = self.size_pred_head(net).transpose(2, 1).view(
            [batch_size, num_proposal, 3])  # (batch_size, num_proposal, 3)

        # class
        sem_cls_scores = self.sem_cls_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_class)

        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}heading_scores'] = heading_scores
        end_points[f'{prefix}heading_residuals_normalized'] = heading_residuals_normalized
        end_points[f'{prefix}heading_residuals'] = heading_residuals
        end_points[f'{prefix}pred_size'] = pred_size
        end_points[f'{prefix}sem_cls_scores'] = sem_cls_scores

        return center, pred_size


class VectorPredictHead(nn.Module):
    def __init__(self, num_yaw_bin, num_pitch_bin, num_proposal, seed_feat_dim=256, terminal_classification=False):
        super().__init__()

        self.num_yaw_bin = num_yaw_bin
        self.num_pitch_bin = num_pitch_bin
        self.num_proposal = num_proposal
        self.seed_feat_dim = seed_feat_dim
        self.terminal_classification = terminal_classification

        # Object proposal/detection
        # Objectness scores (1), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(seed_feat_dim)
        self.conv2 = torch.nn.Conv1d(seed_feat_dim, seed_feat_dim, 1)
        self.bn2 = torch.nn.BatchNorm1d(seed_feat_dim)

        self.objectness_scores_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        if self.terminal_classification:
            self.terminal_classification_head = torch.nn.Conv1d(seed_feat_dim, 1, 1)
        self.center_residual_head = torch.nn.Conv1d(seed_feat_dim, 3, 1)
        self.yaw_class_head = torch.nn.Conv1d(seed_feat_dim, num_yaw_bin, 1)
        self.yaw_residual_head = torch.nn.Conv1d(seed_feat_dim, num_yaw_bin, 1)
        self.pitch_class_head = torch.nn.Conv1d(seed_feat_dim, num_pitch_bin, 1)
        self.pitch_residual_head = torch.nn.Conv1d(seed_feat_dim, num_pitch_bin, 1)


    def forward(self, features, base_xyz, end_points, prefix=''):
        """
        Args:
            features: (B,C,num_proposal)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        batch_size = features.shape[0]
        num_proposal = features.shape[-1]
        net = F.relu(self.bn1(self.conv1(features)))
        net = F.relu(self.bn2(self.conv2(net)))
        # objectness
        objectness_scores = self.objectness_scores_head(net).transpose(2, 1)  # (batch_size, num_proposal, 1)
        if self.terminal_classification:
            terminal_scores = self.terminal_classification_head(net).transpose(2, 1)
            end_points[f'{prefix}terminal_scores'] = terminal_scores
        # center
        center_residual = self.center_residual_head(net).transpose(2, 1)  # (batch_size, num_proposal, 3)
        center = base_xyz + center_residual  # (batch_size, num_proposal, 3)

        # heading
        yaw_scores = self.yaw_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        yaw_residuals_normalized = self.yaw_residual_head(net).transpose(2, 1)
        yaw_residuals = yaw_residuals_normalized * (np.pi / self.num_yaw_bin)
        pitch_scores = self.pitch_class_head(net).transpose(2, 1)  # (batch_size, num_proposal, num_heading_bin)
        # (batch_size, num_proposal, num_heading_bin) (should be -1 to 1)
        pitch_residuals_normalized = self.pitch_residual_head(net).transpose(2, 1)
        pitch_residuals = pitch_residuals_normalized * (np.pi / (2 * self.num_pitch_bin))

        # # get yaw and pitch value for pos embed
        # pred_yaw_class = torch.argmax(yaw_scores, -1)  # batch_size, num_proposal
        # pred_yaw_class = pred_yaw_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        # pred_size = torch.gather(size_recover, 2, pred_yaw_class)  # batch_size, num_proposal, 1, 3
        # pred_size = pred_size.squeeze_(2)  # batch_size, num_proposal, 3
        # DC.class2angle()


        end_points[f'{prefix}base_xyz'] = base_xyz
        end_points[f'{prefix}objectness_scores'] = objectness_scores
        end_points[f'{prefix}center'] = center
        end_points[f'{prefix}yaw_scores'] = yaw_scores
        end_points[f'{prefix}yaw_residuals_normalized'] = yaw_residuals_normalized
        end_points[f'{prefix}yaw_residuals'] = yaw_residuals
        end_points[f'{prefix}pitch_scores'] = pitch_scores
        end_points[f'{prefix}pitch_residuals_normalized'] = pitch_residuals_normalized
        end_points[f'{prefix}pitch_residuals'] = pitch_residuals

        return center, objectness_scores


class PositionEmbeddingLearned3D(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, channels=128, image_patch_size=(2, 5, 5)):
        super().__init__()
        self.orig_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        self.row_embed = nn.Embedding(int(np.ceil(127 / image_patch_size[1])), channels)
        self.col_embed = nn.Embedding(int(np.ceil(127 / image_patch_size[2])), channels)
        self.depth_embed = nn.Embedding(int(np.ceil(19 / image_patch_size[0])), channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.depth_embed.weight)

    def forward(self, x):
        # B*C*H*W*D
        d, h, w = x.shape[-3:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        k = torch.arange(d, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.depth_embed(k)
        # pos = torch.cat([
        #     x_emb.unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1),
        #     y_emb.unsqueeze(1).unsqueeze(1).repeat(1, w, d, 1),
        #     z_emb.unsqueeze(0).unsqueeze(0).repeat(h, w, 1, 1),
        # ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        pos = torch.cat([
            x_emb.unsqueeze(0).unsqueeze(0).repeat(d, h, 1, 1),
            y_emb.unsqueeze(0).unsqueeze(2).repeat(d, 1, w, 1),
            z_emb.unsqueeze(1).unsqueeze(1).repeat(1, h, w, 1),
        ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        # pdb.set_trace()
        return pos[:, :self.orig_channels, ...]


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on 3D data.
    """

    def __init__(self, channels=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.orig_channels = channels
        self.channels = int(np.ceil(channels / 6) * 2)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, src):
        mask = torch.zeros_like(src[:, 0], dtype=torch.bool)
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            x_embed = (x_embed - 0.5) / (x_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = (y_embed - 0.5) / (y_embed[:, :, -1:, :] + eps) * self.scale
            z_embed = (z_embed - 0.5) / (z_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.channels, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.channels)  # check 6 or something else?

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(4)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(4)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=4).flatten(4)
        pos = torch.cat((pos_y, pos_x, pos_z), dim=4).permute(0, 4, 1, 2, 3)
        return pos[:, :self.orig_channels, ...]


class PatchEmbed(nn.Module):
    """
    Patch embedding block
    """

    def __init__(self, patch_size, in_chans=16, embed_dim=48, norm_layer=nn.LayerNorm, ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
        # pdb.set_trace()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
            x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
        return x