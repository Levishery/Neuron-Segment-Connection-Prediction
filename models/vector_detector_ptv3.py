import pdb

import numpy
import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import DBSCAN
import pickle
import warnings
from cloudvolume import CloudVolume
import concurrent.futures
import tifffile as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from model import PointTransformerV3
from .transformer import TransformerDecoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, VectorPredictHead, \
    PositionEmbeddingLearned3D, PositionEmbeddingSine3D, PatchEmbed
# from .modules import PointsObjClsModule, FPSModule, PositionEmbeddingLearned, VectorPredictHead, \
#     PositionEmbeddingLearned3D, PositionEmbeddingSine3D, PatchEmbed
from .imagemodel import *


class GroupFreeVectorDetector_ptv3(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_yaw_bin, num_pitch_bin,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned',
                 image_position_embedding='xyz_learned', image_model_path=None, token_patch_size=(2, 5, 5), image_batch_size=4, terminal_classification=False, dbscan_eps=300):
        super().__init__()

        self.num_yaw_bin = num_yaw_bin
        self.num_pitch_bin = num_pitch_bin
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.terminal_classification = terminal_classification
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.image_position_embedding = image_position_embedding
        self.use_image_feature = True if image_model_path is not None else False
        self.return_offset = True
        if self.use_image_feature:
            print('load cloud volume data')
            self.image_volume = CloudVolume('file:///h3cstore_nt/fafbv14/fafb_v14_orig_sharded', mip=2, cache=False) if image_model_path is not None else None
            self.token_patch_size = token_patch_size
            self.sample_pool = concurrent.futures.ThreadPoolExecutor(max_workers=16)
            self.image_batch_size = image_batch_size
            self.dbscan_eps = dbscan_eps
        self.grid_size = 0.02
        # pdb.set_trace()

        # Backbone point feature learning
        # Input point number is not constrained. It's possible to remove downsample.
        self.backbone_net = PointTransformerV3(in_channels=self.input_feature_dim + 3, enable_flash=False, 
                                                    dec_depths=(2, 2, 2, 2),
                                                    dec_channels=(288, 64, 128, 256),
                                                    dec_num_head=(4, 4, 8, 16),
                                                    dec_patch_size=(1024, 1024, 1024, 1024),)

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        self.proposal_head = VectorPredictHead(self.num_yaw_bin, self.num_pitch_bin, num_proposal, 288, terminal_classification=self.terminal_classification)

        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)
        if self.use_image_feature:
            self.decoder_image_key_proj = PatchEmbed(patch_size=self.token_patch_size, in_chans=16, embed_dim=288)
            self.dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=3)
            with open(image_model_path, 'rb') as f:
                [self.cfg_image_model, self.ckpt_image_model] = pickle.load(f)
            self.cfg_image_model.defrost()
            print(f"{image_model_path} loaded successfully !!!")
            self.image_model = build_model(self.cfg_image_model, device='cuda')
            self.load_image_model(self.ckpt_image_model)
            self.image_model.eval()
            self.patch_size = np.array(self.cfg_image_model.MODEL.INPUT_SIZE)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            # xyz+yaw+pitch
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(5, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        if self.image_position_embedding == 'none':
            self.decoder_image_posembeds = [None for i in range(num_decoder_layers)]
        elif self.image_position_embedding == 'xyz_learned':
            self.decoder_image_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_image_posembeds.append(PositionEmbeddingLearned3D(288))
        elif self_position_embedding == 'sine':
            self.decoder_image_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_image_posembeds.append(PositionEmbeddingSine3D(288))
        else:
            raise NotImplementedError(f"image_position_embedding not supported {self.image_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i], image_posembed=self.decoder_image_posembeds[i],
                    use_image_feature=self.use_image_feature, return_offset=self.return_offset
                ))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(VectorPredictHead(self.num_yaw_bin, self.num_pitch_bin, num_proposal, 288, terminal_classification=self.terminal_classification))

        # Init
        self.init_weights()
        self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward(self, inputs, return_offset=False):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        """
        For ptv3 a data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        input_ptv3 = {}
        B, N, C = inputs['point_clouds'].shape
        input_ptv3['feat'] = inputs['point_clouds'].reshape(B*N, C)
        input_ptv3['coord'] = input_ptv3['feat'][:,:3]
        input_ptv3['grid_size'] = self.grid_size
        input_ptv3['batch'] = (torch.arange(B * N, device=inputs['point_clouds'].device) // N)
        out_feature = self.backbone_net(input_ptv3)
        # end_points = self.backbone_net(inputs['point_clouds'], end_points)

        # Query Points Generation
        # import pdb; pdb.set_trace()
        # for pointnet, the number of seed is 1024(2048/2); However, for ptv3, the pooling produce different point numbers.
        # Therefore we use seed number of 2048 for ptv3 for simplicity. 
        c_out = out_feature.feat.shape[1]
        points_xyz = out_feature.coord.reshape(B,N,3)
        points_features = out_feature.feat.reshape(B,N,c_out).transpose(1,2)
        xyz = out_feature.coord.reshape(B,N,3)
        features = out_feature.feat.reshape(B,N,c_out).transpose(1,2).contiguous()
        end_points['seed_inds'] = torch.arange(N, device=inputs['point_clouds'].device).repeat(B, 1)
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError

        # Proposal
        proposal_center, objectness_scores = self.proposal_head(cluster_feature,
                                                                base_xyz=cluster_xyz,
                                                                end_points=end_points,
                                                                prefix='proposal_')  # N num_proposal 3

        # sample image feature from proposal centers
        if self.use_image_feature:
            proposal_cluster_offsets, proposal_cluster_labels, proposal_cluster_centers, cluster_label_centers = \
                self.get_image_sample_positions(proposal_center, inputs['centroid'], inputs['scale'])
            # pdb.set_trace()
            image_patches = self.sample_image_patches_parallel(cluster_label_centers)
            image_features = self.get_image_features(image_patches)
            # pdb.set_trace()

        base_xyz = proposal_center.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
            if self.use_image_feature:
                # pdb.set_trace()
                image_key = self.decoder_image_key_proj(image_features)

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            # Transformer Decoder Layer
            if self.use_image_feature:
                query_cluster_offsets, query_cluster_labels = self.get_query_offsets(proposal_cluster_offsets,
                                                                                     proposal_cluster_labels, base_xyz,
                                                                                     proposal_center.detach().clone(),
                                                                                     inputs['scale'].cpu())
                reference_points, input_spatial_shapes, level_start_index, query_mask = self.get_reference_points(
                    query_cluster_offsets.to(dtype=torch.float32), query_cluster_labels, image_key)
                query, sampling_offset = self.decoder[i](query, key, query_pos, key_pos, reference_points=reference_points,
                                        image_key=image_key, input_spatial_shapes=input_spatial_shapes,
                                        input_level_start_index=level_start_index, query_mask=query_mask, input_padding_mask=None)
                end_points[f'{prefix}sampling_offset'] = sampling_offset
            else:
                query, _ = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, objectness_scores = self.prediction_heads[i](query,
                                                                   base_xyz=cluster_xyz,
                                                                   end_points=end_points,
                                                                   prefix=prefix)

            base_xyz = base_xyz.detach().clone()

        if return_offset:
            self.visualize_offsets(proposal_cluster_labels, proposal_cluster_centers, query_cluster_offsets, image_patches, end_points, inputs['centroid'], inputs['scale'])

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def get_image_sample_positions(self, proposal_centers, centroids, scales):
        # get image sampling positions by dbscan clustering
        proposal_cluster_centers = []
        cluster_labels_centers = []
        proposal_cluster_offsets = []
        proposal_cluster_labels = []
        for b in range(proposal_centers.shape[0]):
            proposal_center = proposal_centers[b, :, :].detach().cpu().numpy() * scales[b].cpu().numpy() + centroids[
                b].cpu().numpy()
            proposal_cluster_center = np.zeros(proposal_center.shape)
            if self.dbscan_eps > 0:
                labels = self.dbscan.fit_predict(proposal_center)
            else:
                labels = np.array(range(proposal_centers.shape[1]))
            # compute cluster centers
            cluster_labels_center = {}
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label != -1:
                    cluster_points = proposal_center[labels == label, :]
                    # pdb.set_trace()
                    center = np.mean(cluster_points, axis=0)
                    proposal_cluster_center[labels == label, :] = center
                    cluster_labels_center[label] = center
            proposal_cluster_offset = proposal_center - proposal_cluster_center
            proposal_cluster_centers.append(torch.unsqueeze(torch.from_numpy(proposal_cluster_center), dim=0).cuda())
            cluster_labels_centers.append(cluster_labels_center)
            proposal_cluster_offsets.append(torch.unsqueeze(torch.from_numpy(proposal_cluster_offset), dim=0).cuda())
            proposal_cluster_labels.append(torch.unsqueeze(torch.from_numpy(labels), dim=0).cuda())

        return torch.cat(proposal_cluster_offsets, dim=0), torch.cat(proposal_cluster_labels, dim=0), torch.cat(
            proposal_cluster_centers, dim=0), cluster_labels_centers

    def sample_image_patches_parallel(self, locs):
        image_patches = []
        index_dict = {}
        center_cord_raw_list = []
        for b in range(len(locs)):
            centers = locs[b]
            for label in list(centers.keys()):
                center_cord_raw = (centers[label] / np.asarray(self.image_volume.resolution)).astype(np.int32)
                center_cord_raw_list.append(center_cord_raw)
        futures = [self.sample_pool.submit(self._process_load_image_patch, center_cord_raw, index) for index, center_cord_raw in enumerate(center_cord_raw_list)]
        for future in concurrent.futures.as_completed(futures):
            (result, index) = future.result()
            index_dict[index] = result
        for i in range(len(index_dict)):
            image_patches.append(index_dict[i])
        return image_patches

    def sample_image_patches(self, locs):
        image_patches = []
        center_cord_raw_list = []
        for b in range(len(locs)):
            centers = locs[b]
            for label in list(centers.keys()):
                center_cord_raw = (centers[label] / np.asarray(self.image_volume.resolution)).astype(np.int32)
                center_cord_raw_list.append(center_cord_raw)
                image_patches.append(self.load_image_patch(center_cord_raw))
        return image_patches

    def _process_load_image_patch(self, center_cord_raw, index):
        try:
            patch = self.image_volume[
                    center_cord_raw[0] - self.patch_size[1] // 2:center_cord_raw[0] + self.patch_size[1] // 2 + 1,
                    center_cord_raw[1] - self.patch_size[2] // 2:center_cord_raw[1] + self.patch_size[2] // 2 + 1,
                    center_cord_raw[2] - self.patch_size[0] // 2:center_cord_raw[2] + self.patch_size[0] // 2 + 1]
            patch = np.expand_dims(np.transpose((np.array(patch) / 255.0).astype(np.float32).squeeze(), (2, 0, 1)),
                                   0)
            patch = torch.from_numpy((patch - 0.5) / 0.5).unsqueeze(dim=0)  # [b,c,d,w,h]
        except:
            patch = torch.zeros([1, 1, 17, 129, 129])
            print(f'raw cord {center_cord_raw} cannot be sampled!')
        assert patch.shape == torch.Size([1, 1, 17, 129, 129]), f'patch size is {patch.shape}, ' \
                                              f'raw cord {center_cord_raw} '
        return patch, index

    def load_image_patch(self, center_cord_raw):
        patch = self.image_volume[
                center_cord_raw[0] - self.patch_size[1] // 2:center_cord_raw[0] + self.patch_size[1] // 2 + 1,
                center_cord_raw[1] - self.patch_size[2] // 2:center_cord_raw[1] + self.patch_size[2] // 2 + 1,
                center_cord_raw[2] - self.patch_size[0] // 2:center_cord_raw[2] + self.patch_size[0] // 2 + 1]
        patch = np.expand_dims(np.transpose((np.array(patch) / 255.0).astype(np.float32).squeeze(), (2, 0, 1)),
                               0)
        patch = torch.from_numpy((patch - 0.5) / 0.5).unsqueeze(dim=0)  # [b,c,d,w,h]
        return patch

    def get_image_features(self, image_patches):
        image_features = []
        total_batch = int(np.ceil(len(image_patches)/self.image_batch_size))
        for b in range(total_batch):
            with torch.no_grad():
                batch_data = torch.cat(image_patches[b*self.image_batch_size:(b+1)*self.image_batch_size], dim=0).cuda(non_blocking=True)
                # pdb.set_trace()
                image_feature = self.image_model(batch_data)  # [b=1,c=1,d,w,h]
            image_features.append(image_feature)
        return torch.cat(image_features, dim=0)

    def load_image_model(self, checkpoint):
        model = self.image_model
        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()

            # show model keys that do not match pretrained_dict
            if not model_dict.keys() == pretrained_dict.keys():
                warnings.warn("Module keys in model.state_dict() do not exactly "
                              "match the keys in pretrained_dict!")
                for key in model_dict.keys():
                    if not key in pretrained_dict:
                        print(key)

            # 1. filter out unnecessary keys by name
            pretrained_dict = {k: v for k,
                                        v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict (if size match)
            for param_tensor in pretrained_dict:
                if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                    model_dict[param_tensor] = pretrained_dict[param_tensor]
                else:
                    warnings.warn("Parameter size in model.state_dict() do not exactly "
                                  "match the keys in pretrained_dict!")
            # 3. load the new state dict
            model.load_state_dict(model_dict)

    def get_reference_points(self, proposal_cluster_offsets, proposal_cluster_labels, image_features):
        # get reference point position for every query
        resolution = np.array([self.image_volume.resolution[2], self.image_volume.resolution[0],
                               self.image_volume.resolution[1]]) * self.token_patch_size
        B_img, C, D, H, W = image_features.shape
        B, num_query = proposal_cluster_labels.shape
        image_patch_index = 0
        # get the patch index
        batch_idx = []
        label_idx = []
        for b in range(B):
            num_patch = len(torch.unique(proposal_cluster_labels[b, proposal_cluster_labels[b, :] > -1]))
            batch_idx += num_patch * [b]
            label_idx += range(num_patch)
            image_patch_index += num_patch

        input_spatial_shapes = torch.as_tensor([(D, H, W)], device=image_features.device)
        # Determine indices of batches that mark the start of a new feature level,
        # In our implementation len(level)= max cluster number in the batch.
        level_start_index = torch.cat(
            (input_spatial_shapes.new_zeros((1,)), input_spatial_shapes.prod(1).cumsum(0)[:-1]))
        cluster_center_point = ((np.asarray(self.cfg_image_model.MODEL.INPUT_SIZE) - np.array(
            [1, 1, 1])) / 2) / self.token_patch_size
        # point cloud: x,y,z; image: z,x,y
        proposal_points = proposal_cluster_offsets[:, :, [2, 0, 1]] / torch.tensor(resolution).cuda() + torch.tensor(
            cluster_center_point).to(dtype=torch.float32).cuda()
        reference_point = proposal_points / torch.tensor([D, H, W]).cuda()
        # reference_point = torch.cat([torch.unsqueeze(proposal_cluster_labels, dim=2), reference_point], dim=2)
        # (N, Length_{query}, n_level = 1, 4)
        reference_point = torch.unsqueeze(reference_point, dim=2)
        query_masks = []
        for img_patch_idx in range(B_img):
            query_mask = proposal_cluster_labels[batch_idx[img_patch_idx]] == label_idx[img_patch_idx]
            query_masks.append(query_mask)
        return reference_point, input_spatial_shapes, level_start_index, {'batch_idx': batch_idx, 'masks': query_masks}

    def get_query_offsets(self, proposal_cluster_offsets, proposal_cluster_labels, base_xyz, proposal_xyz, scales):
        # pdb.set_trace()
        query_cluster_labels = proposal_cluster_labels
        query_cluster_offsets = proposal_cluster_offsets + (proposal_xyz - base_xyz) * scales[0]
        return query_cluster_offsets, query_cluster_labels

    def add_decreasing_circle(self, image, center, radius):
        # Create a grid of coordinates
        grid = np.mgrid[[slice(0, s) for s in image.shape]]
        # Calculate the distance from the center for each pixel
        distances = np.sqrt(np.sum((grid - np.array(center)[:, None, None, None]) ** 2, axis=0))
        # Normalize distances to range from 1 to 0
        normalized_distances = 1 - distances / radius
        # Clip values to ensure they are between 0 and 1
        normalized_distances = np.clip(normalized_distances, 0, 1)
        # Update the image with the decreasing circle
        image = image + normalized_distances

        return image

    def get_corss(self, image, center):
        cross_image = np.zeros(image.shape)
        center = [int(center[0]), int(center[1]), int(center[2])]
        cross_image[center[0], center[1], center[2]] = 1
        cross_image[center[0], center[1] + 1, center[2]] = 1
        cross_image[center[0], center[1] - 1, center[2]] = 1
        cross_image[center[0], center[1], center[2] - 1] = 1
        cross_image[center[0], center[1], center[2] + 1] = 1

        return cross_image

    def get_crop_index(self, center, half_patch_size, volume_size):
        patch = self.create_decreasing_array([5, 11, 11])
        x_bound = np.clip(np.asarray([center[0] - half_patch_size[0], center[0] + half_patch_size[0] + 1]), 0,
                          volume_size[0]).astype(np.int32)
        y_bound = np.clip(np.asarray([center[1] - half_patch_size[1], center[1] + half_patch_size[1] + 1]), 0,
                          volume_size[1]).astype(np.int32)
        z_bound = np.clip(np.asarray([center[2] - half_patch_size[2], center[2] + half_patch_size[2] + 1]), 0,
                          volume_size[2]).astype(np.int32)
        return [x_bound, y_bound, z_bound]

    def set_attention_point(self, cord, patch):
        attention_mask = np.zeros(patch.shape)
        [z_bound, x_bound, y_bound] = self.get_crop_index(cord, [2, 5, 5], patch.shape)
        attention_mask[z_bound[0]:z_bound[1], x_bound[0]:x_bound[1], y_bound[0]:y_bound[1]] = 1
        return attention_mask

    def visualize_offsets(self, proposal_cluster_labels, proposal_cluster_centers, query_cluster_offsets, image_patches, end_points, centriods, scales):
        # (end_points['0head_base_xyz'][0,30,:]*end_points['scale']+end_points['centroid']).cpu()/torch.tensor([4,4,40]) + (end_points['1head_sampling_offset'][0,30,0,0,:,[1,2,0]].cpu())*torch.tensor([20,20,2])
        image_patch = image_patches[3][0, 0, :].numpy()
        show_rgb = np.stack((image_patch, image_patch, image_patch), axis=-1)
        show_rgb = (show_rgb - show_rgb.min()) * 120
        sampling_offsets = end_points['1head_sampling_offset']
        index0 = np.where(proposal_cluster_labels.cpu().numpy() == 2)[0]
        index1 = np.where(proposal_cluster_labels.cpu().numpy() == 2)[1]
        for i in range(len(index1)):
            query_cluster_offset = query_cluster_offsets[0, i, :].cpu()
            reference_cord = torch.tensor([64, 64, 8]) + query_cluster_offset / torch.tensor([16, 16, 40])
            reference_cord = reference_cord[[2, 0, 1]]
            attention_map = np.zeros(image_patch.shape)
            for j in range(6):
                for k in range(4):
                    attention_offset = query_cluster_offset/torch.tensor([16,16,40]) + sampling_offsets[0, i, j, 0, k, [1,2,0]].cpu() * torch.tensor([5,5,2])
                    attention_cord = torch.tensor([64, 64, 8]) + attention_offset
                    # x, y, z -> z, x, y
                    attention_cord = attention_cord[[2, 0, 1]]
                    attention_map = self.add_decreasing_circle(attention_map, attention_cord, 8)
                    # attention_map = attention_map + 0.5 * self.set_attention_point(attention_cord, attention_map)
            print(attention_map.max())
            reference_cross = self.get_corss(attention_map, reference_cord)
            x = show_rgb \
                + 100 * np.stack((attention_map, np.zeros(attention_map.shape), np.zeros(attention_map.shape)), axis=-1) \
                + 100 * np.stack((np.zeros(attention_map.shape), reference_cross, np.zeros(attention_map.shape)), axis=-1)
            x = numpy.clip(x, 0, 254)
            tf.imsave(f'image{i}.tif', x.astype(np.uint8))
            if i == 11:
                for l in range(17):
                    tf.imsave(f'image{i}_{l}.tif', show_rgb[l,32:-32, 32:-32].astype(np.uint8))
                    tf.imsave(f'attention{i}_{l}.tif', x[l, 32:-32, 32:-32].astype(np.uint8))

        pdb.set_trace()
