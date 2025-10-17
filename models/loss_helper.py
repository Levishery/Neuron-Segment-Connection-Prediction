import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import pdb

DEBUG = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from .losses import smoothl1_loss, l1_loss, SigmoidFocalClassificationLoss


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, M (pointnet output point num)
    seed_xyz = end_points['seed_xyz']  # B, M, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, M
    gt_center = end_points['center_label'][:, :, 0:3]  # B, MAX_NUM_OBJ, 3
    gt_size = end_points['size_gts'][:, :, 0:3]  # B, MAX_NUM_OBJ, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]  # pointnet output point num
    K2 = gt_center.shape[1]  # MAX_NUM_OBJ

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss


def compute_points_vector_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, M (pointnet output point num)
    seed_xyz = end_points['seed_xyz']  # B, M, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, M
    gt_center = end_points['center_label'][:, :, 0:3]  # B, MAX_NUM_OBJ, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]  # pointnet output point num
    K2 = gt_center.shape[1]  # MAX_NUM_OBJ

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    # mask the distance with object assignment mask and set the distance of background points to 100.
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2;
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    # get topk inds and mask with objectness mask. The none object boxes inds are set to -1.
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    # since none object boxes inds are set to -1, they are assigned to the last point,
    # so init an extra last point and remove.
    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    # mask the labels out of the box
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound (if an instance do not generate any seed, it can not be detected)
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss


def compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    for prefix in prefixes:
        # Associate proposal and GT objects
        seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        query_points_sample_inds = end_points['query_points_sample_inds'].long()

        B = seed_inds.shape[0]
        K = query_points_sample_inds.shape[1]  # num query
        K2 = gt_center.shape[1]  # max object num

        seed_obj_gt = torch.gather(end_points['point_obj_mask'], 1, seed_inds)  # B,num_seed
        query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points

        point_instance_label = end_points['point_instance_label']  # B, num_points
        seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
        query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points

        objectness_mask = torch.ones((B, K)).cuda()

        # Set assignment
        # In this implementation all queries inited with a seed point inside bbox is considered to have object
        object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
        object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

        end_points[f'{prefix}objectness_label'] = query_points_obj_gt
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment
        total_num_proposal = query_points_obj_gt.shape[0] * query_points_obj_gt.shape[1]
        end_points[f'{prefix}pos_ratio'] = \
            torch.sum(query_points_obj_gt.float().cuda()) / float(total_num_proposal)
        end_points[f'{prefix}neg_ratio'] = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - end_points[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        cls_loss_src = criterion(objectness_scores.transpose(2, 1).contiguous().view(B, K, 1),
                                 query_points_obj_gt.unsqueeze(-1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points


def compute_box_and_sem_cls_loss(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(end_points['heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(end_points[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(end_points['heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            end_points[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label

        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        if size_cls_agnostic:
            pred_size = end_points[f'{prefix}pred_size']
            size_label = torch.gather(
                end_points['size_gts'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            if size_loss_type == 'smoothl1':
                size_loss = size_delta * smoothl1_loss(size_error,
                                                       delta=size_delta)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_loss = l1_loss(size_error)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError
        else:
            size_class_label = torch.gather(end_points['size_class_label'], 1,
                                            object_assignment)  # select (B,K) from (B,K2)
            criterion_size_class = nn.CrossEntropyLoss(reduction='none')
            size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1),
                                                   size_class_label)  # (B,K)
            size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

            size_residual_label = torch.gather(
                end_points['size_residual_label'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

            size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
            size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                        1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
            size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
            predicted_size_residual_normalized = torch.sum(
                end_points[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled,
                2)  # (B,K,3)

            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
                0)  # (1,1,num_size_cluster,3)
            mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
            size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

            size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

            if size_loss_type == 'smoothl1':
                size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                           delta=size_delta)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            end_points[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + size_loss
        else:
            end_points[f'{prefix}size_cls_loss'] = size_class_loss
            end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points


def compute_vector_loss(end_points, config, num_decoder_layers,
                        center_loss_type='smoothl1', center_delta=1.0,
                        heading_loss_type='smoothl1', yaw_delta=1.0, pitch_delta=1.0, terminal_classification=False):
    """ Compute 3D bounding box and semantic classification loss.
    """
    num_yaw_bin = config.num_yaw_bin
    num_pitch_bin = config.num_pitch_bin

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    vector_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        if terminal_classification:
            # or use focal classification loss
            terminal_class_label = torch.gather(end_points['terminal_class_label'], 1, object_assignment) # (B, K, 1) from (B, K2, 1)
            criterion_terminal_class = nn.BCEWithLogitsLoss(reduction='none')
            terminal_cls_loss = criterion_terminal_class(torch.squeeze(end_points[f'{prefix}terminal_scores'], dim=2), terminal_class_label)

            terminal_cls_loss = torch.sum(terminal_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)


        # Compute heading loss
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        # yaw
        yaw_class_label = torch.gather(end_points['yaw_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        yaw_class_loss = criterion_heading_class(end_points[f'{prefix}yaw_scores'].transpose(2, 1),
                                                     yaw_class_label)  # (B,K)

        yaw_class_loss = torch.sum(yaw_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        yaw_residual_label = torch.gather(end_points['yaw_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        yaw_residual_normalized_label = yaw_residual_label / (np.pi / num_yaw_bin)  # 2pi/num_yaw_bin/2 -> -1~1

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        yaw_label_one_hot = torch.cuda.FloatTensor(batch_size, yaw_class_label.shape[1],
                                                       num_yaw_bin).zero_()
        yaw_label_one_hot.scatter_(2, yaw_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        yaw_residual_normalized_error = torch.sum(
            end_points[f'{prefix}yaw_residuals_normalized'] * yaw_label_one_hot,
            -1) - yaw_residual_normalized_label  # compute residual error only for the gt bin

        if heading_loss_type == 'smoothl1':
            yaw_residual_normalized_loss = yaw_delta * smoothl1_loss(yaw_residual_normalized_error,
                                                                             delta=yaw_delta)  # (B,K)
            yaw_residual_normalized_loss = torch.sum(
                yaw_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            yaw_residual_normalized_loss = l1_loss(yaw_residual_normalized_error)  # (B,K)
            yaw_residual_normalized_loss = torch.sum(
                yaw_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # pitch
        pitch_class_label = torch.gather(end_points['pitch_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        pitch_class_loss = criterion_heading_class(end_points[f'{prefix}pitch_scores'].transpose(2, 1),
                                                     pitch_class_label)  # (B,K)
        pitch_class_loss = torch.sum(pitch_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        pitch_residual_label = torch.gather(end_points['pitch_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        pitch_residual_normalized_label = pitch_residual_label / (np.pi / (2*num_pitch_bin))  # pi/num_pitch_bin/2 -> -1~1

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        pitch_label_one_hot = torch.cuda.FloatTensor(batch_size, pitch_class_label.shape[1],
                                                       num_pitch_bin).zero_()
        pitch_label_one_hot.scatter_(2, pitch_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        pitch_residual_normalized_error = torch.sum(
            end_points[f'{prefix}pitch_residuals_normalized'] * pitch_label_one_hot,
            -1) - pitch_residual_normalized_label  # compute residual error only for the gt bin

        if heading_loss_type == 'smoothl1':
            pitch_residual_normalized_loss = pitch_delta * smoothl1_loss(pitch_residual_normalized_error,
                                                                             delta=pitch_delta)  # (B,K)
            pitch_residual_normalized_loss = torch.sum(
                pitch_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            pitch_residual_normalized_loss = l1_loss(pitch_residual_normalized_error)  # (B,K)
            pitch_residual_normalized_loss = torch.sum(
                pitch_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}yaw_cls_loss'] = yaw_class_loss
        end_points[f'{prefix}yaw_reg_loss'] = yaw_residual_normalized_loss
        end_points[f'{prefix}pitch_cls_loss'] = pitch_class_loss
        end_points[f'{prefix}pitch_reg_loss'] = pitch_residual_normalized_loss
        if terminal_classification:
            vector_loss = center_loss + 0.1 * yaw_class_loss + yaw_residual_normalized_loss + 0.1 * pitch_class_loss + pitch_residual_normalized_loss + 0.1 * terminal_cls_loss
            end_points[f'{prefix}terminal_cls_loss'] = terminal_cls_loss
        else:
            vector_loss = center_loss + 0.1 * yaw_class_loss + yaw_residual_normalized_loss + 0.1 * pitch_class_loss + pitch_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = vector_loss

        vector_loss_sum += vector_loss

    return vector_loss_sum, end_points


def get_loss(end_points, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0,
             size_cls_agnostic=False):
    """ Loss functions
    """
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(end_points, query_points_obj_topk)

        end_points['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0

    # Obj loss
    objectness_loss_sum, end_points = \
        compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers)

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta,
        size_cls_agnostic=size_cls_agnostic)
    end_points['sum_heads_box_loss'] = box_loss_sum
    end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

    # means average proposal with prediction loss
    loss = query_points_generator_loss_coef * query_points_generation_loss + \
           1.0 / (num_decoder_layers + 1) * (
                   obj_loss_coef * objectness_loss_sum + box_loss_coef * box_loss_sum + sem_cls_loss_coef * sem_cls_loss_sum)
    loss *= 10

    end_points['loss'] = loss
    return loss, end_points

def get_loss_vector(end_points, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, vector_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0, yaw_delta=1.0, pitch_delta=1.0,
             heading_loss_type='smoothl1', terminal_classification=False):
    """ Loss functions
    """
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_vector_cls_loss_hard_topk(end_points, query_points_obj_topk)

        end_points['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0

    # Obj loss
    objectness_loss_sum, end_points = \
        compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers)

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    vector_loss_sum, end_points = compute_vector_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        heading_loss_type=heading_loss_type, yaw_delta=yaw_delta,
        pitch_delta=pitch_delta, terminal_classification=terminal_classification)
    end_points['sum_heads_vector_loss'] = vector_loss_sum

    # means average proposal with prediction loss
    loss = query_points_generator_loss_coef * query_points_generation_loss + \
           1.0 / (num_decoder_layers + 1) * (
                   obj_loss_coef * objectness_loss_sum + vector_loss_coef * vector_loss_sum)
    loss *= 10

    end_points['loss'] = loss
    return loss, end_points