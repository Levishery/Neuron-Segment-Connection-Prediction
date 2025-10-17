""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import pdb
import sys
import numpy as np
import torch
import math
import cc3d
import pandas as pd
from numba import jit
import pickle
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_cls, eval_det_multiprocessing
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box

sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
from sunrgbd_utils import extract_pc_in_box3d



def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def sigmoid(x):
    ''' Numpy function for softmax'''
    s = 1 / (1 + np.exp(-x))
    return s


def parse_predictions(end_points, config_dict, prefix="", size_cls_agnostic=False):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points[f'{prefix}center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points[f'{prefix}heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points[f'{prefix}heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)

    if size_cls_agnostic:
        pred_size = end_points[f'{prefix}pred_size']  # B, num_proposal, 3
    else:
        pred_size_class = torch.argmax(end_points[f'{prefix}size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(end_points[f'{prefix}size_residuals'], 2,
                                          pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                             3))  # B,num_proposal,1,3
        pred_size_residual.squeeze_(2)

    pred_sem_cls = torch.argmax(end_points[f'{prefix}sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = softmax(end_points[f'{prefix}sem_cls_scores'].detach().cpu().numpy())  # B,num_proposal,10

    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    # pred_size_check = end_points[f'{prefix}pred_size']  # B,num_proposal,3
    # pred_bbox_check = end_points[f'{prefix}bbox_check']  # B,num_proposal,3

    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            if size_cls_agnostic:
                box_size = pred_size[i, j].detach().cpu().numpy()
            else:
                box_size = config_dict['dataset_config'].class2size( \
                    int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = end_points[f'{prefix}objectness_scores'].detach().cpu().numpy()
    obj_prob = sigmoid(obj_logits)[:, :, 0]  # (B,K)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            # assert (len(pick) > 0)
            if len(pick) > 0:
                pred_mask[i, nonempty_box_inds[pick]] = 1
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                cur_list += [(ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                             for j in range(pred_center.shape[1]) if
                             pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls[i, j].item(), pred_corners_3d_upright_camera[i, j], obj_prob[i, j]) \
                                       for j in range(pred_center.shape[1]) if
                                       pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])

    return batch_pred_map_cls


def bio_constrained_candidates_v1(vol_ffn1, seg_id, end_point, skel_vector, radius=500, theta=0.3216, dust_thresh=200):
    ### a simple implementation of biological constrained candidate finding
    end_point = end_point // [16, 16, 40]
    block_shape = np.asarray([2 * (radius // 16), 2 * (radius // 16), 2 * (radius // 40)])
    ffn_volume = vol_ffn1[end_point[0] - block_shape[0] // 2:end_point[0] + block_shape[0] // 2 + 1,
                 end_point[1] - block_shape[1] // 2:end_point[1] + block_shape[1] // 2 + 1,
                 end_point[2] - block_shape[2] // 2:end_point[2] + block_shape[2] // 2 + 1].squeeze()
    ffn_volume = cc3d.dust(ffn_volume, threshold=dust_thresh, connectivity=26, in_place=False)
    mask = np.zeros(block_shape + [1, 1, 1])
    center = np.asarray(mask.shape) // 2
    resolution = np.asarray([16, 16, 40])
    mask = Traverse_end_point(theta, mask, radius, center, skel_vector, resolution)
    candidates = np.setdiff1d(np.unique(mask * ffn_volume), np.array([0, seg_id]))
    return set(np.asarray(candidates).astype(np.int64)), end_point, end_point + skel_vector * radius // [16, 16, 40]


def bio_constrained_candidates_v2(vol_ffn1_data, test_bbox, padding, candidate_masks, mask_indexes, seg_id, end_point_raw, skel_vector, angle, radius, dust_thresh=200, move_back=0):
    ### v2: predfine the masks 10h -> 20min
    end_point_voxel = end_point_raw / [4, 4, 40] - skel_vector * move_back // [4, 4, 40]
    end_point = np.round((end_point_voxel / [4, 4, 1] - np.asarray(test_bbox[0]) / [4, 4, 1] + np.asarray(padding))).astype(np.int)
    block_shape = np.asarray([2 * (radius // 16), 2 * (radius // 16), 2 * (radius // 40)])
    ffn_volume = vol_ffn1_data[end_point[0] - block_shape[0] // 2:end_point[0] + block_shape[0] // 2 + 1,
                 end_point[1] - block_shape[1] // 2:end_point[1] + block_shape[1] // 2 + 1,
                 end_point[2] - block_shape[2] // 2:end_point[2] + block_shape[2] // 2 + 1].squeeze()
    # ffn_volume = cc3d.dust(ffn_volume, threshold=dust_thresh, connectivity=26, in_place=False)
    yaw_class = int(max(min(np.round(360 * (angle[0]+np.pi)/(2*np.pi)), 359), 0))
    pitch_class = int(max(min(np.round(180 * (angle[1] + np.pi/2) / np.pi), 179), 0))
    mask = candidate_masks[mask_indexes[yaw_class, pitch_class]]
    candidates = np.setdiff1d(np.unique(mask * ffn_volume), np.array([0, seg_id]))
    # pdb.set_trace()
    return set(np.asarray(candidates).astype(np.int64)), end_point_voxel, end_point_voxel + skel_vector * radius // [4, 4, 40]


def scan_sphere(vol_ffn1_data, test_bbox, padding, sphere_mask, seg_id, end_point_raw):
    ### v2: predfine the masks 10h -> 20min
    end_point_voxel = np.asarray(end_point_raw) / np.asarray([4,4,40])
    end_point = np.round((end_point_voxel /  np.asarray([4, 4, 1]) - np.asarray(test_bbox[0]) /  np.asarray([4, 4, 1]) + np.asarray(padding))).astype(np.int)
    block_shape = sphere_mask.shape
    ffn_volume = vol_ffn1_data[end_point[0] - block_shape[0] // 2:end_point[0] + block_shape[0] // 2 + 1, end_point[1] - block_shape[1] // 2:end_point[1] + block_shape[1] // 2 + 1, end_point[2] - block_shape[2] // 2:end_point[2] + block_shape[2] // 2 + 1].squeeze()
    candidates = np.setdiff1d(np.unique(sphere_mask * ffn_volume), np.array([0, seg_id]))
    return set(np.asarray(candidates).astype(np.int64))


@jit(nopython=True)
def Traverse_end_point(theta, mask, radius, center, skel_vector, resolution):
    cos_theta = math.cos(theta)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                cord = np.asarray([x, y, z])
                vector = (cord - center) * resolution
                distance = math.sqrt(sum(vector * vector)) + 1e-4
                if distance > radius:
                    continue
                norm_vector = vector / distance
                if sum(norm_vector * skel_vector) < cos_theta:
                    continue
                mask[x, y, z] = 1
    return mask


@jit(nopython=True)
def Traverse_end_point_cylinder(theta, mask, radius, center, skel_vector, resolution):
    sin_theta = math.sin(theta)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                cord = np.asarray([x, y, z])
                vector = (cord - center) * resolution
                distance = math.sqrt(sum(vector * vector)) + 1e-4
                if distance > radius:
                    continue
                norm_vector = vector / distance
                sin_phi = math.sqrt(1 - (sum(norm_vector * skel_vector)) * (sum(norm_vector * skel_vector)))
                if sin_phi * distance > radius * sin_theta or sum(norm_vector * skel_vector) < 0:
                    continue
                mask[x, y, z] = 1
    return mask

@jit(nopython=True)
def Traverse_end_point_sphere(mask, radius, center, resolution):
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            for z in range(mask.shape[2]):
                cord = np.asarray([x, y, z])
                vector = (cord - center) * resolution
                distance = math.sqrt(sum(vector * vector)) + 1e-4
                if distance > radius:
                    continue
                mask[x, y, z] = 1
    return mask


def get_candidate_masks(parameters, DATASET_CONFIG, data_path, scan_area='cone'):
    # precompute the candidate masks

    [move_back, prediction_thresh, biological_theta, biological_radius] = parameters
    if scan_area == 'cone':
        file_name_masks = f'{biological_theta}_{biological_radius}_masks.pkl'
        file_name_indexes = f'{biological_theta}_{biological_radius}_indexes.pkl'
    elif scan_area == 'cylinder':
        file_name_masks = f'{biological_theta}_{biological_radius}_cyl_masks.pkl'
        file_name_indexes = f'{biological_theta}_{biological_radius}_cyl_indexes.pkl'
    elif scan_area == 'sphere':
        file_name_masks = f'{biological_theta}_{biological_radius}_sphere_masks.pkl'
        file_name_indexes = f'{biological_theta}_{biological_radius}_sphere_indexes.pkl'
    else:
        raise ValueError(f"{scan_area} not support. ")
    if os.path.exists(os.path.join(data_path, file_name_masks)):
        pickle_filename = os.path.join(data_path, file_name_masks)
        with open(pickle_filename, 'rb') as f:
            mask_list = pickle.load(f)
        print(f"{pickle_filename} loaded successfully !!!")
        pickle_filename = os.path.join(data_path, file_name_indexes)
        with open(pickle_filename, 'rb') as f:
            class_mask_dict = pickle.load(f)
        print(f"{pickle_filename} loaded successfully !!!")
    else:
        mask_list = []
        class_mask_dict = {}
        block_shape = np.asarray([2 * (biological_radius // 16), 2 * (biological_radius // 16), 2 * (biological_radius // 40)])
        mask_record = np.zeros(block_shape + [1, 1, 1], dtype=np.bool)
        mask = np.zeros(block_shape + [1, 1, 1], dtype=np.bool)
        center = np.asarray(mask.shape) // 2
        resolution = np.asarray([16, 16, 40])

        # -pi ~ pi
        yaw_bin = 360
        # -pi/2 ~ pi/2
        pitch_bin = 180

        index = -1
        if scan_area != 'sphere':
            for j in tqdm(range(pitch_bin)):
                for i in range(yaw_bin):
                    yaw = -np.pi + 2*np.pi*i/yaw_bin
                    pitch = -np.pi/2 + np.pi*j/pitch_bin
                    vector = DATASET_CONFIG.Euler2vector(yaw, pitch)
                    mask = np.zeros(block_shape + [1, 1, 1], dtype=np.bool)
                    if scan_area == 'cone':
                        mask = Traverse_end_point(biological_theta, mask, biological_radius, center, vector, resolution)
                    elif scan_area == 'cylinder':
                        mask = Traverse_end_point_cylinder(biological_theta, mask, biological_radius, center, vector, resolution)
                    if np.any(mask != mask_record):
                        mask_list.append(mask)
                        index = index + 1
                        mask_record = mask
                    class_mask_dict[i, j] = index
        else:
            mask = np.zeros(block_shape + [1, 1, 1], dtype=np.bool)
            mask_list = [Traverse_end_point_sphere(mask, biological_radius, center, resolution)]
            for j in tqdm(range(pitch_bin)):
                for i in range(yaw_bin):
                    class_mask_dict[i, j] = 0

        # pdb.set_trace()
        pickle_filename = os.path.join(data_path, file_name_masks)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(mask_list, f)
            print(f"{pickle_filename} saved successfully !!")
        pickle_filename = os.path.join(data_path, file_name_indexes)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(class_mask_dict, f)
            print(f"{pickle_filename} saved successfully !!")
    return mask_list, class_mask_dict


def parse_predictions_to_vector(end_points, DATASET_CONFIG, prefix=""):

    result_batch = []
    b_size = len(end_points['centroid'])
    yaw_class_batch = torch.argmax(end_points[prefix + 'yaw_scores'].detach().cpu(), -1)  # B,num_proposal
    yaw_residual_batch = torch.gather(end_points[prefix + 'yaw_residuals'].detach().cpu(), 2, yaw_class_batch.unsqueeze(-1))  # B,num_proposal,1
    pitch_class_batch = torch.argmax(end_points[prefix + 'pitch_scores'].detach().cpu(), -1)  # B,num_proposal
    pitch_residual_batch = torch.gather(end_points[prefix + 'pitch_residuals'].detach().cpu(), 2, pitch_class_batch.unsqueeze(-1))  # B,num_proposal,1
    instance_obj_mask_batch = sigmoid(end_points['last_objectness_scores'].detach().cpu().numpy())[:, :, 0]
    if 'last_terminal_scores' in list(end_points.keys()):
        instance_terminal_mask_batch = sigmoid(end_points['last_terminal_scores'].detach().cpu().numpy())[:, :, 0]
    else:
        instance_terminal_mask_batch = np.ones(instance_obj_mask_batch.shape)
    prediction_centers_batch = end_points[prefix + 'center'].detach().cpu()

    for i in range(b_size):
        vector_prediction = {}
        instance_num = instance_obj_mask_batch[i, :].shape[0]
        centroid = end_points['centroid'][i].detach().cpu()
        scale = end_points['scale'][i].detach().cpu()
        vector_prediction['seg_id'] = int(end_points['seg_id'][i].detach().cpu())

        center_data = np.zeros([instance_num, 3])
        vector_data = np.zeros([instance_num, 3])
        angle_data = np.zeros([instance_num, 2])
        score_data = np.zeros([instance_num, 1])
        terminal_score_data = np.zeros([instance_num, 1])
        for j in range(instance_num):
            center_data[j, :] = (prediction_centers_batch[i, j, :]) * scale + centroid
            yaw = DATASET_CONFIG.class2angle(yaw_class_batch[i, j], yaw_residual_batch[i, j], type='yaw')
            pitch = DATASET_CONFIG.class2angle(pitch_class_batch[i, j], pitch_residual_batch[i, j], type='pitch')
            vector_data[j, :] = - DATASET_CONFIG.Euler2vector(yaw, pitch)
            score_data[j, :] = float(instance_obj_mask_batch[i, j])
            terminal_score_data[j, :] = float(instance_terminal_mask_batch[i, j])
            angle_data[j, 0] = - yaw
            angle_data[j, 1] = - pitch

        vector_prediction['center'] = center_data
        vector_prediction['vector'] = vector_data
        vector_prediction['score'] = score_data
        vector_prediction['terminal_score'] = terminal_score_data
        vector_prediction['angle'] = angle_data
        result_batch.append(vector_prediction)

    return result_batch


def inbox(center, test_bbox):
    cord_start = test_bbox[0] * [4, 4, 40]
    cord_end = test_bbox[1] * [4, 4, 40]
    inside_box = np.logical_and.reduce((center[0] >= cord_start[0], center[0] <= cord_end[0],
                                center[1] >= cord_start[1], center[1] <= cord_end[1],
                                center[2] >= cord_start[2], center[2] <= cord_end[2]))
    return inside_box

def get_candidates(instance, parameters, candidate_masks, mask_indexes, vol_ffn1_data, padding, test_bbox):
    [move_back, prediction_thresh, biological_theta, biological_radius] = parameters
    result_set = set({})
    candidates_per_vector = []
    terminal_score = []
    starts = []
    ends = []
    instance_num = instance['score'].shape[0]
    for i in range(instance_num):
        if instance['score'][i] > prediction_thresh and inbox(instance['center'][i], test_bbox):
            # candidates, start, end = bio_constrained_candidates_v1(vol_ffn1_data, instance['seg_id'], instance['center'][i], instance['vector'][i], radius=biological_radius, theta=biological_theta)
            candidates, start, end = bio_constrained_candidates_v2(vol_ffn1_data, test_bbox, padding, candidate_masks, mask_indexes, instance['seg_id'], instance['center'][i], instance['vector'][i], instance['angle'][i], radius=biological_radius, move_back=move_back)
            result_set = result_set | candidates
            candidates_per_vector.append(candidates)
            starts.append(start)
            ends.append(end)
            terminal_score.append(instance['terminal_score'][i])
    return result_set, starts, ends, candidates_per_vector, terminal_score

def get_candidates_surface_points(point_cloud, scale, centroid, padding, sphere_mask, seg_id, vol_ffn1_data, test_bbox):
    result_set = set({})
    point_cloud = point_cloud * scale + centroid
    instance_num = point_cloud.shape[0]
    for i in range(instance_num):
        if inbox(point_cloud[i], test_bbox):
            candidates = scan_sphere(vol_ffn1_data, test_bbox, padding, sphere_mask, seg_id, point_cloud[i])
            result_set = result_set | candidates
    return result_set

def get_candidates_with_terminal_thresh(candidate_dict_all_vectors, terminal_score_dict, thresh):
    candidate_dict = {}
    for instace_id in list(candidate_dict_all_vectors.keys()):
        result_set = set({})
        candidate_per_segment_all_vectors = candidate_dict_all_vectors[instace_id]
        terminal_score_all_vectors = terminal_score_dict[instace_id]
        for i in range(len(terminal_score_all_vectors)):
            if terminal_score_all_vectors[i] < thresh:
                result_set = result_set | candidate_per_segment_all_vectors[i]
        candidate_dict[instace_id] = result_set
    return candidate_dict

import pandas as pd
import numpy as np
from collections import defaultdict

import pandas as pd
import numpy as np
from collections import defaultdict

def get_recall_record(candidate_dict, start_dict, end_dict, baseline_recall_record, record_path='None', write_csv=False, logger=None):
    # --- START: 修正部分 1 (使用整数键进行预处理) ---
    ground_truth_connections = defaultdict(set)
    valid_baseline = baseline_recall_record[baseline_recall_record.iloc[:, 3] > 0.5]

    for _, row in valid_baseline.iterrows():
        # --- FIX: Use original integer types for keys ---
        query, pos = row.iloc[0], row.iloc[1]
        ground_truth_connections[query].add(pos)
        ground_truth_connections[pos].add(query)
    # --- END: 修正部分 1 ---

    hit = 0
    upper_hit = 0
    total_candidate_num = 0
    seg_id_records = []
    candidate_path = record_path + '_candidates.csv'
    hit_path = record_path + '_hit.csv'

    recalled_connections = defaultdict(set)

    for i in range(len(baseline_recall_record)):
        # --- FIX: Use original integer types throughout the loop ---
        query = baseline_recall_record.iloc[i][0]
        pos = baseline_recall_record.iloc[i][1]

        upper_hit += baseline_recall_record.iloc[i][3]
        if baseline_recall_record.iloc[i][3] > 0.5:
            # 防御性检查，确保 key 存在（此逻辑来自您的原始代码，予以保留）
            if query not in candidate_dict:
                candidate_dict[query] = set({})
            if pos not in candidate_dict:
                candidate_dict[pos] = set({})

            # 核心命中判断，现在使用整数键，可以正确工作
            if query in candidate_dict[pos] or pos in candidate_dict[query]:
                hit += 1
                # --- START: 修正部分 2 (使用整数键记录命中) ---
                recalled_connections[query].add(pos)
                recalled_connections[pos].add(query)
                # --- END: 修正部分 2 ---
                if write_csv:
                    pd.DataFrame([{'node0_segid': int(query), 'node1_segid': int(pos), 'hit': 1}])\
                        .to_csv(hit_path, mode='a', header=False, index=False)
            else:
                if write_csv:
                    pd.DataFrame([{'node0_segid': int(query), 'node1_segid': int(pos), 'hit': 0}]) \
                        .to_csv(hit_path, mode='a', header=False, index=False)
            
            # CSV 写入和 total_candidate_num 计算
            if write_csv:
                if pos not in seg_id_records:
                    seg_id_records.append(pos)
                    total_candidate_num += len(candidate_dict[pos])
                    pd.DataFrame([{'node0_segid': int(pos), 'candidates': str(candidate_dict[pos]), 'starts': str(start_dict.get(pos, [])), 'ends': str(end_dict.get(pos, []))}]) \
                        .to_csv(candidate_path, mode='a', header=False, index=False)
                if query not in seg_id_records:
                    seg_id_records.append(query)
                    total_candidate_num += len(candidate_dict[query])
                    pd.DataFrame([{'node0_segid': int(query), 'candidates': str(candidate_dict[query]), 'starts': str(start_dict.get(query, [])), 'ends': str(end_dict.get(query, []))}]) \
                        .to_csv(candidate_path, mode='a', header=False, index=False)

    # --- START: 修正部分 3 (计算标准差，现在可以正确工作) ---
    segment_recalls = []
    # 遍历的 ground_truth_connections 现在也使用整数键
    for seg_id, gt_set in ground_truth_connections.items():
        recalled_set = recalled_connections.get(seg_id, set())
        num_gt = len(gt_set)
        if num_gt > 0:
            recall = len(recalled_set) / num_gt
            segment_recalls.append(recall)
    
    recall_std = np.std(segment_recalls) if segment_recalls else 0.0
    print('std:')
    print(recall_std)
    # --- END: 修正部分 3 ---

    num_valid_pairs = len(valid_baseline)
    global_recall = hit / num_valid_pairs if num_valid_pairs > 0 else 0.0

    return [global_recall, total_candidate_num]


def parse_groundtruths(end_points, config_dict, size_cls_agnostic):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    if size_cls_agnostic:
        size_gts = end_points['size_gts']
    else:
        size_class_label = end_points['size_class_label']
        size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    sem_cls_label = end_points['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(heading_class_label[i, j].detach().cpu().numpy(),
                                                                      heading_residual_label[
                                                                          i, j].detach().cpu().numpy())
            if size_cls_agnostic:
                box_size = size_gts[i, j].detach().cpu().numpy()
            else:
                box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                    size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append([(sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j]) for j in
                                 range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i, j] == 1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
