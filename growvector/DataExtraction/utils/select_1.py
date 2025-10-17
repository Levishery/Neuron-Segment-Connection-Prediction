import numpy as np
import math
import pandas as pd
import os
import shutil
from tqdm import tqdm
import random
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import navis
from skimage.transform import resize


def compute_normalized_vector(p_1, p_2):
    end_point_vector = p_1 - p_2
    norm = np.sqrt(sum(end_point_vector * end_point_vector))
    return end_point_vector / norm


def bio_constrained_candidates(vol_ffn1, end_point, skel_vector=np.asarray([0, 0, 0]), radius=500, theta=18.5,
                               move_back=0, use_direct=True):
    ### a simple implementation of biological constrained candidate finding
    end_point = end_point // [4, 4, 1]
    block_shape = np.asarray([2 * (radius // 16), 2 * (radius // 16), 2 * (radius // 40)])
    ffn_volume = vol_ffn1[end_point[0] - block_shape[0] // 2:end_point[0] + block_shape[0] // 2 + 1,
                 end_point[1] - block_shape[1] // 2:end_point[1] + block_shape[1] // 2 + 1,
                 end_point[2] - block_shape[2] // 2:end_point[2] + block_shape[2] // 2 + 1].squeeze()
    cos_theta = math.cos(theta / 360 * 2 * math.pi)
    mask = np.zeros(block_shape + [1, 1, 1])
    center = np.asarray(mask.shape) // 2
    it = np.nditer(mask, flags=['multi_index'])
    while not it.finished:
        x, y, z = it.multi_index
        it.iternext()
        cord = np.asarray([x, y, z])
        vector = (cord - center) * [16, 16, 40] + skel_vector * move_back
        distance = math.sqrt(sum(vector * vector)) + 1e-4
        if distance > radius + move_back:
            continue
        norm_vector = vector / distance
        if sum(norm_vector * skel_vector) < cos_theta and use_direct:
            continue
        mask[x, y, z] = 1
    candidates = np.setdiff1d(np.unique(mask * ffn_volume), 0)
    return np.asarray(candidates)


def refine_center(cord0, cord1, seg_id, vol_ffn1):
    # 将坐标转换为整数
    cord0 = np.round(cord0).astype(int)
    cord1 = np.round(cord1).astype(int)

    # 计算线段的方向向量
    direction = cord1 - cord0

    # 计算线段的长度
    length = np.linalg.norm(direction)

    # 根据线段长度生成一系列坐标点
    points = []
    for i in range(int(length) + 1):
        point = cord0 + direction * (i / length)
        point = np.round(point).astype(int)
        points.append(point)
    points = np.asarray(points)

    # get point segids
    try:
        chunk_start = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]) // np.array(
            [4, 4, 1])
        chunk_end = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])]) // np.array(
            [4, 4, 1]) + np.array([1, 1, 1])
        indices = (points // np.array([4, 4, 1])).astype(int) - chunk_start
        chunk = np.array(
            vol_ffn1[chunk_start[0]:chunk_end[0], chunk_start[1]:chunk_end[1], chunk_start[2]:chunk_end[2]])
        point_seg_ids = np.squeeze(chunk[indices[:, 0], indices[:, 1], indices[:, 2]])

        # use the points at the boundary as the center
        if len(np.where(point_seg_ids == seg_id)[0]) > 0:
            cord_center = points[np.where(point_seg_ids == seg_id)[0][-1]]
        else:
            cord_center = cord0
    except:
        cord_center = cord0
    return cord_center


def fafb_to_block(x, y, z, return_pixel=False):
    '''
    (x,y,z):fafb坐标
    (x_block,y_block,z_block):block块号
    (x_pixel,y_pixel,z_pixel):块内像素号,其中z为帧序号,为了使得z属于中间部分,强制z属于[29,54)
    文件名：z/y/z-xxx-y-xx-x-xx
    '''
    x_block_float = (x + 17631) / 1736 / 4
    y_block_float = (y + 19211) / 1736 / 4
    z_block_float = (z - 15) / 26
    x_block = math.floor(x_block_float)
    y_block = math.floor(y_block_float)
    z_block = math.floor(z_block_float)
    x_pixel = (x_block_float - x_block) * 1736
    y_pixel = (y_block_float - y_block) * 1736
    z_pixel = (z - 15) - z_block * 26
    while z_pixel < 28:
        z_block = z_block - 1
        z_pixel = z_pixel + 26
    if return_pixel:
        return x_block, y_block, z_block, x_pixel, y_pixel, z_pixel
    else:
        return x_block, y_block, z_block


def get_vector_files(tree, filename_vector, filename_connector, filename_weight, vol_ffn1, get_gt_recall=False):
    """
    get grown vectors df for segments in a neuron
    :param tree: mapped tree neuron
    :return: grown vector df, the cords are in voxel space and the vectors are in normalized physical space
    """
    thresh = 0
    df = pd.read_csv(filename_connector, index_col=False)
    df_weight = pd.read_csv(filename_weight, index_col=False)
    df_vector = {'node0_segid': [], 'node1_segid': [],
                 'cord': [], 'cord0': [], 'vector0': [], 'cord1': [], 'vector1': [],
                 'score': [], 'neuron_id': []}
    df_vector = pd.DataFrame(df_vector)
    neuron_id = tree.id
    stat_distance = []
    hit = 0
    total_grow = 0
    if len(df) > 10:
        for i in df.index:
            try:
                weight1 = min(df_weight[str(int(df['node1_segid'][i]))][0], 40)
                weight0 = min(df_weight[str(int(df['node0_segid'][i]))][0], 40)
                score = min(weight0, weight1)
                if score > thresh:
                    cord0 = np.array(df['node0_cord'][i][1:-1].split(', ')).astype(float)
                    cord1 = np.array(df['node1_cord'][i][1:-1].split(', ')).astype(float)
                    cord0[2] = cord0[2] * 10
                    cord1[2] = cord1[2] * 10
                    cord = (cord0 + cord1) / 2
                    # stat_distance.append(np.linalg.norm(cord0 - cord1) * 4)
                    vector0 = compute_normalized_vector(cord0, cord)
                    vector1 = compute_normalized_vector(cord1, cord)
                    # node 0 is seg_start, which is a larger segment.
                    cord0[2] = cord0[2] / 10
                    cord1[2] = cord1[2] / 10
                    cord = (cord0 + cord1) / 2
                    center_node0 = refine_center(cord0, cord1, df['node0_segid'][i], vol_ffn1)
                    center_node1 = refine_center(cord1, cord0, df['node1_segid'][i], vol_ffn1)

                    if get_gt_recall:
                        hit_flag0 = df['node0_segid'][i] in bio_constrained_candidates(vol_ffn1, center_node0, vector0)
                        hit_flag1 = df['node1_segid'][i] in bio_constrained_candidates(vol_ffn1, center_node1, vector1)
                        hit = hit + int(hit_flag0) + int(hit_flag1)
                        total_grow = total_grow + 2

                    df_vector = df_vector.append(
                        {'node0_segid': int(df['node0_segid'][i]), 'node1_segid': int(df['node1_segid'][i]),
                         'cord': center_node0, 'cord0': cord0, 'vector0': vector0, 'cord1': cord1, 'vector1': vector1,
                         'score': score, 'neuron_id': neuron_id}, ignore_index=True)
                    df_vector = df_vector.append(
                        {'node0_segid': int(df['node1_segid'][i]), 'node1_segid': int(df['node0_segid'][i]),
                         'cord': center_node1, 'cord0': cord1, 'vector0': vector1, 'cord1': cord0, 'vector1': vector0,
                         'score': score, 'neuron_id': neuron_id}, ignore_index=True)
            except:
                continue
        for node in tree.ends.index:
            leaf_point = tree.ends.node_id[node]
            leaf_parent = tree.ends.parent_id[node]
            node_leaf = tree.nodes['node_id'] == leaf_point
            node_parent = tree.nodes['node_id'] == leaf_parent
            node0_segid = tree.nodes['seg_id'][node_leaf].item()
            cord0 = np.array([tree.nodes['x'][node_leaf].item(), tree.nodes['y'][node_leaf].item(),
                              tree.nodes['z'][node_leaf].item()])
            cord1 = np.array([tree.nodes['x'][node_parent].item(), tree.nodes['y'][node_parent].item(),
                              tree.nodes['z'][node_parent].item()])
            cord0[2] = cord0[2] * 10
            cord1[2] = cord1[2] * 10
            cord = cord0 + (cord0 - cord1) / 2
            vector0 = compute_normalized_vector(cord0, cord)
            vector1 = compute_normalized_vector(cord1, cord0)
            cord0[2] = cord0[2] / 10
            cord1[2] = cord1[2] / 10
            cord = cord0 + (cord0 - cord1) / 2
            center_node0 = refine_center(cord0, cord0 + (cord0 - cord1)*2, node0_segid, vol_ffn1)
            # row0 = pd.DataFrame(
            #     [{'node0_segid': int(node0_segid), 'node1_segid': int(-1),
            #       'cord': cord, 'cord0': cord0, 'vector0': vector0, 'cord1': cord1, 'vector1': vector1,
            #       'score': -1, 'neuron_id': neuron_id}])
            # row0.to_csv(filename_vector, mode='a', header=False, index=False)
            df_vector = df_vector.append({'node0_segid': int(node0_segid), 'node1_segid': int(-1),
                                          'cord': center_node0, 'cord0': cord0, 'vector0': vector0, 'cord1': cord1,
                                          'vector1': vector1,
                                          'score': -1, 'neuron_id': neuron_id}, ignore_index=True)
        df_vector.to_csv(filename_vector)
    else:
        print('invalid neuron', neuron_id)
    if get_gt_recall:
        return df_vector, [hit, total_grow]
    else:
        return df_vector
