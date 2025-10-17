import pdb

import numpy as np
import math
import pandas as pd
import os
import shutil
from tqdm import tqdm
import tifffile as tf
import random
import imageio
import glob
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
import navis
import h5py
from scipy.spatial import cKDTree
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
from cloudvolume import CloudVolume
from skimage.transform import resize
# pip install connected-components-3d
import cc3d
import cv2
from queue import Queue, LifoQueue, PriorityQueue
from .select_1 import compute_normalized_vector


def find_perpendicular_plane(normal_vector, point_on_plane):
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    a, b, c = normal_vector
    d = -np.dot(normal_vector, point_on_plane)
    return a, b, c, d


def spc(vol, segid, filename_pcd, lx, ly, lz, cx, cy, cz):
    temp1 = 0
    vol = np.where(vol == segid, 255, vol)
    vol = np.where(vol != 255, 0, vol)
    fid1 = open(filename_pcd, 'w')
    for i in range(0, int(lz) * 2):
        data_tmp0 = vol[:, :, i]
        data_tmp0 = data_tmp0.astype(np.uint8)
        contours0 = cv2.findContours(data_tmp0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary0 = np.array(contours0[0])
        boundary0 = [y for x in boundary0 for y in x]
        boundary0 = np.array(boundary0).squeeze()
        if (len(boundary0.shape) == 1):
            continue
        temp1 += len(boundary0)
        for n in range(0, len(boundary0)):
            # fid1.write(str((boundary0[n, 1]*16 + cx*4 - lx*16)*sample_factor) + "\t" + str(boundary0[n, 0]*16 + cy*4 - ly*16)*sample_factor)
            # fid1.write("\t" + str((i + cz - lz))*sample_factor)
            fid1.write(str((boundary0[n, 1] + cx - lx) * 4) + "\t" + str((boundary0[n, 0] + cy - ly) * 4))
            fid1.write("\t" + str((i + cz - lz) * 10))
            fid1.write("\n")
    fid1.close()
    del vol
    with open(filename_pcd, 'r+') as fid_:
        content = fid_.read()
        fid_.seek(0, 0)
        fid_.write(
            'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n'.format(
                temp1) + content)
        fid_.close()


def partition_once(skel, p_vectors, p_cords, df_vector, method='branch'):
    """
    partition the skel once and record partition information
    Input: segment skel to partition, partition cords & vectors
    parameters: method = 'branch' or 'longest_path'
    Returns: partitioned skel

    """
    if method == 'branch':
        main_branchpoint = navis.find_main_branchpoint(skel, method='longest_neurite')
        main_branchpoint = main_branchpoint[0] if isinstance(main_branchpoint, list) else main_branchpoint
        childs = list(skel.nodes[skel.nodes.parent_id == main_branchpoint].node_id.values)
        assert len(childs) == 2, "branch point contains more than 2 child"
        try:
            split = navis.cut_skeleton(skel, [childs[0], main_branchpoint])
        except:
            skel = navis.reroot_skeleton(skel, childs[1])
            split = navis.cut_skeleton(skel, [childs[0], main_branchpoint])
        child = childs[0]
        grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
        p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
        p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])
        p_vector = p_cords_grandchild - p_cords_child
        p_cord = (p_cords_child + p_cords_grandchild) / 2
        p_vectors.append(p_vector)
        p_cords.append(p_cord)

        child = childs[1]
        grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
        p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
        p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])

    elif method == 'longest_path':
        # Find the two most distal points
        leafs = skel.leafs.node_id.values
        dists = navis.geodesic_matrix(skel, from_=leafs)[leafs]

        # This might be multiple values
        mx = np.where(dists == np.max(dists.values))
        start = dists.columns[mx[0][0]]

        # Reroot to one of the nodes that gives the longest distance
        skel.reroot(start, inplace=True)
        if isinstance(skel, navis.NeuronList): skel = skel[0]
        segments = navis.graph_utils._generate_segments(skel, weight='weight')
        longest_path = segments[0]
        cut_point = longest_path[int(len(longest_path) / 2)]
        # the cut point should not be a branching point. If so, use it's parent
        offset = 0
        while cut_point in skel.branch_points.node_id.values:
            offset += 1
            cut_point = longest_path[int(len(longest_path) / 2 + offset)]
        split_ = navis.cut_skeleton(skel, cut_point)
        assert len(split_) == 2
        # the order is [distal, proximal]
        # while function cut_skeleton preserves cut_point in both results, we want none-overlap skeletons.
        # therefore we assure the cut_point is not a branching point and remove the cut_point in the distal skeleton.
        split = navis.NeuronList([split_[1], navis.remove_nodes(split_[0], cut_point)])

        child = cut_point
        grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
        p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                    skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
        p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                         skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])

    else:
        raise ValueError("Invalid method")

    # visulize cut point
    # fig, ax = navis.plot2d(skel, method='2d', view=('x', '-y'))
    # cut_coords = skel.nodes.set_index('node_id').loc[cut_point, ['x', 'y']].values
    # ax.annotate('cut point',
    #             xy=(cut_coords[0], -cut_coords[1]),
    #             color='red',
    #             xytext=(cut_coords[0], -cut_coords[1] - 2000), va='center', ha='center',
    #             arrowprops=dict(shrink=0.1, width=2, color='red'),
    #             )
    # plt.show()
    p_vector = compute_normalized_vector(p_cords_child, p_cords_grandchild)
    p_cord = (p_cords_child + p_cords_grandchild) / 2
    p_vectors.append(p_vector)
    p_cords.append(p_cord)
    df_vector = df_vector.append({'node0_id': int(child), 'node1_id': int(grand_child),
                 'cord': p_cord, 'cord0': p_cords_child, 'vector0': p_vector, 'cord1': p_cords_grandchild, 'vector1': -p_vector,
                 'score': 400, 'neuron_id': 0}, ignore_index=True)
    df_vector = df_vector.append({'node0_id': int(grand_child), 'node1_id': int(child),
                 'cord': p_cord, 'cord0': p_cords_grandchild, 'vector0': -p_vector, 'cord1': p_cords_child, 'vector1': p_vector,
                 'score': 400, 'neuron_id': 0}, ignore_index=True)

    return split, df_vector

def segment_partition(segment_id, vol_ffn1, thresh=20000):
    """
    Input: segment id, partition cords & vectors
    Returns: partition segment point cloud

    """

    # cut the first and second child with their grand child
    # raw voxel cord
    p_vectors = []
    p_cords = []
    patitioned_skels = []
    df_vector = pd.DataFrame({'node0_id': [], 'node1_id': [],
                 'cord': [], 'cord0': [], 'vector0': [], 'cord1': [], 'vector1': [],
                 'score': [], 'neuron_id': []})
    try:
        skel = vol_ffn1.skeleton.get(segment_id, as_navis=True)[0]
    except:
        # the segment is too small
        return df_vector, patitioned_skels

    navis.heal_skeleton(skel, inplace=True)
    skel_queue = Queue(maxsize=0)
    skel_queue.put(skel)
    while not skel_queue.empty():
        skel = skel_queue.get()
        if skel.cable_length > thresh:
            splits, df_vector = partition_once(skel, p_vectors, p_cords, df_vector, method='longest_path')
        else:
            splits = []
        # for s in splits:
        #     s.to_swc(os.path.join('/braindat/lab/liusl/flywire/log', str(s.cable_length) + '.swc'))
        # assert len(splits) == 3, "result in more than three branches"
        for split in splits:
            if split.cable_length > thresh:
                skel_queue.put(split)
            else:
                patitioned_skels.append(split)

    return df_vector, patitioned_skels


def assign_nearest(points, skel_partition):
    """
    Input: N*3 points (xyz), partitioned skeletons
    assign the points to the nearest skeleton
    Returns: N*4 points (xyz, skel id)

    """
    skel_points = np.concatenate(skel_partition)  # 将所有小点集合并成一个大点集
    skel_indices = np.concatenate([np.full(len(sk), i) for i, sk in enumerate(skel_partition)])  # 记录每个点属于哪个小点集

    tree = cKDTree(skel_points)  # 构建kd树以进行最近点搜索
    dists, indices = tree.query(points)  # 查询每个大点集中的点最近的小点集

    assigned_points = np.zeros((len(points), 4))
    assigned_points[:, :3] = points
    assigned_points[:, 3] = skel_indices[indices]

    patitioned_points = []
    for i in range(len(skel_partition)):
        patitioned_points.append(points[assigned_points[:, 3] == i])
    return assigned_points, patitioned_points


def pointcloud_partition(pc, patitioned_skels):
    """
    Input: original point cloud, partitioned skeletons
    partition through measuring the distance between the points and skeletons
    Returns: partition segment point cloud

    """
    pc_points = np.transpose(np.asarray([pc.elements[0].data['x'], pc.elements[0].data['y'], pc.elements[0].data['z']]),
                             [1, 0])
    if len(patitioned_skels) > 0:
        skel_partition = []
        for skel in patitioned_skels:
            skel_partition.append(np.transpose(np.asarray([skel.nodes['x'], skel.nodes['y'], skel.nodes['z']]), [1, 0]))
        assigned_points, patitioned_points = assign_nearest(pc_points, skel_partition)
    else:
        patitioned_points = [pc_points]
    return patitioned_points

def vector_partition(df_vector, df_partition_vector, patitioned_skels, segid, ignore_ids):
    """
    Input: full vector df, partitioned skeletons, ignore_ids
    partition through measuring the distance between the vector start point and skeletons. ignore segment in ignore_ids
    Returns: selected and partitioned vector df

    """
    score_list = []
    partitioned_vector = []
    if len(patitioned_skels) > 0:
        skel_partition = []
        for skel in patitioned_skels:
            skel_partition.append(np.transpose(np.asarray([skel.nodes['x'], skel.nodes['y'], skel.nodes['z']]), [1, 0]))

        neuron_id = df_vector['neuron_id'][0]
        df_seg_vector = df_vector[df_vector['node0_segid'] == segid]
        vector_starts = np.asarray(list(df_seg_vector['cord0']))
        vector_starts[:, 0] = vector_starts[:,0]*4
        vector_starts[:, 1] = vector_starts[:, 1] * 4
        vector_starts[:, 2] = vector_starts[:, 2] * 40
        assigned_points, patitioned_points = assign_nearest(vector_starts, skel_partition)

        for i in range(len(skel_partition)):
            score = 0
            df_vector = {'node0_segid': [], 'node1_segid': [],
                         'cord': [], 'cord0': [], 'vector0': [], 'cord1': [], 'vector1': [],
                         'score': [], 'neuron_id': []}
            df_vector = pd.DataFrame(df_vector)
            # get assigned vectors
            df_vector = df_vector.append(df_seg_vector[assigned_points[:, 3] == i], ignore_index=True)
            partitioned_vector.append(df_vector)
            for i in range(len(df_vector)):
                if df_vector.iloc[i][7] > 0:
                    score += df_vector.iloc[i][7]
                else:
                    score += 10
            score_list.append(score)

        skel_partition_nodeid = []
        for skel in patitioned_skels:
            skel_partition_nodeid.append(np.asarray(skel.nodes.node_id))
        skel_nodeids = np.concatenate(skel_partition_nodeid)  # 将所有node集合并成一个大点集
        skel_indices = np.concatenate([np.full(len(sk), i) for i, sk in enumerate(skel_partition_nodeid)])  # 记录每个node属于哪个skel
        for i in range(len(df_partition_vector)):
            row = df_partition_vector.iloc[i]
            skel_index = skel_indices[skel_nodeids == row['node0_id']]
            skel_target_index = skel_indices[skel_nodeids == row['node1_id']]
            assert len(skel_index) == 1 and len(skel_target_index) == 1
            seg_0 = segid*1000 + skel_index[0]
            seg_1 = segid*1000 + skel_target_index[0]
            row_partition_vector = {'node0_segid': seg_0, 'node1_segid': seg_1,
                         'cord': row['cord'], 'cord0': row['cord0'], 'vector0': row['vector0'], 'cord1': row['cord1'], 'vector1': row['vector1'],
                         'score': row['score'], 'neuron_id': neuron_id}
            partitioned_vector[skel_index[0]] = partitioned_vector[skel_index[0]].append(row_partition_vector, ignore_index=True)
            score_list[skel_index[0]] += 10
    else:
        score = 0
        partitioned_vector.append(df_vector[df_vector['node0_segid'] == segid])
        for i in range(len(partitioned_vector[0])):
            if partitioned_vector[0].iloc[i][7] > 0:
                score += partitioned_vector[0].iloc[i][7]
            else:
                score += 10
        score_list = [score]

    # remove the vectors related to the segments in ignore_ids
    partitioned_vector_dropped = []
    for partitioned_vector_item in partitioned_vector:
        remove_index = []
        for i in range(len(partitioned_vector_item)):
            if partitioned_vector_item.iloc[i][1] in ignore_ids:
                remove_index.append(partitioned_vector_item.index[i])
        partitioned_vector_dropped.append(partitioned_vector_item.drop(remove_index))

    return partitioned_vector_dropped, score_list


def segment_partition_cxw():
    """
    Input: segment id, partition cords & vectors
    Returns: partition segment point cloud

    """
    navis.patch_cloudvolume()
    segment_id = 4098055229
    vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True)  #
    vol_ffn1.parallel = 8
    vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
    vol_ffn1.skeleton.meta.refresh_info()
    vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
    vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)
    skel = vol_ffn1.skeleton.get(segment_id, as_navis=True)
    # fig, ax = navis.plot2d(skel, method='2d', view=('x', '-y'))
    # plt.show()
    # pruned = skel.prune_by_strahler(to_prune=[1,2], inplace=False)
    # fig, ax = skel.plot2d(color='red', view=('x', '-y'))
    # fig, ax = pruned.plot2d(color='green', ax=ax, linewidth=1, view=('x', '-y'))
    main_branchpoint = navis.find_main_branchpoint(skel)
    childs = list(skel.nodes[skel.nodes.parent_id == main_branchpoint[0]].node_id.values)
    split = navis.cut_skeleton(skel, [childs[0], main_branchpoint[0]])
    fig, ax = split[0].plot2d(color='red', view=('x', '-y'))
    fig, ax = split[1].plot2d(color='green', ax=ax, linewidth=1, view=('x', '-y'))
    fig, ax = split[2].plot2d(color='blue', ax=ax, linewidth=1, view=('x', '-y'))
    plt.savefig('/braindat/lab/wangcx/fafb/img/1.png')
    plt.show()

    # cut the first and second child with their grand child
    # raw voxel cord
    p_vectors = []
    p_cords = []

    child = childs[0]
    grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
    p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
    p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                     skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                     skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])
    p_vector = p_cords_grandchild - p_cords_child
    p_cord = (p_cords_child + p_cords_grandchild) / 2
    p_vectors.append(p_vector)
    p_cords.append(p_cord)

    child = childs[1]
    grand_child = skel.nodes[skel.nodes.parent_id == child].node_id.values[0]
    p_cords_child = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == child].item() / 4,
                                skel.nodes['y'][skel.nodes['node_id'] == child].item() / 4,
                                skel.nodes['z'][skel.nodes['node_id'] == child].item() / 40])
    p_cords_grandchild = np.asarray([skel.nodes['x'][skel.nodes['node_id'] == grand_child].item() / 4,
                                     skel.nodes['y'][skel.nodes['node_id'] == grand_child].item() / 4,
                                     skel.nodes['z'][skel.nodes['node_id'] == grand_child].item() / 40])
    p_vector = p_cords_grandchild - p_cords_child
    p_cord = (p_cords_child + p_cords_grandchild) / 2
    p_vectors.append(p_vector)
    p_cords.append(p_cord)
    print(p_cords, p_vectors)

    n = len(p_cords)
    size = [480, 480, 120]
    for i in range(0, n):
        sx, ex = int(p_cords[i][0] // 4 - size[0]), int(p_cords[i][0] // 4 + size[0])
        sy, ey = int(p_cords[i][1] // 4 - size[1]), int(p_cords[i][1] // 4 + size[1])
        sz, ez = int(p_cords[i][2] - size[2]), int(p_cords[i][2] + size[2])
        p_cord = p_cords[i] // [4, 4, 1] - [sx, sy, sz]
        p_vector = p_vectors[i] // [4, 4, 1]
        a, b, c, d = find_perpendicular_plane(p_vector, p_cord)
        r = 100
        vol = vol_ffn1[sx:ex, sy:ey, sz:ez]
        vol = np.where(vol == segment_id, 255, vol)
        vol = np.where(vol != 255, 0, vol)
        x, y, z = np.meshgrid(range(int(p_cord[0] - r), int(p_cord[0] + r)),
                              range(int(p_cord[1] - r), int(p_cord[1] + r)),
                              range(int(p_cord[2] - r), int(p_cord[2] + r)))
        plane = a * x + b * y + c * z + d
        index = np.argwhere(plane == 0) + [p_cord[0] - r, p_cord[1] - r, p_cord[2] - r]
        index = index.astype(int)
        del x, y, z, plane
        for k in range(index.shape[0]):
            vol[index[k][0], index[k][1], index[k][2]] = 0
        vol1 = cc3d.connected_components(np.squeeze(vol), connectivity=26)
        del vol
        unique_elements, counts = np.unique(vol1, return_counts=True)
        for j in range(1, unique_elements.shape[0]):
            filename = "/braindat/lab/wangcx/fafb/img/{}_{}_{}.ply".format(segment_id, i, j)
            spc(vol1, segid=j, filename_pcd=filename, lx=480, ly=480, lz=120, cx=p_cords[i][0], cy=p_cords[i][1],
                cz=p_cords[i][2])
        # segid = np.max(counts[1:-1])
        # filename= "/braindat/lab/wangcx/fafb/img/{}_{}.ply".format(segment_id, i)
        # spc(vol1, segid=segid, filename_pcd=filename, lx=480, ly=480, lz=120, cx=p_cords[i][0], cy=p_cords[i][1], cz=p_cords[i][2])


if __name__ == '__main__':
    segment_partition()
