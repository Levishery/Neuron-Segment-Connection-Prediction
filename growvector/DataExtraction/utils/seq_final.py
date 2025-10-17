import math
import time
import random
import struct
import skeletor as sk
import navis
from tqdm import tqdm
import collections
import pandas as pd
from cloudvolume import Skeleton
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
from cloudvolume import CloudVolume
import cloudvolume
from PIL import Image
import numpy as np
import csv
import os


def find_path(skel, node_index):
    path = []
    node_pos = []
    navis.strahler_index(skel)                          
    path.append(skel.nodes['seg_id'][node_index])
    pos = [skel.nodes['x'][node_index], skel.nodes['y'][node_index], skel.nodes['z'][node_index]]
    node_pos.append(pos)
    while 1:
        parent_node_id = skel.nodes['parent_id'][node_index]
        try:
            parent_node_index = skel.nodes.index[np.where(skel.nodes['node_id'].values == parent_node_id)].item()
        except:
            break
        #parent_node_index = skel.nodes.index[parent_id]
        path.append(skel.nodes['seg_id'][parent_node_index])
        pos = [skel.nodes['x'][parent_node_index], skel.nodes['y'][parent_node_index], skel.nodes['z'][parent_node_index]]
        node_pos.append(pos)
        node_index = parent_node_index
    #node_index = skel.nodes.index[np.where(skel.nodes['node_id'].values == parent_node_id)].item()
    #path.append(skel.nodes['seg_id'][node_index])
    return path, node_pos

def filter(seg, pos):
    seg = np.asarray(seg)
    pos = np.asarray(pos)
    seg_count=collections.Counter(seg)
    for key, value in seg_count.items():
        index = np.where(seg==key)
        try:
            s, e = index[0][1], index[0][-1]
            seg = np.delete(seg, np.arange(s,e+1))
            pos = np.delete(pos, np.arange(s,e+1))
        except:
            continue
    l = len(seg)
    return l, seg, pos

def get_removed(seg):
    removed_seg = []
    seg = np.asarray(seg)
    seg_count=collections.Counter(seg)
    for key, value in seg_count.items():
        index = np.where(seg==key)
        try:
            s, e = index[0][1], index[0][-1]
            removed_seg = np.setdiff1d(np.unique(seg[s:e+1]), [key])
        except:
            continue
    return removed_seg

def decompose_tree(tree):
    paths = []
    stack = [(tree.root[0], [])]

    while stack:
        node, path = stack.pop()
        node_index = tree.nodes.index[np.where(tree.nodes['node_id'].values == node)].item()
        path.append(tree.nodes['seg_id'][node_index])

        children = tree.nodes.index[np.where(tree.nodes['parent_id'].values == node)]
        if len(children) > 1:  # 判断是否为分支点
            paths.append(path)
            for child in children:
                path = []  # 重置当前路径为空列表
                child_id = tree.nodes['node_id'][child]
                path.append(tree.nodes['seg_id'][node_index])
                stack.append((child_id, path.copy()))

        elif len(children) == 0:
            paths.append(path)
        else:
            child = children[0]
            child_id = tree.nodes['node_id'][child]
            stack.append((child_id, path.copy()))

    return paths

def get_ignore_segid(mapped_skel):
    # the warpped segment, usually mito, should be removed
    removed_seg = []
    paths = decompose_tree(mapped_skel)
    for path in paths:
        remove_in_path = get_removed(path)
        # if len(remove_in_path) > 0:
        #     print('debug')
        for seg in remove_in_path:
            removed_seg.append(seg)
    seg_long = [int(key) for key, value in mapped_skel.segment_length.items() if value > 5]
    removed_seg = np.setdiff1d(removed_seg, seg_long)
    return removed_seg


if __name__ == "__main__":
    # 可以改进的：1. soma 2. filter thresh
    #create_logger(name='l1', file='/braindat/lab/liusl/flywire/log/flywire2fafbffn_test.log', sub_print=True, file_level='DEBUG')
    #target_tree_path = '/braindat/lab/liusl/flywire/test-skel/tree_data'
    #target_connector_path = '/braindat/lab/wangcx/flywire/sequence/720575940630732087'
    img_path = '/braindat/lab/wangcx/fafb/test/img'
    flywire_skel_path = '/braindat/lab/liusl/flywire/test-skel/tree_data'
    id_dir ='/braindat/lab/wangcx/fafb/test/id'
    sequence_dir = '/braindat/lab/wangcx/fafb/test/seq_final'
    file_gt_skels = os.listdir(flywire_skel_path)
    #random.shuffle(file_gt_skels)
    # get k neurons per iter
    for file_gt_skel in file_gt_skels:
        #mapped_skel = navis.read_json('/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data/720575940614462879.json')
        mapped_skel = navis.read_json(os.path.join(flywire_skel_path, file_gt_skel))
        mapped_skel = mapped_skel[0]
        if mapped_skel.n_nodes < 100:
            print(f'#W#{mapped_skel.id} is not a neuron.')
            continue
        #mapped_skel.soma = None
        """
        id_path = os.path.join(id_dir, str(mapped_skel.id)+'.csv')
        if not os.path.exists(id_path):
            ids = mapped_skel.nodes['seg_id'][1:-1].values
            data2 = pd.DataFrame(data = ids)
            data2.to_csv(id_path,header=False, index=None)
        else :
            continue
        """
        os.makedirs(os.path.join(sequence_dir, str(mapped_skel.id)))
        leafs = mapped_skel.leafs.index.values
        navis.strahler_index(mapped_skel)  
        i=0
        r = []
        for leaf in leafs:
            path, node_pos = find_path(mapped_skel, leaf)
            l, new_seg, new_pos = filter(path, node_pos)
            sequence_file = os.path.join(sequence_dir, str(mapped_skel.id), '%s'%l + '_%s.csv'%i)
            if l < 1 :
                continue
            else:
                data = pd.DataFrame(data = [new_seg.tolist(), new_pos.tolist()])
                data.to_csv(sequence_file, mode='a', header=False, index=None)
                i = i+1
