# 8*8*40; 16*16*40; 512*512*640
import navis
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import networkx as nx
from cloudvolume import CloudVolume
from sklearn.neighbors import NearestNeighbors
#from cilog import create_logger
import time
import math
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
import pickle
from matplotlib import pyplot as plt
import os
import json


def connection_weight(skel, node0, node1):
    """
    Compute connection length of the two directions by cutting the tree
    :param skel: mapped tree neuron
    :param edge: connector
    :return: connection length of the two directions
    """
    for subtree in skel.subtrees:
        if node0 and node1 in subtree:
            if navis.distal_to(skel, node0, node1):
                sub_tree0, sub_tree1 = navis.cut_skeleton(skel, node0)
                return sub_tree0.n_nodes, sub_tree1.n_nodes - 1
            else:
                sub_tree1, sub_tree0 = navis.cut_skeleton(skel, node1)
                return sub_tree0.n_nodes - 1, sub_tree1.n_nodes
    print(f'#E#cannot find {node0} in {skel.id}')

def get_connector(skel):
    """
    :param skel: mapped tree neuron
    :return: connector table, wrapped nodes filtered by connection record (all_connection)
    """
    connector_table = {'node0_id': [], 'node1_id': [], 'node0_cord': [], 'node1_cord': [], 'node0_segid': [],
                       'node1_segid': [], 'node0_weight': [], 'node1_weight': [], 'Strahler order': []}
    connector_table = pd.DataFrame(connector_table)
    all_connection = []
    for edge in skel.edges:
        node0 = skel.nodes['node_id'] == edge[0]
        node1 = skel.nodes['node_id'] == edge[1]
        all_connection.append([skel.nodes['seg_id'][node0].item(), skel.nodes['seg_id'][node1].item()])
    for edge in skel.edges:
        node0 = skel.nodes['node_id'] == edge[0]
        node1 = skel.nodes['node_id'] == edge[1]
        if skel.nodes['seg_id'][node0].item() != skel.nodes['seg_id'][node1].item():
            if [skel.nodes['seg_id'][node1].item(), skel.nodes['seg_id'][node0].item()] not in all_connection:
                # filter wrapped connector
                weight0, weight1 = connection_weight(skel, edge[0], edge[1])
                try:
                    connector_table.loc[len(connector_table.index)] = [edge[0], edge[1],
                                                                   [skel.nodes['x'][node0].item(),
                                                                    skel.nodes['y'][node0].item(),
                                                                    skel.nodes['z'][node0].item()],
                                                                   [skel.nodes['x'][node1].item(),
                                                                    skel.nodes['y'][node1].item(),
                                                                    skel.nodes['z'][node1].item()],
                                                                   skel.nodes['seg_id'][node0].item(),
                                                                   skel.nodes['seg_id'][node1].item(),
                                                                   weight0, weight1,
                                                                   skel.nodes.strahler_index[node1].item()]
                except:
                    print(f'#W#Maybe is not neuron!')
                    return None
    return connector_table
    

def get_segment_info(skel):
    segment_node_dict = {}
    segment_index_dict = {}
    for node in tqdm(range(max(skel.nodes.index))):
        try:
            seg_id = skel.nodes['seg_id'][node]
            if seg_id in segment_node_dict:
                segment_node_dict[seg_id].append(skel.nodes['node_id'][node])
                segment_index_dict[seg_id].append(node)
            else:
                segment_node_dict[seg_id] = [skel.nodes['node_id'][node]]
                segment_index_dict[seg_id] = [node]
        except:
            continue
    skel.segment_bbox, skel.segment_center = segment_bbox(skel, segment_index_dict)
    return skel

def segment_bbox(skel, segment_index_dict):
    """
    :param segment_node_dict: segments -> nodes
    :return: segment_weight_dict: segments -> AABB
    """
    segment_bbox_dict = {}
    segment_center_dict = {}
    for seg_id in segment_index_dict:
        x, y, z = (max(skel.nodes['x'][segment_index_dict[seg_id]])- min(skel.nodes['x'][segment_index_dict[seg_id]])),\
                  (max(skel.nodes['y'][segment_index_dict[seg_id]])- min(skel.nodes['y'][segment_index_dict[seg_id]])),\
                  (max(skel.nodes['z'][segment_index_dict[seg_id]])-min(skel.nodes['z'][segment_index_dict[seg_id]]))
        cx, cy, cz = (max(skel.nodes['x'][segment_index_dict[seg_id]])+ min(skel.nodes['x'][segment_index_dict[seg_id]]))/2,\
                  (max(skel.nodes['y'][segment_index_dict[seg_id]])+ min(skel.nodes['y'][segment_index_dict[seg_id]]))/2,\
                  (max(skel.nodes['z'][segment_index_dict[seg_id]])+min(skel.nodes['z'][segment_index_dict[seg_id]]))/2
        segment_bbox_dict[seg_id]= [x, y, z]
        segment_center_dict[seg_id]= [cx, cy, cz]
    return segment_bbox_dict, segment_center_dict

def get_save(skel):
    save_table = {'seg_id': [], 'segment_bbox': [],'segment_center': [], 'segment_length': [], 'segment_block': []}
    save_table = pd.DataFrame(save_table)
    for seg_id in skel.segment_bbox:
        cord = skel.segment_center[seg_id]
        x_b, y_b, z_b = fafb_to_block(cord[0], cord[1], cord[2])
        save_table.loc[len(save_table.index)] = [seg_id,
                                                 skel.segment_bbox[seg_id],
                                                 skel.segment_center[seg_id],
                                                 skel.segment_length[str(seg_id)],
                                                 (x_b, y_b, z_b)]
    return save_table

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

if __name__ == "__main__":
    # 可以改进的：1. soma 2. filter thresh
    #create_logger(name='l1', file='/braindat/lab/liusl/flywire/log/flywire2fafbffn_test.log', sub_print=True, file_level='DEBUG')
    #target_tree_path = '/braindat/lab/liusl/flywire/test-skel/tree_data'
    target_connector_path = '/braindat/lab/wangcx/fafb/test/connector_data'
    target_save_path = '/braindat/lab/wangcx/fafb/test/save_data'
    flywire_skel_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
    file_gt_skels = os.listdir(flywire_skel_path)
    #random.shuffle(file_gt_skels)
    # get k neurons per iter
    
    for file_gt_skel in file_gt_skels[148:]:
        #mapped_skel = navis.read_json('/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data/720575940623502323.json')
        mapped_skel = navis.read_json(os.path.join(flywire_skel_path, file_gt_skel))
        mapped_skel = mapped_skel[0]
        if mapped_skel.n_nodes < 100:
            print(f'#W#{mapped_skel.id} is not a neuron.')
            continue
        mapped_skel.soma = None
        filename = os.path.join(target_connector_path, str(mapped_skel.id) + '_connector.csv')
        if not os.path.exists(filename):
            print(f'#I#building segment tree for {mapped_skel.id}')
            start = time.time()
            mapped_skel = get_segment_info(mapped_skel)
            connector_table = get_connector(mapped_skel)
            save_table = get_save(mapped_skel)
            con_time = time.time()
            print(f'#D#connector computing time {con_time - start}')
            filename_connector = os.path.join(target_connector_path, str(mapped_skel.id) + '_connector.csv')
            filename_save = os.path.join(target_save_path, str(mapped_skel.id) + '_save.csv')
            if connector_table is not None:
                connector_table.to_csv(filename_connector, index=False)
                save_table.to_csv(filename_save, index=False)
                print(f'#D#saving time {time.time() - con_time}')
