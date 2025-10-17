# 点与点cd>10
# /braindat/lab/liusl/flywire/block_data/30_percent
import os
from plyfile import PlyData
import navis
import time
import numpy as np
import h5py
from cilog import create_logger
from tqdm import tqdm
from matplotlib import pyplot as plt
from cloudvolume.datasource.precomputed.skeleton.sharded import ShardedPrecomputedSkeletonSource
from cloudvolume import CloudVolume

from utils import get_segment_info, get_save, get_complete_pc, get_ignore_segid, get_vector_files, \
    segment_partition, pointcloud_partition, vector_partition


def get_vectors(mapped_skel, filename_save, filename_vector, filename_connector, filename_weight, dir_pc, dir_target, fig_dir, vol_ffn1, score_index_path):
    """
    get grown vectors for each segment in a neuron
    :param mapped_skel: mapped tree neuron
    :return: grown vector file path, saved as h5 files (numpy points of point cloud and vectors)
    """
    # get segment bbox for pc sampling
    get_gt_recall = False
    print(f'#I#building grown vector data for {mapped_skel.id}')
    start = time.time()
    ignore_ids = get_ignore_segid(mapped_skel)
    mapped_skel = get_segment_info(mapped_skel)
    save_table = get_save(mapped_skel)
    save_table.to_csv(filename_save, index=False)

    # get connector info
    if get_gt_recall:
        df_vector, [hit, total_grow] = get_vector_files(mapped_skel, filename_vector, filename_connector, filename_weight, vol_ffn1, get_gt_recall)
        f_recall = open('/braindat/lab/liusl/flywire/flywire_growvector/v1/gt_recall.txt', 'a+')
        f_recall.writelines(str(int(mapped_skel.id)) + '#' + str(hit) + '/' + str(total_grow) + '\n')
        f_recall.flush()
    else:
        df_vector = get_vector_files(mapped_skel, filename_vector, filename_connector, filename_weight, vol_ffn1,
                                     get_gt_recall)

    # sample pc for each segment
    os.makedirs(dir_pc, exist_ok=True)
    os.makedirs(dir_target, exist_ok=True)
    vector_num = []
    f_score = open(score_index_path, 'a+')
    for i in range(0, len(save_table)):
        row = save_table.iloc[i]
        # ignore small segments
        if row[3] > 1.5 and (int(row[0]) not in ignore_ids):
            try:
                print(f'#I#Sampling point cloud for {row[0]}')
                filename_pcd = get_complete_pc(row, dir_pc, vol_ffn1)
                # filename_pcd = '/braindat/lab/liusl/flywire/flywire_neuroskel/point_cloud/720575940637798003/1745678884.ply'
                pcd = PlyData.read(filename_pcd)
                print(f'#I#Segment parition for {row[0]}')
                df_partition_vector, patitioned_skels = segment_partition(row[0], vol_ffn1)
                patitioned_pc = pointcloud_partition(pcd, patitioned_skels)
                print(f'#I#Vector partition for {row[0]}')
                partitoned_vectors, score_list = vector_partition(df_vector, df_partition_vector, patitioned_skels, int(row[0]), ignore_ids)
                if len(partitoned_vectors) == 1 and len(partitoned_vectors[0]) == 1:
                    continue
                print(f'#I#Saving {row[0]}')
                for index, pc in enumerate(patitioned_pc):
                    if len(partitoned_vectors[index]['score']) > 1:
                        pc_id = int(row[0]) * 1000 + index
                        filename = os.path.join(dir_target, str(pc_id)+'.h5')
                        # save path&score dict
                        f_score.writelines(filename + '#' + str(score_list[index]) + '#'
                                           + str(score_list[index]/len(partitoned_vectors[index]['score'])) + '\n')
                        with h5py.File(filename, 'w') as f:
                            # 写入 point_cloud 数据
                            f.create_dataset('point_cloud', data=pc)
                            # 写入 vectors 数据
                            f.create_dataset('center', data=np.asarray(list(partitoned_vectors[index]['cord'])))
                            f.create_dataset('vector', data=np.asarray(list(partitoned_vectors[index]['vector0'])))
                            f.create_dataset('score', data=np.asarray(list(partitoned_vectors[index]['score'])))
                            f.create_dataset('seg_weight', data=row[3])
                        vector_num.append(len(partitoned_vectors[index]['score']))
            except:
                print(f'#W#cannot process segment {row[0]}')
        # else:
        #     df_seg_vector = df_vector[df_vector['node0_segid'] == int(row[0])]
        #     if len(df_seg_vector) == 1:
        #         print(len(df_seg_vector))
    plt.hist(vector_num, bins=np.unique(np.asarray(vector_num)))
    plt.savefig(fig_dir + str(mapped_skel.id) + '.pdf')
    plt.clf()
    f_score.flush()
    f_score.close()
    print(f'#D# Neuron time {time.time() - start}')


if __name__ == "__main__":
    prefix = 'test/'  # used v2 in experiment
    create_logger(name='l1', file='/braindat/lab/liusl/flywire/flywire_growvector/' + prefix + 'vector_gt.log', sub_print=True,
                  file_level='DEBUG')
    target_dir = '/braindat/lab/liusl/flywire/flywire_growvector/' + prefix + 'grow_vector'
    fig_dir = '/braindat/lab/liusl/flywire/flywire_growvector/' + prefix + 'stat_figs/'
    target_bbox_path = '/braindat/lab/liusl/flywire/flywire_growvector/' + prefix + 'bbox_data'
    target_vector_path = '/braindat/lab/liusl/flywire/flywire_growvector/' + prefix +'vector_data'
    flywire_skel_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/tree_data'
    flywire_connector_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/connector_data'
    flywire_weight_path = '/braindat/lab/liusl/flywire/flywire_neuroskel/visualization'
    pc_path = '/braindat/lab/liusl/flywire/flywire_growvector/' + prefix + 'point_cloud'
    score_index_path = '/braindat/lab/liusl/flywire/flywire_growvector/' + prefix + 'score_path.txt'
    file_gt_skels = os.listdir(flywire_skel_path)
    # file_gt_skels = ['720575940616769078.json', '720575940618983083.json', '720575940630355148.json']

    navis.patch_cloudvolume()
    vol_ffn1 = CloudVolume('file:///braindat/lab/lizl/google/google_16.0x16.0x40.0', cache=True, parallel=True)  #
    vol_ffn1.parallel = 8
    vol_ffn1.meta.info['skeletons'] = 'skeletons_32nm'
    vol_ffn1.skeleton.meta.refresh_info()
    vol_ffn1.skeleton.meta.info['sharding']['hash'] = 'murmurhash3_x86_128'
    vol_ffn1.skeleton = ShardedPrecomputedSkeletonSource(vol_ffn1.skeleton.meta, vol_ffn1.cache, vol_ffn1.config)

    for file_gt_skel in file_gt_skels:
        mapped_skel = navis.read_json(os.path.join(flywire_skel_path, file_gt_skel))
        mapped_skel = mapped_skel[0]
        if mapped_skel.n_nodes < 100:
            print(f'#W#{mapped_skel.id} is not a neuron.')
            continue
        vol_ffn1.cache.flush()
        filename_bbox = os.path.join(target_bbox_path, str(mapped_skel.id) + '_seginfo.csv')
        filename_vector = os.path.join(target_vector_path, str(mapped_skel.id) + '_vector.csv')
        filename_connector = os.path.join(flywire_connector_path, str(mapped_skel.id) + '_connector.csv')
        filename_weight = os.path.join(flywire_weight_path, str(mapped_skel.id) + '_save.csv')

        dir_pc = os.path.join(pc_path, str(mapped_skel.id))
        dir_target = os.path.join(target_dir, str(mapped_skel.id))
        if not os.path.exists(filename_bbox):
            get_vectors(mapped_skel, filename_bbox, filename_vector, filename_connector, filename_weight, dir_pc, dir_target, fig_dir, vol_ffn1, score_index_path)
