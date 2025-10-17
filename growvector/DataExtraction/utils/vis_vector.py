import matplotlib.pyplot as plt
from plyfile import PlyData
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv
import os
import h5py
import open3d as o3d
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors

def upsample_point_cloud(point_cloud, npoint, eps=1e-5):
    # 获取输入点云的维度
    N, D = point_cloud.shape

    # 如果目标点数小于等于输入点数，直接返回输入点云
    if npoint <= N:
        return point_cloud

    # 使用最近邻算法寻找每个点的最近邻点
    nbrs = NearestNeighbors(n_neighbors=2).fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)

    # 计算每个点的权重，权重为与最近邻点的距离
    weights = distances[:, 1]

    # 归一化权重
    weights /= np.sum(weights)

    # 生成新的点云
    upsampled_point_cloud = np.empty((npoint, D))

    # 复制输入点云到新的点云中
    upsampled_point_cloud[:N] = point_cloud

    # 根据权重进行插值
    for i in range(N, npoint):
        # 随机选择一个输入点的索引
        index = np.random.choice(N, p=weights.flatten())

        # 获取最近邻点的索引和距离
        nearest_index = indices[index][1]
        nearest_distance = distances[index][1]

        # 计算插值权重
        weight = np.random.uniform()

        # 根据插值权重进行插值
        upsampled_point_cloud[i] = point_cloud[index] + weight * (point_cloud[nearest_index] - point_cloud[index])

    return upsampled_point_cloud


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def sample_point_cloud(point_cloud, npoint, eps=1e-5):
    # 获取输入点云的维度
    N, D = point_cloud.shape

    # 如果目标点数小于等于输入点数，直接返回输入点云
    if npoint <= N:
        return farthest_point_sample(point_cloud, npoint)
    else:
        return upsample_point_cloud(point_cloud, npoint)



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def merge_rows(csv_file):
    col = {}
    vector = {}
    weight = {}
    nei = {}
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            segid = row[0]
            # row[2] = ' '.join(row[2].split())
            row[2] = list(map(float, row[2][1:-1].split()))
            row[3] = list(map(float, row[3][1:-1].split()))
            row[-2] = float(row[-2])
            if segid in col:
                nei[segid].append(row[1])
                col[segid].append(row[2])
                vector[segid].append(row[3])
                weight[segid].append(row[-2])
            else:
                nei[segid] = [row[1]]
                col[segid] = [row[2]]
                vector[segid] = [row[3]]
                weight[segid] = [row[-2]]
    return col, vector, weight, nei


def read_ply_cloud(filename):
    ply_data = PlyData.read(filename)
    points = ply_data['vertex'].data.copy()
    print(points.shape)
    cloud = np.empty([points.shape[0], 3])
    for i in range(len(points)):
        point = points[i]
        p = np.array([point[0], point[1], point[2]])
        cloud[i] = p
    return np.array(cloud)


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat


def get_arrow(col, vector, x, y, z):
    n, c = col.shape
    vec = np.zeros((n, c))
    for i in range(0, n):
        vec_Arr = vector[i]
        vec_len = np.linalg.norm(vec_Arr)
        vec[i] = vec_Arr / vec_len * [x // 10, y // 10, z // 10]
    return vec


def get_example_arrow_o3d(begin, vector, delta, scale):
    z_unit_Arr = np.array([0, 0, 1])
    begin = begin
    # end = np.add(begin,vec)
    vec_Arr = vector * 30 * scale

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.6,
        cone_radius=0.6,
        cylinder_height=1.4,
        cylinder_radius=0.4
    )
    color = np.asarray([0, 1, 0]) if delta > 0 else np.asarray([0, 0, 1])
    mesh_arrow.paint_uniform_color(color)
    mesh_arrow.compute_vertex_normals()

    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    # mesh_arrow.translate(0.5*(np.array(end) - np.array(begin)))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow


def get_arrow_o3d(begin, vector, delta, scale):
    z_unit_Arr = np.array([0, 0, 1])
    begin = begin
    # end = np.add(begin,vec)
    vec_Arr = vector*scale

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=60, origin=[0, 0, 0])

    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.6,
        cone_radius=0.6,
        cylinder_height=1.4,
        cylinder_radius=0.4
    )
    # color =  np.asarray([1,0,0])*delta + np.asarray([0,0,1])*(1-delta) if delta>0 else np.asarray([0,1,0])
    # color = np.asarray([0, 1, 0]) if delta > 0 else np.asarray([0, 0, 1])
    color = np.asarray([0, 1, 0])
    mesh_arrow.paint_uniform_color(color)
    # mesh_arrow.paint_uniform_color([0, 0, 0])
    mesh_arrow.compute_vertex_normals()

    rot_mat = caculate_align_mat(vec_Arr)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    # mesh_arrow.translate(0.5*(np.array(end) - np.array(begin)))
    mesh_arrow.translate(np.array(begin))  # 0.5*(np.array(end) - np.array(begin))
    return mesh_arrow


def score2delta(scores):
    #min_score = min(scores)
    #max_score = max(scores)
    min_score = 0
    max_score = max(scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) if score>0 else -1 for score in scores]
    normalized_scores = [np.min([score/150, 1]) for score in scores]
    return normalized_scores


def vis_plt(col, vector, x, y, z, xlist, ylist, zlist):
    len = get_arrow(col, vector, x, y, z)
    # len = np.full((col.shape[0], 3), 200)
    # print(len)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xlist, ylist, zlist)
    ax.quiver(col[:, 0], col[:, 1], col[:, 2], len[:, 0], len[:, 1], len[:, 2], color=[(1, 0, 0, 0.8)])
    # Add text annotations
    for loc, annotation in zip(col, score_data):
        ax.text(loc[0], loc[1], loc[2], annotation, color='black')
    plt.show()


def vis_o3d(col, vector, point, weight, scale):
    point_clouds = []
    fps = o3d.geometry.PointCloud()  # 定义点云
    fps.points = o3d.utility.Vector3dVector(point)
    color = np.random.rand(3)
    fps.paint_uniform_color(color)
    # fps.paint_uniform_color([0.0, 1.0, 0.0])
    point_clouds.append(fps)
    col = np.array(col)
    vector = np.array(vector)
    weight = score2delta(np.array(weight))
    key_points = o3d.geometry.PointCloud()  # 定义点云
    key_points.points = o3d.utility.Vector3dVector(col)  # 定义点云坐标位置
    key_points.paint_uniform_color([0.0, 0.0, 0.0])
    # point_clouds.append(key_points)
    vector_points = o3d.geometry.PointCloud()  # 定义点云
    vector_points.points = o3d.utility.Vector3dVector(vector)  # 定义点云坐标位置
    vector_points.paint_uniform_color([1.0, 0.0, 0.0])
    geometry_list = []
    for i in range(0, col.shape[0]):
        mesh_arrow = get_example_arrow_o3d(col[i], vector[i], weight[i], scale)
        print(vector[i], col[i], weight[i])
        geometry_list.append(mesh_arrow)
    o3d.visualization.draw_geometries(
        point_clouds + geometry_list
    )
    print('\n')


def get_scale(point):
    xlist = point[:, 0]
    ylist = point[:, 1]
    zlist = point[:, 2]
    x, y, z = max(xlist) - min(xlist), max(ylist) - min(ylist), max(zlist) - min(zlist)
    # vis_plt(col, vector, x, y, z, xlist, ylist, zlist)
    scale = max(x, y, z) / 1000
    return scale


if __name__ == "__main__":
    path = r'/h3cstore_nt/JaneChen/Point-detect/group_free_ptv3/Group_Free/experiment/fafb/group_free/test_1733326221/17852604/visualize/0'
    files = os.listdir(path)
    for file in files:
        print(file)
        filename = os.path.join(path, file)
        # filename = r'F:\flywire数据\point_det\4339962635001.h5'
        with h5py.File(filename, 'r') as f:
            # 读取 point_cloud 数据
            pc_data = f['point_cloud'][:]

            # 读取 vectors 数据
            center_data = f['center'][:]
            vector_data = f['vector'][:]
            score_data = f['score'][:]

        point = sample_point_cloud(pc_data, 2048)
        # point = pc_data
        center_data[:, 0] = center_data[:, 0] * 4
        center_data[:, 1] = center_data[:, 1] * 4
        center_data[:, 2] = center_data[:, 2] * 40
        col = np.array(center_data)
        vector = -np.array(vector_data)
        weight = np.array(score_data)
        scale = get_scale(point)
        vis_o3d(col, vector, point, weight, scale)
