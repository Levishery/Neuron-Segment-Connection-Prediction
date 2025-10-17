import open3d as o3d
import numpy as np
from plyfile import PlyData
import numpy as np

path = r'F:\data\evaluation\6957776911.ply'
pcd = o3d.io.read_point_cloud(path)
ids = PlyData.read(path).elements[0].data['x']
ids = np.expand_dims(ids, axis=1)
c1 = np.zeros(ids.shape)
c2 = np.ones(ids.shape)
y = np.concatenate([c1, c1, c2], axis=1)
pcd.colors = o3d.utility.Vector3dVector(y)
o3d.visualization.draw_geometries([pcd])