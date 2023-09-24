import os
import trimesh
import polyscope as ps
import open3d as o3d
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cup_dir = "/home/ikun/obj/cup_reshaped_upright"
mesh0 = trimesh.load(f'{cup_dir}/1/model.obj')
mesh1 = trimesh.load(f'{cup_dir}/2/model.obj')

mesh0.apply_scale(0.3)
mesh1.apply_scale(0.3)

trans_mat = np.eye(4)
trans_mat[0:3, -1] = np.array([0, 0, 0.14])
mesh1.apply_transform(trans_mat)

pcd1 = mesh0.sample(5000)
pcd2 = mesh1.sample(5000)

query_pts = np.random.normal(0.0, 0.006, size=(500, 3)) + np.array([0, 0, 0.085])

coords = np.array([[0.1, 0, 0, 0],
                   [0., 0.1, 0, 0],
                   [0, 0, 0.1, 0],
                   [1, 1, 1, 1]])
trans_pose = np.eye(4)
trans_pose[0:3, -1] = np.array([0, 0, 0.085])
coords = trans_pose @ coords
coords = coords[0:3, :]
coords = coords.T
nodes = coords

ps.init()
ps.set_up_dir("z_up")

ps.init()
ps.set_up_dir("z_up")

ps.register_curve_network("edge_x_ref", nodes[[0, 3]], np.array([[0, 1]]), enabled=True, radius=0.002,
                          color=(1, 0, 0))
ps.register_curve_network("edge_y_ref", nodes[[1, 3]], np.array([[0, 1]]), enabled=True, radius=0.002,
                          color=(0, 1, 0))
ps.register_curve_network("edge_z_ref", nodes[[2, 3]], np.array([[0, 1]]), enabled=True, radius=0.002,
                          color=(0, 0, 1))

ps.register_point_cloud("gripper", pcd1, radius=0.005, enabled=True)
ps.register_point_cloud("obj", pcd2, radius=0.005, enabled=True, color=(1, 0, 0))
ps.register_point_cloud("pts", query_pts, radius=0.005, enabled=True, color=(0, 0, 1))
# ps.register_point_cloud("place", place_pos.reshape((1, 3)), radius=0.004, enabled=True)

ps.show()

# path = os.path.join(BASE_DIR, "examples/interact_pts")
# np.savez(path, ref=query_pts, obj1=pcd1, obj2=pcd2)
