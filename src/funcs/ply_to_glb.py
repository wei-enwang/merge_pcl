import open3d as o3d
import os

data_path = os.path.join(os.getcwd(), "../clouds/")

for file in os.listdir(data_path):
    if file.endswith(".ply"):
        mesh = o3d.io.read_triangle_mesh(os.path.join(data_path, file))
        o3d.io.write_triangle_mesh(os.path.join(data_path, file[:-3]+"glb"), mesh)