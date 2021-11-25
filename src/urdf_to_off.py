import trimesh
import os
import glob

trimesh.util.attach_to_log()

output_dir = "./data/obj/"
obj_list = sorted(glob.glob("../ig_dataset/objects/*/*/shape/visual/*.obj"))
# print(obj_list)
for obj in obj_list:

    fileName = os.path.basename(obj)
    mesh = trimesh.load(obj, force='mesh')
        # save as .off, retain file name
    mesh.export(output_dir+fileName[:-3]+"off")