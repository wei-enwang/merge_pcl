import trimesh
import os.path
import glob

trimesh.util.attach_to_log()

output_dir = "../occupancy_network/external/mesh-fusion/output/"
obj_list = sorted(glob.glob("./data/obj/ShapeNetCore.v2/*/*/model/model_normalized.obj"))
print(len(obj_list))
obj = obj_list[0]
# for obj in obj_list:

fileName = os.path.split(os.path.dirname(obj))[-2]
mesh = trimesh.load(obj, force='mesh')
    # save as .off, retain file name
mesh.export(output_dir+fileName+".off")