import trimesh
import os
import os.path
import glob

trimesh.util.attach_to_log()

output_dir = "../occupancy_networks/external/mesh-fusion/occ_src/raw/"
obj_list = sorted(glob.glob("./data/obj/ShapeNetCore.v2/*/*/models/model_normalized.obj"))
# print(len(obj_list))

for obj in obj_list:
    fileName = os.path.join(os.path.normpath(obj).split(os.sep)
                 [-4], os.path.normpath(obj).split(os.sep)[-3])
    mesh = trimesh.load(obj, force='mesh')
    # save as .off, retain file name
    directory = output_dir+fileName+".off"
    
    if not os.path.exists(os.path.dirname(directory)):
        os.makedirs(os.path.dirname(directory))
    mesh.export(directory)
