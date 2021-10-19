import numpy as np
from create_scene import Scene

save_data_dir = "data/imgs/"
num_scenes = 10
scene = Scene()

for i in range(num_scenes):
    new_scene = scene.raw_rgbd()
    np.save(save_data_dir+str(i)+'.npy', new_scene)