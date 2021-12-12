import numpy as np
import os.path
import glob
import json
from create_scene import Scene
from imageio import imwrite


def all_latent_generator(objects, data_dir, latent_size=128, save=True):
    """
    Every object has a latent representation
        objects: a list of objects
        data_dir: the place where the latent should be stored
    """

    # Make sure don't overwrite existing latents
    if os.path.exists(data_dir+"trial_latents.json"):
        print("Latents already exist. Generation aborted...\n")
        return

    latents = {}
    for obj in objects:
        latents[obj] = list(np.random.random_sample(size=latent_size))
        # sanity check
        assert len(latents[obj]) == latent_size
    
    
    if save:
        with open(data_dir+"trial_latents.json", "w") as data:
            json.dump(latents, data)


save_data_dir = "./data/imgs/"
save_latent_dir = "./data/latents/"
# change this to 10000 when training
num_scenes = 10
max_obj_num = 10

img_size = 224
scene = Scene()

# object_name_list = ['basket', 'chair', 'chest', 'fridge', 'sofa', 'plant', 'piano', 'guitar', 'toilet', 
# 'floor_lamp']
# max_obj_num = len(object_name_list)

# obj_list = []
# for name in object_name_list:

obj_list = sorted(glob.glob("./data/obj/ShapeNetCore.v2/*/*/model/model_normalized.obj"))

# Generate latents for the first time
# all_latent_generator(obj_list, save_latent_dir)

# TODO: edit the following 
with open(save_latent_dir+"trial_latents.json", "r") as f:
    current_latent_dict = json.load(f)

for i in range(num_scenes):

    # Select number of objects in this scene
    num_obj = np.random.randint(max_obj_num)+1
    used_objs = np.random.choice(obj_list, size=num_obj, replace=False)

    # Make new view angle (pitch)
    scene.set_random_view_matrix()

    scene.place_objects_no_table(used_objs, size=1.5, vis=False)
    
    # Get new scene
    scene_rgbd_img = scene.raw_rgbd(w=img_size, h=img_size)

    # below for debug purposes
    # imwrite("test"+str(i)+".png", test_img)
    # print(f"The shape of rgbd image: {scene_rgbd_img.shape}") 

    # save rgbd image(data)
    np.save(save_data_dir+str(i)+'.npy', scene_rgbd_img)

    # save object latents (label)
    # to allow vectorize operations in during PyTorch training, 
    # duplicate the number of latents in ground truth so that the dimensions stays the same
    latents = []
    
    for obj in used_objs:
        latents.append(current_latent_dict[obj])
    
    for _ in range(max_obj_num-len(latents)):
        latents.append(current_latent_dict[used_objs[-1]])

    assert len(latents) == max_obj_num
    np.save(save_latent_dir+str(i)+'.npy', np.array(latents))

    scene.reset_scene()