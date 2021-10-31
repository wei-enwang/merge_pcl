from imageio import imwrite
import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import os.path as osp
import os

import glob
import random

import transformations
from utils import enable_gravity, set_default_camera, in_collision, load_model, set_pose, set_quat, \
        Pose, Point, get_pose, get_center_extent, sample_placement, \
        random_placement, get_aabb, single_collision, rotate_quat, stable_z, quat_from_euler, remove_body
from camera_functions import get_rays_np


yaw = 160
pitch = -25
distance = 2

FLOOR_MODEL = "short_floor.urdf"
BASE_URL = os.path.abspath(os.path.dirname(__file__))
object_set = sorted(glob.glob("models/daily_object/*/model_meshlabserver_normalized.obj"))

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
CIRCULAR_LIMITS = -np.pi, np.pi

def add_labels(seg_im, im, map_dict):
    idxs = np.unique(seg_im)

    # image data preprocessing
    im = np.reshape(im, (IMAGE_WIDTH, IMAGE_HEIGHT, 4))
    im = np.clip(im, 0, 255).astype(np.uint8)

    for idx, name in map_dict.items():
        if (idxs == idx).sum() > 0:
            mask = np.equal(seg_im, idx)
            idx = np.arange(mask.size)
            select_idx = idx[mask.flatten()]
            y = select_idx // IMAGE_HEIGHT
            x = select_idx % IMAGE_WIDTH
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            y_center = (y_min + y_max) // 2
            cv2.putText(im, name, (x_min, y_center), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    
    return im

def process_pybullet_image(img, image_width, image_height, onlyrgb=True):

    img_saveable = np.reshape(img, (image_width, image_height, 4))
    if onlyrgb:
        img_saveable = np.clip(img_saveable[:,:,:-1], 0, 255).astype(np.uint8)
    else:
        img_saveable = np.clip(img_saveable, 0, 255).astype(np.uint8)
    return img_saveable

class Scene(object):
    def __init__(self, timestep=1./60, grid_size=16.):

        if not p.isConnected():
            p.connect(p.DIRECT)

        # set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        p.setTimeStep(timestep)
        enable_gravity()
        # p.setRealTimeSimulation(1)
        set_default_camera()

        self.nearVal = 0.02
        self.farVal = 5.0

        self.floor = load_model(FLOOR_MODEL, scale=1.0, fixed_base=True)
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll([0, 0, 0.1], distance, yaw, pitch, 0, 2)
        self.projMatrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=self.nearVal, farVal=self.farVal)
        
        self.objects = []
        self.table_ready = False
        self.grid_size = grid_size
        

    def set_table(self, states, action=None):
        """
        Given a scene represented as a list of objects of specified
        size and location, returns an image visualization as well
        indications of whether the scene is stable or not

        A given obj should be a dictionary with attributes:
        s corresponding to the shape of the object
        t cooresponding to the translation of the object
        c corresponding to the color of the object
        """
        
        grid_size = self.grid_size
        blocks = states
        density = 100
        scale_factor = 1. / grid_size

        base_blocks = []
        map_dict = {}
        
        # Extract the surface of the table
        table = False

        for obj_name, obj in list(blocks.items()):
            s, t, c = obj['s'], obj['t'], obj['c']
            shape = obj['shape']

            s = np.array(s) * scale_factor
            s = tuple(s)
            x, y, z = t
            x, y = -1. + x * 2. / grid_size + 1. / grid_size, -1. + y * 2. / grid_size + 1. / grid_size
            z = z * 2. / grid_size

            if shape == "cube":
                vis_obj = p.createVisualShape(p.GEOM_BOX, halfExtents=s)
                phys_obj = p.createCollisionShape(p.GEOM_BOX, halfExtents=s)
            elif shape == "cylinder":
                table = True
                s = np.array(s)*3
                vis_obj = p.createVisualShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/cylinder.obj") , meshScale=s)
                phys_obj = p.createCollisionShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/cylinder.obj") , meshScale=s)
            elif shape == "half_sphere":
                s = np.array(s)*2
                vis_obj = p.createVisualShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/hemisphere.obj") , meshScale=s)
                phys_obj = p.createCollisionShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/hemisphere.obj") , meshScale=s)
            elif shape == "wedge_1":
                s = np.array(s)*2
                vis_obj = p.createVisualShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/wedge_1.obj"), meshScale=s)
                phys_obj = p.createCollisionShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/wedge_1.obj"), meshScale=s)
            elif shape == "wedge_2":
                s = np.array(s)*2
                vis_obj = p.createVisualShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/wedge_2.obj"), meshScale=s)
                phys_obj = p.createCollisionShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/wedge_2.obj"), meshScale=s)
            elif shape == "half_cylinder":
                s = np.array(s)*2
                vis_obj = p.createVisualShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/half_cylinder.obj"), meshScale=s)
                phys_obj = p.createCollisionShape(p.GEOM_MESH, fileName=osp.join(BASE_URL, "models/table/half_cylinder.obj"), meshScale=s)

            mass = np.array(s).prod() * density
            obj = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=phys_obj,
                                    baseVisualShapeIndex=vis_obj)
            
            if table:
                self.table = obj
                table = False

            # Compute geometric center of object
            (_, _, z_center) , _ = get_center_extent(obj)
            # Compute geometric center of mass of object
            (_, _, z_com), _ = get_pose(obj)
            offset = z_center - z_com

            p.changeVisualShape(obj, -1, rgbaColor=tuple(c))
            set_pose(obj, Pose(Point(x=x, y=y, z=(z-offset))))
            base_blocks.append(obj)

            map_dict[obj] = obj_name

        # The table is set up 
        self.table_ready = True

    def place_objects(self, objects, vis=False):
        """
        Call this method after set_table, objects are in the form of a list 
        Returns the image after the objects were placed
        """
        if not self.table_ready:
            assert "Call set_table() before placing objects on the table"


        for item, arg in objects.items():
            local = arg
            if local:
                visShape = p.createVisualShape(p.GEOM_MESH, 
                fileName=osp.join(BASE_URL, item), 
                meshScale=[0.2, 0.2, 0.2])
                colShape = p.createCollisionShape(p.GEOM_MESH, 
                fileName=osp.join(BASE_URL, item),
                meshScale=[0.2, 0.2, 0.2])
            else:
                visShape = p.createVisualShape(p.GEOM_MESH, 
                fileName=item, 
                meshScale=[0.1, 0.1, 0.1])
                colShape = p.createCollisionShape(p.GEOM_MESH, 
                fileName=item,
                meshScale=[0.1, 0.1, 0.1])                

            mass = 2
            obj_id = p.createMultiBody(baseMass=mass, 
                                    baseCollisionShapeIndex=colShape,
                                    baseVisualShapeIndex=visShape)

            sample_placement(obj_id, self.table)
            while single_collision(obj_id):
                sample_placement(obj_id, self.table)

            self.objects.append(obj_id)

            # rotate_quat(obj_id, transformations.random_quaternion())
        
        if vis:
            return self.take_snapshot()


    def place_objects_no_table(self, objects, size=2, vis=False):
        if self.table_ready:
            assert "This method expcts no table."

        coords = np.random.uniform(-size/2, size/2, (len(objects), 2))

        for i,item in enumerate(objects):
            obj_id = p.loadURDF(item, globalScaling=0.3, useFixedBase=True)
            
            z = stable_z(obj_id, self.floor)
            set_pose(obj_id, (np.array([coords[i, 0], coords[i, 1], z]),
                              quat_from_euler([0,0,np.random.uniform(*CIRCULAR_LIMITS)])))

            while single_collision(obj_id):
                coords[i] = np.random.uniform(-size/2, size/2, (2,))
                set_pose(obj_id, (np.array([coords[i, 0], coords[i, 1], z]),
                                  quat_from_euler([0,0,np.random.uniform(*CIRCULAR_LIMITS)])))

            self.objects.append(obj_id)
        if vis:
            return self.take_snapshot()


    def set_view_matrix(self, distance_, yaw_, pitch_, roll_):
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll([0, 0, 0.1], distance_, yaw_, pitch_, roll_, 2)

    def set_random_view_matrix(self, distance_=distance, yaw_=yaw):
        pitch_ = random.randint(-30, -10)
        self.set_view_matrix(distance_=distance_, yaw_=yaw_, pitch_=pitch_, roll_=0)

    def get_segmented_pointcloud(self, viewMat=None, projMat=None, jsonify=False):
        """
        Return a dictionary consists of pointclouds of objects in the camera scene defined by viewMatrix and projectionMatrix. Use default matrices of the scene created during initialization if not given.

        If jsonify is true, the pointclouds are returned in the form of python lists.
        """
        
        if viewMat is None:
            viewMat = self.viewMatrix
        if projMat is None:
            projMat = self.projMatrix
        
        _, _, img, depth_img, seg_img = p.getCameraImage(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMat)

        far = self.farVal
        near = self.nearVal
        # the raw depth value from getCameraImage is distorted, apply the following transform to retrieve the value
        depth = far * near / (far  - (far - near) * np.array(depth_img).reshape((IMAGE_WIDTH, IMAGE_HEIGHT)))

        viewMatrix = np.array(viewMat).reshape((4, 4)).transpose()
        cam2world = np.linalg.inv(viewMatrix)
        focal = 0.5 * IMAGE_WIDTH / np.tan(0.5 * np.pi / 3)

        rays_o, rays_d = get_rays_np(IMAGE_WIDTH, IMAGE_HEIGHT, focal, cam2world)
        generalPointcloud = depth[:,:,None] * rays_d + rays_o

        # Extract the pointcloud of each object
        pointcloud_map = {}
        for object_id in self.objects:

            filter_mask = np.array(seg_img).reshape((IMAGE_WIDTH, IMAGE_HEIGHT)) == object_id
            # the number of points need in order to be registered in our data is temporarily set to 5
            if np.sum(filter_mask) > 5:

                points_xyz = generalPointcloud[filter_mask]
            
                if jsonify:
                    pointcloud_map[object_id] = points_xyz.tolist()
                else:
                    pointcloud_map[object_id] = points_xyz

        return pointcloud_map
    
    def take_snapshot(self, viewMat=None, projMat=None):
        """
        Return the image taken by a specified or default camera
        """
        if viewMat is None:
            viewMat = self.viewMatrix
        if projMat is None:
            projMat = self.projMatrix

        _, _, img, _, _ = p.getCameraImage(IMAGE_WIDTH, IMAGE_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMat, renderer=p.ER_TINY_RENDERER)

        return process_pybullet_image(img, IMAGE_WIDTH, IMAGE_HEIGHT)
    
    def raw_rgbd(self, viewMat=None, projMat=None):
        """
        Return the raw_rgbd image in the form of numpy arrays
        (for PyTorch)
        """
        if viewMat is None:
            viewMat = self.viewMatrix
        if projMat is None:
            projMat = self.projMatrix

        _, _, rgb_img, depth_img, _ = p.getCameraImage(IMAGE_WIDTH, IMAGE_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMat, renderer=p.ER_TINY_RENDERER)

        # still need to reshape image
        rgb_img = process_pybullet_image(rgb_img, IMAGE_WIDTH, IMAGE_HEIGHT)
        # multiply depth_img by 255 so that torch.ToTensor() can handle the entire rgbd image
        rgbd_img = np.concatenate((rgb_img, np.array(255*depth_img).reshape((IMAGE_WIDTH, IMAGE_HEIGHT,1))), axis=-1)

        return rgbd_img

    def random_reset_object(self, obj_id):
        """
        Randomly reset an object's position and orientation.
        """
        assert obj_id in self.objects
        sample_placement(obj_id, self.table)
        while single_collision(obj_id):
            sample_placement(obj_id, self.table)

    def shuffle_objects(self, vis=False):
        """
        Give all objects random positions and orientations.
        """
        for obj_id in self.objects:
            self.random_reset_object(obj_id)
        
        if vis:
            return self.take_snapshot()

    def reset_scene(self, new_objects=None):
        for obj_id in self.objects:
            remove_body(obj_id)
        if new_objects is not None:
            self.place_objects_no_table(new_objects)


def parse_steps(data):
    blocks = {}

    for i in range(data.shape[0]):
        name = "Block{}".format(i)
        x, y, z, s_x, s_y, s_z, cylinder, top, r, g, b, a = data[i]
        if cylinder:
            shape = "cylinder"
        else:
            shape = "cube"

        if top:
            semantic_part = 'top'
        else:
            semantic_part = 'leg'

        s = [s_x, s_y, s_z]
        t = [x + 8, y + 8, z]
        rgba = [r, g, b, a]
        
        obj_info = {}
        obj_info['shape'] = shape
        obj_info['s'] = np.array(s)
        obj_info['t'] = np.array(t)
        obj_info['c'] = np.array(rgba)
        obj_info['semantic_part'] = semantic_part

        blocks[name] = obj_info

    return blocks


if __name__ == "__main__":

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    p.connect(p.DIRECT)

    data = np.load("table.npy")
    blocks = parse_steps(data[0])

    num_objects = 5
    random_objects = dict.fromkeys(random.sample(object_set, num_objects), True)  

    scene = Scene()
    scene.set_table(blocks)

    img = scene.place_objects(random_objects, vis=True)
    imwrite("scene_after_action.png", img)
    objectPointCloud_dict = scene.get_segmented_pointcloud(jsonify=True)

    viewMatrix2 = p.computeViewMatrixFromYawPitchRoll([0, 0, 1], distance, yaw-180, pitch, 0, 2)

    img2 = scene.take_snapshot(viewMat=viewMatrix2)
    imwrite("second_scene.png", img2)
    print("Scene updated!")

    objectPointCloud_dict2 = scene.get_segmented_pointcloud(viewMat=viewMatrix2, jsonify=True)

    fig = plt.figure()
    allpointsDiagram = fig.add_subplot(131, projection='3d')
    pointsDiagram = fig.add_subplot(132, projection='3d')
    
    for obj_id in objectPointCloud_dict.keys():
        ci = obj_id % 6
        obj_cloud = np.array(objectPointCloud_dict[obj_id])

        num_points = obj_cloud.shape[0]
        pointsDiagram.scatter3D(obj_cloud[:,0], obj_cloud[:,1], obj_cloud[:,2],
                                c=[colors[ci] for _ in range(num_points)], s=0.001)
        allpointsDiagram.scatter3D(obj_cloud[:,0], obj_cloud[:,1], obj_cloud[:,2],
                                c=[colors[ci] for _ in range(num_points)], s=0.001)
        # plt.show()

    pointsDiagram2 = fig.add_subplot(133, projection='3d')
    
    for obj_id in objectPointCloud_dict2.keys():
        ci = obj_id % 6
        obj_cloud = np.array(objectPointCloud_dict2[obj_id])

        num_points = obj_cloud.shape[0]
        pointsDiagram2.scatter3D(obj_cloud[:,0], obj_cloud[:,1], obj_cloud[:,2],
                                c=[colors[ci] for _ in range(num_points)], s=0.001)
        allpointsDiagram.scatter3D(obj_cloud[:,0], obj_cloud[:,1], obj_cloud[:,2],
                                c=[colors[ci] for _ in range(num_points)], s=0.001)

    plt.show()

    # save the two pointcloud for icp or other future analysis
    with open("pointclouds_scene1.json", "w") as data:
        json.dump(objectPointCloud_dict, data)

    with open("pointclouds_scene2.json", "w") as data:
        json.dump(objectPointCloud_dict2, data)
