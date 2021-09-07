import os
import glob
import json

import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from imageio import imwrite

from create_scene import Scene, parse_steps
from pointcloud import random_icp, icp, partial_icp

yaw = 160
pitch = -30
distance = 2.5

FLOOR_MODEL = "short_floor.urdf"
BASE_URL = os.path.abspath(os.path.dirname(__file__))
object_set = sorted(glob.glob("models/bag/*/model.obj"))

if __name__=="__main__":
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    p.connect(p.DIRECT)

    data = np.load("table.npy")
    blocks = parse_steps(data[0])

    num_objects = 5
    random_objects = dict.fromkeys(random.sample(object_set, num_objects), True)  

    scene = Scene()
    scene.set_table(blocks)

    img = scene.place_objects(random_objects, vis=True)
    imwrite("scene1.png", img)
    print("First scene complete!")
    orig_pcl = scene.get_segmented_pointcloud(jsonify=True)

    img2 = scene.shuffle_objects(vis=True)
    imwrite("shuffled_scene2.png", img2)
    print("Second scene complete!")
    shuffled_pcl = scene.get_segmented_pointcloud(jsonify=True)
    
    # label_cnt = 1
    for item in scene.objects:

        src_cloud = np.array(orig_pcl[item])
        dst_cloud = np.array(shuffled_pcl[item])

        n_A = src_cloud.shape[0]
        n_B = dst_cloud.shape[0]       
        
        if n_A > n_B:
            src_cloud, dst_cloud = dst_cloud, src_cloud
            n_A, n_B = n_B, n_A

        # rotate the cloud (after first icp) 180 degrees to resolve convergence problems
        rot1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        rot3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        mir1_cloud = np.dot(src_cloud, rot1)
        mir2_cloud = np.dot(src_cloud, rot2)
        mir3_cloud = np.dot(src_cloud, rot3)

        # icp params
        trials = 1
        lamda = 2
        tol = 1e-5

        results = []
        # icp
        results.append(random_icp(src_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol))
        results.append(random_icp(mir1_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol))
        results.append(random_icp(mir2_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol))
        results.append(random_icp(mir3_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol))
        
        # T1, err1, frac1 = partial_icp(src_cloud, dst_cloud, tolerance=1e-5, lam=1.5)
        # T2, err2, frac2 = partial_icp(mir_cloud, dst_cloud, tolerance=1e-5, lam=1.5)
        
        min_err = np.Inf
        # compare which orientation has the best result
        for i, rec in enumerate(results):
            T, err, frac = rec
            if err < min_err:
                min_T = T
                min_err = err
                min_frac = frac
                min_i = i

        if min_i == 0:
            chosen_cloud = src_cloud
        elif min_i == 1:
            chosen_cloud = mir1_cloud
        elif min_i == 2:
            chosen_cloud = mir2_cloud
        elif min_i == 3:
            chosen_cloud = mir3_cloud
        
        fig = plt.figure()

        # homo_cloud is in homo coords, has dimensions 4*n
        homo_cloud = np.concatenate((chosen_cloud.T, np.ones((1,n_A))))
        transformed_second_cloud = np.dot(min_T, homo_cloud)
        fig.suptitle(f"Fractional RMSD:{min_err:.3g}, final fraction:{min_frac:.3f}")
        

        plotA = fig.add_subplot(221, projection='3d')
        plotB = fig.add_subplot(222, projection='3d')
        plotB2 = fig.add_subplot(223, projection='3d')
        plotAB = fig.add_subplot(224, projection='3d')
    
        ci = item % 6
        ci2 = (ci+1)%6
        plotA.scatter3D(dst_cloud[:,0], dst_cloud[:,1], dst_cloud[:,2],
                                    c=[colors[ci] for _ in range(n_B)], s=0.005)
        plotB.scatter3D(src_cloud[:,0], src_cloud[:,1], src_cloud[:,2],
                                    c=[colors[ci2] for _ in range(n_A)], s=0.005)
        plotB2.scatter3D(transformed_second_cloud[0,:], 
                        transformed_second_cloud[1,:],
                        transformed_second_cloud[2,:],
                                    c=[colors[ci2] for _ in range(n_A)], s=0.005)

        plotAB.scatter3D(dst_cloud[:,0], dst_cloud[:,1], dst_cloud[:,2],
                                    c=[colors[ci] for _ in range(n_B)], s=0.005)
        plotAB.scatter3D(transformed_second_cloud[0,:], 
                        transformed_second_cloud[1,:],
                        transformed_second_cloud[2,:],
                                    c=[colors[ci2] for _ in range(n_A)], s=0.005)
        
        plt.show()
        # plt.savefig(f"trial4/obj_{label_cnt}.png")
        # label_cnt += 1

    plt.show()
    # with open("cloud1.json", "w") as data:
    #     json.dump(orig_pcl, data)

    # with open("cloud2.json", "w") as data:
    #     json.dump(shuffled_pcl, data)
    
    with open("objects.json", "w") as data:
        json.dump(random_objects, data)