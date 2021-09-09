import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json
import pybullet as p
from imageio import imwrite
import open3d as o3d

from pointcloud import random_icp, merge_cloud, HomoTo3D, compute_occupancy
from create_scene import Scene, parse_steps


if __name__=="__main__":

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    num_scenes = 10

    # icp params
    trials = 5
    lamda = 1.8
    tol = 1e-6
    max_iter = 50
    stop_thresh = 0.0015

    data_path = "clouds/"
    mid_iteration = 5

    # merge params
    radius = 0.002
    err_threshold = 0.003
    voxel_threshold = 0.003
    dict_of_clouds = {}

    p.connect(p.DIRECT)

    data = np.load("table.npy")
    blocks = parse_steps(data[0])

    with open("objects.json", ) as data:
        objects = json.load(data)

    scene = Scene()
    scene.set_table(blocks)

    img = scene.place_objects(objects, vis=True)    
    orig_pcl = scene.get_segmented_pointcloud(jsonify=False)
    imwrite("multiscene_results/trial6/scene_0.png", img)
    print("New image registered!")

    for i in range(num_scenes):

        # scene.set_random_view_matrix()
        img = scene.shuffle_objects(vis=True)
        imwrite(f"multiscene_results/trial6/scene_{i+1}.png", img)
        shuffled_pcl = scene.get_segmented_pointcloud(jsonify=False)
        print("New image registered!")
    
        for item in scene.objects:

            if not item in shuffled_pcl.keys():
                continue # item not registered

            if not item in dict_of_clouds.keys():
                dst_cloud = compute_occupancy(np.array(orig_pcl[item]), tresh=voxel_threshold)
            else:
                dst_cloud = dict_of_clouds[item]

            src_cloud = compute_occupancy(np.array(shuffled_pcl[item]), tresh=voxel_threshold)

            n_A = dst_cloud.shape[0]
            n_B = src_cloud.shape[0]

            if n_A == 0 or n_B == 0:
                continue

            if n_A < n_B:
                src_cloud, dst_cloud = dst_cloud, src_cloud
                n_A, n_B = n_B, n_A

            # rotate the cloud (after first icp) 180 degrees to resolve convergence problems
            rot1 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            rot2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            rot3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

            mir1_cloud = np.dot(src_cloud, rot1)
            mir2_cloud = np.dot(src_cloud, rot2)
            mir3_cloud = np.dot(src_cloud, rot3)

            # icp
            min_T, min_err, min_frac = random_icp(src_cloud, dst_cloud, max_iter=max_iter, stop_thresh=stop_thresh, trials=trials, lam=lamda, tolerance=tol)

            min_i = 0

            if min_err >= stop_thresh:
                T, err, frac = random_icp(mir1_cloud, dst_cloud, max_iter=max_iter,
                stop_thresh=stop_thresh, trials=trials, lam=lamda, tolerance=tol)
                if err < min_err:
                    min_T, min_err, min_frac = T, err, frac
                    min_i = 1
                if min_err >= stop_thresh:
                    T, err, frac = random_icp(mir2_cloud, dst_cloud, max_iter=max_iter, 
                    stop_thresh=stop_thresh, trials=trials, lam=lamda, tolerance=tol)
                    if err < min_err:
                        min_T, min_err, min_frac = T, err, frac
                        min_i = 2
                    if min_err >= stop_thresh:
                        T, err, frac = random_icp(mir3_cloud, dst_cloud, max_iter=max_iter, 
                        stop_thresh=stop_thresh, trials=trials, lam=lamda, tolerance=tol)
                        if err < min_err:
                            min_T, min_err, min_frac = T, err, frac
                            min_i = 3

            if min_err >= 0.015:
                continue

            if min_i == 0:
                chosen_cloud = src_cloud
            elif min_i == 1:
                chosen_cloud = mir1_cloud
            elif min_i == 2:
                chosen_cloud = mir2_cloud
            elif min_i == 3:
                chosen_cloud = mir3_cloud
                
            if chosen_cloud is None:
                continue

            # homo_cloud is in homo coords, has dimensions 4*n
            homo_cloud = np.concatenate((chosen_cloud.T, np.ones((1,n_B))))
            transformed_cloud = np.dot(min_T, homo_cloud)
            transformed_cloud = HomoTo3D(transformed_cloud, transpose=True)

            bigCloud = merge_cloud(transformed_cloud, dst_cloud, r=radius)

            if min_err < err_threshold:
                dict_of_clouds[item] = bigCloud
            else:
                dict_of_clouds[item] = dst_cloud

            if i+1 == mid_iteration:
                # save in progress pointcloud
                mid_pcd = o3d.geometry.PointCloud()
                mid_pcd.points = o3d.utility.Vector3dVector(dict_of_clouds[item])
                o3d.io.write_point_cloud(data_path+str(item)+"_mid.ply", mid_pcd)

    for item in scene.objects:

        orig_cloud = compute_occupancy(np.array(orig_pcl[item]), tresh=voxel_threshold)
        final_cloud = dict_of_clouds[item]


        n1 = orig_cloud.shape[0]
        n2 = final_cloud.shape[0]

        fig = plt.figure()
        fig.suptitle(f"Size of first cloud:{n1}, size of final cloud:{n2}")
        plotA = fig.add_subplot(121, projection='3d')
        plotB = fig.add_subplot(122, projection='3d')

        ci = int(item)%6
        ci2 = (ci+1)%6

        plotA.scatter3D(orig_cloud[:,0], orig_cloud[:,1], orig_cloud[:,2],
                        c=[colors[ci] for _ in range(n1)], s=0.01)

        plotB.scatter3D(final_cloud[:,0], 
                        final_cloud[:,1],
                        final_cloud[:,2],
                        c=[colors[ci2] for _ in range(n2)], s=0.01)

        # plot_merged_A.scatter3D(bigCloud[:,0], bigCloud[:,1], bigCloud[:,2],
        #                         c=[colors[ci] for _ in range(bigCloud.shape[0])], s=0.005)
        plt.show()

        orig_pcd = o3d.geometry.PointCloud()
        final_pcd = o3d.geometry.PointCloud()
        orig_pcd.points = o3d.utility.Vector3dVector(orig_cloud)
        final_pcd.points = o3d.utility.Vector3dVector(final_cloud)

        o3d.io.write_point_cloud(data_path+str(item)+"_first.ply", orig_pcd)
        o3d.io.write_point_cloud(data_path+str(item)+"_final.ply", final_pcd)
