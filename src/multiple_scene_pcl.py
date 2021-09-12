import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json
import pybullet as p
from imageio import imwrite
import open3d as o3d
from funcs.save import np_save_as_json
from pointcloud import random_icp, merge_cloud, HomoTo3D, compute_occupancy
from create_scene import Scene, parse_steps


if __name__=="__main__":

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    num_scenes = 5

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
    err_threshold = 0.005
    merge_threshold = 0.002
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
    imwrite("multiscene_results/trial7/scene_0.png", img)
    print("New image registered!")

    # hypo_count stands for how many hypotheses we have
    hypo_count = 0

    for _, val in orig_pcl.items():
        dict_of_clouds[hypo_count] = []
        dict_of_clouds[hypo_count].append(compute_occupancy(np.array(val), tresh=voxel_threshold))

        hypo_count = hypo_count + 1

    # Pass scenes into program
    for i in range(num_scenes):

        # scene.set_random_view_matrix()
        img = scene.shuffle_objects(vis=True)
        imwrite(f"multiscene_results/trial7/scene_{i+1}.png", img)
        shuffled_pcl = scene.get_segmented_pointcloud(jsonify=False)
        print("New image registered!")
        
        buffer_new_item = {}
        for k, cloud in shuffled_pcl.items():

            old_item_detected = False

            # if not item in dict_of_clouds.keys():
            #     
            # else:
            #     dst_cloud = dict_of_clouds[item]

            src_cloud = compute_occupancy(np.array(cloud), tresh=voxel_threshold)
            n_B = src_cloud.shape[0]

            if n_B == 0:
                continue

            # Match the current cloud with each stored hypothesis
            for key, cloud_arr in dict_of_clouds.items():

                cloud_buffer = None
                for j, dst_cloud in enumerate(cloud_arr):
                    n_A = dst_cloud.shape[0]
                    if n_A == 0:
                        continue


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

                    # the clouds differ too much, test the next cloud in memory
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
                    # transform back to standard dimensions
                    transformed_cloud = HomoTo3D(transformed_cloud, transpose=True)

                    if min_err < merge_threshold:
                        # update cloud hypothesis
                        cloud_arr[j] = merge_cloud(transformed_cloud, dst_cloud, r=radius)
                        old_item_detected = True

                    elif min_err < err_threshold:
                        
                        if old_item_detected:
                            # either merged with other clouds or already registered
                            continue
                        # add to hypothesis list
                        cloud_buffer = transformed_cloud
                        old_item_detected = True          

                # append hypotheses to list
                if not cloud_buffer is None:
                    cloud_arr.append(cloud_buffer)

            # cloud not recognized
            if not old_item_detected:
                buffer_new_item[hypo_count] = []
                buffer_new_item[hypo_count].append(src_cloud)
                hypo_count += 1

        # scene analyze complete, update items' hypothesis
        dict_of_clouds.update(buffer_new_item)

                # if i+1 == mid_iteration:
                #     # save in progress pointcloud
                #     mid_pcd = o3d.geometry.PointCloud()
                #     mid_pcd.points = o3d.utility.Vector3dVector(dict_of_clouds[item])
                #     o3d.io.write_point_cloud(data_path+str(item)+"_mid.ply", mid_pcd)

    np_save_as_json(dict_of_clouds, data_path+"hypo_clouds.json")

    for item in dict_of_clouds.keys():

        # orig_cloud = compute_occupancy(np.array(orig_pcl[item]), tresh=voxel_threshold)
        cloud_list = dict_of_clouds[item]
        print(f"Item {item}: Number of clouds: {len(cloud_list)}")
        

        # fig = plt.figure()
        # fig.suptitle(f"Size of first cloud:{n1}, size of final cloud:{n2}")
        # plotA = fig.add_subplot(121, projection='3d')
        # plotB = fig.add_subplot(122, projection='3d')

        # ci = int(item)%6
        # ci2 = (ci+1)%6

        # plotA.scatter3D(orig_cloud[:,0], orig_cloud[:,1], orig_cloud[:,2],
        #                 c=[colors[ci] for _ in range(n1)], s=0.01)

        # plotB.scatter3D(final_cloud[:,0], 
        #                 final_cloud[:,1],
        #                 final_cloud[:,2],
        #                 c=[colors[ci2] for _ in range(n2)], s=0.01)

        # # plot_merged_A.scatter3D(bigCloud[:,0], bigCloud[:,1], bigCloud[:,2],
        # #                         c=[colors[ci] for _ in range(bigCloud.shape[0])], s=0.005)
        # plt.show()

        # orig_pcd = o3d.geometry.PointCloud()
        # final_pcd = o3d.geometry.PointCloud()
        # orig_pcd.points = o3d.utility.Vector3dVector(orig_cloud)
        # final_pcd.points = o3d.utility.Vector3dVector(final_cloud)

        # o3d.io.write_point_cloud(data_path+str(item)+"_first.ply", orig_pcd)
        # o3d.io.write_point_cloud(data_path+str(item)+"_final.ply", final_pcd)


# data = {"models/daily_object/1/model_meshlabserver_normalized.obj": true, "models/daily_object/8/model_meshlabserver_normalized.obj": true, "models/daily_object/119/model_meshlabserver_normalized.obj": true, "models/daily_object/136/model_meshlabserver_normalized.obj": true, "models/daily_object/139/model_meshlabserver_normalized.obj": true}