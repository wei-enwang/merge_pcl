import json
import open3d as o3d

def cloud_to_ply(dict_of_clouds, save_dir="", suffix=""):
    for item in dict_of_clouds.keys():
        cloud_list = dict_of_clouds[item]
        print(f"Item {item}: Number of clouds: {len(cloud_list)}")
        
        for i, hypo in enumerate(cloud_list):
  
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(hypo)

            o3d.io.write_point_cloud(save_dir+str(item)+"_"+str(i)+suffix+".ply", pcd)


if __name__ == "__main__":
    with open("hypo_clouds.json", ) as data:
        dict_of_clouds = json.load(data)
    
    cloud_to_ply(dict_of_clouds)