import numpy as np
import matplotlib.pyplot as plt
import random
import os
import json
import trimesh as tm
from trimesh.exchange.obj import export_obj
from trimesh.util import write_encoded
from sklearn.neighbors import NearestNeighbors

# Maybe consider rgb in icp as well

class PointCloud(object):
    def __init__(self, points=None):
        self.points = points
        self.num_points, self.dim = points.shape
        

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: N1xm array of points
        dst: N2xm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''
    # The numpy brute force method is much slower than the kneighbors approach
    # all_distances = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=-1)
    # distances = np.min(all_distances, axis=1)
    # indices = np.argmin(all_distances, axis=1)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


# If cloud A and B have different number of points, the matching performs badly, perhaps 
# it is more difficult to calculate a correct transformation matrix in best_fit_transform
def icp(A, B, init_pose=None, max_iter=50, tolerance=1e-6, outlier=True, lam=0.95, verbose=False):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Fractional icp: based on https://users.cs.duke.edu/~tomasi/papers/phillips/phillips3DIM07.pdf
    Input:
        A: N1xm numpy array of source mD points
        B: N2xm numpy array of destination mD point
        (N1 <= N2)
        init_pose: (m+1)x(m+1) homogeneous transformation
        tolerance: convergence criteria
        outlier: perform fractional icp if True
        frac: the percentage of points considered in the icp variant
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # assert A.shape == B.shape

    # get number of points and dimensions
    n1, m = A.shape
    n2, _ = B.shape

    # make points homogeneous, copy them to maintain the originals
   
    src = np.ones((m+1,n1))
    dst = np.ones((m+1,n2))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_frmsd = np.Inf
    final_T = np.identity(m+1)
    frac = 1

    for i in range(max_iter):
        # find the nearest neighbors between the current source and destination points
        # The coordinates are divided by the fourth component (homogeneous coords)
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        if outlier:
            n_points = int(np.floor(frac*n1))

            # find the point pairs that have the shortest distances
            order = np.argsort(distances)

            chosen_indices = indices[order]
            distances = distances[order]
            T, _, _ = best_fit_transform(src[:m, order[:n_points]].T,
                                         dst[:m, chosen_indices[:n_points]].T)

        else: 
            # compute the transformation between the current source and nearest destination points
            T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)
        final_T = np.dot(T, final_T)

        # turn distance into error function and update fraction f
        FRMSD = np.square(distances)
        for k in range(1, n1):
            FRMSD[k] = (FRMSD[k-1]*k+FRMSD[k])/(k+1)
        
        for k in range(n1):
            FRMSD[k] = np.sqrt(FRMSD[k])/(((k+1)/n1)**lam)
        
        index = np.argmin(FRMSD)
        min_frmsd = FRMSD[index]
        frac = (index+1)/n1
        if prev_frmsd - min_frmsd < tolerance:
            break

        prev_frmsd = min_frmsd
        if verbose:
            print(f"Matching iteration {i+1}, error:{prev_frmsd}, fraction:{frac}\n")

    # # calculate final transformation
    # T,_,_ = best_fit_transform(A, src[:m,:].T)
    # print(distances)
    return final_T, min_frmsd, frac


def partial_icp(A, B, max_iter=50, tolerance=1e-5, num_points=10000, outlier=True, lam=0.95, verbose=False):
    """
    This icp variant randomly pick a subset from each of the two pointclouds set and try to match the two subsets.

    Pointcloud A and B should be in the format of numpy arrays with dimensions:
        A: n1 * d
        B: n2 * d
    n1, n2 need not be the same
    """

    num_points = min(num_points, A.shape[0])
    num_points = min(num_points, B.shape[0])

    if num_points == A.shape[0]:
        A_points = A
    else:
        random_ids = random.sample(range(0, A.shape[0]), num_points)
        A_points = A[random_ids]
    
    if num_points == B.shape[0]:
        B_points = B
    else:
        random_ids = random.sample(range(0, B.shape[0]), num_points)
        B_points = B[random_ids]


    return icp(A_points, B_points, max_iter=max_iter, tolerance=tolerance, outlier=outlier, lam=lam, verbose=verbose)

def random_icp(A, B, max_iter=50, stop_thresh=0.0015, tolerance=1e-5, trials=10, outlier=True, lam=2, verbose=False):
    """
    This method calls icp() multiple times and find the best matching.

    Return:
    Transformation matrix and the error
    """

    chosen_T = None
    chosen_err = np.Inf
    for i in range(trials):
        T, err, frac = partial_icp(A, B, max_iter=max_iter, tolerance=tolerance, outlier=outlier, lam=lam, verbose=verbose)

        if err < chosen_err:
            chosen_T = T
            chosen_err = err
            chosen_frac = frac

        print(f"Trial {i}, Final FRMSD={err}, fraction={frac}")
        if err <= stop_thresh or err >= 0.015:
            break
        
    return chosen_T, chosen_err, chosen_frac

def merge_cloud(src, dst, r=0.001):
    '''
    Merge clouds src and dst by removing points from cloud src that are within distance of r 
    from any point in cloud dst
    Input:
        src: N1xm array of points
        dst: N2xm array of points
        r: radius to preserve points
    Output:
        merged_cloud: The merged cloud, N3xm array of points
    '''
    assert src.shape[1] == dst.shape[1]

    neigh = NearestNeighbors(radius=r) 
    neigh.fit(src)

    indices = np.concatenate(neigh.radius_neighbors(dst, return_distance=False)).ravel()
    uni_indices = np.unique(indices)
    
    # print(uni_indices.shape[0])
    if uni_indices.shape[0] > 0:
        updated_src = np.delete(src, uni_indices, axis=0)
        merged_cloud = np.vstack((updated_src, dst))
    else:
        merged_cloud = np.vstack((src, dst))

    return merged_cloud        

def HomoTo3D(src, transpose=False):
    '''
    Transform pointcloud from homogeneous coordinates to 3d coordinates.
    Input:
        src: Cloud to transform, has shape 4*n
        transpose: If true, output cloud has dimensions n*3, n*3 otherwise
    Output:
        cloud: Transformed cloud
    '''
    assert src.shape[0] == 4

    if transpose:
        cloud = src[:-1, :].T
    else:
        cloud = src[:-1, :]
    return cloud

def quantize(points, tresh=0.01):
    inv_tresh = 1. / tresh
    scale_points = points * inv_tresh
    scale_points = np.around(scale_points, 0)
    scale_points = tresh * scale_points

    return scale_points

def compute_occupancy(points, tresh=0.01):
    # Computes a voxel grid of points k
    scale_points = quantize(points, tresh=tresh)

    hash = 100*scale_points[:, 0]+10*scale_points[:, 1]+scale_points[:, 2]
    val, idx = np.unique(hash, return_index=True)
    scale_select = scale_points[idx]

    points_select = scale_select
    return points_select


def save_as_obj(points, path, fileName):
    
    orig_mesh = tm.PointCloud(points).convex_hull

    obj = export_obj(orig_mesh)
    obj_path = os.path.join(path, fileName)
    with open(obj_path, 'w') as f:
        # save obj files
        write_encoded(f, obj)
        # save the MTL and images                                
        # for k, v in data.items():
        #     with open(os.path.join(path, k), 'wb') as f:
        #         f.write(v)


if __name__=="__main__":

    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    with open("cloud1.json", ) as data:
        orig_pcl = json.load(data)

    with open("cloud2.json", ) as data:
        shuffled_pcl = json.load(data)

    for item in orig_pcl.keys():

        src_cloud = compute_occupancy(np.array(orig_pcl[item]), tresh=0.002)
        dst_cloud = compute_occupancy(np.array(shuffled_pcl[item]), tresh=0.002)

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
        trials = 3
        lamda = 2
        tol = 1e-5
        verbose = True

        results = []
        # icp
        results.append(random_icp(src_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol, verbose=verbose))
        results.append(random_icp(mir1_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol, verbose=verbose))
        results.append(random_icp(mir2_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol, verbose=verbose))
        results.append(random_icp(mir3_cloud, dst_cloud, trials=trials, lam=lamda, tolerance=tol, verbose=verbose))
        
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
        

        # homo_cloud is in homo coords, has dimensions 4*n
        homo_cloud = np.concatenate((chosen_cloud.T, np.ones((1,n_A))))
        transformed_cloud = np.dot(min_T, homo_cloud)
        transformed_cloud = HomoTo3D(transformed_cloud, transpose=True)

        AB_cloud = merge_cloud(transformed_cloud, dst_cloud)
        n_C = AB_cloud.shape[0]
        print(f"cloud sizes: {n_A}, {n_B}, {n_C}")

        fig = plt.figure()
        fig.suptitle(f"Fractional RMSD:{min_err:.3g}, final fraction:{min_frac:.3f}")
        plotA = fig.add_subplot(221, projection='3d')
        plotB = fig.add_subplot(222, projection='3d')
        plotAB = fig.add_subplot(223, projection='3d')
        plot_merged_A = fig.add_subplot(224, projection='3d')

        
        ci = int(item) % 6
        ci2 = (ci+1)%6
        plotA.scatter3D(dst_cloud[:,0], dst_cloud[:,1], dst_cloud[:,2],
                        c=[colors[ci] for _ in range(n_B)], s=0.005)

        plotB.scatter3D(transformed_cloud[:,0], 
                        transformed_cloud[:,1],
                        transformed_cloud[:,2],
                        c=[colors[ci2] for _ in range(n_A)], s=0.005)

        plotAB.scatter3D(dst_cloud[:,0], dst_cloud[:,1], dst_cloud[:,2],
                        c=[colors[ci] for _ in range(n_B)], s=0.005)

        plotAB.scatter3D(transformed_cloud[:,0], 
                        transformed_cloud[:,1],
                        transformed_cloud[:,2],
                        c=[colors[ci2] for _ in range(n_A)], s=0.005)

        plot_merged_A.scatter3D(AB_cloud[:,0], AB_cloud[:,1], AB_cloud[:,2],
                                c=[colors[ci] for _ in range(AB_cloud.shape[0])], s=0.005)
        plt.show()
    
    plt.show()