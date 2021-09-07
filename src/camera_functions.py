import pybullet as p
import math
import numpy as np

def getRayFromTo(mouseX, mouseY):

    width, height, viewMat, projMat, cameraUp, camForward, horizon, vertical, _, _, dist, camTarget = p.getDebugVisualizerCamera()
    camPos = [
      camTarget[0] - dist * camForward[0], camTarget[1] - dist * camForward[1],
      camTarget[2] - dist * camForward[2]
      ]
    farPlane = 10000
    rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]), (camTarget[2] - camPos[2])]
    invLen = farPlane * 1. / (math.sqrt(rayForward[0] * rayForward[0] + rayForward[1] *
                                        rayForward[1] + rayForward[2] * rayForward[2]))
    rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
    rayFrom = camPos
    oneOverWidth = float(1) / float(width)
    oneOverHeight = float(1) / float(height)
    dHor = [horizon[0] * oneOverWidth, horizon[1] * oneOverWidth, horizon[2] * oneOverWidth]
    dVer = [vertical[0] * oneOverHeight, vertical[1] * oneOverHeight, vertical[2] * oneOverHeight]
    rayToCenter = [
        rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
    ]
    rayTo = [
        rayFrom[0] + rayForward[0] - 0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] -
        float(mouseY) * dVer[0], rayFrom[1] + rayForward[1] - 0.5 * horizon[1] + 0.5 * vertical[1] +
        float(mouseX) * dHor[1] - float(mouseY) * dVer[1], rayFrom[2] + rayForward[2] -
        0.5 * horizon[2] + 0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
    ]
    return rayFrom, rayTo


def get_rays_np(W, H, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

def point_from_cam_to_world(depth_np_arr, img_width, img_height, projection_matrix, view_matrix):
    stepX = 1
    stepY = 1        
    pointCloud = np.empty(shape=(img_width*img_height, 3))
    projectionMatrix = np.asarray(projection_matrix).reshape([4,4],order='F')
    viewMatrix = np.asarray(view_matrix).reshape([4,4],order='F')
    tran_pix_world = np.linalg.inv(np.matmul(projectionMatrix, viewMatrix))
    for h in range(0, img_height, stepY):
        for w in range(0, img_width, stepX):

            x = (2*w - img_width)/img_width
            y = -(2*h - img_height)/img_height  # be careful！ depth and its corresponding position
            z = 2*depth_np_arr[h,w] - 1
            pixPos = np.asarray([x, y, z, 1])
            position = np.matmul(tran_pix_world, pixPos)

            pointCloud[h, w] = position[:-1] / position[3]

    return pointCloud