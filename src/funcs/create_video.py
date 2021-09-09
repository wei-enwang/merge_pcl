import cv2
import numpy as np
import glob
 
img_array = []
data_path = "../multiscene_results/trial6/"
for filename in glob.glob(data_path+"*.png"):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
fourcc = cv2.VideoWriter_fourcc(*'H264')
fps = 1
out = cv2.VideoWriter(data_path+"demo.mp4", fourcc, fps, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
print("video completed")

