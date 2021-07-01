# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:27:47 2021

@author: jyotm
"""
# from utils import one_hot2dist
import numpy as np
from plyfile import PlyData, PlyElement

seq = 8
print(f"{seq:0>2}")

point_name = "D:/OpenEDS 2021/val/0024_pose_1/"

print("Load a ply point cloud, print it, and render it")
# train_labels = np.load(point_name+"labels.npy")
# label = np.fromfile(point_name+"labels.npy")
# print(train_labels.shape,label.shape)
# print(label.max(), train_labels[:10])


plydata = PlyData.read(point_name + 'pointcloud.ply')
pts = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)    
print(pts.shape)
print(plydata['vertex'])