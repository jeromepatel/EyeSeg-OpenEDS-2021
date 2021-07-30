# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 23:46:28 2021

@author: jyotm
"""

from pathlib import Path

import numpy as np
import torch
from scipy.spatial.ckdtree import cKDTree as kdtree
from torch.utils.data import Dataset 
import os
from plyfile import PlyData, PlyElement
import open3d as o3d
# from PIL import Image
from torchvision import transforms
# import cv2
import random
import os.path as osp
# from utils import one_hot2dist
import copy

finalSizes = []

class eyeSegmentation(Dataset):
    def __init__(self, filepath, split='train',transform=None,npoints = 5000, testrun=True,**args):
        self.transform = transform
        self.filepath= osp.join(filepath,split)
        self.split = split
        self.sweeps = []
        self.listalldir = []
        self.npoints = npoints
        
        for file in os.listdir(self.filepath):   
            if os.path.isdir(osp.join(self.filepath,file)):
                pointCloudPath = osp.join(self.filepath,file,"pointcloud.ply")
                plydata = PlyData.read(pointCloudPath)
                pts = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)
                if self.split == 'test':
                    label = np.zeros((pts.shape[0],))
                else:
                    labels_path = osp.join(self.filepath, file, "labels.npy")
                    label = np.load(labels_path)
                #print(label.shape[0])
                if label.shape[0] == 9290 :
                    print(file,label.shape[0], "remove outlier")
                elif label.shape[0] == 67931:
                    print(file,label.shape[0], "remove outlier")
                elif label.shape[0] <= 10096:
                     print(file,label.shape[0], "cuation: check for test set point sizes")
                finalSizes.append(pts.shape[0])
                self.listalldir.append(file)
                metaData = file.split("_")
                self.sweeps.append((metaData[0],metaData[2]))
            else:
                raise IOError 
                        
        self.testrun = testrun
        

    def __len__(self):
        if self.testrun:
            return 10
        return len(self.listalldir)

    def __getitem__(self, idx):
        seqDir = self.listalldir[idx]
        pointCloudPath = osp.join(self.filepath,seqDir,"pointcloud.ply")
        
        plydata = PlyData.read(pointCloudPath)
        pts = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)
        print(pts.shape)
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale
        
        # labels_path = osp.join(self.filepath, seqDir, "labels.npy")
        # label = np.load(labels_path)
        
        label = np.zeros((120,))
        # features = np.ones([pts.shape[0], 1])
        
        # if self.split == 'test':
        #     label = np.zeros(pts.shape)
        
        # label = label[choice]
        
        return pts, label
               
# seq_path = "D:/OpenEDS 2021/val/0024_pose_1"
# # for sweep in seq_path.iterdir():
# #     print(seq_str, sweep.stem)

eyeSeg = eyeSegmentation("D:/OpenEDS 2021","val")
DataLoader = torch.utils.data.DataLoader(eyeSeg, batch_size=1, shuffle=True)
# for point,label in DataLoader:
#     print(point.shape)

## To print the visualization results of Pointcloud sizes

finalSizes.sort()
print(finalSizes[:10],finalSizes[-10:])
# with open("NumPoints.txt","w") as f:
#     f.write("\n".join(map(str,finalSizes)))
    
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# with open("NumPoints.txt","r") as f:
#     str_data = list(map(int,f.read().split("\n")))
#     print(str_data)
#     str_data.pop()
# sns.distplot(str_data)

sns.distplot(finalSizes)