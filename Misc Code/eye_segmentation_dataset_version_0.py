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
from PIL import Image
from torchvision import transforms
import cv2
import random
import os.path as osp
# from utils import one_hot2dist
import copy



class eyeSegmentation(Dataset):
    def __init__(self, filepath, split='train',transform=None,testrun=True,**args):
        self.transform = transform
        self.filepath= osp.join(filepath,split)
        self.split = split
        self.sweeps = []
        self.listalldir = []
        
        
        for file in os.listdir(self.filepath):   
            if os.path.isdir(osp.join(self.filepath,file)):
                # print(os.listdir(osp.join(self.filepath,file)))
                self.listalldir.append(file)
                metaData = file.split("_")
                self.sweeps.append((metaData[0],metaData[2]))
            else:
                raise IOError 
                        
    

        
        for seq in self.listalldir:
            # seq_str = f"/00{seq:0>2}" + "_pose_1/" 
            labels_path = osp.join(self.filepath, seq, "labels.npy")
            label = np.fromfile(labels_path)
            print(label.shape)
            print("asdaga",labels_path)
            # for sweep in os.listdir(seq_path):
                # print(seq, sweep)
                # print(self.sweeps[0])
            break
        # self.list_files=listalldir
        labels_path = osp.join(self.filepath, "0024_pose_1/labels.npy")
        point_name = "D:/OpenEDS 2021/val/0024_pose_1/"
        
        print("Load a ply point cloud, print it, and render it")
        train_labels = np.load(point_name+"labels.npy")
        label = np.fromfile(labels_path)
        print(label.shape, labels_path, label.shape, train_labels)

        self.testrun = testrun
        

    def __len__(self):
        if self.testrun:
            return 10
        return len(self.listalldir)

    def __getitem__(self, idx):
        seqDir = self.listalldir[idx]
        pointCloudPath = osp.join(self.filepath,seqDir,"pointcloud.ply")
        
        
        labels_path = osp.join(self.filepath, seq, "labels.npy")
        label = np.fromfile(labels_path, dtype=np.int32)
        

        if self.split != 'test':
            labelpath = osp.join(self.filepath,'labels',self.list_files[idx]+'.npy')
            label = np.load(labelpath)    
            label = np.resize(label,(W,H))
            label = Image.fromarray(label)     
               
seq_path = "D:/OpenEDS 2021/val/0024_pose_1"
# for sweep in seq_path.iterdir():
#     print(seq_str, sweep.stem)

eyeSeg = eyeSegmentation("D:/OpenEDS 2021/","val")