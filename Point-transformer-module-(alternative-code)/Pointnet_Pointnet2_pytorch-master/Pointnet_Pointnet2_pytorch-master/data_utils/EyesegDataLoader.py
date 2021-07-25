import numpy as np
import os
from torch.utils.data import Dataset
import torch
# from pointnet_util import farthest_point_sample, pc_normalize
# from fartheset_point_sample_numpy import FPS
# import json
import os.path as osp
from plyfile import PlyData, PlyElement
from collections import Counter

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


class EyeSegDataset(Dataset):
    def __init__(self, root='./data/val', npoints=2500, split='train', class_choice=None, normal_channel=False, use_val = False):
        self.npoints = npoints
        # self.root = root
        # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.split = split
        self.normal_channel = normal_channel
        self.folder =  self.split
        self.data_dir = os.path.join(root, self.folder)
        #use val for testing during inference
        self.use_val = use_val
        self.datapath = []
        for file in os.listdir(self.data_dir):   
            if os.path.isdir(osp.join(self.data_dir,file)):
                self.datapath.append(file)
            else:
                print("wrong file path, please reconsider your life choices")
                raise IOError 

        # self.classes = {}
        # for i in self.cat.keys():
        #     self.classes[i] = self.classes_original[i]

        # print(self.classes)
        # print(self.datapath)
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Pupil' : [0], 'Iris': [1], 'Sclera': [2], 'Eye-lashes': [3], 'Background': [4]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 5


    def __getitem__(self, index):
        if index in self.cache:
            point_set, fn,image, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            data = self.load_point_cloud(fn)
            image = self.load_image(fn)
            image = image.reshape(1,image.shape[0], image.shape[1])
            # assert image.shape[0] == 2048
            point_set = data[:, 0:3]
            if self.split != "test":
                labels_path = osp.join(self.data_dir, fn, "labels.npy")
                seg = np.load(labels_path).astype(np.int8) 
            else:
                seg = np.zeros((point_set.shape[0],))
            # seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set,fn,image, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # if self.split == 'test':
        #     return point_set,fn,seg
        if self.split != 'train' and self.npoints > point_set.shape[0]:
            choice = np.random.choice(len(seg),point_set.shape[0],replace=False)
        elif self.split == 'train' and self.npoints > point_set.shape[0]:
            choice = np.random.choice(len(seg), self.npoints, replace=True)
        else:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
        
        
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        
        #alternative sampling using farthest point sampling 
        # f = FPS(point_set)
        # point_set = f.compute_fps(self.npoints)
        
        # elif self.split =='val':
        #     self.npoints = 5024
        #     choice = np.random.choice(len(seg), self.npoints, replace=True)
        #     # resample
        #     point_set = point_set[choice, :]
        #     seg = seg[choice]
        
        if self.split != 'train' and self.use_val:
            return point_set,fn, image, seg
        else:
            return point_set,seg

    def load_point_cloud(self,filepath):
        pointCloudPath = osp.join(self.data_dir,filepath,"pointcloud.ply")
        plydata = PlyData.read(pointCloudPath)
        return np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)

    def load_image(self,filepath):
        image_path = osp.join(self.data_dir, filepath, "image.npy")
        image = np.load(image_path).astype(np.float32)
        return image

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    data = EyeSegDataset('data', split='val',npoints=10096)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point,fn,img, label in DataLoader:
        print(point.shape)
        # print(label.shape)
        k = label.data.numpy().astype(int)
        meta_data = Counter(list(k[0]))
        print(meta_data)
        # print(Counter(list(k[0])))
        # print([m.shape for m in k])