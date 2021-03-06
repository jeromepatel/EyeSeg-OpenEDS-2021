import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
from fartheset_point_sample_numpy import FPS
import json
import os.path as osp
from plyfile import PlyData, PlyElement
from collections import Counter 
import matplotlib.pyplot as plt
from tqdm import tqdm

class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


class EyeSegDataset(Dataset):
    def __init__(self, root='./data/val', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        # self.root = root
        # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.split = split
        self.normal_channel = normal_channel
        self.folder =  self.split
        self.data_dir = os.path.join(root, self.folder)

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
        if self.split == 'test' and self.npoints > point_set.shape[0]:
            choice = np.random.choice(len(seg),point_set.shape[0],replace=False)
        elif self.split != 'test' and self.npoints > point_set.shape[0]:
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

        return point_set,fn, image, seg

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
    data = EyeSegDataset('data', split='train',npoints=10096)
    lst = {i:0 for i in range(5)}
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    for point,fn,img, label in tqdm(DataLoader):
        # print(point.shape)
        # print(label.shape)
        k = label.data.numpy().astype(int)
        meta_data = Counter(list(k[0]))
        # print(meta_data)
        for k in meta_data.keys():
            lst[k] += 1
        # print(Counter(list(k[0])))
        # print([m.shape for m in k])
    print(lst)
    # plt.bar(list(lst.keys()), list(lst.values()), width = 0.9, color = "red")
    # plt.show()