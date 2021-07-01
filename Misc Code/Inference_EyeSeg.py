# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 17:27:05 2021

@author: jyotm
"""
import torch
import os
import hydra
import numpy
import shutil
import importlib
import logging
import omegaconf
import sys
from tqdm import tqdm
import open3d as o3d
from sklearn.preprocessing import OneHotEncoder
from torchviz import make_dot

filepath = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Point-Transformers-method/"
# filepath = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Misc Code/Trained-Model"
# print(os.listdir(filepath))
sys.path.append(os.path.abspath(filepath))

import dataset

# print(checkpoint)


def visualize_ptc(xyz, train_labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    train_labels = train_labels.reshape(train_labels.shape[0],1)
    print(f"this is the train labels shape, {train_labels.shape} with unique values {len(set(train_labels[:,0]))}")
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(train_labels)
    
    red = (onehot_encoded[:,0]  + onehot_encoded[:,2]  * 3 ) / 4
    green = (onehot_encoded[:,1] + onehot_encoded[:,3] * 3) / 4
    blue = (onehot_encoded[:,3] + onehot_encoded[:,4] ) / 2
    colours = np.vstack([red,green,blue]).transpose()
    
    pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcd])

@hydra.main(config_path= filepath + 'config', config_name='eyeseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)
    args.input_dim = 3
    args.num_class = 5
    
    
    # print(args.pretty())
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    checkpoint = torch.load(filepath + 'log/eyeseg/Hengshuang/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Using pretrain model')
    
    root = hydra.utils.to_absolute_path(filepath + 'data/')
    print(root)
    VAL_DATASET = dataset.EyeSegDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=2)
    classifier.eval()
    
    with torch.no_grad():
        for batch_id, (points,fn, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points,  target = points.float().data.cuda(), target.long().data.numpy()
            # print(batch_id, points.shape, target.shape)
            pred_seg = classifier(points)
            
            # print(batch_id, pred_seg.shape)
    
    # x = torch.randn(1, 1800, 3).cuda()
    # y = classifier(x)
    # print(classifier)
    # make_dot(y.mean(), params=dict(classifier.named_parameters()),show_attrs=True, show_saved=True)
    # print(torch.cuda.memory_summary())
        
    
    
if __name__ == '__main__':
    main()