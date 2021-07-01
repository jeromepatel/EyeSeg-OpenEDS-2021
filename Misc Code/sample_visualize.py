# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:10:08 2021

@author: jyotm
"""

from sklearn.preprocessing import OneHotEncoder
# from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
import torch
import os
# os.chdir('C:/')
# print(os.listdir())
filepath = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Point-Transformers-method/log/eyeseg/Hengshuang"
# filepath = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Misc Code/Trained-Model"
print(os.listdir(filepath))

def compute_mean_iou(flat_pred, flat_label):
    '''
    compute mean intersection over union (IOU) over all classes
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return: mean IOU
    '''
    unique_labels = np.unique(flat_label)
    num_unique_labels = len(unique_labels)

    # print(num_unique_labels, "num unique values")
    Intersect = np.zeros(num_unique_labels)
    Union = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = flat_pred == val
        label_i = flat_label == val

        # print(set(label_i),set(pred_i), "prediciton set")
    
        # print(np.logical_and(label_i, pred_i))
        Intersect[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        Union[index] = float(np.sum(np.logical_or(label_i, pred_i)) )

    print(Intersect,Union)
    mean_iou = np.mean(Intersect / Union)
    return mean_iou

Show_target = 1 # "both"

# print(os.listdir(filepath))
a = torch.load(filepath+"/first_batch.pth")
print(a.keys())
# print("Load a ply point cloud, print it, and render it")
# pcd = o3d.io.read_point_cloud( filepath + "pointcloud.ply")


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


xyz = np.array(a['points'].cpu())
print(xyz.shape, "is the shape of points")
xyz = xyz[0]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)


# print(a["target"][0][0:10])
# print(to_categorical(a['target'],5)[0][0:10])
train_preds = None

if Show_target == "both":
    train_labels = np.array(a['target'].cpu())
    train_preds = a['seg_pred'].cpu()
    train_preds = train_preds[0]
    print(train_preds.shape)

elif Show_target:
    train_labels = np.array(a['target'].cpu())
else:
    train_labels = np.array(a['seg_pred'].cpu())

train_labels = train_labels[0]


if Show_target == "both":
    train_preds = torch.argmax(train_preds, dim=1).data.numpy()
    print(train_labels.shape,train_preds.shape)
    print("----------------\n",compute_mean_iou(train_preds, train_labels), "\n------------------")
    
    
    
# train_labels = train_preds
# Show_target = True
    
# train_labels = np.load(filepath+"labels.npy")
# # train_labels = train_labels[choice]
# print(train_labels.shape)
if Show_target:
    train_labels = train_labels.reshape(train_labels.shape[0],1)
    print(f"this is the train labels shape, {train_labels.shape} with unique values {len(set(train_labels[:,0]))}")
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(train_labels)
else:
    onehot_encoded = train_labels

print(onehot_encoded.shape)
red = (onehot_encoded[:,0]  + onehot_encoded[:,2]  * 3 ) / 4
green = (onehot_encoded[:,1] + onehot_encoded[:,3] * 3) / 4
blue = (onehot_encoded[:,3] + onehot_encoded[:,4] ) / 2

# pcd.points = o3d.utility.Vector3dVector(point_set)
colours = np.vstack([red,green,blue]).transpose()
print(colours.shape)
if Show_target != "both":
    pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcd])