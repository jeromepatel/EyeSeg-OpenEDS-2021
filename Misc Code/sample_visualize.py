# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:10:08 2021

@author: jyotm (apologies for the mess)
"""

from sklearn.preprocessing import OneHotEncoder
# from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np
import torch
import os
from collections import Counter
import statistics as stat
# os.chdir('C:/')
# print(os.listdir())
filepath = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Point-Transformers-method/log/eyeseg/Hengshuang"
# filepath = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Misc Code/Trained-Model"
files = os.listdir(filepath)

if len(files) > 5:
    print(files[:5],files[-5:])
else:
    print(files)
#remove redundant files 
files = [f for f in files if "_pose_" in f]
#sample point cloud to visualize
sample_batch = 8
interpolated_preds = True
samplepath = filepath + f"/{files[sample_batch]}"

def compute_mean_iou(flat_pred, flat_label):
    '''
    compute mean intersection over union (IOU) over all classes
    :param flat_pred: flattened prediction matrix
    :param flat_label: flattened label matrix
    :return: mean IOU
    '''
    val =  stat.mode(list(flat_pred))
    print("the mode of all classes is: ", val)
    print(Counter(list(flat_pred)))
    
    unique_labels = np.unique(flat_pred)
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
        
    print(Intersect)
    # print(Intersect,Union)
    mean_iou = np.mean(Intersect[:] / Union[:])
    return mean_iou

Show_target =  "both"       # 0, 1, "both" (for computing iou)

# print(os.listdir(filepath))
a = torch.load(samplepath+"/first_batch.pth")
print(a.keys())

# print("Load a ply point cloud, print it, and render it")
# pcd = o3d.io.read_point_cloud( filepath + "pointcloud.ply")

#converting labels into one hot
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    print("the shape is ",new_y)
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


#visualize the onehot label point cloud or target label point cloud 
def visualize(pcd,labels,Isonehot = True):
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(pcd)
    
    #keep onehot if its in ONEHOT else convert categorical to onehot
    if not Isonehot:
        #convert categorical into oneHot
        categorical = to_categorical(torch.Tensor(labels), 5).data.numpy()
    else:
        categorical = labels
    
    '''
    columns are classes, rows are points, we assign each class to a rgb mix (or row) channel
        0 1 0 0 0   => r
        1 0 0 0 0   => g
        0 0 1 0 0   => b
        0 0 0 0 1   => r & g
        0 0 0 1 0   => g & b
        
        final r,g,b 
        r = 0 1 0 0 1
        g = 1 0 0 1 1
        b = 0 0 1 1 0
        
    '''
    # red = categorical[:,0]  + categorical[:,3]  + categorical[:,4]
    # green = categorical[:,1] + categorical[:,3] 
    # blue = categorical[:,2] + categorical[:,4] + categorical[:,4]
    
    #above colours are not good looking but the concept is same for below colours
    red = ( categorical[:,0] * 2  + categorical[:,2]  * 3 ) / 5
    green = (categorical[:,1] * 2 + categorical[:,3] ) / 3
    blue = (categorical[:,3] + categorical[:,4] * 2  ) / 3

    colours = np.vstack([red,green,blue]).transpose()
    sparse_pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([sparse_pcd],left=-50)
    
    
    
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

xyz = np.array(a['points'].cpu())
print(xyz.shape, "is the shape of points")
xyz = xyz[0]
# xyz = pc_normalize(xyz)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)


# print(a["target"][0][0:10])
# print(to_categorical(a['target'],5)[0][0:10])
train_preds = None

#looad pc and labels
if Show_target == "both":
    train_labels = np.array(a['target'].cpu())
    train_preds = a['seg_pred'].cpu()
    train_preds = train_preds[0]
    print(train_preds.shape)

elif Show_target:
    train_labels = np.array(to_categorical(a['target'].cpu(),5))
    print(f"this is the train labels shape, {train_labels.shape}, and current sample size {a['target'].cpu().data.numpy()[0,:].shape} with unique values {len(set(a['target'].cpu().data.numpy()[0,:]))}")
else:
    train_preds = np.array(a['seg_pred'].cpu())
    print(f"the prediction labels size is {train_labels.shape} ")
    
if Show_target == "both":
    train_preds = torch.argmax(train_preds, dim=1).data.numpy()
    # print(train_labels.shape,train_preds.shape)
    print("----------------\n",compute_mean_iou(train_preds, train_labels), "\n------------------")
    
    
    
# train_labels = train_preds
# Show_target = True
    
#in case of reading samples from dataset directory instead of prediction dir, we can use next few lines
# train_labels = np.load(filepath+"labels.npy")
# print(train_labels.shape)

# if Show_target:
    #alternative to torch one hot encode function, this uses sklearn one hot encoding
    # train_labels = train_labels.reshape(train_labels.shape[0],1)
    # onehot_encoder = OneHotEncoder(sparse=False)
    # onehot_encoded = onehot_encoder.fit_transform(train_labels)


if Show_target == 1: 
    train_labels = train_labels[0]
    onehot_encoded = train_labels
elif Show_target == 0:
    train_preds = train_preds[0]
    onehot_encoded = train_preds
else:
    print("no one hot vector needed for show_target = both")   

# print(onehot_encoded.shape)
# red = onehot_encoded[:,0]
# green = onehot_encoded[:,1]
# blue = onehot_encoded[:,2]


if Show_target != "both":
    red = (onehot_encoded[:,0]  + onehot_encoded[:,2]  * 3 ) / 4
    green = (onehot_encoded[:,1] + onehot_encoded[:,3] * 3) / 4
    blue = (onehot_encoded[:,3] + onehot_encoded[:,4] ) / 2

# I was too lazy to use visualize function at this point, coz I first wrote this, so use that function instead of below code
# pcd.points = o3d.utility.Vector3dVector(point_set)
if Show_target != "both":
    colours = np.vstack([red,green,blue]).transpose()
    # print(colours.shape)
    pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.visualization.draw_geometries([pcd])


def interpolate_dense_labels(sparse_points, sparse_labels, dense_points, k=3):
    #calculate the dense point cloud labels from sparse points and corrresponding sparse labels
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(sparse_points)
    sparse_pcd_tree = o3d.geometry.KDTreeFlann(sparse_pcd)
            
    dense_labels = []
    for dense_point in dense_points:
        _, sparse_indexes, _ = sparse_pcd_tree.search_knn_vector_3d(
            dense_point, k
        )
        knn_sparse_labels = sparse_labels[sparse_indexes]
        dense_label = np.bincount(knn_sparse_labels).argmax()
        dense_labels.append(dense_label)
    return np.asarray(dense_labels)

if interpolated_preds:
    #read the pc ply file
    orig_data_fp =  "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Point-Transformers-method/data/val"
    orig_pc_filepath = orig_data_fp + f"/{files[sample_batch]}/pointcloud.ply"
    # print(orig_pc_filepath)
    #convert and extract points
    pcd_orig = o3d.io.read_point_cloud(orig_pc_filepath)
    pcd_orig_points =  np.asarray(pcd_orig.points)
    print(f"shape of original pcd is {pcd_orig_points.shape}")
    pcd_orig_points = pc_normalize(pcd_orig_points)
    
    for pt in xyz:
        if pt in pcd_orig_points:
            print("the poinnt is present ",pt)
    
    print("agsdag",np.mean(xyz, axis=0), np.mean(pcd_orig_points,axis = 0))
    #read original target label file
    orig_target = orig_data_fp + f"/{files[sample_batch]}/labels.npy"
    labels = np.load(orig_target)
    print(f"the full dense target shape is : {labels.shape}")
    
    
    print("the counter is ",Counter(labels) )
    #calculate interpolation of dense pcd from sparse labels 
    interpolated_preds = interpolate_dense_labels(xyz, train_preds, pcd_orig_points,k = 15)
    print(f"interpolation complete, the shape is {interpolated_preds.shape}")
    print(f"the MIOU is:\n", compute_mean_iou(interpolated_preds, labels))
    
    visualize(pcd_orig_points,labels,Isonehot=False)
    # visualize(pcd_orig_points,interpolated_preds,Isonehot=False)
    