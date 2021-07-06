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
sample_batch = 7
interpolated_preds = True
samplepath = filepath + f"/{files[sample_batch]}"

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

    # print(Intersect,Union)
    mean_iou = np.mean(Intersect / Union)
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
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


#visualize the onehot label point cloud or target label point cloud 
def visualize(pcd,labels,onehot = True):
    pass

xyz = np.array(a['points'].cpu())
print(xyz.shape, "is the shape of points")
xyz = xyz[0]

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
    
    #read original target label file
    orig_target = orig_data_fp + f"/{files[sample_batch]}/labels.npy"
    labels = np.load(orig_target)
    print(f"the full dense target shape is : {labels.shape}")
    
    #calculate interpolation of dense pcd from sparse labels 
    interpolated_preds = interpolate_dense_labels(xyz, train_preds, pcd_orig_points,k = 5)
    print(f"interpolation complete, the shape is {interpolated_preds.shape}")
    print(f"the MIOU is:\n", compute_mean_iou(interpolated_preds, labels))
    
    