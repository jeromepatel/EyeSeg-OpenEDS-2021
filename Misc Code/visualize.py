import numpy as np
# import laspy as lp
# from plyfile import PlyData, PlyElement
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import os

visualize =  1

point_name = "C:/Users/jyotm/Documents/OpenEDS 2021 3d point cloud segmentation/Point-Transformers-method/data/val/0024_pose_1/"
print(os.listdir(point_name))

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud( point_name+ "pointcloud.ply")

# pointCloudPath = osp.join(point_name,"pointcloud.ply")
# plydata = PlyData.read(pointCloudPath)
# pts = np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)

pts = np.asarray(pcd.points)

# choice = np.random.choice(len(pts), 4096, replace=True)
# point_set = pts[choice, :]

# point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
# dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
# point_set = point_set / dist


train_labels = np.load(point_name+"labels.npy")
# train_labels = train_labels[choice]
print(train_labels.shape)
train_labels = train_labels.reshape(train_labels.shape[0],1)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(train_labels)
red = (onehot_encoded[:,0]  + onehot_encoded[:,2]  *3 ) / 4
green = (onehot_encoded[:,1] + onehot_encoded[:,3] *3) / 4
blue = (onehot_encoded[:,3] + onehot_encoded[:,4] ) / 2

# pcd.points = o3d.utility.Vector3dVector(point_set)
colours = np.vstack([red,green,blue]).transpose()
print(colours.shape)
if visualize:
    pcd.colors = o3d.utility.Vector3dVector(colours)
print(pcd)


if visualize:
    o3d.visualization.draw_geometries([pcd])
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.65)
a = np.asarray(downpcd.points)
print("size of downsampled aray is ",a.shape)
if visualize:
    o3d.visualization.draw_geometries([downpcd])

print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30))
if visualize:
    o3d.visualization.draw_geometries([downpcd])

print("Print a normal vector of the 0th point")
print(downpcd.normals[0])
print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals)[:10, :])
print("")


train_image = np.load(point_name+"image.npy")
print(train_image.shape)
# plt.imshow(train_image)

train_labels = np.load(point_name+"labels.npy")
print(set(train_labels))
print(train_labels.shape)



