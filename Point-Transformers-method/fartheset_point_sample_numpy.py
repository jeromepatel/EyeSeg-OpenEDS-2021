# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:29:43 2021

@author: jyot (modified from https://github.com/corochann/chainer-pointnet/blob/master/chainer_pointnet/utils/sampling.py)
"""
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
 
 
def read_ply_points(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    points = np.stack([data['x'], data['y'], data['z']], axis=1)
    return points
 
 
class FPS:
    def __init__(self, points):
        self.points = np.unique(points, axis=0)
 
    def get_min_distance(self, a, b):
        distance = []
        for i in range(a.shape[0]):
            dis = np.sum(np.square(a[i] - b), axis=-1)
            distance.append(dis)
        distance = np.stack(distance, axis=-1)
        distance = np.min(distance, axis=-1)
        return np.argmax(distance)
 
    @staticmethod
    def get_model_corners(model):
        min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
        min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
        min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return corners_3d
 
    def compute_fps(self, K):
        # Calculate the center point
        corner_3d = self.get_model_corners(self.points)
        center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
        A = np.array([center_3d])
        B = np.array(self.points)
        t = []
                 # K  
        for i in range(K):
            max_id = self.get_min_distance(A, B)
            A = np.append(A, np.array([B[max_id]]), 0)
            B = np.delete(B, max_id, 0)
            t.append(max_id)
        return A
        # print(A)
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(A[:, 0], A[:, 1], A[:, 2], cmap='Greens')
        # plt.show()
 