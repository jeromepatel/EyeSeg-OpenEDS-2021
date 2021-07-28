# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 22:21:06 2021

@author: jyotm
usage, eg: python test_eyesemseg.py --log_dir 2021-07-25_22-08 --test_dir test --save_labels True
"""
import argparse
import os
from data_utils.EyesegDataLoader import EyeSegDataset
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np
import os.path as osp
from plyfile import PlyData
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['Pupil' , 'Iris', 'Sclera', 'Eye-lashes', 'Background']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=10096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--save_labels',type= bool, default = False, help='save prediction labels into folders')
    parser.add_argument('--valid',type=bool, default = True, help='validate result on valid dataset [default: True]')
    parser.add_argument('--test_dir', type=str,default='val', help='test dir')
    parser.add_argument('--k', type=int, default=5, help='k for interpolation [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
     
    return parser.parse_args()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def read_densePoints(root,testDir, filepath):
    pointCloudPath = osp.join(root,testDir,filepath,"pointcloud.ply")
    plydata = PlyData.read(pointCloudPath)
    return np.array(np.transpose(np.stack((plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z'])))).astype(np.float32)

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
    return np.asarray(dense_labels, dtype='uint8')

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    experiment_dir = 'log/eye_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/eyeseg/'
    NUM_CLASSES = 5
    NUM_POINT = args.num_point
    BATCH_SIZE = args.batch_size
    SAVE_LABELS = args.save_labels
    TEST_DIR = args.test_dir
    num_votes = args.num_votes
    if TEST_DIR == 'test':
        args.valid = False
    
    TEST_DATASET = EyeSegDataset(root=root, npoints=args.num_point, split=TEST_DIR, use_val = True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=2)
    # log_string("The number of test data is: %d" % len(TEST_DATASET))
    

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    
    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)] 
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        
        filenames = []
        for batch_id, (points,filepath,img, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            filepath = filepath[0]
            points = points.data.numpy()
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            total_pred = np.zeros((1,points.shape[2]))
            for i in range(num_votes):
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                pred_val = np.argmax(pred_val, 2)
                total_pred += pred_val
            pred_val = total_pred / num_votes
            pred_val = np.rint(pred_val).astype(int)
            
            #convert to full size labels from sparse points, dense points, and sparse labels
            #read dense points
            dense_points = read_densePoints(root,TEST_DIR,filepath)
            dense_points = pc_normalize(dense_points)
            sparse_points = points.transpose(1,2).cpu().data.numpy()
            sparse_points = sparse_points[0]
            sparse_label = pred_val.reshape(pred_val.shape[1],)
            
            if sparse_label.shape[0] >= dense_points.shape[0]:
                dense_label = sparse_label.copy()
                log_string(f"skipped an interpolation with {sparse_label.shape[0]} points")
            else:
                dense_label = interpolate_dense_labels(sparse_points, sparse_label, dense_points, k = args.k)
            # print(sparse_label.shape, sparse_points.shape, dense_points.shape,dense_label.shape)
            
            if args.valid:
                labels_path = f"{root}{TEST_DIR}/{filepath}/labels.npy"
                batch_label = np.load(labels_path).astype(np.int8)
                # print(batch_label.shape)
        
            if SAVE_LABELS:
                #save interpolated predictions
                outputs_dir = experiment_dir + '/output/'
                os.makedirs(outputs_dir,exist_ok=True)
                np.save(outputs_dir + f"{filepath}.npy",dense_label)
                filenames.append(outputs_dir + f"{filepath}.npy")

            if args.valid:
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((dense_label == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((dense_label == l) | (batch_label == l)))
    
        if args.valid:
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            mIoU_4_class = np.mean(np.array(total_correct_class[:4]) / (np.array(total_iou_deno_class[:4], dtype=np.float) + 1e-6))
            
            
            log_string('eval 4 classes mean IoU: %f' % (mIoU_4_class))
            log_string('eval point avg class IoU: %f' % (mIoU))
            
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
    
        
        if SAVE_LABELS:
            with open("submissionFilesSemSeg.txt","w") as f:
                f.write("\n".join(filenames))
        
if __name__ == '__main__':
    args = parse_args()
    main(args)


    
