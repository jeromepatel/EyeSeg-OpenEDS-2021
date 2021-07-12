import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import os.path as osp
from pathlib import Path
from tqdm import tqdm
from dataset import EyeSegDataset
import hydra
import omegaconf
from plyfile import PlyData
import open3d as o3d
from pointnet_util import pc_normalize

seg_classes = {'Pupil' :[0], 'Iris': [1], 'Sclera': [2], 'Eye-lashes' :[3], 'Background':[4]}
seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

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

@hydra.main(config_path='config', config_name='eyeseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    # print(args.pretty())

    root = hydra.utils.to_absolute_path('data/')
    args.batch_size = 1
    args.test_split = 'val'

    # TRAIN_DATASET = EyeSegDataset(root=root, npoints=args.num_point, split='train', normal_channel=args.normal)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    TEST_DATASET = EyeSegDataset(root=root, npoints=args.num_point, split=args.test_split, normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    '''MODEL LOADING'''
    # args.input_dim = (6 if args.normal else 3) + 16
    args.input_dim = 3 #(6 if args.normal else 3)
    args.num_class = 5
    num_category = 5
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    torch.cuda.empty_cache()
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Using pretrain model..........')
    except:
        logger.info('No existing model, please check your model.')
        start_epoch = 0

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size
    
    SAVE_LABELS = False

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        # lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        # logger.info('Learning rate:%f' % lr)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        # momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        # if momentum < 0.01:
        #     momentum = 0.01
        # print('BN momentum updated to: %f' % momentum)
        # classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)] 
            total_iou_deno_class = [0 for _ in range(num_part)] 
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            classifier = classifier.eval()
            filenames = []
            
            for batch_id, (points,filepath, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                filepath = filepath[0]
                cur_batch_size, NUM_POINT, _ = points.size()
                points,  target = points.float().cuda(), target.long().cuda()
                seg_pred = classifier(points,torch.Tensor(np.zeros((1,2048,2048))).cuda())
                sparse_labels = torch.argmax(seg_pred[0], dim=1).cpu().data.numpy()
                assert sparse_labels.shape[0] != 1 
                #run interpolation
                
                #convert to full size labels from sparse points, dense points, and sparse labels
                
                #read dense points
                dense_points = read_densePoints(root,args.test_split,filepath)
                dense_points = pc_normalize(dense_points)
                sparse_points = points.cpu().data.numpy()
                sparse_points = sparse_points.reshape(sparse_points.shape[1],3)
                
                # print(sparse_labels.shape, sparse_points.shape, dense_points.shape)
                
                dense_labels = interpolate_dense_labels(sparse_points, sparse_labels, dense_points)
                
                if SAVE_LABELS:
                    #save interpolated predictions
                    os.makedirs("output/",exist_ok=True)
                    np.save(f"output/{filepath}.npy",dense_labels)
                    filenames.append(f"Point-Transformers-method/log/eyeseg/Hengshuang/output/{filepath}.npy")
                # print(dense_labels.shape)
                
                seg_pred_intmd = seg_pred.contiguous().view(-1, num_category)
                pred_choice = seg_pred_intmd.cpu().data.max(1)[1].numpy()         
                
                # print(sparse_labels.shape, pred_choice.shape,seg_pred_intmd.shape)
                # print(sparse_labels == pred_choice)
                
                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                
                
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (args.batch_size * args.num_point)
                
                for l in range(num_category):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))
            
                print(total_correct_class[0]/total_seen)
            
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
            logger.info('Training accuracy for sparse labels: %f' % (total_correct / float(total_seen)))
            logger.info('The mean IOU is %f' %(mIoU))
            
            iou_per_class_str = '------- IoU --------\n'
            for l in range(num_category):
                iou_per_class_str += 'class %s IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (5 - len(seg_label_to_cat[l])),
                    total_correct_class[l] / float(total_iou_deno_class[l]))
            
            logger.info(iou_per_class_str)
            
                
                
                # os.makedirs(f"{filepath}/",exist_ok = True)
                # test_dict = {'points':points,'target':target,'seg_pred':seg_pred}
                # torch.save(test_dict,f"{filepath}/first_batch.pth")
                # cur_pred_val_logits = sparse_labels
                # cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                # target = target.cpu().data.numpy()
                # # print(target.shape, "adsdgs")
                # for i in range(cur_batch_size):
                #     cat = seg_label_to_cat[target[i, 0]]
                #     logits = cur_pred_val_logits[i, :, :]
                #     cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                # correct = np.sum(cur_pred_val == target)
                # total_correct += correct
                # total_seen += (cur_batch_size * NUM_POINT)

            with open("submissionFiles.txt","w") as f:
                f.write("\n".join(filenames))
            
        #     test_metrics['accuracy'] = total_correct / float(total_seen)
    

        # logger.info('Epoch %d test Accuracy: %f' % (
        #     epoch + 1, test_metrics['accuracy'])
        # )
        global_epoch += 1


if __name__ == '__main__':
    main()