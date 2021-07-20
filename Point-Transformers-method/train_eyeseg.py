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

from pathlib import Path
from tqdm import tqdm
from dataset import EyeSegDataset
import hydra
import omegaconf


seg_classes = {'Pupil' :[0], 'Iris': [1], 'Sclera': [2], 'Eye-lashes' :[3], 'Background':[4]}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
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

@hydra.main(config_path='config', config_name='eyeseg')
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # os.environ['HYDRA_FULL_ERROR'] = str(1)
    logger = logging.getLogger(__name__)

    # print(args.pretty())

    root = hydra.utils.to_absolute_path('data/')

    TRAIN_DATASET = EyeSegDataset(root=root, npoints=args.num_point, split='train', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    TEST_DATASET = EyeSegDataset(root=root, npoints=args.num_point, split='val', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=1)


    '''MODEL LOADING'''
    # args.input_dim = (6 if args.normal else 3) + 16
    args.input_dim = (4 if args.normal else 3)
    args.num_class = 5
    num_category = 5
    num_part = args.num_class
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    # torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)
    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    weights = [3.0, 2.5, 2.5, 2.0, 0.5]
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    try:
        checkpoint = torch.load('best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_avg_iou = 0
   

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points,fn,img, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            # points = points.data.numpy()
            # print(points.shape, img.shape)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            # k = torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1)
            # print(points.shape, label, target.shape,k.shape)
            points,  target = points.float().cuda(), target.long().cuda()
            optimizer.zero_grad()
            seg_pred = classifier(points,img.cuda())
            # seg_pred = classifier(points)
            del points
            # seg_pred = classifier(torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))
            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.num_point))
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train accuracy is: %.5f' % train_instance_acc)

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

            for batch_id, (points,filepath,img, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                filepath = filepath[0]
                cur_batch_size, NUM_POINT, _ = points.size()
                points,  target = points.float().cuda(), target.long().cuda()
                seg_pred = classifier(points,img.cuda())
                sparse_labels = torch.argmax(seg_pred[0], dim=1).cpu().data.numpy()
                assert sparse_labels.shape[0] != 1 

                seg_pred_intmd = seg_pred.contiguous().view(-1, num_category)
                pred_choice = seg_pred_intmd.cpu().data.max(1)[1].numpy()         
                
                # print(sparse_labels.shape, pred_choice.shape,seg_pred_intmd.shape)
                # print(sparse_labels == pred_choice)
                
                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                
                
                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (args.batch_size * NUM_POINT)
                
                for l in range(num_category):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))
                

        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32) + 1e-6))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['miou'] = mIoU
        logger.info('Test accuracy for sparse labels: %f' % (test_metrics['accuracy']))
        logger.info('The mean IOU is %f' %(mIoU))
        
        iou_per_class_str = '\n------- IoU --------\n'
        for l in range(num_category):
            iou_per_class_str += 'class %s IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (5 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
            
        logger.info(iou_per_class_str)
        
        if (test_metrics['miou'] >= best_avg_iou):
            logger.info('Save model...')
            savepath = 'best_model.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'miou': test_metrics['miou'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['miou'] > best_avg_iou:
            best_avg_iou = test_metrics['miou']
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best mIOU is: %.5f' % best_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    main()