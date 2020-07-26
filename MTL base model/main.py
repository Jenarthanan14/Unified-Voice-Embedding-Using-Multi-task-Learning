# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import socket
import timeit
import cv2
from datetime import datetime
import imageio
import numpy as np

# PyTorch includes
import torch
import torch.optim as optim
from torch.nn.functional import interpolate
import torch.nn as nn
import torchvision.transforms as transforms


# Custom includes
# from attention.fblib.util.helpers import generate_param_report
from DB_wav_reader import read_feats_structure
import configure as c
from SR_Dataset import read_MFB, TruncatedInputfromMFB, ToTensorInput, ToTensorDevInput, DvectorDataset, collate_fn_feat_padded
from utils import lr_poly
from loss import BalancedCrossEntropyLoss, SoftMaxwithLoss, NormalsLoss, DepthLoss
# from attention.experiments.dense_predict import common_configs
# from attention.fblib.util.mtl_tools.multitask_visualizer import TBVisualizer, visualize_network
# from attention.fblib.util.model_resources.flops import compute_gflops
# from attention.fblib.util.model_resources.num_parameters import count_parameters
from utils import AverageMeter

# Custom optimizer
from select_used_modules import make_closure

# Configuration file
# from attention.experiments.dense_predict.pascal_resnet import config as config

# Network file
import model.deeplab_se_resnet_multitask as se_resnet_multitask

# Tensorboard include
from tensorboardX import SummaryWriter

def load_dataset(val_ratio):
    # Load training set and validation set
    
    
    # Split training set into training set and validation set according to "val_ratio"
    train_DB, valid_DB = split_train_dev(c.TRAIN_FEAT_DIR, val_ratio)
    # print(train_DB)
    # print(valid_DB)
    
    file_loader = read_MFB # numpy array:(n_frames, n_dims)
     
    transform = transforms.Compose([
        TruncatedInputfromMFB(), # numpy array:(1, n_frames, n_dims)
        ToTensorInput() # torch tensor:(1, n_dims, n_frames)
    ])
    transform_T = ToTensorDevInput()
   
    
    speaker_list_train = sorted(set(train_DB['speaker_id'])) # len(speaker_list) == n_speakers
    spk_to_idx_train = {spk: i for i, spk in enumerate(speaker_list_train)}
    speaker_list_valid = sorted(set(valid_DB['speaker_id']))  # len(speaker_list) == n_speakers
    spk_to_idx_valid = {spk: i for i, spk in enumerate(speaker_list_valid)}
    # print(speaker_list)
    # print(spk_to_idx)
    train_dataset = DvectorDataset(DB=train_DB, loader=file_loader, transform=transform, spk_to_idx=spk_to_idx_train)
    valid_dataset = DvectorDataset(DB=valid_DB, loader=file_loader, transform=transform_T, spk_to_idx=spk_to_idx_valid)
    
    n_classes = len(speaker_list_train) # How many speakers? 240
    return train_dataset, valid_dataset, n_classes

def split_train_dev(train_feat_dir, valid_ratio):
    train_valid_DB = read_feats_structure(train_feat_dir)
    total_len = len(train_valid_DB) # 148642
    valid_len = int(total_len * valid_ratio/100.)
    train_len = total_len - valid_len
    shuffled_train_valid_DB = train_valid_DB.sample(frac=1).reset_index(drop=True)
    # Split the DB into train and valid set
    train_DB = shuffled_train_valid_DB.iloc[:train_len]
    valid_DB = shuffled_train_valid_DB.iloc[train_len:]
    # Reset the index
    train_DB = train_DB.reset_index(drop=True)
    valid_DB = valid_DB.reset_index(drop=True)
    print('\nTraining set %d utts (%0.1f%%)' %(train_len, (train_len/total_len)*100))
    print('Validation set %d utts (%0.1f%%)' %(valid_len, (valid_len/total_len)*100))
    print('Total %d utts' %(total_len))
    
    return train_DB, valid_DB


def get_net_resnet(arch,resume_epoch,num_classes):
    """
    Define the network (standard Deeplab ResNet101) and the trainable parameters
    """
    SAVE_MODEL_DIR='model_saved/'
    if arch== 'se_res26':
        network = se_resnet_multitask.se_resnet26
    elif arch == 'se_res50':
        network = se_resnet_multitask.se_resnet50
    elif arch == 'se_res101':
        network = se_resnet_multitask.se_resnet101

    # print('Creating ResNet model: {}'.format(p.NETWORK))
    tasks_names=['identification','clustering']
    squeeze_enc = False
    squeeze_dec = False
    adapters = False
    width_decoder = 256
    norm_per_task = False



    net = network(tasks=tasks_names, n_classes=num_classes, pretrained='scratch', classifier='uber',
                  output_stride=8, train_norm_layers=True, width_decoder=width_decoder,
                  squeeze_enc=squeeze_enc, squeeze_dec=squeeze_dec, adapters=adapters,
                  norm_per_task=norm_per_task, dscr_type=None)

    if resume_epoch !=0:
        print("Initializing weights from: {}".format(
            os.path.join(SAVE_MODEL_DIR, 'models', 'model_epoch-' + str(resume_epoch - 1) + '.pth')))
        state_dict_checkpoint = torch.load(
            os.path.join(SAVE_MODEL_DIR, 'models', 'model_epoch-' + str(resume_epoch- 1) + '.pth')
            , map_location=lambda storage, loc: storage)

        net.load_state_dict(state_dict_checkpoint)

    return net

def get_train_params(net, p,tasks_names,dscr_type):
    train_params = [{'params': se_resnet_multitask.get_lr_params(net, part='backbone', tasks=tasks_names),
                     'lr': p[0]},
                    {'params': se_resnet_multitask.get_lr_params(net, part='decoder', tasks=tasks_names),
                     'lr': p[0] * p[1]},
                    {'params': se_resnet_multitask.get_lr_params(net, part='task_specific', tasks=tasks_names),
                     'lr': p[0] * p[2]}]
    if dscr_type is not None:
        train_params.append(
            {'params': se_resnet_multitask.get_lr_params(net, part='discriminator', tasks=tasks_names),
             'lr': p[0] * p[1]})

    return train_params

def get_loss(p, task=None):
    if task == 'edge':
        criterion = BalancedCrossEntropyLoss(size_average=True)
    elif task == 'identification' or task == 'human_parts':
        criterion = SoftMaxwithLoss()
    elif task == 'normals':
        criterion = NormalsLoss(normalize=True, size_average=True, norm=p['normloss'])
    elif task == 'clustering':
        criterion = BalancedCrossEntropyLoss(size_average=True)
    elif task == 'depth':
        criterion = DepthLoss()
    elif task == 'albedo':
        criterion = torch.nn.L1Loss(reduction='elementwise_mean')
    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, albedo, or normals')

    return criterion

def accuracy(output, target, topk=(1,), ignore_label=255):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = (target != ignore_label).sum().item()
    if batch_size == 0:
        return -1

    _, pred = output.topk(maxk, 1, True, True)
    if pred.shape[-1] == 1:
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        correct = pred.eq(target.unsqueeze(1))

    res = []
    for _ in topk:
        correct_k = correct[:].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def main():
    # p = config.create_config()

    gpu_id = 0
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    # p.TEST.BATCH_SIZE = 32

    # Setting parameters
    n_epochs = 100
    resume_epoch=0
    lr = 0.001 # Initial learning rate
    lr_dec=1
    lr_tsk=1
    wd = 1e-4 # Weight decay (L2 penalty)
    batch_size = 10 # Batch size for training
    valid_batch_size = 5 # Batch size for validation
    log_interval=84
    use_shuffle = True # Shuffle for training or not
    multi_task_loss={'identification':1, 'clustering':1}
    train_momentum=0.9
    SAVE_MODEL_DIR='model_saved/'
    tasks_names = ['identification', 'clustering']
    dscr_type=None
    print("Total training epochs: {}".format(n_epochs))
    # print(p)
    # print('Training on {}'.format(p['train_db_name']))

    snapshot = 2  # Store a model every snapshot epochs
    test_interval = 100 # Run on test set every test_interval epochs
    seed=2
    torch.manual_seed(seed)
    use_test=True
    val_ratio=10
    train_dataset, valid_dataset, num_classes = load_dataset(val_ratio)
    exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    if not os.path.exists(os.path.join(SAVE_MODEL_DIR, 'models')):
        if resume_epoch == 0:
            os.makedirs(os.path.join(SAVE_MODEL_DIR, 'models'))
    n_classes={'identification':num_classes, 'clustering':num_classes}
    net = get_net_resnet('se_res26',resume_epoch,n_classes)

    net.to(device)

    if resume_epoch != n_epochs:
        criteria_tr = {}
        criteria_ts = {}

        running_loss_tr = {task: 0. for task in tasks_names}
        running_loss_ts = {task: 0. for task in tasks_names}
        curr_loss_task = {task: 0. for task in tasks_names}
        counter_tr = {task: 0 for task in tasks_names}
        counter_ts = {task: 0 for task in tasks_names}

        # Discriminator loss variables for logging
        running_loss_tr_dscr = 0
        running_loss_ts_dscr = 0

        # Logging into Tensorboard
        log_dir = os.path.join(SAVE_MODEL_DIR, 'models',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)

        # Training parameters and their optimizer
        p=[lr,lr_dec,lr_tsk]
        train_params = get_train_params(net, p,tasks_names,dscr_type)

        optimizer = optim.SGD(train_params, lr=lr, momentum=train_momentum, weight_decay=wd)

        for task in tasks_names:
            # Losses
            criteria_tr[task] = get_loss(p, task)
            criteria_ts[task] = get_loss(p, task)
            criteria_tr[task].to(device)
            criteria_ts[task].to(device)

        # Preparation of the data loaders
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                       shuffle=use_shuffle)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                    batch_size=valid_batch_size,
                                                    shuffle=False,
                                                    collate_fn = collate_fn_feat_padded)
        # Train variables
        num_img_tr = len(train_loader)
        num_img_ts = len(valid_loader)

        print("Training Network")
        # Main Training and Testing Loop
        for epoch in range(resume_epoch, n_epochs):
            top1_dscr = AverageMeter()
            start_time = timeit.default_timer()

            # One training epoch
            net.train()

            alpha = 2. / (1. + np.exp(-10 * ((epoch + 1) / n_epochs))) - 1  # Ganin et al for gradient reversal

            train_acc = {}
            n_correct={}
            n_total={}
            losses={}
            for task in tasks_names:
                train_acc[task] = AverageMeter()
                n_correct[task], n_total[task] = 0, 0
                losses[task]=AverageMeter()

            if dscr_type is not None:
                print('Value of alpha: {}'.format(alpha))

            for ii, (sample) in enumerate(train_loader):
                curr_loss_dscr = 0
                inputs, targets = sample  # target size:(batch size,1), input size:(batch size, 1, dim, win)
                current_sample = inputs.size(0)  # batch size
                targets[0] = targets[0].view(-1)  # target size:(batch size)
                targets[1] = targets[1].view(-1)  # target size:(batch size)
                tasks = net.tasks
                gt_elems = {x: targets[tasks.index(x)] for x in tasks}

                outputs = {}
                for task in tasks_names:
                    # Forward pass
                    output = {}
                    features = {}
                    spk_embeddings = {}
                    output[task], features[task],spk_embeddings[task] = net.forward(inputs,task=task)
                    losses_tasks, losses_dscr, outputs_dscr, grads, task_labels \
                        = net.compute_losses(output, features, criteria_tr, gt_elems, alpha, multi_task_loss)

                    loss_tasks = losses_tasks[task]
                    running_loss_tr[task] += losses_tasks[task].item()
                    curr_loss_task[task] = losses_tasks[task].item()

                    counter_tr[task] += 1

                    # Store output for logging
                    # outputs[task] = output[task].detach()

                    # print stuff
                    n_correct[task] += (torch.max(output[task], 1)[1].long().view(gt_elems[task].size()) == gt_elems[
                        task]).sum().item()
                    n_total[task] += current_sample
                    train_acc_temp = 100. * n_correct[task] / n_total[task]
                    train_acc[task].update(train_acc_temp, inputs.size(0))
                    losses[task].update(loss_tasks.item(), inputs.size(0))
                    if ii % log_interval == 0:
                        print('  * Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                              'Loss {loss.avg:.4f}\t'
                              'Task {task}\t'
                              'Accuracy {train_acc.avg:.4f}\t'.format(
                            epoch, ii * len(inputs), len(train_loader.dataset),
                                   100. * ii / len(train_loader),
                            loss=losses[task],task=task, train_acc=train_acc[task]))

                        file1 = open("output/train_result.txt", "a")
                        file1.write('\n *Train Epoch: {:3d} [{:8d}/{:8d} ({:3.0f}%)]\t'
                                    'Task {task}\t'
                                    'Loss {loss.avg:.4f}\t'
                                    'Acc {train_acc.avg:.4f}'.format(
                            epoch, ii * len(inputs), len(train_loader.dataset),
                                   100. * ii / len(train_loader),
                            loss=losses[task], task=task, train_acc=train_acc[task]))
                        file1.close()
                        

                    if dscr_type is not None:
                        # measure loss, accuracy and record accuracy for discriminator
                        loss_dscr = losses_dscr[task]
                        running_loss_tr_dscr += losses_dscr[task].item()
                        curr_loss_dscr += loss_dscr.item()

                        prec1 = accuracy(outputs_dscr[task].data, task_labels[task], topk=(1,))
                        if prec1 != -1:
                            top1_dscr.update(prec1[0].item(), task_labels[task].size(0))

                        loss = (1 - p['dscr_w']) * loss_tasks + p['dscr_w'] * loss_dscr
                    else:
                        loss = loss_tasks

                    # Backward pass inside make_closure to update only weights that were used during fw pass
                    optimizer.zero_grad()
                    optimizer.step(closure=make_closure(loss=loss, net=net))
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                if ii % num_img_tr == num_img_tr - 1:
                    lr_ = lr_poly(lr, iter_=epoch, max_iter=n_epochs)
                    print('(poly lr policy) learning rate: {0:.6f}'.format(lr_))
                    train_params = get_train_params(net, p,tasks_names,dscr_type)
                    optimizer = optim.SGD(train_params, lr=lr_, momentum=train_momentum, weight_decay=wd)
                    optimizer.zero_grad()

            # Save the model
            if (epoch % snapshot) == snapshot - 1 and epoch != 0:
                torch.save(net.state_dict(), os.path.join(SAVE_MODEL_DIR, 'models',
                                                          'model_epoch-' + str(epoch) + '.pth'))

            # One testing epoch
            if use_test and epoch % test_interval == (test_interval - 1):
                print('Testing Phase')
                top1_dscr = AverageMeter()
                net.eval()
                val_acc={}
                n_correct={}
                n_total={}
                for task in tasks_names:
                    val_acc[task] = AverageMeter()
                    n_correct[task],n_total[task]=0,0
               
                for ii, (sample) in enumerate(valid_loader):

                    inputs, targets = sample
                    current_sample = inputs.size(0)  # batch size
                    temp1, temp2 = [], []
                    labels1 = list(targets)
                    for x in range(len(labels1)):
                        temp1.append(labels1[x][0])
                        temp2.append(labels1[x][1])
                    temp1 = tuple(temp1)
                    temp2 = tuple(temp2)
                    temp1 = torch.stack(temp1, 0)
                    temp2 = torch.stack(temp2, 0)
                    temp1 = temp1.view(-1)
                    temp2 = temp2.view(-1)
                    labelstemp = []
                    labelstemp.append(temp1)
                    labelstemp.append(temp2)
                    # task_gts = list(sample.keys())
                    tasks = net.tasks
                    # print(tasks)
                    gt_elems = {x: labelstemp[tasks.index(x)] for x in tasks}

                    outputs = {}
                    for task in tasks_names:
                        output = {}
                        features = {}
                        spk_embedding = {}
                        output[task], features[task],spk_embedding[task] = net.forward(inputs,task=task)
                        losses_tasks, losses_dscr, outputs_dscr, grads, task_labels \
                            = net.compute_losses(output, features, criteria_tr, gt_elems, alpha, p)

                        running_loss_ts[task] += losses_tasks[task].item()
                        counter_ts[task] += 1

                        # for logging
                        # outputs[task] = output[task].detach()

                        n_correct[task] += (torch.max(output[task], 1)[1].long().view(gt_elems[task].size()) == gt_elems[
                            task]).sum().item()
                        n_total[task] += current_sample
                        val_acc_temp = 100. * n_correct[task] / n_total[task]
                        val_acc[task].update(val_acc_temp, inputs.size(0))
                        print('  * Validaton: '
                              'epoch {epoch}\t'
                              'Task {task}\t'
                              'Loss {loss:.4f}\t'
                              'Accuracy {valid_acc.avg:.4f}\t'.format(
                            epoch=epoch, task=task, loss=losses_tasks[task], valid_acc=val_acc[task]) )

                        file2 = open("output/validation_result.txt", "a")
                        file2.write('\n *Validaton: '
                                    'epoch {epoch}\t'
                                    'Task {task}\t'
                                    'Loss {loss:.4f}\t'
                                    'Accuracy {valid_acc.avg:.4f}\t'.format(
                            epoch=epoch, task=task, loss=losses_tasks[task], valid_acc=val_acc[task]))
                        file2.close()

                        if dscr_type is not None:
                            running_loss_ts_dscr += losses_dscr[task].item()

                            # measure accuracy and record loss for discriminator
                            prec1 = accuracy(outputs_dscr[task].data, task_labels[task], topk=(1,))
                            if prec1 != -1:
                                top1_dscr.update(prec1[0].item(), task_labels[task].size(0))


        writer.close()

    # Generate Results
    # net.eval()
    # _, _, transforms_infer = config.get_transformations(p)
    # for db_name in p['infer_db_names']:
    #
    #     valid_loader = config.get_valid_loader(p, db_name=db_name, transforms=transforms_infer, infer=True)
    #     save_dir_res = os.path.join(p['save_dir'], 'Results_' + db_name)
    #
    #     print('Testing Network')
    #     # Main Testing Loop
    #     with torch.no_grad():
    #         for ii, sample in enumerate(valid_loader):
    #
    #             img, meta = sample['image'], sample['meta']
    #
    #             # Forward pass of the mini-batch
    #             inputs = img.to(device)
    #             tasks = net.tasks
    #
    #             for task in tasks:
    #                 output, _ = net.forward(inputs, task=task)
    #
    #                 save_dir_task = os.path.join(save_dir_res, task)
    #                 if not os.path.exists(save_dir_task):
    #                     os.makedirs(save_dir_task)
    #
    #                 output = interpolate(output, size=(inputs.size()[-2], inputs.size()[-1]),
    #                                      mode='bilinear', align_corners=False)
    #                 output = common_configs.get_output(output, task)
    #
    #                 for jj in range(int(inputs.size()[0])):
    #                     if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == 255:
    #                         continue
    #
    #                     fname = meta['image'][jj]
    #
    #                     result = cv2.resize(output[jj], dsize=(meta['im_size'][1][jj], meta['im_size'][0][jj]),
    #                                         interpolation=p.TASKS.INFER_FLAGVALS[task])
    #
    #                     imageio.imwrite(os.path.join(save_dir_task, fname + '.png'), result.astype(np.uint8))

    # if p.EVALUATE:
    #     common_configs.eval_all_results(p)


if __name__ == '__main__':
    main()
