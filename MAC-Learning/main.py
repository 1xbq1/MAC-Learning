#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
from sklearn.cluster import KMeans

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import torch.nn.functional as F


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=24,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser

def del_tensor_ele(arr,index_list):
    index1 = index2 = 0
    arr_mid = None
    for ind in index_list:
        index2 = ind
        arr_mid = torch.cat((arr_mid, arr[index1:index2]),dim=0) if arr_mid is not None else arr[index1:index2]
        index1 = index2 + 1
    arr_mid = torch.cat((arr_mid, arr[index1:]),dim=0) if arr_mid is not None else arr[index1:]
    return arr_mid

def square_distance(matrix1, matrix2):
    """
    matrix1 (d x n)
    matrix2 (d x m)
    return a matrix (n x m) where [i,j] is going to be the squared distance between the point i-th point in matrix1 and
     the j-th point in matrix2.
    """
    # TODO: error check that the dimensions match

    # Get the distances
    (d, n) = matrix1.size()
    (d, m) = matrix2.size()

    # Matrix that shows the distance between two points
    distances = torch.zeros([n, m])

    # For each point in the first matrix calculate the squared distance between it and each point in the other matrix
    for i in range(n):
        for j in range(m):
            #distances[i, j] = torch.norm(matrix1[:, i] - matrix2[:, j])**2
            distances[i, j] = torch.sum(torch.pow(matrix1[:, i] - matrix2[:, j], 2))

    return distances

def get_Z(data_matrix, anchor_matrix, closest_anchors, weight_flag=0, num_iterations=0):
    """
    data_matrix: is (d x n) matrix of the input data; where 'n' is the number of samples and 'd' is the dimension of
    each sample.
    anchor_matrix: is the (d x m) matrix of anchors; where 'm' is the number of anchors and 'd' is the dimension of each
    anchor.  The anchors are generally not samples in the data_matrix but are derived from it.
    closest_anchors must be less than the total number of anchors; this is 's' variable; it's how many closest anchors
    we look for each sample

    Return The Z matrix (weight) which is (n x m).  Represents the weighted connection between samples and anchors.
    """

    # Extract important parameters from the dimensions of the matrices
    (dimensions, num_anchors) = anchor_matrix.size()
    num_samples = data_matrix.size(1)

    # The Z matrix. This matrix will define the weighted connections between the samples and the anchors.
    # The idea is that closer anchors will have heavier weights, for a given sample, than farther ones.
    weight_matrix = torch.zeros((num_samples, num_anchors))

    # Calculate the pairwise squared distances; passing in two matrices; we are calculating the distance between every
    # sample and every anchor
    # NOTE: Can this be done simply in numpy? I see they have a pdist function but that is not going to take in two
    # separate matrices
    distances = square_distance(data_matrix, anchor_matrix)

    # Track the distances of al the closest anchors for each sample
    distances_closest_anchors = torch.zeros((num_samples, closest_anchors))
    indices_closest_anchors = torch.zeros((num_samples, closest_anchors))
    
    distances_sort, indices_sort = torch.sort(distances, dim=1)
    distances_closest_anchors[:,:] = distances_sort[:,:closest_anchors]
    indices_closest_anchors[:,:] = indices_sort[:,:closest_anchors]
    # We will want to find the 'closest_anchors' number of closest anchors for each point; both the value and the indices
    '''for i in range(0, closest_anchors):
        # For each row (sample) determine the min values and associated indices for the closest anchor
        # NOTE: we can probably find the top closest_anchors for each in one go
        min_values, min_indices = min_matrix(distances)
        distances_closest_anchors[:, i] = min_values
        indices_closest_anchors[:, i] = min_indices

        # Now we are going to effectively make sure that we do re-use the same anchors by setting those distances to
        # infinity for each
        # NOTE: faster way of doing this? so you can use an array for the indices into the matrix; that would probably
        # be much faster.
        distances = apply_to_all_rows(distances, min_indices)'''

    # Apply the kernel
    if weight_flag == 0:
        # sigma = mean(val(:,s).^0.5);
        # We calculate "sigma" which is going to be the equal to average of the square root of the maximum of min value
        # for each sample.  The last column of "distances_closest_anchors" will be the furthest away of the closest
        # anchors.
        sigma = torch.mean(distances_closest_anchors[:, -1]**0.5)

        # NOTE: Possible error?  Maybe missing a () around the sigma^2.  Otherwise the 1/1 is not needed.
        #val = exp(-val/(1/1*sigma^2));
        distances_closest_anchors = torch.exp(-1*distances_closest_anchors/(sigma**2))

        #val = repmat(sum(val,2).^-1,1,s).*val;
        distances_closest_anchors = torch.transpose(((torch.sum(distances_closest_anchors, dim=1)**-1).repeat(closest_anchors, 1)), 0, 1)*distances_closest_anchors

    else:
        # TODO: Apply LAE
        pass

    # Now we need to set the Z matrix; the indices_closest_anchors has the same number of rows as Z but fewer columns;
    # the values in that matrix at [i,j] corresponds to the column we are setting in Z and the value we are setting
    # there is going to by [i,j] in the distances_closest_anchors matrix.
    # TODO: Use better indexing
    for i in range(num_samples):
        for j in range(closest_anchors):
            weight_matrix[i, int(indices_closest_anchors[i, j])] = distances_closest_anchors[i, j]

    return weight_matrix

def get_W(Z,Z_ind):
    # NOTE: Formula I have for the W creation from Z -- in MATLAB -- W = Z*diag(sum(Z).^-1)*transpose(Z)
    # diag(V) returns a square diagonal matrix with the elements of vector V on the main diagonal
    # so sum(Z) will return a vector where each element in the vector is the sum of that associated column in Z.
    # then .^-1 does an element-wise 1/x on each x
    # Z is a (n x m) matrix and we want to return the (n x n) matrix, W.
    # In dimension terms we have (n x n) = (n x m) * (m x m) * (m x n)

    # sum of the columns; make sure that Z is a float type array; produces a vector of length "m"; one element per column
    column_vector_sum = torch.sum(Z.clone().detach().requires_grad_(True), dim=0)

    # take each element to the power of -1
    power_vector = torch.pow(column_vector_sum, -1)
    
    w1 = torch.mm(Z_ind.to(torch.float32), torch.diag(power_vector).to(torch.float32))

    # now put this power vector as the diagonal of matrix of zeros; multiple by the transpose
    return torch.mm(w1, Z_ind.t().to(torch.float32))

def getins(output):
    if isinstance(output, tuple):
        output, l1 = output
        l1 = l1.mean()
    else:
        l1 = 0
    return output, l1

class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                # self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                # self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                pass
                # self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=False,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
    
    def info_nce_loss0(self, features):
        
        labels = torch.cat([torch.arange(features.shape[0]//2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(self.output_device)
        
        features = F.normalize(features, dim=1)
        
        similarity_matrix = torch.matmul(features, features.T)
        
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.output_device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape
        
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.output_device)
        
        logits = logits / 0.07
        
        return self.loss(logits, labels)
    
    def anchor_graph(self, features, indexs): # 2batchsize * num_class
        
        k = 400 # number of anchors
        s = 50 # it's how many closest anchors we look for each sample
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features.detach().cpu().numpy())
        anchors = torch.tensor(kmeans.cluster_centers_).cuda(self.output_device)
        
        #after clustering, adjust the sequence order to match the original data set
        features_copy = torch.zeros(features.size())
        for i in range(features.shape[0]):
            features_copy[indexs[i]] = features[i]
        
        features_copy = features_copy.cuda(self.output_device)
        z_mid = get_Z(features_copy.t(), anchors.t(), s).cuda(self.output_device)
        
        _, z_index = z_mid.max(1)
        z_index = z_index.unsqueeze(1)
        
        return z_index, z_mid
    
    def info_nce_loss(self, features, index): # anchor contrastive loss
        #global z_, w_
        z_1 = self.z_[index]
        z_ = z_1.repeat(2,1)
        
        with torch.no_grad():
            z_ind = self.z_w[index]
            w_1 = get_W(self.z_w, z_ind).cuda(self.output_device)
        
        w_ = w_1.repeat(2,2)
        
        z_last = (z_ == z_.t()).float()
        z_last = z_last.cuda(self.output_device)
        
        features = F.normalize(features, dim=1)
        similarity_matrix = (w_.cuda(self.output_device)).mul(torch.exp(torch.matmul(features, features.T)/0.07)) #逐点相乘
        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).cuda(self.output_device)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        z_last = z_last[~mask].view(z_last.shape[0], -1)
        
        positives = similarity_matrix.mul(z_last)
        
        softmax = torch.sum(positives, dim=1).reshape(-1,1)/torch.sum(similarity_matrix, dim=1).reshape(-1,1)
        loss = -torch.sum(torch.log(softmax))/similarity_matrix.shape[0]
        return loss
    
    def info_nce_loss_i(self, featuresi, features, index):
        V, _, _ = featuresi.size()
        loss = None
        for v in range(V):
            features_ = torch.cat((featuresi[v], features), dim=0)
            
            lo = self.info_nce_loss(features_, index)
            loss = loss + lo if loss is not None else lo
        return loss/V
    
    def info_nce_loss_i0(self, featuresi, features):
        V, _, _ = featuresi.size()
        loss = None
        for v in range(V):
            features_ = torch.cat((featuresi[v], features), dim=0)
            
            lo = self.info_nce_loss0(features_)
            loss = loss + lo if loss is not None else lo
        return loss/V
    
    def info_nce_loss_ii(self, featuresi, featuresj, index):
        V, _, _ = featuresi.size()
        loss = None
        for v in range(V):
            features_ = torch.cat((featuresi[v], featuresj[v]), dim=0)
            
            lo = self.info_nce_loss(features_, index)
            loss = loss + lo if loss is not None else lo
        return loss/V
    
    def info_nce_loss_ii0(self, featuresi, featuresj):
        V, _, _ = featuresi.size()
        loss = None
        for v in range(V):
            features_ = torch.cat((featuresi[v], featuresj[v]), dim=0)
            
            lo = self.info_nce_loss0(features_)
            loss = loss + lo if loss is not None else lo
        return loss/V
    
    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        # self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        features_all = None
        index_all = None
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            i_a, g_a, t_a, i_s, g_s, t_s = self.model(data)
            i_a = i_a.permute(1, 0, 2).contiguous()
            i_s = i_s.permute(1, 0, 2).contiguous()
            i_a, i_a_l = getins(i_a)
            g_a, g_a_l = getins(g_a)
            t_a, t_a_l = getins(t_a)
            i_s, i_s_l = getins(i_s)
            g_s, g_s_l = getins(g_s)
            t_s, t_s_l = getins(t_s)
            
            #global z_, w_
            V, _, _ = i_a.size()
            sum_i_a = None
            sum_i_s = None
            for v in range(V):
                sum_i_a = sum_i_a + i_a[v] if sum_i_a is not None else i_a[v]
                sum_i_s = sum_i_s + i_s[v] if sum_i_s is not None else i_s[v]
            # sum_i_a = sum_i_a / V
            # sum_i_s = sum_i_s / V
            features_sum = sum_i_a + g_a + t_a + sum_i_s + g_s + t_s
            output = features_sum
            features_all = torch.cat((features_all, features_sum), dim=0) if features_all is not None else features_sum
            index_all = torch.cat((index_all, index), dim=0) if index_all is not None else index
            
            # output = self.model(data)
            # if batch_idx == 0 and epoch == 0:
            #     self.train_writer.add_graph(self.model, output)
            # if isinstance(output, tuple):
            #     output, l1 = output
            #     l1 = l1.mean()
            # else:
            #     l1 = 0
            
            #increase SimCLR contrastive Loss
            if epoch == 0:
                con_loss_aa_it = self.info_nce_loss_i0(i_a, t_a)
                con_loss_ss_it = self.info_nce_loss_i0(i_s, t_s)
                con_loss_aa_ig = self.info_nce_loss_i0(i_a, g_a)
                con_loss_ss_ig = self.info_nce_loss_i0(i_s, g_s)
                            
                features_tga = torch.cat((t_a, g_a), dim=0)
                con_loss_aa_tg = self.info_nce_loss0(features_tga)
                            
                features_tgs = torch.cat((t_s, g_s), dim=0)
                con_loss_ss_tg = self.info_nce_loss0(features_tgs)
                            
                con_loss_as_ii = self.info_nce_loss_ii0(i_a, i_s)
                            
                features_ttas = torch.cat((t_a, t_s), dim=0)
                con_loss_as_tt = self.info_nce_loss0(features_ttas)
                            
                features_ggas = torch.cat((g_a, g_s), dim=0)
                con_loss_as_gg = self.info_nce_loss0(features_ggas)
            else:
                con_loss_aa_it = self.info_nce_loss_i(i_a, t_a, index)
                con_loss_ss_it = self.info_nce_loss_i(i_s, t_s, index)
                con_loss_aa_ig = self.info_nce_loss_i(i_a, g_a, index)
                con_loss_ss_ig = self.info_nce_loss_i(i_s, g_s, index)
            
                features_tga = torch.cat((t_a, g_a), dim=0)
                con_loss_aa_tg = self.info_nce_loss(features_tga, index)
            
                features_tgs = torch.cat((t_s, g_s), dim=0)
                con_loss_ss_tg = self.info_nce_loss(features_tgs, index)
            
                con_loss_as_ii = self.info_nce_loss_ii(i_a, i_s, index)
            
                features_ttas = torch.cat((t_a, t_s), dim=0)
                con_loss_as_tt = self.info_nce_loss(features_ttas, index)
            
                features_ggas = torch.cat((g_a, g_s), dim=0)
                con_loss_as_gg = self.info_nce_loss(features_ggas, index)
            
            con_loss = (con_loss_aa_it + con_loss_ss_it + con_loss_aa_ig + con_loss_ss_ig + con_loss_aa_tg + con_loss_ss_tg + con_loss_as_ii + con_loss_as_tt + con_loss_as_gg)/9
            
            N = label.size(0)
            n_list = []
            for n in range(N):
                if label[n].data.item() == -1:
                    n_list.append(n)
            #print(label)
            output = del_tensor_ele(output, n_list)
            label = del_tensor_ele(label, n_list)
            #print(label)
            if output.size(0) > 0:
                reg_loss = self.loss(output, label)
                loss = con_loss + reg_loss + i_a_l + g_a_l + t_a_l + i_s_l + g_s_l + t_s_l
            else:
                loss = con_loss + i_a_l + g_a_l + t_a_l + i_s_l + g_s_l + t_s_l
            
            # loss = self.loss(output, label) + l1

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            # self.train_writer.add_scalar('acc', acc, self.global_step)
            # self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            # self.train_writer.add_scalar('loss_l1', l1, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            # self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()
        
        with torch.no_grad():
            # if (epoch % 20) == 0:
            if epoch == 0:
                self.z_, self.z_w = self.anchor_graph(features_all, index_all)

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    # output = self.model(data)
                    i_a, g_a, t_a, i_s, g_s, t_s = self.model(data)
                    i_a = i_a.permute(1, 0, 2).contiguous()
                    i_s = i_s.permute(1, 0, 2).contiguous()
                    i_a, i_a_l = getins(i_a)
                    g_a, g_a_l = getins(g_a)
                    t_a, t_a_l = getins(t_a)
                    i_s, i_s_l = getins(i_s)
                    g_s, g_s_l = getins(g_s)
                    t_s, t_s_l = getins(t_s)
                    
                    V, _, _ = i_a.size()
                    sum_i_a = None
                    sum_i_s = None
                    for v in range(V):
                        sum_i_a = sum_i_a + i_a[v] if sum_i_a is not None else i_a[v]
                        sum_i_s = sum_i_s + i_s[v] if sum_i_s is not None else i_s[v]
                    # i_a = sum_i_a / V
                    # i_s = sum_i_s / V
                    output = sum_i_a + g_a + t_a + g_s + t_s + sum_i_s
                    
                    # if isinstance(output, tuple):
                    #     output, l1 = output
                    #     l1 = l1.mean()
                    # else:
                    #     l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            # self.lr_scheduler.step(loss)
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                pass
                # self.val_writer.add_scalar('loss', loss, self.global_step)
                # self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                # self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    processor = Processor(arg)
    processor.start()
