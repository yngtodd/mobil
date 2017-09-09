'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#from train_test import train, test
from data_utils import trainloader, testloader

#from utils import progress_bar
from torch.autograd import Variable
from mobilenet import Block, MobileNet 

import numpy as np
from skopt import dump
from skopt import gp_minimize
from space_division import HyperSpace
from mpi4py import MPI


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# Training
def train(epoch, net, optimizer, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


def objective(space):
    block_kernel1, block_stride1, block_kernel2, block_stride2, kernel_size1, stride1, learning_rate = space

    block_kernel1 = int(block_kernel1)
    block_stride1 = int(block_stride1)
    block_kernel2 = int(block_kernel2)
    block_stride2 = int(block_stride2)
    kernel_size1 = int(kernel_size1)
    stride1 = int(stride1)
    learning_rate = float(learning_rate)

    block = Block(in_planes=64, out_planes=64, block_kernel1=block_kernel1, block_stride1=block_stride1, 
                  block_kernel2=block_kernel2, block_stride2=block_stride2)
    net = MobileNet(block, kernel_size1=kernel_size1, stride1=stride1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    total_loss = 0.0
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(epoch, net, optimizer, criterion)
        test_loss = test(epoch, net)
        total_loss += test_loss

    return total_loss / args.num_epochs


def main():
    if rank == 0:
        hyperparameters = {'blockKernel1': np.arange(2,10),
                           'blockStride1': np.arange(1, 10),
                           'blockKernerl2': np.arange(1, 10),
                           'blockStride2': np.arange(1, 10),
                           'kernerlSize1': np.arange(1, 10),
                           'stride1': np.arange(1, 10),
                           'learningRate': np.linspace(0.001, 0.1)}

        hyperspace = HyperSpace(hyperparameters)
        all_intervals = hyperspace.fold_space()
        hyperspaces = hyperspace.hyper_permute(all_intervals)
        subspace_keys, subspace_boundaries = hyperspace.format_hyperspace(hyperspaces)
    else:
        subspace_keys, subspace_boundaries = None, None

    space = comm.scatter(subspace_boundaries, root=0)

    # Gaussian process minimization (see scikit-optimize skopt module for other optimizers)
    res_gp = gp_minimize(objective, space, n_calls=20, random_state=0, verbose=True)
    # Each worker will write their results to disk
    dump(res_gp, 'hyper_results/gp_subspace_' + str(rank))


if __name__=='__main__':
    main()
