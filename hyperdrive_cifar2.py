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
from cnn2layer import CNN

import numpy as np
from skopt.callbacks import DeadlineStopper
from skopt import gp_minimize
from skopt import dump
from space_division import HyperSpace
from mpi4py import MPI


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def objective(space):
    kernel_size1, stride1, dropout1, kernel_size2, stride2, dropout2, learning_rate = space

    # Hyper Parameters
    num_epochs = 10
    kernel_size1 = int(kernel_size1)
    stride1 = int(kernel_size1)
    dropout1 = float(dropout1)
    kernel_size2 = int(kernel_size2)
    stride2 = int(stride2)
    dropout2 = float(dropout2)
    learning_rate = float(learning_rate)

    cnn = CNN()
    cnn.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, 60000//128, loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in testloader:
        images = Variable(images).cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    test_accuracy = 100 * correct / total
    return loss.data[0]


def main():
    if rank == 0:
        hyperparameters = {'kernelSize1': np.arange(2,10),
                           'stride1': np.arange(1, 5),
                           'dropout1': np.linspace(0.0, 0.8),
                           'kernelSize2': np.arange(2,10),
                           'stride2': np.arange(1, 5),
                           'dropout2': np.linspace(0.0, 0.8),
                           'learningRate': np.linspace(0.001, 0.1)}

        hyperspace = HyperSpace(hyperparameters)
        all_intervals = hyperspace.fold_space()
        hyperspaces = hyperspace.hyper_permute(all_intervals)
        subspace_keys, subspace_boundaries = hyperspace.format_hyperspace(hyperspaces)
    else:
        subspace_keys, subspace_boundaries = None, None

    space = comm.scatter(subspace_boundaries, root=0)

    deadline = DeadlineStopper(18000)
    # Gaussian process minimization (see scikit-optimize skopt module for other optimizers)
    res_gp = gp_minimize(objective, space, n_calls=50, callback=deadline, random_state=0, verbose=True)
    # Each worker will write their results to disk
    dump(res_gp, '/lustre/atlas/proj-shared/csc237/ygx/safari_zone/vision/pytorch/cifar2/mobilenet/hyper_results/gp_subspace_' + str(rank))


if __name__=='__main__':
    main()
