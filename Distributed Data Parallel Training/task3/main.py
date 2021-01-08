import argparse
import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import time
from utilities import itr_meta


device = "cpu"
torch.set_num_threads(5)
Epoch = 1

parser = argparse.ArgumentParser(description='Part 3')
parser.add_argument('--master-ip', required=True, help='ip address of the rank 0 node')
parser.add_argument('--num-nodes', required=True, type=int, help='number of nodes in the group')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--resume', default='', type=str, metavar='PATH', required=False,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--port', required=False, help='port to use for this task', default='34785')

def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    loss = 0
    meta_data = itr_meta()
    # remember to exit the train loop at end of the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        meta_data.start_phase('forward')
        output = model(data)
        loss = criterion(output, target)
        meta_data.end_phase('forward')
        
        meta_data.start_phase('backward')
        loss.backward()
        meta_data.end_phase('backward')
        
        meta_data.start_phase('optimize')
        optimizer.step()
        meta_data.end_phase('optimize')

        if batch_idx % 20 == 0:
            meta_data.start_iter()
            print('Training Progress: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    meta_data.print_time_stats()
    return None

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def setup(args):
    # process groups setup
    os.environ['MASTER_ADDR'] = args.master_ip
    os.environ['MASTER_PORT'] = "25610"
    os.environ["GLOO_SOCKET_IFNAME"]='eth1'
    print "Master-IP: "+ args.master_ip
    print "num_nodes: "+ str(args.num_nodes)
    print "rank: "+ str(args.rank)
    # Initializes the default distributed process group
    dist.init_process_group("gloo", init_method = 'tcp://'+ args.master_ip +':25610', rank=args.rank, world_size=int(args.num_nodes))


def cleanup():
    # clean process groups
    dist.destroy_process_group()

def main():
    args = parser.parse_args()
    # Sets a certain seed for generating random numbers
    seed = 314
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    batch_size = 256/args.num_nodes # batch for one node
    
    # distributed model setup
    setup(args)
    model = mdl.VGG11()
    print("Setup Finished")
    
    # Distributed Data Parallel Training using Built in Module with cpu
    # DDP uses collective communications in the torch.distributed package to synchronize 
    # gradients and buffers transparently. That is, Gradient synchronization communications 
    # take place during the backward pass and overlap with the backward computation 
    ddp_model = DDP(model)
    ddp_model.to(device)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    # data loading and preprocess
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    print("Data Loaded")
    
    #Sampler that restricts data loading to a subset of the dataset.
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)

    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=args.num_nodes,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=(train_sampler is None),
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=args.num_nodes,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
                                              
    #  ensure shuffling work properly across multiple epochs
    train_sampler.set_epoch(Epoch)
    
    print("Training Started")
    start = time.time()
    for epoch in range(Epoch):
        e_s = time.time()
        train_model(ddp_model, train_loader, optimizer, training_criterion, epoch)
        e_d = time.time()
        print("Duration of Epoch "+str(epoch)+" :", e_d-start)
        test_model(ddp_model, test_loader, training_criterion)
    print("Total Duration"+str(epoch)+" :", e_d-start)
    cleanup()

if __name__ == "__main__":
    main()
