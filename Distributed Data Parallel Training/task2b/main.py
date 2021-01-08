import os
import time
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import torch.distributed as td
import argparse
from utilities import itr_meta


device = "cpu"
torch.set_num_threads(5) 


              
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', required=True, help='ip address of the rank 0 node')
    parser.add_argument('--num-nodes', required=True, help='number of nodes in the group')
    parser.add_argument('--rank', required=True, help='the process identifier for this instance')
    parser.add_argument('--port', required=False, help='port to use for this task', default='34785')
    return parser.parse_args()


def train_model(model, train_loader, optimizer, criterion, epoch, world_size):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    
    meta_data = itr_meta()
    for batch_idx, (data, target) in enumerate(train_loader):
        #pull in data and run through the model
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        

        #determine loss and update the convolution weights
        meta_data.start_phase('forward')
        output = model(data)
        loss = criterion(output, target)
        meta_data.end_phase('forward')

        meta_data.start_phase('backward')
        loss.backward()
        
        meta_data.start_phase('communication')
        # sync gradients across all nodes
        for parameter in model.parameters():
            td.all_reduce(parameter.grad, op=td.ReduceOp.SUM)
            parameter.grad = torch.div(parameter.grad,world_size)

        meta_data.end_phase('communication')
        meta_data.end_phase('backward')

        meta_data.start_phase('optimize')
        optimizer.step()
        meta_data.end_phase('optimize')
    
        #collecting timing info and output iterative progress
        if batch_idx % 20 == 0:
            meta_data.start_iter()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * world_size * len(data), len(train_loader.dataset),
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
            

def main():

    #pull in parameters and set variables needed for initialization
    args = parse_args()
    print('main.py launched with master ip: ' + str(args.master_ip) +' num nodes: ' + str(args.num_nodes) + ' rank: ' + str(args.rank) + ' port: ' + str(args.port))
    
    world_size = int(args.num_nodes)
    os.environ["GLOO_SOCKET_IFNAME"]='eth1'
    
    batch_size = 256/world_size #batch size is 256 across all nodes, so need to divide by number of nodes
    
    random_seed = 314
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    #initialize distributed group
    td.init_process_group('gloo', init_method='tcp://' + args.master_ip + ':' + args.port, 
        world_size=world_size, rank=int(args.rank))
   


    #start model setup
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
    #
    # added for distributed process
    dist_train_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
    #
    #add the sampler to the test_loader
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=dist_train_sampler,
                                                    #shuffle=True,
                                                    pin_memory=True
                                                    )
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True
                                              )
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    
    #save off model to check if model is initialized the same across nodes
    #torch.save(model.state_dict(), './node'+args.rank+'_'+'initial_model_weights')
    
    # running training for one epoch    
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch, world_size)
        test_model(model, test_loader, training_criterion)
    
    #the final model should be the same across all nodes as well
    #torch.save(model.state_dict(), './node'+args.rank+'_'+'final_model_weights')

if __name__ == "__main__":
    main()
