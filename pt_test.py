import os, sys, time, math
import csv

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

import numpy as np

import pt_learn as core

def main():
    transform = transforms.ToTensor()
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    pth_path = "./fc_cpu.pth"
    #
    net = core.Net()
    net.load_state_dict(torch.load(pth_path))
    net.eval()    
    print(summary(net, (1, 28, 28)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #print(predicted, labels)
            correct += (predicted == labels).sum().item()
        #
    #
    print("accuracy: %f %%" % (100.0 * correct / total))

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#

