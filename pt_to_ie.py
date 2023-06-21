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
    #transform = transforms.ToTensor()
    #testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    
    pth_path = "./fc_cpu.pth"
    net = core.Net()
    net.load_state_dict(torch.load(pth_path))
    print(summary(net, (1, 28, 28)))

    csv_path = "./wi-fc.csv"
    with open(csv_path, "w") as f:
        writer = csv.writer(f, lineterminator='\n')
            
        for name, param in net.named_parameters():
            print(name, param.shape)
            w = param.detach().numpy()
            data = w.tolist()
            if data:
                writer.writerows(data)
            else:
                print("error")
            #
        #
    #

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#

