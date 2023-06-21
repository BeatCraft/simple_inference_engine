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

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc_1 = nn.Linear(784, 512, bias=False)
        self.fc_2 = nn.Linear(512, 256, bias=False)
        self.fc_3 = nn.Linear(256, 10, bias=False)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return F.log_softmax(x, dim=1)

def main():
    device = torch.device("cpu")
    
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    
    net = Net()

    pth_path = "./fc_cpu.pth"
    #
    if os.path.exists(pth_path):
        print("resume")
        net.load_state_dict(torch.load(pth_path), strict=False)
    else:
        print("new")
    #
    print(net)
    print(summary(net, (1, 28, 28)))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    loop = 10
    for epoch in range(loop):
        sum_loss = 0.0
        k = 0
        for i, data in enumerate(trainloader, 0):
            sum_loss = 0.0
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            k = k + 1
        #
        avg = sum_loss / float(k)
        print(epoch, avg)
    #
    torch.save(net.state_dict(), pth_path)

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#

