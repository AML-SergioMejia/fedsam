import copy
import numpy as np
import os
import re
import torch
import torch.nn as nn
from PIL import Image
from typing import Callable

import torch.nn.functional as functional
import torch.nn.init as init

def conv3_stride1(input, output):
    '''
    ## PARAMS
    - input = Number of input channels
    - output = Number of output channels
    ## RETURNS
    - conv3 = returned layer
    '''
    conv3 = nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, bias=False)
    return(conv3)

def conv3_stride2(input, output):
    '''
    ## PARAMS
    - input = Number of input channels
    - output = Number of output channels
    ## RETURNS
    - conv3 = returned layer
    '''
    conv3 = nn.Conv2d(input, output, kernel_size=3, stride=2, padding=1, bias=False)
    return(conv3)

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')

class BasicBlock(nn.Module):
    '''
    - Class that represents a basic block on the ResNet architecture, 
      composed of 2 convolutional layers with a Normalization and Activation layer between those
    '''
    def __init__(self, in_channels, in_size, residual=True, norm_type="BATCH") -> None:
        '''
        ## PARAMS
        - in_channels: Number of channels for the block
        - in_size: Size (width or height) of one square sample
        - residual: Whether to add a residual connection or not
        - norm_type: Which type of normalization to use - "BATCH" or "GROUP"
        '''
        super().__init__()
        self.in_channels = in_channels
        self.in_size = in_size
        self.residual = residual

        self.model = nn.Sequential(
            conv3_stride1(in_channels, in_channels),
            nn.BatchNorm2d(in_channels) if norm_type == "BATCH" else nn.GroupNorm(2, in_channels),
            nn.ReLU(),
            conv3_stride1(in_channels, in_channels),
            nn.BatchNorm2d(in_channels) if norm_type == "BATCH" else nn.GroupNorm(2, in_channels)
        )

    def forward(self, x):
        y = self.model(x)
        if self.residual:
            y += x
        return functional.relu(y)


class DownsampleBlock(nn.Module):
    '''
    - Class that represents a basic block responsible for downsampling on the ResNet architecture, 
      composed of 2 convolutional layers (one with stride 2 and other with stride 1) with a Normalization and Activation layer between those
    '''
    def __init__(self, in_channels, in_size, residual=True, option='A', norm_type="BATCH") -> None:
        '''
        ## PARAMS
        - in_channels: Number of channels for the block
        - in_size: Size (width or height) of one square sample
        - residual: Whether to add a residual connection or not
        - option: Way to create the downsample residual connection (A or B)
        - norm_type: Which type of normalization to use - "BATCH" or "GROUP"
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 2

        self.in_size = in_size
        self.residual = residual
        self.option = option

        self.model = nn.Sequential(
            conv3_stride2(in_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels) if norm_type == "BATCH" else nn.GroupNorm(2, self.out_channels),
            nn.ReLU(),
            conv3_stride1(self.out_channels, self.out_channels),
            nn.BatchNorm2d(self.out_channels) if norm_type == "BATCH" else nn.GroupNorm(2, self.out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(self.out_channels) if norm_type == "BATCH" else nn.GroupNorm(2, self.out_channels)
        )

    def forward(self, x):
        '''
        - Performs the forwarding through the net considering the residual connection performed between 2 different size layers (the next layer being 1/2 the size of the previous)
        '''
        y = self.model(x)
        if self.residual:
            if self.option == 'A':
                shortcut = functional.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.out_channels//4, self.out_channels//4), "constant", 0)
                y += shortcut
            else:
                shortcut = self.shortcut(x)
                y += shortcut  
        return functional.relu(y)

class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device, num_blocks=3, option='B', norm_type="GROUP"):
        '''
        ResNet20 implementation
        ### PARAMS
        - num_blocks: Number of basic blocks composing the network
        - num_classes: Number of outputs of the network
        - option: Type of shortcut on downsample blocks (A: Padding, B: Conv layer)
        - norm_type: Which type of normalization to use - "BATCH" or "GROUP"
        '''
        super().__init__()  
        self.num_blocks = num_blocks
        self.norm_type = norm_type
        self.lr = lr
        self.device = device
        
        self.conv1 = conv3_stride1(3, 16)
        self.norm1 = nn.BatchNorm2d(16) if norm_type == "BATCH" else nn.GroupNorm(2, 16)
        
        # Layer 1 (32x32x16) -> Layer 2 (16x16x32) -> Layer 3 (8x8x64) -> AvgPool (8x8) -> FC (64 x num_classes)
        self.layer1 = nn.Sequential(*[BasicBlock(16, 32, norm_type=norm_type) for _ in range(self.num_blocks)])
        self.downsample1 = DownsampleBlock(16, 32, option=option, norm_type=norm_type)
        self.layer2 = nn.Sequential(*[BasicBlock(32, 16, norm_type=norm_type) for _ in range(self.num_blocks-1)])
        self.downsample2 = DownsampleBlock(32, 16, option=option, norm_type=norm_type)
        self.layer3 = nn.Sequential(*[BasicBlock(64, 8, norm_type=norm_type)  for _ in range(self.num_blocks-1)])
        self.avgpool = nn.AvgPool2d(kernel_size=(8,8), stride=(1,1))
        self.fc = nn.Linear(64, num_classes) # Fully connected layer
        self.softmax = nn.Softmax()

        self.apply(_weights_init)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = functional.relu(x)
        
        # =========================== #

        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        x = self.layer3(x)
        
        # =========================== #

        x = self.avgpool(x)           # Average pooling 8x8
        x = x.reshape(x.shape[0], -1) # Flatten to 64
        x = self.fc(x)                # Fully connected 100
        #x = self.softmax(x)

        return x
    
    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size

#     cosineLR  | cosineWR  
# BN   0.6499   |    0
# GN     0      |    0

'''
class ClientModel(nn.Module):
    def __init__(self, lr, num_classes, device):
        super(ClientModel, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.lr = lr

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*5*5, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, self.num_classes)
        )

        self.size = self.model_size()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x

    def model_size(self):
        tot_size = 0
        for param in self.parameters():
            tot_size += param.size()[0]
        return tot_size
'''