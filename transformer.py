#!/usr/bin/env python3

import numpy as np
import torch
import torchvision as thv
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import random

class FilterTokenizer(nn.Module):
    def __init__(
        self,
        input_shape=(32,32),
        in_channels=512,
        out_channels=1024,
        num_tokens=16,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))

    def forward(x):
        assert len(x.shape) < 4, "please provide sample in batches"
        out = nn.Softmax(self.conv1(x))
        for i in range(x.shape[0]):
            out[i] = out[i]*x[i].permute(1,2,0)
        return out

class BasicBlock(nn.Module):
    def __init__(
        self,
        input_shape=(32,32),
        kernel_shape=(7,7),
        in_channels=16,
        out_channels=16,
        stride=2,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_shape, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_shape, stride)


    def forward(x):
        assert len(x.shape) < 4, "please provide sample in batches"
        out = nn.ReLU()(self.conv1(x))
        out = x + nn.ReLU()(self.conv2(out))
        return out


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        input_shape=(32,32),
        in_channels=256,
        out_channels=256,
        stride=2,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, (1,1), stride)
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride)
        self.conv3 = nn.Conv2d(64, out_channels, (1,1), stride)


    def forward(x):
        assert len(x.shape) < 4, "please provide sample in batches"
        out = nn.ReLU()(self.conv1(x))
        out = nn.ReLU()(self.conv2(x))
        out = x + nn.ReLU()(self.conv3(out))
        return out


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_nodes=1024,
        out_nodes=1000
    ):
        super(SegmentationHead, self).__init__()
        self.fc = nn.Linear(in_nodes,out_nodes)

    def forward(x):
        assert len(x.shape) < 3, "please provide sample in batches (NxLxC)"
#         L = int(x.shape[1])
#         C = int(x.shape[2])
        out = torch.mean(x, dim=1)
        out = nn.ReLU()(self.fc(x))
        return out

class Transformer(nn.Module):
    def __init__(
        self
    ):
        super(Transformer, self).__init__()

    def forward():
