#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn


def check_x(x):
    assert len(x.shape) == 4, "please provide sample in batches"


class ViTransformer(nn.Module):
    def __init__(
        self,
        num_tokens=16,
        num_channels=1024,
        task="classification",
    ):
        super(ViTransformer, self).__init__()
        self.Tokenizer = FilterTokenizer(num_channels, num_tokens)
        self.Transformer = Transformer(num_channels, num_tokens)

        if task=="classification":
            self.Projector = nn.Identity()
        elif task=="segmentation":
            self.Projector = Projector(num_channels)
        else:
            exit("`task` must be 'classification' or 'segmentation'")

    def forward(self, x):
        tokens_in = self.Tokenizer(x)
        tokens_out = self.Transformer(tokens_in)
        print(tokens_out.shape)
        out = self.Projector(tokens_out)
        return out


class FeatExtractor(nn.Module):
    def __init__(
        self,
        in_channels=3,
        res_blocks=None,
        kernel_shape=(7, 7)
    ):
        super(FeatExtractor, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_shape, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        if not res_blocks:
            # bb_11 = BasicBlock(in_channels=64, out_channels=64)
            # bb_12 = BasicBlock(in_channels=64, out_channels=64)
            # bb_21 = BasicBlock(in_channels=64, out_channels=128)
            # bb_22 = BasicBlock(in_channels=128, out_channels=128)
            # bb_31 = BasicBlock(in_channels=128, out_channels=256)
            # bb_32 = BasicBlock(in_channels=256, out_channels=256)
            # self.res_blocks = nn.Sequential(bb_11, bb_12, bb_21, bb_22, bb_31, bb_32)
            bb = BasicBlock(in_channels=64, out_channels=256)
            self.res_blocks = nn.Sequential(bb)
        else:
            assert len(res_blocks) != 3, "Transformer uses 3 basic/bottleneck blocks"
            self.res_blocks = res_blocks

    def forward(self, x):
        out = self.maxpool(self.conv(x))
        for md in self.res_blocks:
            out = md(out)
        return out


class FilterTokenizer(nn.Module):
    def __init__(
        self,
        in_channels=512,
        num_tokens=16,
    ):
        super(FilterTokenizer, self).__init__()
        self.linear = nn.Linear(in_channels, num_tokens, bias=False)

    def forward(self, x):
        check_x(x)
        _x = x.permute(0,2,3,1)
        _out = self.linear(_x)
        _out = nn.Softmax(dim=2)(torch.flatten(_out, start_dim=1, end_dim=2)).permute(0,2,1)
        out = torch.zeros(x.shape[0],_out.shape[1], x.shape[1])
        for i in range(x.shape[0]):
            _x_i = torch.permute(x[i],(1, 2, 0))
            _x_i = torch.flatten(_x_i, start_dim=0, end_dim=1)
            out[i] = torch.matmul(_out[i],_x_i) # out[i].shape==(L, H, W) x[i].shape=(C, H, W)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels=1024,
        num_tokens=16,
    ):
        super(Transformer, self).__init__()
        self.key = nn.Linear(in_channels, num_tokens, bias=False)
        self.query = nn.Linear(in_channels, num_tokens, bias=False)
        self.weight1 = nn.Linear(in_channels, in_channels, bias=False)
        self.weight2 = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        key = self.key(x)
        query = self.query(x).transpose(1,2)
        self_attention = nn.Softmax(dim=1)(torch.matmul(key, query))
        _out = x + torch.matmul(self_attention, x)
        out = _out + self.weight2(nn.ReLU()(self.weight1(_out)))
        return out


class Projector(nn.Module):
    def __init__(
        self,
        in_channels=1024,
    ):
        super(Projector, self).__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)

    def forward(self, x, out_tokens):
        key = self.key(out_tokens).transpose(1,2)
        query = self.query(x.permute(0,2,3,1))
        projection = nn.Softmax(torch.matmul(query, key))
        out = x + torch.matmul(projection, out_tokens)
        print(out.shape)
        return out


class BasicBlock(nn.Module):
    def __init__(
        self,
        kernel_shape=(3,3),
        in_channels=16,
        out_channels=16,
        stride=1,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_shape, stride, padding=int(kernel_shape[0]/2))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_shape, stride, padding=int(kernel_shape[0]/2))
        self.res_conv = nn.Identity()
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, (1,1))

    def forward(self, x):
        check_x(x)
        out = nn.ReLU()(self.conv1(x))
        out = nn.ReLU()(self.conv2(out))
        out = self.res_conv(x) + out
        return out


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels=256,
        out_channels=256,
        stride=2,
    ):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, (1,1), stride)
        self.conv2 = nn.Conv2d(64, 64, (3,3), stride, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, (1,1), stride)
        self.res_conv = nn.Identity()
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x):
        check_x(x)
        out = nn.ReLU()(self.conv1(x))
        out = nn.ReLU()(self.conv2(out))
        out = nn.ReLU()(self.conv3(out))
        out = self.res_conv(x) + out
        return out


class ClassificationHead(nn.Module):
    def __init__(
        self,
        num_tokens=16,
        out_nodes=1000,
    ):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(num_tokens,out_nodes)

    def forward(self, x):
        print(x.shape)
        out = torch.mean(x, dim=2)
        out = nn.ReLU()(self.fc(out))
        return out


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_nodes=1024,
        out_nodes=1000
    ):
        super(SegmentationHead, self).__init__()
        self.fc = nn.Linear(in_nodes,out_nodes)

    def forward(self, x):
        out = torch.mean(x, dim=1)
        out = nn.ReLU()(self.fc(out))
        return out
