#!/usr/bin/env python3

from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.transformer import *


def main():
    trainset = CIFAR10(root='./data', train=True,
               download=True, transform=T.Compose([ T.ToTensor(), T.Resize((64,64))]))
    trainloader = DataLoader(trainset, batch_size=1) ## use batch_size=1 for debugging

    model = nn.Sequential(FeatExtractor(3),
                          ViTransformer(num_channels=256,task="classification"),
                          ClassificationHead())
    model.eval()
    for data in trainloader:
        x = data[0]
        y = data[1]
        out = model(x)
        break

main()