#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 14 16:23:39 2023
@author: Yongjian Tang, Yun Di
"""

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch import mul
from torch.utils.data import DataLoader


class Neural_network(nn.Module):
    """the class of the simple neural network architecture, using pytorch"""

    def __init__(self, width):
        """define the basic structure of neural network

        width : int
                the parameter to define the width of each layer in neural network
        """
        super().__init__()
        input_size = 2
        output_size = 2
        self.layer1 = nn.Linear(input_size, width)
        self.layer2 = nn.Linear(width, 2 * width)
        self.layer3 = nn.Linear(2 * width, 3 * width)
        self.layer4 = nn.Linear(3 * width, 2 * width)
        self.layer5 = nn.Linear(2 * width, 2 * width)
        self.layer6 = nn.Linear(2 * width, width)
        self.layer7 = nn.Linear(width, output_size)
        self.bn1 = nn.BatchNorm1d(width)
        self.bn2 = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.weights_init()

    def weights_init(self):
        """xavier weight initialization to ensure a more stable output of neural network"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """forward function of the neural network

        Parameters
        ----------
        x : float
            the coordinate of the pedestrian before one iteration of neural network

        Returns
        -------
        x : float
            the coordinate of the pedestrian after one iteration of neural network
        """
        x = func.leaky_relu(self.layer1(x))
        x = func.leaky_relu(self.layer2(x))
        x = func.leaky_relu(self.layer3(x))
        x = func.leaky_relu(self.layer4(x))
        x = func.leaky_relu(self.layer5(x))
        x = func.leaky_relu(self.layer6(x))
        x = self.layer7(x)
        return x
