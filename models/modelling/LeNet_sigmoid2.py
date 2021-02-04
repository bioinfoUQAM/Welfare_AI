# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:42:43 2020

@author: Yesmin
"""

from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, (3, 3), padding=2)
        self.conv1_bn = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=2)
        self.conv2_bn = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(32 * 49 * 2, 120)
        self.dense1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.dense2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x))), (2, 2))

        # x = F.interpolate(x, size=(48,1), mode='bilinear')
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.dense1_bn(self.fc1(x)))

        x = F.relu(self.dense2_bn(self.fc2(x)))
        # x = self.fc3(x)
        x = self.sigmoid(self.fc3(x))  # if BCELoss is used
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
