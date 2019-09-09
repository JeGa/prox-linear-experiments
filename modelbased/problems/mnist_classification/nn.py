import torch
import torch.nn.functional as F

import modelbased.problems.models.nn as nn


class SimpleConvNetMNIST(nn.BaseNetwork):
    def __init__(self, h, device):
        super(SimpleConvNetMNIST, self).__init__(device)

        self.output_dim = 10

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

        self.h = h

        self.to(self.device)

    def forward(self, x):
        x = self.h(F.max_pool2d(self.conv1(x), 2))
        x = self.h(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)

        x = self.h(self.fc1(x))
        x = self.fc2(x)

        return x
