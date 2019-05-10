import torch

import modelbased.problems.models.nn as nn


class SimpleFullyConnected(nn.BaseNetwork):
    def __init__(self, layers, h, device):
        super(SimpleFullyConnected, self).__init__(device)

        self.fclayers = []

        for i in range(len(layers) - 2):
            submodule = torch.nn.Linear(layers[i], layers[i + 1])

            self.fclayers.append(submodule)
            self.add_module('fc' + str(i), submodule)

        self.linear_layer = torch.nn.Linear(layers[-2], layers[-1])

        self.output_dim = layers[-1]

        self.h = h
        self.layer_size = layers

        self.to(self.device)

    def forward(self, x):

        a = self.h(self.fclayers[0](x))

        for i in range(1, len(self.fclayers)):
            a = self.h(self.fclayers[i](a))

        return self.linear_layer(a)
