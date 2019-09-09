import torch
from torch.nn import functional as F
import torch.random
import logging

import modelbased.problems.spirals_classification.nn as spirals_nn


class LogisticRegression:
    def __init__(self, m=2):
        """
        :param m: Sample dimension.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using device: {}.".format(device))

        layers = [m, 12, 12, 12, 2]
        self.net = spirals_nn.SimpleFullyConnected(layers, F.relu, device)

    @staticmethod
    def L_binary_cross_entropy_per_class(v, y_targets):
        """
        Logistic loss applied per sample and per output class.
        This is the same as pytorchs BCEWithLogitsLoss/binary_cross_entropy_with_logits.

        :param v: shape = (n, 2).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: L(v), shape = (n, 1).
        """
        r = torch.log(1 + torch.exp(-v))
        return 0.5 * (y_targets * r + (1 - y_targets) * (v + r)).sum(1)

    @staticmethod
    def L_binary_cross_entropy(v, y_targets):
        """
        Logistic loss applied per sample.

        :param v: shape = (n, 1).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: L(v), shape = (n, 1).
        """
        # TODO
        # torch.log(1 + torch.exp(-v))[y_targets == 1].unsqueeze(1)
        raise NotImplementedError()

    @classmethod
    def h(cls, a, y_targets):
        """
        :param a: shape = (n, 2).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: h(a) torch tensor.
        """
        n = a.size(0)

        return (1 / n) * cls.L_binary_cross_entropy_per_class(a, y_targets).sum()

    def c(self, u, x):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, m).

        :return: c(u) torch tensor, shape = (n, 2).
        """
        return self.net.f(u, x)

    def loss(self, u, x, y_targets):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, m).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: L(u) torch tensor.
        """
        return self.h(self.c(u, x), y_targets)

    def grad(self, u, x, yt):
        with self.net.param_eval(u):
            self.net.zero_grad()
            loss = self.h(self.net.forward(x), yt)
            loss.backward()

        return self.net.gradient

    @staticmethod
    def description():
        return {
            'loss function': 'binary cross-entropy (per class).',
            'prediction function': 'shallow fully connected net.',
        }

    def run(self, loader, **kwargs):
        raise NotImplementedError()
