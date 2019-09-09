import logging

import torch
from torch.nn import functional as F

import modelbased.problems.mnist_classification.misc as misc
import modelbased.problems.mnist_classification.nn as mnist_nn

logger = logging.getLogger(__name__)


class SVM_OVA:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: {}.".format(device))

        self.net = mnist_nn.SimpleConvNetMNIST(F.relu, device)

    @staticmethod
    def L(v, y_targets):
        """
        Apply the one-versus-all SVM loss.

        :param v: shape = (n, c).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.

        :return: L(v), shape = (n, 1).
        """
        _max = torch.max(torch.zeros(1, device=v.device), 1 - v * y_targets)

        return _max.sum(1, keepdim=True)

    @classmethod
    def h(cls, a, y_targets, n=None, torch_tensor=False):
        """
        :param a: shape = (n, c).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param n: The factor the sum is multiplied with (1/N).
        :param torch_tensor: If true, returns a torch tensor, else a scalar.

        :return: h(a) torch tensor or scalar.
        """
        if not n:
            n = a.size(0)

        ret = (1 / n) * cls.L(a, y_targets).sum()

        if torch_tensor:
            return ret
        else:
            return ret.item()

    def c(self, u, x, set_param=False):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, channels, ysize, xsize).
        :param set_param: If True, set the network parameters, if False just evaluate and keep the old parameters.

        :return: c(u), shape = (n, c).
        """
        return self.net.f(u, x, set_param)

    @staticmethod
    def reg(u, lam):
        """
        Squared l2-norm regularizer.

        :param u: shape = (d, 1).
        :param lam: Scalar regularizer weight factor.

        :return: lam * 0.5 * norm(u)**2 as scalar.
        """
        return lam * 0.5 * misc.sql2norm(u)

    def loss(self, u, x, y_targets, lam):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param lam: Scalar regularizer weight factor.

        :return: L(u).
        """
        return self.h(self.c(u, x), y_targets) + self.reg(u, lam)

    def grad(self, u, x, y_targets, lam):
        with self.net.param_eval(u):
            self.net.zero_grad()
            loss = self.h(self.net.forward(x), y_targets, torch_tensor=True) + self.reg(u, lam)
            loss.backward()

        return self.net.gradient

    def _predict(self, u, x):
        x = x.to(self.net.device)

        y = self.net.f(u, x)

        return y.argmax(1)

    def predict(self, x):
        return self._predict(self.net.params, x)

    @staticmethod
    def description():
        return {
            'loss function': 'SVM one-versus-all.',
            'prediction function': 'shallow convolutional network,',
        }
