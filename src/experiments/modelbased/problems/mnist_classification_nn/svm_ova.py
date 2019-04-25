import modelbased.utils.misc

import modelbased.data.mnist as mnist_data

import modelbased.problems.mnist_classification_nn.nn as mnist_nn

import torch
from torch.nn import functional as F


class SVM_OVA:
    def __init__(self, subset, batchsize):
        transform = lambda x: modelbased.utils.misc.one_hot(x, 10, hot=1, off=-1)

        trainloader, _, _, _, _ = mnist_data.load('datasets/mnist', subset, 10, batchsize, 10,
                                                  one_hot_encoding=transform)

        self.trainloader = trainloader
        self.net = mnist_nn.SimpleConvNetMNIST(F.relu)

    @staticmethod
    def L(v, y_targets):
        """
        Apply the one-versus-all SVM loss.

        :param v: shape = (n, c).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.

        :return: L(v), shape = (n, 1).
        """
        _max = torch.max(torch.zeros(1), 1 - v * y_targets)

        return _max.sum(1, keepdim=True)

    @classmethod
    def h(cls, a, y_targets, n=None):
        """
        :param a: shape = (n, c).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param n: The factor the sum is multiplied with (1/N).

        :return: h(a) scalar.
        """
        if not n:
            n = a.size(0)

        return (1 / n) * cls.L(a, y_targets).sum().item()

    def c(self, u, x, set_param=False):
        """
        :param u: shape (d, 1).
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
        return lam * 0.5 * (u ** 2).sum().item()

    def loss(self, u, x, y_targets, lam):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param lam: Scalar regularizer weight factor.

        :return: L(u)
        """
        return self.h(self.c(u, x), y_targets) + self.reg(u, lam)

    def solve_linearized_subproblem(self, uk, tau, x, y_targets, lam):
        """
        Solve the inner linearized subproblem using gradient ascent on the dual problem.

        :param uk: shape = (d, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.

        :param lam: Scalar regularizer weight factor.

        :return: Solution argmin L_lin(u).
        """
        n = x.size(0)

        #J = self.net.

    def run(self):
        u = self.net.params

        for x, yt in self.trainloader:
            print(self.loss(u, x, yt, 1))

            J = self.net.Jacobian(u, x)
            print(J.size())


def run():
    cls = SVM_OVA(5, 5)

    cls.run()
