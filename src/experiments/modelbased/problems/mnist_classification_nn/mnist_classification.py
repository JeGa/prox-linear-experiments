import modelbased.data.mnist as mnist_data
import modelbased.problems.mnist_classification_nn.nn as mnist_nn
import modelbased.utils

import modelbased.solver.torch_gradient_descent
import modelbased.solver.gradient_descent

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt


class ClassificationMNIST:
    def __init__(self):
        trainloader, _, _, _, _ = mnist_data.load('../datasets/mnist', 15, 10, one_hot_encoding=True)

        self.trainloader = trainloader
        self.net = mnist_nn.SimpleConvNetMNIST(F.relu)

    def f(self, u, x, set=False):
        if set:
            self.net.params = u
            return self.net(x)
        else:
            with self.net.no_param_set(u):
                return self.net(x)

    @staticmethod
    def loss(y, yt):
        """
        :param y: shape = (N, c).
        :param yt: shape = (N, c).

        :return: l1 norm loss.
        """
        return (torch.norm(y - yt, p=1, dim=1)).sum()

    def gradient(self, u, x, yt):
        self.net.zero_grad()

        loss = self.loss(self.f(u, x, set=True), yt)

        loss.backward()

        return self.net.gradients

    @staticmethod
    def Jf(net, y, retain_graph=False):
        """
        :param y: shape = (N, c).
        :param retain_graph: If False, does not retain graph after call.

        :return: Jacobian with shape (N*c, size(params)).
        """
        net.zero_grad()

        ysize = y.numel()
        J = torch.empty(ysize, net.numparams())

        for i, yi in enumerate(y):  # Samples.
            for j, c in enumerate(yi):  # Classes.
                # Free the graph at the last backward call.
                if i == y.size(0) - 1 and j == yi.size(0) - 1:
                    c.backward(retain_graph=retain_graph)
                else:
                    c.backward(retain_graph=True)

                start_index = 0
                for p in net.parameters():
                    J[i * y.size()[1] + j, start_index:start_index + p.numel()] = torch.reshape(p.grad, (-1,))

                    start_index += p.numel()

                net.zero_grad()

        return J

    def solve_linearized_subproblem(self):
        pass

    def run_fullbatch(self):
        params = modelbased.utils.Params(
            max_iter=1,
            mu_min=10,
            tau=2,
            sigma=0.8,
            eps=1e-3)

        params = modelbased.utils.Params(
            max_iter=500, sigma=0.001)

        #params = modelbased.utils.Params(
        #    max_iter=500, beta=0.5, gamma=1e-3)

        x, yt = iter(self.trainloader).next()

        def f(u):
            y = self.f(torch.from_numpy(u), x)
            return self.loss(y, yt).item()

        def G(u):
            return self.gradient(torch.from_numpy(u), x, yt).numpy()

        x0 = self.net.params.detach().numpy()

        u, losses = modelbased.solver.gradient_descent.fixed_stepsize(x0, f, G, params)

        self.net.params = torch.from_numpy(u)

        # J = self.Jf(self.net, y, retain_graph=True)

        self.predict_class()

        plot_losses(losses)

    def predict_class(self):
        for x, yt in self.trainloader:
            y = self.net(x).detach().numpy()

            y_class = np.argmax(y, 1)
            yt_class = np.argmax(yt.numpy(), 1)

            print(y_class, yt_class)


def plot_losses(losses):
    plt.figure()
    plt.plot(range(len(losses)), losses, linewidth=0.4)

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    plt.show()


def run():
    cls = ClassificationMNIST()

    cls.run_fullbatch()
