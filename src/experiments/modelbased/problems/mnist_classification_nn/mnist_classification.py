import modelbased.data.mnist as mnist_data
import modelbased.problems.mnist_classification_nn.nn as mnist_nn
import modelbased.utils

import modelbased.solver.torch_gradient_descent
import modelbased.solver.gradient_descent
import modelbased.solver.prox_descent
import modelbased.solver.utils
import modelbased.solver.projected_gradient

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging


class ClassificationMNIST:
    def __init__(self):
        trainloader, _, _, _, _ = mnist_data.load('datasets/mnist', 10, 10, one_hot_encoding=True)

        self.trainloader = trainloader
        self.net = mnist_nn.SimpleConvNetMNIST(F.relu)

    def f(self, u, x, set_param=False):
        if set_param:
            self.net.params = u
            return self.net(x)
        else:
            with self.net.param_eval(u):
                return self.net(x)

    def h(self, y, tensor=False):
        """
        :param y: shape = (N*c, 1).
        :param tensor: If true, returns tensor, if False returns scalar.

        :return: h(y).
        """
        if tensor:
            return y.norm(p=1)
        else:
            return y.norm(p=1).item()

    def c(self, u, x, yt, set_param=False):
        """
        :param u: shape (M, 1).
        :param x: shape = (batchsize, channels, ysize, xsize).
        :param yt: shape = (batchsize, classes).
        :param set_param: If True, set the network parameters, if False just evaluate and keep the old parameters.

        :return: c(u), shape = (N*c, 1) in row-major order.
        """
        return (self.f(u, x, set_param=set_param) - yt).view(-1, 1)

    def loss(self, u, x, yt, set_param=False, tensor=False):
        """
        :param u: shape (M, 1).
        :param x: shape = (batchsize, channels, ysize, xsize).
        :param yt: shape = (batchsize, classes).
        :param set_param: See function c.
        :param tensor: See function h.

        :return: l1 norm loss.
        """
        return self.h(self.c(u, x, yt, set_param=set_param), tensor=tensor)

    def gradient(self, u, x, yt):
        self.net.zero_grad()

        loss = self.loss(u, x, yt, set_param=True, tensor=True)

        loss.backward()

        return self.net.gradient

    @staticmethod
    def Jf(net, y, retain_graph=False):
        """
        :param y: shape = (N, c).
        :param retain_graph: If False, does not retain graph after call.

        :return: Jacobian with shape (N*c, M) in row-major order.
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

        return J.detach()

    def solve_linearized_subproblem(self, uk, tau, x, yt):
        """
        Solve the linearized subproblem using gradient ascent on the dual problem.
        This is required for ProxDescent.

        :param uk: Torch tensor with shape = (M, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: Torch tensor with shape = (batchsize, channels, ysize, xsize).
        :param yt: Torch tensor with shape = (batchsize, classes).

        :return: Approximate solution \argmin L(u).
        """
        N = yt.numel()

        with torch.enable_grad():
            y = self.f(uk, x, set_param=True)

            # (batchsize*classes, M).
            Jfuk = ClassificationMNIST.Jf(self.net, y, retain_graph=True)

        yhat = (yt - y).view(-1, 1) + Jfuk.mm(uk)

        # Primal problem.
        def P(u):
            return self.h(Jfuk.mm(u) - yhat) + (1 / (2 * tau)) * ((u - uk) ** 2).sum()

        # Dual problem.
        def D(p):
            p = torch.from_numpy(p)

            c = tau * Jfuk.t().mm(p) - uk
            loss = (1 / (2 * tau)) * (c ** 2).sum() + yhat.t().mm(p)

            return loss.item()

        def gradD(p):
            p = torch.from_numpy(p)

            return (Jfuk.mm(tau * Jfuk.t().mm(p) - uk) + yhat).numpy()

        p = np.ones((N, 1), dtype=np.float32)

        params = modelbased.utils.Params(
            max_iter=5000,
            eps=1e-10,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-10)

        p_new, losses = modelbased.solver.projected_gradient.armijo(p, D, gradD,
                                                                    modelbased.solver.utils.proj_max,
                                                                    params)

        p_new = torch.from_numpy(p_new)

        # Get primal solution.
        u = uk - tau * Jfuk.t().mm(p_new)

        logging.info("D(p0): {}".format(D(p)))
        logging.info("D(p*): {}".format(D(p_new.numpy())))

        logging.info("P(uk): {}".format(P(uk)))
        logging.info("P(u*): {}".format(P(u)))

        # TODO plot_losses(losses)

        # Loss of the linearized sub-problem without the proximal term.
        linloss = self.h(Jfuk.mm(u) - yhat)

        return u.detach(), linloss

    def optimize_pd(self):
        """
        Optimize using ProxDescent.
        """
        params = modelbased.utils.Params(max_iter=10, mu_min=1, tau=2, sigma=0.5, eps=1e-8)

        x, yt = iter(self.trainloader).next()

        u_init = self.net.params.detach()

        def loss(u):
            return self.loss(u, x, yt)

        def subsolver(u, tau):
            return self.solve_linearized_subproblem(u, tau, x, yt)

        proxdescent = modelbased.solver.prox_descent.ProxDescent(params, loss, subsolver)

        with torch.no_grad():
            u_new, losses = proxdescent.prox_descent(u_init)

        self.net.params = u_new

        self.predict_class()
        plot_losses(losses, 'prox_descent')

    def optimize_gd(self, armijo=False):
        """
        Optimize using gradient descent.
        """
        x, yt = iter(self.trainloader).next()

        def f(u):
            return self.loss(torch.from_numpy(u), x, yt)

        def G(u):
            return self.gradient(torch.from_numpy(u), x, yt).numpy()

        u_init = self.net.params.detach().numpy()

        if armijo:
            params = modelbased.utils.Params(max_iter=500, beta=0.5, gamma=1e-8)
            u_new, losses = modelbased.solver.gradient_descent.armijo(u_init, f, G, params)
        else:
            params = modelbased.utils.Params(max_iter=1500, sigma=0.001)
            u_new, losses = modelbased.solver.gradient_descent.fixed_stepsize(u_init, f, G, params)

        self.net.params = torch.from_numpy(u_new)

        self.predict_class()
        plot_losses(losses, 'gradient_descent')

    def run_fullbatch(self):
        self.optimize_gd()

    def predict_class(self):
        for x, yt in self.trainloader:
            y = self.net(x).detach().numpy()

            y_class = np.argmax(y, 1)
            yt_class = np.argmax(yt.numpy(), 1)

            correct = (y_class == yt_class).sum()

            print("Correctly classified: {}/{}.".format(correct, y_class.shape[0]))


def plot_losses(losses, filename):
    plt.figure()
    plt.plot(range(len(losses)), losses, linewidth=0.4)

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)
    plt.title(filename)

    plt.show()
    plt.savefig('modelbased/results/' + filename)


def run():
    cls = ClassificationMNIST()

    cls.run_fullbatch()
