import modelbased.data.mnist as mnist_data
import modelbased.problems.mnist_classification_nn.nn as mnist_nn
import modelbased.utils.trainrun
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.utils.misc

import modelbased.solvers.torch_gradient_descent
import modelbased.solvers.gradient_descent
import modelbased.solvers.prox_descent
import modelbased.solvers.utils
import modelbased.solvers.projected_gradient

import torch
from torch.nn import functional as F

import numpy as np
import matplotlib.pyplot as plt
import logging


class l1normNoreg:
    def __init__(self, subset, batchsize):
        transform = lambda x: modelbased.utils.misc.one_hot(x, 10)

        trainloader, _, _, _, _ = mnist_data.load('datasets/mnist', subset, 10, batchsize, 10,
                                                  one_hot_encoding=transform)

        self.trainloader = trainloader
        self.net = mnist_nn.SimpleConvNetMNIST(F.relu)

    def L(self, u, x, yt, set_param=False, tensor=False):
        """
        :param u: shape (M, 1).
        :param x: shape = (batchsize, channels, ysize, xsize).
        :param yt: shape = (batchsize, classes).
        :param set_param: See function c.
        :param tensor: See function h.

        :return: l1 norm loss.
        """
        return self.h(self.c(u, x, yt, set_param=set_param), tensor=tensor)

    @staticmethod
    def h(y, tensor=False):
        """
        :param y: shape = (c*N, 1).
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
        return (self.net.f(u, x, set_param=set_param) - yt).view(-1, 1)

    def gradient(self, u, x, yt):
        self.net.zero_grad()

        loss = self.L(u, x, yt, set_param=True, tensor=True)

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

    def solve_linearized_subproblem(self, uk, tau, x, yt, verbose=False):
        """
        Solve the linearized subproblem using gradient ascent on the dual problem.
        This is required for ProxDescent.

        :param uk: Torch tensor with shape = (M, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: Torch tensor with shape = (batchsize, channels, ysize, xsize).
        :param yt: Torch tensor with shape = (batchsize, classes).

        :param verbose: Prints primal and dual loss after optimizing.

        :return: Approximate solution \argmin L(u).
        """
        N = yt.numel()

        with torch.enable_grad():
            y = self.net.f(uk, x, set_param=True)

            # (batchsize*classes, M).
            Jfuk = l1normNoreg.Jf(self.net, y, retain_graph=True)

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

        params = modelbased.utils.misc.Params(
            max_iter=5000,
            eps=1e-10,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-10)

        p_new, losses = modelbased.solvers.projected_gradient.armijo(p, D, gradD,
                                                                     modelbased.solvers.utils.proj_max,
                                                                     params)

        p_new = torch.from_numpy(p_new)

        # Get primal solution.
        u = uk - tau * Jfuk.t().mm(p_new)

        if verbose:
            logging.info("D(p0): {}".format(D(p)))
            logging.info("D(p*): {}".format(D(p_new.numpy())))

            logging.info("P(uk): {}".format(P(uk)))
            logging.info("P(u*): {}".format(P(u)))

        # TODO plot_losses(losses)

        # Loss of the linearized sub-problem without the proximal term.
        linloss = self.h(Jfuk.mm(u) - yhat)

        return u.detach(), linloss

    def optimize_pd(self, num_epochs):
        """
        Optimize using ProxDescent.
        """
        params = modelbased.utils.misc.Params(max_iter=1, mu_min=1, tau=2, sigma=0.5, eps=1e-8)

        def step_fun(x, yt):
            u_init = self.net.params.detach()

            def loss(u):
                return self.L(u, x, yt)

            def subsolver(u, tau):
                return self.solve_linearized_subproblem(u, tau, x, yt)

            proxdescent = modelbased.solvers.prox_descent.ProxDescent(params, loss, subsolver)

            with torch.no_grad():
                u_new, losses = proxdescent.prox_descent(u_init)

            self.net.params = u_new

            return losses

        def interval_fun(epoch, iteration, total_losses):
            logging.info("[{}], {}/{}: Loss={:.6f}.".format(iteration, epoch, num_epochs, total_losses[-1]))

        total_losses = modelbased.utils.trainrun.run(num_epochs, self.trainloader, step_fun,
                                                     interval_fun=interval_fun, interval=1)

        filename = modelbased.utils.misc.append_time('prox_descent')

        modelbased.utils.misc.plot_losses(total_losses, filename)
        modelbased.utils.yaml.write(filename, total_losses, params)
        self.predict_class()

    def optimize_gd(self, num_epochs, armijo=False):
        if armijo:
            params = modelbased.utils.misc.Params(max_iter=1, beta=0.5, gamma=1e-8)
        else:
            params = modelbased.utils.misc.Params(max_iter=1, sigma=0.001)

        def step_fun(x, yt):
            def f(u):
                return self.L(torch.from_numpy(u), x, yt)

            def G(u):
                return self.gradient(torch.from_numpy(u), x, yt).numpy()

            u_init = self.net.params.detach().numpy()

            if armijo:
                u_new, losses = modelbased.solvers.gradient_descent.armijo(u_init, f, G, params)
            else:
                u_new, losses = modelbased.solvers.gradient_descent.fixed_stepsize(u_init, f, G, params)

            self.net.params = torch.from_numpy(u_new)

            return losses

        def interval_fun(epoch, iteration, total_losses):
            logging.info("[{}], {}/{}: Loss={:.6f}.".format(iteration, epoch, num_epochs, total_losses[-1]))

        total_losses = modelbased.utils.trainrun.run(num_epochs, self.trainloader, step_fun,
                                                     interval_fun=interval_fun, interval=1)

        filename = modelbased.utils.misc.append_time('gradient_descent')

        modelbased.utils.misc.plot_losses(total_losses, filename)
        modelbased.utils.yaml.write(filename, total_losses, params)
        self.predict_class()

    def predict_class(self):
        correct = 0
        num = 0

        for x, yt in self.trainloader:
            y = self.net(x).detach().numpy()

            y_class = np.argmax(y, 1)
            yt_class = np.argmax(yt.numpy(), 1)

            correct += (y_class == yt_class).sum()
            num += y_class.shape[0]

        print("Correctly classified: {}/{}.".format(correct, num))


def run():
    cls = l1normNoreg(None, 5)

    # cls.optimize_gd(10)
    cls.optimize_pd(1)
