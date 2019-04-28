import torch
from torch.nn import functional as F
import torchvision.utils
import logging
import numpy as np
import math
import os

import modelbased.data.mnist as mnist_data
import modelbased.problems.mnist_classification_nn.nn as mnist_nn
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.utils.trainrun
import modelbased.solvers.utils
import modelbased.solvers.projected_gradient
import modelbased.solvers.prox_descent

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def sql2norm(x):
    return (x ** 2).sum().item()


class SVM_OVA:
    def __init__(self, subset, batchsize):
        def transform(x):
            return modelbased.utils.misc.one_hot(x, 10, hot=1, off=-1)

        trainloader, _, _, _, _ = mnist_data.load('datasets/mnist', subset, 10, batchsize, 10,
                                                  one_hot_encoding=transform)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: {}.".format(device))

        self.trainloader = trainloader
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
        return lam * 0.5 * sql2norm(u)

    def loss(self, u, x, y_targets, lam):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param lam: Scalar regularizer weight factor.

        :return: L(u)
        """
        return self.h(self.c(u, x), y_targets) + self.reg(u, lam)

    def solve_linearized_subproblem(self, uk, tau, x, y_targets, lam, verbose=True):
        """
        Solve the inner linearized subproblem using gradient ascent on the dual problem.

        :param uk: shape = (d, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.

        :param lam: Scalar regularizer weight factor.

        :param verbose: If True, prints primal and dual loss before and after optimization.

        :return: Solution argmin L_lin(u), loss of linearized subproblem without proximal term.
        """
        n = x.size(0)
        c = y_targets.size(1)

        # (c*n, d).
        J = self.net.Jacobian(uk, x)

        # (n, c).
        yhat = self.net.f(uk, x) - J.mm(uk).view(n, c)

        # Primal problem.
        def P(_u):
            return self.h(J.mm(_u).view(n, c) + yhat, y_targets) + (
                    lam * 0.5 * sql2norm(_u) + tau * 0.5 * sql2norm(_u - uk))

        # Dual problem.
        def D(_p):
            dloss = (y_targets * _p.view(n, c)).sum() + (
                    1 / (2 * (lam + tau)) * sql2norm(-J.t().mm(_p) + tau * uk) - yhat.view(1, -1).mm(_p))

            return dloss.item()

        def gradD(_p):
            return y_targets.view(-1, 1) + (1 / (lam + tau)) * J.mm(J.t().mm(_p) - tau * uk) - yhat.view(-1, 1)

        def proj(_x):
            # (c*n, 1).
            x_aug = y_targets.view(-1, 1) * n * _x

            x_aug_projected = modelbased.solvers.utils.torch_proj(x_aug, -1, 0)

            x_projected = x_aug_projected * y_targets.view(-1, 1) / n

            return x_projected

        # (c*n, 1).
        p = proj(torch.empty(c * n, 1, device=uk.device))

        params = modelbased.utils.misc.Params(
            max_iter=5000,
            eps=1e-6,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-6)

        p_new, losses = modelbased.solvers.projected_gradient.armijo(p, D, gradD, proj, params, tensor_type='pytorch')

        # Get primal solution.
        u = (1 / (lam + tau)) * (tau * uk - J.t().mm(p_new))

        if verbose:
            logger.info("D(p0): {}".format(D(p)))
            logger.info("D(p*): {}".format(D(p_new)))

            logger.info("P(uk): {}".format(P(uk)))
            logger.info("P(u*): {}".format(P(u)))

        # Loss of the linearized sub-problem without the proximal term.
        linloss = self.h(J.mm(u).view(n, c) + yhat, y_targets) + self.reg(u, lam)

        return u.detach(), linloss

    def run(self):
        params = modelbased.utils.misc.Params(
            max_iter=1,
            mu_min=0.01,
            tau=2,
            sigma=0.7,
            eps=1e-6)

        num_epochs = 2
        lam = 0.01

        def step_fun(x, yt):
            u_init = self.net.params

            def loss(u):
                return self.loss(u, x, yt, lam)

            def subsolver(u, tau):
                return self.solve_linearized_subproblem(u, tau, x, yt, lam, verbose=False)

            proxdescent = modelbased.solvers.prox_descent.ProxDescent(params, loss, subsolver)
            u_new, losses = proxdescent.prox_descent(u_init, tensor_type='pytorch', verbose=True)

            self.net.params = u_new

            return losses

        def interval_fun(epoch, iteration, _total_losses):
            logger.info("[{}], {}/{}: Loss={:.6f}.".format(iteration, epoch, num_epochs, _total_losses[-1]))

        total_losses = modelbased.utils.trainrun.run(num_epochs, self.trainloader, step_fun, self.net.device,
                                                     interval_fun=interval_fun, interval=1)

        filename = modelbased.utils.misc.append_time('prox_descent')

        modelbased.utils.misc.plot_losses(total_losses, filename)
        modelbased.utils.yaml.write(filename, total_losses, params)

        return self.net.params

    def predict(self, u, x):
        # (n, c).
        y = self.net.f(u, x)

        return y.argmax(1)


def plot(x, y, yt, nrow=6):
    """
    Plot the given x images in a grid with the corresponding predicted and ground truth labels.

    :param x: shape = (n, channels, ysize, xsize).
    :param y: shape = (c).
    :param yt: shape = (c).
    :param nrow: Number of images per row.
    """
    n = x.size(0)

    if n < nrow:
        nrow = n

    img_data = np.transpose(torchvision.utils.make_grid(x, nrow=nrow, padding=0, normalize=True), (1, 2, 0))

    plt.figure()

    plt.imshow(img_data)
    plt.tight_layout()

    grid_y = math.ceil(n / nrow)

    for i in range(1, grid_y + 1):
        for j in range(1, nrow + 1):
            plt.text(j * 28 - 4, i * 28 - 7, str(yt[(i - 1) * nrow + j - 1].item()), size=16, color='red')
            plt.text(j * 28 - 4, i * 28 - 1, str(y[(i - 1) * nrow + j - 1].item()), size=16, color='yellow')

            if (i - 1) * nrow + j == n:
                break

    filename = modelbased.utils.misc.append_time('mnist_results')
    plt.savefig(os.path.join('modelbased', 'results', 'plots', filename + '.png'))


def run():
    cls = SVM_OVA(30, 5)

    u_new = cls.run()

    for x, yt in cls.trainloader:
        yt_predict = yt.argmax(1)
        y_predict = cls.predict(u_new, x)

        plot(x, y_predict, yt_predict)
