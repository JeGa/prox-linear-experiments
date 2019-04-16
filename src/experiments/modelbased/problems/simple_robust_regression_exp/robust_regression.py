import numpy as np
import matplotlib.pylab as plt
import scipy.optimize
import logging

import modelbased.data.noise_from_model
import modelbased.utils.misc
import modelbased.solver.prox_descent
import modelbased.solver.projected_gradient
import modelbased.solver.utils

logger = logging.getLogger(__name__)


class LaplaceNoise1d:
    def __init__(self, x, y_targets):
        self.x = x
        self.y_targets = y_targets

    @staticmethod
    def f(u, x):
        """
        Exponential model function.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).

        :return: shape = (N, 1).
        """
        N = x.shape[0]

        a, b, _ = LaplaceNoise1d.split_params(u)

        A = np.tile(a, (1, N))
        y = b.T.dot(np.exp(-A.dot(np.diag(np.squeeze(x))))).T

        return y

    @staticmethod
    def Jf(u, x):
        """
        Jacobian of the model function at u.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).

        :return: shape = (N, 2P).
        """
        N = x.shape[0]

        a, b, P = LaplaceNoise1d.split_params(u)

        # (N, P).
        A = np.tile(a, (1, N))
        exp = np.exp(-A.dot(np.diag(np.squeeze(x)))).T

        B = np.tile(b.T, (N, 1))
        X = np.tile(x, (1, P))

        Jf_a = -B * exp * X

        # (N, P).
        Jf_b = exp

        J = np.concatenate((Jf_a, Jf_b), axis=1)

        return J

    @staticmethod
    def _check_grad(x):
        err = []

        for i in range(x.shape[0]):
            def f(u):
                u = np.expand_dims(u, 1)
                return LaplaceNoise1d.f(u, x)[i].squeeze()

            def g(u):
                u = np.expand_dims(u, 1)
                return LaplaceNoise1d.Jf(u, x)[i].squeeze()

            P_model = 4
            u0 = np.ones((2 * P_model, 1)).squeeze()

            err.append(scipy.optimize.check_grad(f, g, u0))

        print(np.mean(err))

    @staticmethod
    def split_params(u):
        P = int(u.shape[0] / 2)

        a = u[:P]
        b = u[P:]

        return a, b, P

    @staticmethod
    def L(x, y_targets):
        """
        Applies the element-wise l1-norm loss function.

        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: L(x), shape = (N, 1).
        """
        return np.abs(x - y_targets)

    @staticmethod
    def h(a, y_targets):
        """
        :param a: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: h(a).
        """
        N = a.shape[0]

        return (1 / N) * LaplaceNoise1d.L(a, y_targets).sum()

    @staticmethod
    def c(u, x):
        """
        Just wraps the model function.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).

        :return: c(u), shape = (N, 1).
        """
        return LaplaceNoise1d.f(u, x)

    @staticmethod
    def reg(u, lam):
        """
        Squared l2-norm regularizer.

        :param u: shape = (2P, 1).
        :param lam: Scalar weight factor.

        :return: lam * 0.5 * norm(u)**2.
        """
        return lam * 0.5 * (u ** 2).sum()

    @staticmethod
    def loss(u, x, y_targets, lam):
        """
        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).
        :param lam: Scalar regularizer weight factor.

        :return: L(w).
        """
        return LaplaceNoise1d.h(LaplaceNoise1d.c(u, x), y_targets).squeeze() + LaplaceNoise1d.reg(u, lam).squeeze()

    @staticmethod
    def solve_linearized_subproblem(uk, tau, x, y_targets, lam):
        """
        Solve the inner linearized subproblem using gradient ascent on the dual problem.

        :param uk: shape = (2P, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :param lam: Scalar regularizer weight factor.

        :return: Solution argmin L(u).
        """
        N = x.shape[0]

        # (N, 2P). Jf(u^k).
        Jfuk = LaplaceNoise1d.Jf(uk, x)

        # (N, 1). Inner constant part per iteration.
        yhat = LaplaceNoise1d.f(uk, x) - Jfuk.dot(uk)

        # Primal problem.
        def P(_u):
            return LaplaceNoise1d.h(Jfuk.dot(_u) + yhat, y_targets) + \
                   LaplaceNoise1d.reg(_u, lam) + \
                   (tau / 2) * ((_u - uk) ** 2).sum()

        # Dual problem.
        def D(_p):
            c = -Jfuk.T.dot(_p) + tau * uk
            loss = (1 / (2 * (lam + tau))) * (c ** 2).sum() + (y_targets - yhat).T.dot(_p)

            return loss.squeeze()

        def gradD(_p):
            return (1 / (lam + tau)) * Jfuk.dot(Jfuk.T.dot(_p) - tau * uk) + y_targets - yhat

        def proj(_x):
            return modelbased.solver.utils.proj_max(_x, 1 / N)

        # (N, 1).
        p = proj(np.empty((N, 1)))

        params = modelbased.utils.misc.Params(
            max_iter=5000,
            eps=1e-6,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-6)

        p_new, losses = modelbased.solver.projected_gradient.armijo(p, D, gradD, proj, params)

        # Get primal solution.
        u = (1 / (lam + tau)) * (tau * uk - Jfuk.T.dot(p_new))

        logger.info("D(p0): {}".format(D(p)))
        logger.info("D(p*): {}".format(D(p_new)))

        logger.info("P(uk): {}".format(P(uk)))
        logger.info("P(u*): {}".format(P(u)))

        # Loss of the linearized sub-problem without the proximal term.
        linloss = LaplaceNoise1d.h(Jfuk.dot(u) + yhat, y_targets) + LaplaceNoise1d.reg(u, lam)

        return u, linloss

    def run(self, u_init):
        params = modelbased.utils.misc.Params(
            max_iter=50,
            mu_min=0.01,
            tau=2,
            sigma=0.7,
            eps=1e-6)

        lam = 0.01

        def loss(u):
            return self.loss(u, self.x, self.y_targets, lam)

        def subsolver(u, tau):
            return self.solve_linearized_subproblem(u, tau, self.x, self.y_targets, lam)

        proxdescent = modelbased.solver.prox_descent.ProxDescent(params, loss, subsolver)

        u_new, losses = proxdescent.prox_descent(u_init, verbose=True)

        return u_new


def plot(x, y, y_noisy, y_predict, y_init):
    plt.plot(x, y, label='true')
    plt.scatter(x, y_noisy, marker='x')
    plt.plot(x, y_predict, label='predict')
    plt.plot(x, y_init, label='init')

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    plt.legend()

    plt.show()


def run():
    N = 200
    P_gen = 10
    P_model = 20

    np.random.seed(1112)

    # Generate some noisy data.
    a = 2 * np.random.random((P_gen, 1))
    b = np.random.random((P_gen, 1))
    u = np.concatenate((a, b), axis=0)

    def fun(_x):
        return LaplaceNoise1d.f(u, _x)

    x, y_noisy, y = modelbased.data.noise_from_model.generate(N, fun)

    reg = LaplaceNoise1d(x, y_noisy)

    u_init = 0.1 * np.ones((2 * P_model, 1))
    u_new = reg.run(u_init)

    y_predict = reg.f(u_new, x)
    y_init = reg.f(u_init, x)

    plot(x, y, y_noisy, y_predict, y_init)
