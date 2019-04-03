import numpy as np
import matplotlib.pylab as plt
import scipy.optimize
import logging

import modelbased.data.noise_from_model
import modelbased.utils
import modelbased.solver.prox_descent
import modelbased.solver.projected_gradient
import modelbased.solver.utils


class LaplaceNoise1d:
    """
    Simple 1d robust regression problem:

        \min_u \sum_i^N \| f_i(u) - y_i \|_1

    where u = (a,b) and

        f_i(u) = \sum_j^P a_j \exp(-b_j x_i)

    with data (x_i,y_i) \in R^2.

    Solve the linearized sub-problem (at u^k).

        \min_u \{ L(u) = \sum_i^N \| Jf_i(u^k)*u - \hat{y_i} \|_1 + \frac{1}{2\tau} \| u - u^k \|_2^2 \},

    where \hat{y_i} = y_i - f_i(u^k) + Jf_i(u^k) u^k.

    We use the proximal gradient method on the dual problem:

        -\min_p \{ E(p) = \frac{1}{2 \tau} \| \tau Jf^T p - u^k \|_2^2 + \hat{y}^Tp \}

        s.t. \| p \|_\infty \le 1.
    """

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
    def h(y):
        """
        :param y: shape = (N, 1).

        :return: h(y).
        """
        return np.abs(y).sum()

    @staticmethod
    def c(u, x, y_targets):
        """
        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: c(u), shape = (N, 1).
        """
        y = LaplaceNoise1d.f(u, x)

        c = y - y_targets

        return c

    @staticmethod
    def loss(u, x, y_targets):
        """
        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: L(w).
        """
        return LaplaceNoise1d.h(LaplaceNoise1d.c(u, x, y_targets))

    @staticmethod
    def Jf(u, x):
        """
        Jacobian of the model function at x.

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
    def solve_linearized_subproblem(uk, tau, x, y_targets):
        """
        Solve the inner linearized subproblem using gradient ascent on the dual problem.

        :param uk: shape = (2P, 1).
        :param tau: Proximal weight of the subproblem.
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: Approximate solution \argmin L(u).
        """
        N = x.shape[0]

        # (N, 2P). Jf(u^k).
        Jfuk = LaplaceNoise1d.Jf(uk, x)

        # (N, 1). Inner constant part per iteration.
        yhat = y_targets - LaplaceNoise1d.f(uk, x) + Jfuk.dot(uk)

        # Primal problem.
        def P(u):
            return LaplaceNoise1d.h(Jfuk.dot(u) - yhat) + (1 / (2 * tau)) * ((u - uk) ** 2).sum()

        # Dual problem.
        def D(p):
            c = tau * Jfuk.T.dot(p) - uk
            loss = (1 / (2 * tau)) * (c ** 2).sum() + yhat.T.dot(p)

            return loss.squeeze()

        def gradD(p):
            return Jfuk.dot(tau * Jfuk.T.dot(p) - uk) + yhat

        # (N, 1).
        p = modelbased.solver.utils.proj_max(np.empty((N, 1)))

        params = modelbased.utils.Params(
            max_iter=3000,
            eps=1e-10,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-10)

        p_new, losses = modelbased.solver.projected_gradient.armijo(p, D, gradD,
                                                                    modelbased.solver.utils.proj_max,
                                                                    params)

        # Get primal solution.
        u = uk - tau * Jfuk.T.dot(p_new)

        logging.info("D(p0): {}".format(D(p)))
        logging.info("D(p*): {}".format(D(p_new)))

        logging.info("P(uk): {}".format(P(uk)))
        logging.info("P(u*): {}".format(P(u)))

        # Loss of the linearized sub-problem without the proximal term.
        linloss = LaplaceNoise1d.h(Jfuk.dot(u) - yhat)

        return u, linloss

    def run(self, u_init):
        params = modelbased.utils.Params(
            max_iter=10,
            mu_min=10,
            tau=2,
            sigma=0.8,
            eps=1e-3)

        def loss(u):
            return self.loss(u, self.x, self.y_targets).squeeze()

        def subsolver(u, tau):
            return self.solve_linearized_subproblem(u, tau, self.x, self.y_targets)

        proxdescent = modelbased.solver.prox_descent.ProxDescent(params, loss, subsolver)
        u_new = proxdescent.prox_descent(u_init)

        return u_new


def plot(x, y, y_noisy, y_predict, y_init):
    plt.plot(x, y, label='true')
    plt.scatter(x, y_noisy, marker='x')
    plt.plot(x, y_predict, label='predict')
    plt.plot(x, y_init, label='init')
    plt.legend()
    plt.show()


def run():
    N = 300
    P_gen = 10
    P_model = 20

    # Generate some noisy data.
    a = 2 * np.random.random((P_gen, 1))
    b = np.random.random((P_gen, 1))
    u = np.concatenate((a, b), axis=0)

    def fun(x):
        return LaplaceNoise1d.f(u, x)

    x, y_noisy, y = modelbased.data.noise_from_model.generate(N, fun)

    reg = LaplaceNoise1d(x, y_noisy)

    u_init = 0.1 * np.ones((2 * P_model, 1))
    u_new = reg.run(u_init)
    y_predict = reg.f(u_new, x)

    y_init = reg.f(u_init, x)

    plot(x, y, y_noisy, y_predict, y_init)
