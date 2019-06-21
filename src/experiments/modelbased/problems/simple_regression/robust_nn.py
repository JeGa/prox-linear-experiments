import numpy as np
import scipy.optimize
import logging

import modelbased.data.noise_from_model
import modelbased.utils.misc
import modelbased.solvers.prox_descent_linesearch
import modelbased.solvers.projected_gradient
import modelbased.solvers.utils
from .misc import generate_data, plot

logger = logging.getLogger(__name__)


class RobustRegression:
    @classmethod
    def f(cls, u, x):
        """
        Exponential model function.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).

        :return: shape = (N, 1).
        """
        N = x.shape[0]

        a, b, _ = cls.split_params(u)

        A = np.tile(a, (1, N))
        y = b.T.dot(np.exp(-A.dot(np.diag(np.squeeze(x))))).T

        return y

    @classmethod
    def Jf(cls, u, x):
        """
        Jacobian of the model function at u.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).

        :return: shape = (N, 2P).
        """
        N = x.shape[0]

        a, b, P = cls.split_params(u)

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

    @classmethod
    def _check_grad(cls, x):
        err = []

        for i in range(x.shape[0]):
            def f(u):
                u = np.expand_dims(u, 1)
                return cls.f(u, x)[i].squeeze()

            def g(u):
                u = np.expand_dims(u, 1)
                return cls.Jf(u, x)[i].squeeze()

            P_model = 4
            u0 = np.ones((2 * P_model, 1)).squeeze()

            err.append(scipy.optimize.check_grad(f, g, u0))

        print(np.mean(err))

    @staticmethod
    def split_params(u):
        P = u.shape[0] // 2

        a = u[:P]
        b = u[P:]

        return a, b, P

    @staticmethod
    def L(v, y_targets):
        """
        Applies the element-wise l1-norm loss function.

        :param v: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: L(v), shape = (N, 1).
        """
        return np.abs(v - y_targets)

    @classmethod
    def h(cls, a, y_targets):
        """
        :param a: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: h(a).
        """
        N = a.shape[0]

        return (1 / N) * cls.L(a, y_targets).sum()

    @classmethod
    def c(cls, u, x):
        """
        Just wraps the model function.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).

        :return: c(u), shape = (N, 1).
        """
        return cls.f(u, x)

    @staticmethod
    def reg(u, lam):
        """
        Squared l2-norm regularizer.

        :param u: shape = (2P, 1).
        :param lam: Scalar weight factor.

        :return: lam * 0.5 * norm(u)**2.
        """
        return lam * 0.5 * (u ** 2).sum()

    @classmethod
    def loss(cls, u, x, y_targets, lam):
        """
        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).
        :param lam: Scalar regularizer weight factor.

        :return: L(u).
        """
        return cls.h(cls.c(u, x), y_targets).squeeze() + cls.reg(u, lam).squeeze()

    @classmethod
    def solve_subproblem(cls, uk, tau, x, y_targets, lam):
        """
        Solve the inner linearized subproblem using gradient ascent on the dual problem.

        :param uk: shape = (2P, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :param lam: Scalar regularizer weight factor.

        :return: Solution argmin L_lin(u), loss of linearized subproblem without proximal term.
        """
        N = x.shape[0]

        # (N, 2P). Jf(u^k).
        Jfuk = cls.Jf(uk, x)

        # (N, 1). Inner constant part per iteration.
        yhat = cls.f(uk, x) - Jfuk.dot(uk)

        # Primal problem.
        def P(_u):
            return cls.h(Jfuk.dot(_u) + yhat, y_targets) + \
                   cls.reg(_u, lam) + \
                   (tau / 2) * ((_u - uk) ** 2).sum()

        # Dual problem.
        def D(_p):
            c = -Jfuk.T.dot(_p) + tau * uk
            loss = (1 / (2 * (lam + tau))) * (c ** 2).sum() + (y_targets - yhat).T.dot(_p)

            return loss.squeeze()

        def gradD(_p):
            return (1 / (lam + tau)) * Jfuk.dot(Jfuk.T.dot(_p) - tau * uk) + y_targets - yhat

        def proj(_x):
            return modelbased.solvers.utils.proj_max(_x, 1 / N)

        # (N, 1).
        p = proj(np.empty((N, 1)))

        params = modelbased.utils.misc.Params(
            max_iter=5000,
            eps=1e-6,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-6)

        p_new, losses = modelbased.solvers.projected_gradient.armijo(p, D, gradD, proj, params)

        # Get primal solution.
        u = (1 / (lam + tau)) * (tau * uk - Jfuk.T.dot(p_new))

        logger.info("D(p0): {}".format(D(p)))
        logger.info("D(p*): {}".format(D(p_new)))

        logger.info("P(uk): {}".format(P(uk)))
        logger.info("P(u*): {}".format(P(u)))

        # Loss of the linearized sub-problem without the proximal term.
        linloss = cls.h(Jfuk.dot(u) + yhat, y_targets) + cls.reg(u, lam)

        return u, linloss


class RobustRegressionProxLinearFixed(RobustRegression):
    pass


class RobustRegressionProxLinearLinesearch(RobustRegression):
    @classmethod
    def run(cls, u_init, x, y_targets):
        params = modelbased.utils.misc.Params(
            max_iter=10,
            eps=1e-12,
            proximal_weight=1,
            gamma=0.6,
            delta=0.5,
            eta_max=2)

        lam = 0.0

        def loss(u):
            return cls.loss(u, x, y_targets, lam)

        def subsolver(u, tau):
            return cls.solve_subproblem(u, tau, x, y_targets, lam)

        proxdescent = modelbased.solvers.prox_descent_linesearch.ProxDescentLinesearch(params, verbose=True)

        u_new, losses = proxdescent.run(u_init, loss, subsolver)

        return u_new


class RobustRegressionProxLinearDamping(RobustRegression):
    pass


# params = modelbased.utils.misc.Params(
#     max_iter=50,
#     mu_min=0.01,
#     tau=2,
#     sigma=0.7,
#     eps=1e-6)


def run():
    seed = 4444
    np.random.seed(seed)

    x, y_noisy, y = generate_data(samples=200, P=10, seed=seed)

    # Number of parameters of the prediction function (2P).
    P_model = 20
    u_init = np.random.randn(P_model, 1)

    reg = RobustRegressionProxLinearLinesearch

    u_new = reg.run(u_init, x, y_noisy)

    # TODO: Save results.
    y_predict = reg.f(u_new, x)
    y_init = reg.f(u_init, x)

    plot(x, y, y_noisy, y_predict, y_init)
