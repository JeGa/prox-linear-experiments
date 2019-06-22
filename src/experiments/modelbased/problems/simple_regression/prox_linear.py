import numpy as np
import logging

import modelbased.data.noise_from_model
import modelbased.utils.misc
import modelbased.utils.results
import modelbased.utils.yaml
import modelbased.solvers.prox_descent_linesearch
import modelbased.solvers.projected_gradient
import modelbased.solvers.utils
import modelbased.problems.simple_regression.misc
import modelbased.problems.simple_regression.robust_exp

logger = logging.getLogger(__name__)


class RobustRegressionProxLinear(modelbased.problems.simple_regression.robust_exp.RobustRegression):
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
            return cls.h(Jfuk.dot(_u) + yhat, y_targets) + cls.reg(_u, lam) + (tau / 2) * ((_u - uk) ** 2).sum()

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

    @classmethod
    def run(cls, u_init, data):
        raise NotImplementedError


class FixedStepsize(RobustRegressionProxLinear):
    @classmethod
    def run(cls, u_init, data):
        pass


class Linesearch(RobustRegressionProxLinear):
    @classmethod
    def run(cls, u_init, data):
        params = modelbased.utils.misc.Params(
            max_iter=10,
            eps=1e-12,
            proximal_weight=1,
            gamma=0.6,
            delta=0.5,
            eta_max=2)

        lam = 0.0

        def loss(u):
            return cls.loss(u, data.x, data.y_targets, lam)

        def subsolver(u, tau):
            return cls.solve_subproblem(u, tau, data.x, data.y_targets, lam)

        init_loss = loss(u_init)

        proxdescent = modelbased.solvers.prox_descent_linesearch.ProxDescentLinesearch(params, verbose=True)
        u_new, losses = proxdescent.run(u_init, loss, subsolver)

        losses.insert(0, init_loss)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('robust-regression-prox-linear-linesearch'),
            type='train',
            description={**cls.description(), 'optimization method': 'prox-linear with linesearch.'},
            train_dataset={
                'name': 'generated with seed ' + str(data.seed),
                'size': data.x.shape[0]
            },
            loss={'batch': [losses, list(range(len(losses)))]},
            parameters={**vars(params), 'lambda': lam},
            info=None,
            model_parameters=u_new.tolist()
        )

        return results


class RobustRegressionProxLinearDamping(RobustRegressionProxLinear):
    @classmethod
    def run(cls, u_init, data):
        params = modelbased.utils.misc.Params(
            max_iter=50,
            mu_min=0.01,
            tau=2,
            sigma=0.7,
            eps=1e-6)


def plot_regression(model_parameters, u_init, data):
    y_predict = RobustRegressionProxLinear.f(np.array(model_parameters), data.x)
    y_init = RobustRegressionProxLinear.f(u_init, data.x)

    modelbased.problems.simple_regression.misc.plot_regression(data.x, data.y, data.y_targets, y_predict, y_init)


def run():
    seed = 4444
    np.random.seed(seed)

    data = modelbased.problems.simple_regression.misc.generate_data(samples=200, P=10, seed=seed)

    # Number of parameters of the prediction function (2P).
    P_model = 20
    u_init = np.random.randn(P_model, 1)

    results = Linesearch.run(u_init, data)

    modelbased.utils.yaml.write_result(results)

    plot_regression(results.model_parameters, u_init, data)
