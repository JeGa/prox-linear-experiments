import numpy as np

import modelbased.utils.results
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.solvers.gradient_descent
import modelbased.problems.simple_regression.misc
import modelbased.problems.simple_regression.robust_exp as robust_exp


class RobustRegressionGradientDescent(robust_exp.RobustRegression):
    @classmethod
    def loss_grad(cls, u, x, y_targets, lam):
        """
        Be careful here, since the l1-norm is not differentiable.

        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).
        :param lam: Scalar regularizer weight factor.

        :return: Gradient of the loss at u. Shape = (2P, 1).
        """
        n = x.shape[0]
        _, _, P = cls.split_params(u)

        diff = cls.c(u, x) - y_targets
        diff_norm = np.linalg.norm(diff, ord=1)

        if diff_norm == 0:
            raise RuntimeError('Division by zero.')

        return (1 / n) * (cls.Jf(u, x) * np.tile(diff / diff_norm, (1, 2 * P))).sum(0, keepdims=True).T + lam * u

    @classmethod
    def run(cls, u_init, data):
        lam = 0

        params = modelbased.utils.misc.Params(max_iter=10, beta=0.5, gamma=1e-3, sigma=1)

        def f(_u):
            return cls.loss(_u, data.x, data.y_targets, lam).item()

        def G(_u):
            return cls.loss_grad(_u, data.x, data.y_targets, lam)

        u_new, losses = modelbased.solvers.gradient_descent.fixed_stepsize(u_init, f, G, params, verbose=True)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('robust-regression-gradient-descent'),
            type='train',
            description={**cls.description(), 'optimization method': 'gradient descent with armijo.'},
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


def run():
    seed = 4444
    np.random.seed(seed)

    data = modelbased.problems.simple_regression.misc.generate_data(samples=200, P=10, seed=seed)

    # Number of parameters of the prediction function.
    P_model = 20
    u_init = np.random.randn(P_model, 1)

    results = RobustRegressionGradientDescent.run(u_init, data)

    modelbased.utils.yaml.write_result(results)

    modelbased.problems.simple_regression.misc.plot_regression(results.model_parameters, u_init, data)
