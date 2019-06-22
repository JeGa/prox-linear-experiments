import numpy as np
import scipy.optimize


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
        return (cls.h(cls.c(u, x), y_targets).squeeze() + cls.reg(u, lam).squeeze())

    @staticmethod
    def description():
        return {
            'loss function': 'l1-norm.',
            'prediction function': 'exponential model.',
        }
