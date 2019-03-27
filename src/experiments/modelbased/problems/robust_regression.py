import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


class LaplaceNoise1d:
    """
    Simple 1d robust regression problem:

        \min_u \sum_i^N \| f_i(u) - y_i \|_1

    where u = (a,b) and

        f_i(u) = \sum_j^P a_j \exp(-b_j x_i)

    with data (x_i,y_i) \in R^2.

    It is solved by using ProxDescent. For the subproblems FISTA is used.
    """

    def __init__(self, x, y_targets):
        self.x = x
        self.y_t = y_targets

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
    def loss(u, x, y_targets):
        """
        :param u: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).

        :return: L(w).
        """
        y = LaplaceNoise1d.f(u, x)

        loss = np.abs(y - y_targets).sum()

        return loss, y

    @staticmethod
    def solve_subproblem(uk, x, y_targets, tau):
        """
        Solve the linearized sub-problem (at u^k).

            \min_u \{ L(u) = \sum_i^N \| Jf_i(u^k)*u - \hat{y_i} \|_1 + \frac{1}{2\tau} \| u - u^k \|_2^2 \},

        where \hat{y_i} = y_i - f_i(u^k) + Jf_i(u^k) u^k.

        We use the proximal gradient method on the dual problem:

            -\min_p \{ E(p) = \frac{1}{2 \tau} \| \tau Jf^T p - u^k \|_2^2 + \hat{y}^Tp \}

            s.t. \| p \|_\infty \le 1.

        :param uk: shape = (2P, 1).
        :param x: shape = (N, 1).
        :param y_targets: shape = (N, 1).
        :param tau: Proximal weight of the subproblem.

        :return: Approximate solution \argmin L(u).
        """
        N = x.shape[0]

        # Jf(u^k).
        Jfuk = LaplaceNoise1d.Jf(uk, x)

        # (N, 1).
        yhat = y_targets - LaplaceNoise1d.f(uk, x) + Jfuk.dot(uk)

        # Primal problem.
        def P(u):
            return np.abs(Jfuk.dot(u) - yhat).sum() + (1 / (2 * tau)) * ((u - uk) ** 2).sum()

        # Dual problem.
        def E(p):
            c = tau * Jfuk.T.dot(p) - uk
            loss = (1 / (2 * tau)) * (c ** 2).sum() + yhat.T.dot(p)

            return loss

        def gradE(p):
            return Jfuk.dot(tau * Jfuk.T.dot(p) - uk) + yhat

        def proj_max(v):
            """
            Projection of the given vector onto the unit ball under the max norm.
            Alters the numpy array.

            :param v: shape = (n, 1).

            :return: shape = (n, 1).
            """
            for i in range(v.shape[0]):
                if v[i] >= 1:
                    v[i] = 1
                if v[i] <= -1:
                    v[i] = -1

            return v

        # (N, 1).
        p = proj_max(np.empty((N, 1)))

        def projected_gradient_fixed(p):
            max_iter = 5000
            sigma = 0.0001

            losses = []

            for i in range(max_iter):
                grad = gradE(p)

                p = proj_max(p - sigma * grad)

                losses.append(E(p).squeeze())

            return p, losses

        def projected_gradient_armijo(p):
            max_iter = 3000
            eps = 1e-10
            beta = 0.3
            gamma = 1e-2
            tau = 1e-2
            sigmamin = 1e-10

            losses = []
            stop = False

            for i in range(max_iter):
                grad = gradE(p)
                dtau = p - proj_max(p - tau * grad)

                opt = np.sqrt((dtau ** 2)).sum() / tau
                if opt <= eps:
                    break

                dk = -dtau

                # Armijo.
                i = 0
                while True:
                    sigma = beta ** i

                    if sigma < sigmamin:
                        stop = True
                        break

                    if E(p + sigma * dk) - E(p) <= gamma * sigma * grad.T.dot(dk):
                        p = p + sigma * dk
                        break

                    i += 1

                if stop:
                    break

                loss = E(p).squeeze()
                losses.append(loss)

            return p, losses

        # ==============================================================================================================

        p_new, losses = projected_gradient_armijo(p)

        # Get primal solution.
        u = uk - tau * Jfuk.T.dot(p_new)

        print("E(p0): {}".format(E(p).squeeze()))
        print("E(p*): {}".format(E(p_new).squeeze()))

        print("P(uk): {}".format(P(uk)))
        print("P(u*): {}".format(P(u)))

        # Loss of the linearized sub-problem without the proximal term.
        linloss = np.abs(Jfuk.dot(u) - yhat).sum()

        return u, linloss

    def prox_descent(self, u_init):
        x, y_t = self.x, self.y_t

        max_iter = 10
        mu_min = 10
        mu = mu_min
        tau = 2
        sigma = 0.8
        eps = 1e-3

        u = u_init

        terminate = False

        losses = []
        loss_init, _ = self.loss(u, x, y_t)
        losses.append(loss_init)

        for i in range(max_iter):
            while True:
                loss_old, _ = self.loss(u, x, y_t)

                u_new, linloss = self.solve_subproblem(u, x, y_t, mu ** -1)

                loss_new, _ = self.loss(u_new, x, y_t)

                diff_u = np.sqrt(((u - u_new) ** 2).sum())
                diff_loss = loss_old - loss_new
                diff_lin = loss_old - linloss

                print("L(uk) = {}, L(uk+1) = {}, mu = {}, diff_u = {}.".format(loss_old, loss_new, mu, diff_u))

                if diff_u <= eps:
                    terminate = True
                    break

                if mu >= 1e6:
                    terminate = True
                    break

                # Accept if decrease is sufficiently large.
                if diff_loss >= sigma * diff_lin:
                    mu = max(mu_min, mu / tau)

                    u = u_new
                    break
                else:
                    mu = tau * mu

            loss, y_predict = self.loss(u, self.x, self.y_t)
            losses.append(loss)

            print()
            print("Iteration {}: {}".format(i, loss))
            print()

            if terminate:
                break

        plt.figure()
        plt.plot(range(len(losses)), losses)
        plt.show()

        return u

    @staticmethod
    def split_params(u):
        P = int(u.shape[0] / 2)

        a = u[:P]
        b = u[P:]

        return a, b, P

    def run(self, u_init):
        u = self.prox_descent(u_init)

        loss, y_predict = self.loss(u, self.x, self.y_t)

        return y_predict
