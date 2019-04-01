import numpy as np
import logging


class ProxDescent:
    def __init__(self, params, loss, solve_linearized_subproblem):
        self.params = params
        self.loss = loss
        self.solve_linearized_subproblem = solve_linearized_subproblem

    def prox_descent(self, u_init):
        losses = []

        loss_init = self.loss(u_init)
        losses.append(loss_init)

        mu = self.params.mu_min

        terminate = False
        u = u_init

        for i in range(self.params.max_iter):
            while True:
                loss_old = self.loss(u)

                u_new, linloss = self.solve_linearized_subproblem(u, mu ** -1)

                loss_new = self.loss(u_new)

                diff_u = np.sqrt(((u - u_new) ** 2).sum())

                diff_loss = loss_old - loss_new
                diff_lin = loss_old - linloss

                logging.info("L(uk) = {}, L(uk+1) = {}, diff_u = {}, mu = {}.".format(loss_old, loss_new, diff_u, mu))

                if diff_u <= self.params.eps:
                    terminate = True
                    break

                if mu >= 1e6:
                    terminate = True
                    break

                # Accept if decrease is sufficiently large.
                if diff_loss >= self.params.sigma * diff_lin:
                    mu = max(self.params.mu_min, mu / self.params.tau)

                    u = u_new
                    break
                else:
                    mu = self.params.tau * mu

            loss = self.loss(u)
            losses.append(loss)

            logging.info("Iteration {}: {}".format(i, loss))

            if terminate:
                break

        return u
