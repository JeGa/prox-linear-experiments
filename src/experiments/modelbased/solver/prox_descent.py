import numpy as np
import logging


class ProxDescent:
    def __init__(self, params, loss, solve_linearized_subproblem):
        """
        :param params: Object with parameters:
            max_iter
            mu_min
            tau
            sigma
            eps

        :param loss: Loss function l(u).
        :param solve_linearized_subproblem: Function which solves the linearized subproblem.
            With parameters  solve_linearized_subproblem(u, mu).
        """
        self.params = params
        self.loss = loss
        self.solve_linearized_subproblem = solve_linearized_subproblem

    def prox_descent(self, u_init, verbose=False):
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

                if verbose:
                    logging.info(
                        "L(uk)={:.6f}, L(uk+1)={:.6f}, diff_u={:.6f}, mu={}.".format(loss_old, loss_new, diff_u, mu))

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

            if verbose:
                logging.info("Iteration {}/{}: {:.6f}".format(i, self.params.max_iter, loss))

            if terminate:
                break

        return u, losses
