import logging

import modelbased.solvers.utils

logger = logging.getLogger(__name__)


class ProxDescentDamping:
    def __init__(self, params, loss, solve_linearized_subproblem):
        """
        :param params: Object with parameters:

            max_iter
            eps

            mu_min
            tau
            sigma

        :param loss: Loss function h(c(u)) + f0(u).
        :param solve_linearized_subproblem: Function which solves the linearized subproblem.

            Parameters: u_new, linloss = solve_linearized_subproblem(u, mu).

            Where mu is the weight factor for the proximal term.
        """
        self.params = params
        self.loss = loss
        self.solve_linearized_subproblem = solve_linearized_subproblem

    def run(self, u_init, tensor_type='numpy', verbose=False):
        t, dot, sqrt = modelbased.solvers.utils.ttype(tensor_type)

        losses = []

        loss_init = self.loss(u_init)
        losses.append(loss_init)

        mu = self.params.mu_min

        terminate = False
        u = u_init

        for i in range(self.params.max_iter):
            num_subproblem = 0

            while True:
                loss_old = self.loss(u)

                u_new, linloss = self.solve_linearized_subproblem(u, mu)

                loss_new = self.loss(u_new)

                diff_u = sqrt(((u - u_new) ** 2).sum())

                diff_loss = loss_old - loss_new
                diff_lin = loss_old - linloss

                if verbose:
                    logger.info(
                        "Subproblem {}: L(uk)={:.6f}, L(uk+1)={:.6f}, diff_u={:.6f}, mu={}.".format(num_subproblem,
                                                                                                    loss_old, loss_new,
                                                                                                    diff_u, mu))

                if diff_u <= self.params.eps:
                    terminate = True
                    break

                if mu >= 1e8:
                    terminate = True
                    break

                # Accept if decrease is sufficiently large.
                if diff_loss >= self.params.sigma * diff_lin:
                    mu = max(self.params.mu_min, mu / self.params.tau)

                    u = u_new
                    break
                else:
                    mu = self.params.tau * mu

                num_subproblem += 1

            loss = self.loss(u)
            losses.append(loss)

            if verbose:
                logger.info("Iteration {}/{}: {:.6f}".format(i, self.params.max_iter, loss))

            if terminate:
                break

        return u, losses
