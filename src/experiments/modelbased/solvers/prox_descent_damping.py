import logging
import modelbased.solvers.utils

logger = logging.getLogger(__name__)


# TODO: Use Exceptions for early termination.
class ProxDescentDamping:
    def __init__(self, params, tensor_type='numpy', verbose=False):
        """
        :param params: Object with parameters:

            max_iter
            eps

            mu_min
            tau
            sigma

        :param tensor_type: Type of the Tensors used in the algorithm: 'numpy' or 'pytorch'.
        :param verbose: If True, enables logging on INFO level.
        """
        self.params = params
        self.tensor_type = tensor_type
        self.verbose = verbose

    def run(self, u_init, loss, solve_linearized_subproblem, callback=None):
        """
        :param u_init: Initial parameter guess.
        :param loss: Loss function h(c(u)) + r(u).
        :param solve_linearized_subproblem: Function which solves the linearized subproblem.

            Parameters: u_new, linloss = solve_linearized_subproblem(u, mu).

            Where mu is the weight factor for the proximal term.

        :param callback: Optional callback called in each iteration.

        :return: Solution u, list of losses (if max_iter > 1).
        """
        t, dot, sqrt = modelbased.solvers.utils.ttype(self.tensor_type)

        losses = []

        mu = self.params.mu_min

        terminate = False
        u = u_init
        accepted_mu = None

        for i in range(self.params.max_iter):
            num_subproblem = 0

            while True:
                loss_old = loss(u)

                u_new, linloss = solve_linearized_subproblem(u, mu)

                loss_new = loss(u_new)

                diff_u = sqrt(((u - u_new) ** 2).sum())

                diff_loss = loss_old - loss_new
                diff_lin = loss_old - linloss

                if self.verbose:
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
                    accepted_mu = mu
                    mu = max(self.params.mu_min, mu / self.params.tau)

                    u = u_new
                    break
                else:
                    mu = self.params.tau * mu

                num_subproblem += 1

            loss_value = loss(u)
            losses.append(loss_value)

            if self.verbose:
                logger.info("Iteration {}/{}: {:.6f}".format(i, self.params.max_iter, loss_value))

            if terminate:
                break

            if callback:
                callback(u_new=u, tau=accepted_mu)

        return u, losses
