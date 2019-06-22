import logging
import modelbased.solvers.utils

logger = logging.getLogger(__name__)


class ProxDescentFixed:
    def __init__(self, params, tensor_type='numpy', verbose=False):
        """
        :param params: Object with parameters:

            max_iter: Number of prox-linear iterations.
            sigma: Step size (proximal term weight factor).

        :param tensor_type: Type of the Tensors used in the algorithm: 'numpy' or 'pytorch'.
        :param verbose: If True, enables logging on INFO level.
        """
        self.params = params
        self.tensor_type = tensor_type
        self.verbose = verbose

    def run(self, u_init, loss, solve_subproblem):
        """
        :param u_init: Initial parameter guess.
        :param loss: Loss function h(c(u)) + r(u).
        :param solve_subproblem: Function which solves the linearized subproblem.

            u_new, linloss = solve_subproblem(u, sigma).

            Where sigma is the weight factor for the proximal term.

        :return: Solution u, list of losses (if max_iter > 1).
        """
        t, dot, sqrt = modelbased.solvers.utils.ttype(self.tensor_type)

        losses = []

        u = u_init

        for i in range(self.params.max_iter):
            u, _ = solve_subproblem(u, self.params.sigma)

            loss_value = loss(u)
            losses.append(loss_value)

            if self.verbose:
                logger.info("Iteration {}/{}: {:.6f}".format(i, self.params.max_iter, loss_value))

        return u, losses
