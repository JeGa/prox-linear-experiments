import logging

import modelbased.solvers.utils

logger = logging.getLogger(__name__)


class TerminationException(Exception):
    pass


class ProxDescentLinesearch:
    def __init__(self, params, tensor_type='numpy', verbose=False):
        """
        :param params: Object with parameters:

            max_iter
            eps

            proximal_weight > 0

            Linesearch parameters:
                gamma in (0,1)
                delta in (0,1)
                eta_max > 0

        :param tensor_type: Type of the Tensors used in the algorithm: 'numpy' or 'pytorch'.
        :param verbose: If True, enables logging on INFO level.
        """
        self.params = params
        self.tensor_type = tensor_type
        self.verbose = verbose

    def run(self, u_init, loss, solve_linearized_subproblem):
        """
        :param u_init: Initial parameter guess.
        :param loss: Loss function h(c(u)) + f0(u).
        :param solve_linearized_subproblem: Function which solves the linearized subproblem.

            Parameters: u_new, linloss = solve_linearized_subproblem(u, mu).

            Where mu is the weight factor for the proximal term.

        :return: Solution u, list of losses (if max_iter > 1).
        """
        t, dot, sqrt = modelbased.solvers.utils.ttype(self.tensor_type)

        losses = []

        u = u_init

        try:
            for i in range(self.params.max_iter):
                # Solve model function subproblem approximately such that delta(y, u) < 0.
                y, linloss = solve_linearized_subproblem(u, self.params.proximal_weight)

                # Should be < 0.
                delta = linloss - loss(u)

                # Check for termination.
                if sqrt(((y - u) ** 2).sum()) <= self.params.eps:
                    raise TerminationException("Subproblem solver reached maximum precision.")

                u, eta = self._linesearch(u, y, delta, loss)

                loss_value = loss(u)
                losses.append(loss_value)

                if self.verbose:
                    logger.info(
                        "Iteration [{}/{}], loss={:.6f}, linloss={:.6f}, eta={:.6f}.".format(i, self.params.max_iter,
                                                                                             loss_value,
                                                                                             linloss, eta))
        except TerminationException as e:
            logging.info("Early termination: " + str(e))

        return u, losses

    def _linesearch(self, u, y, delta, loss):
        j = 0

        while True:
            eta = self.params.eta_max * self.params.delta ** j

            u_new = u + eta * (y - u)

            if self.verbose:
                logger.info("Linesearch: eta={:.6f}.".format(eta))

            if self._linesearch_condition(u_new, u, delta, eta, loss):
                break

            if eta <= 1e-8:
                raise TerminationException("Linesearch reached maximum precision.")

            j += 1

        return u_new, eta

    def _linesearch_condition(self, u_new, u, delta, eta, loss):
        if loss(u_new) <= loss(u) + self.params.gamma * eta * delta:
            return True
        else:
            return False
