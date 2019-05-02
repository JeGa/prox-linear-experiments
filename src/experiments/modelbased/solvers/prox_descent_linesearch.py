import logging

import modelbased.solvers.utils

logger = logging.getLogger(__name__)


class TerminationException(Exception):
    pass


class ProxDescentLinesearch:
    def __init__(self, params, loss, solve_linearized_subproblem):
        """

        :param params: Object with parameters:

            max_iter
            eps

            proximal_weight > 0

            Linesearch parameters:
                gamma in (0,1)
                delta in (0,1)
                eta_max > 0

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

        u = u_init

        try:
            for i in range(self.params.max_iter):
                # Solve model function subproblem approximately such that delta(y, u) < 0.
                y, linloss = self.solve_linearized_subproblem(u, self.params.proximal_weight)

                # Should be < 0.
                delta = linloss - self.loss(u)

                # Check for termination.
                if sqrt(((y - u) ** 2).sum()) <= self.params.eps:
                    raise TerminationException("Subproblem solver reached maximum precision.")

                u, eta = self._linesearch(u, y, delta, verbose=verbose)

                loss = self.loss(u)
                losses.append(loss)

                if verbose:
                    logger.info(
                        "Iteration [{}/{}], loss={:.6f}, linloss={:.6f}, eta={:.6f}.".format(i, self.params.max_iter,
                                                                                             loss,
                                                                                             linloss, eta))
        except TerminationException as e:
            logging.info("Early termination: " + str(e))

        return u, losses

    def _linesearch(self, u, y, delta, verbose=False):
        j = 0

        while True:
            eta = self.params.eta_max * self.params.delta ** j

            u_new = u + eta * (y - u)

            if verbose:
                logger.info("Linesearch: eta={:.6f}.".format(eta))

            if self._linesearch_condition(u_new, u, delta, eta):
                break

            if eta <= 1e-8:
                raise TerminationException("Linesearch reached maximum precision.")

            j += 1

        return u_new, eta

    def _linesearch_condition(self, u_new, u, delta, eta):
        if self.loss(u_new) <= self.loss(u) + self.params.gamma * eta * delta:
            return True
        else:
            return False
