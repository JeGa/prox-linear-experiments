import torch
import logging

import modelbased.utils.misc
import modelbased.solvers.utils
import modelbased.solvers.projected_gradient
import modelbased.problems.mnist_classification.svmova_nn as svmova_nn
import modelbased.problems.mnist_classification.misc as misc

logger = logging.getLogger(__name__)


class SVM_OVA_ProxLinear(svmova_nn.SVM_OVA):
    def solve_subproblem(self, uk, tau, x, y_targets, lam, verbose=True, stopping_condition=None):
        """
        Solve the inner linearized subproblem using gradient ascent on the dual problem.

        :param uk: shape = (d, 1).
        :param tau: Proximal weight of the subproblem.

        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.

        :param lam: Scalar regularizer weight factor.

        :param verbose: If True, prints primal and dual loss before and after optimization.

        :param stopping_condition: A function expecting the following parameters:

                stopping_condition(u, linloss)

            It can be used if the subproblem only should be optimized approximately.
            It is evaluated at each iteration of projected gradient descent. If it returns True, it stops optimizing.

        :return: Solution argmin L_lin(u), loss of linearized subproblem without proximal term.
        """
        n = x.size(0)
        c = y_targets.size(1)

        # (c*n, d).
        J = self.net.Jacobian(uk, x)

        # (n, c).
        yhat = self.net.f(uk, x) - J.mm(uk).view(n, c)

        # Primal problem.
        def P(_u, proximal_term=True):
            _linloss = self.h(J.mm(_u).view(n, c) + yhat, y_targets) + self.reg(_u, lam)

            if proximal_term:
                return _linloss + tau * 0.5 * misc.sql2norm(_u - uk)
            else:
                return _linloss

        # Dual problem.
        def D(_p):
            dloss = (y_targets * _p.view(n, c)).sum() + (
                    1 / (2 * (lam + tau)) * misc.sql2norm(-J.t().mm(_p) + tau * uk) - yhat.view(1, -1).mm(_p))

            return dloss.item()

        def gradD(_p):
            return y_targets.view(-1, 1) + (1 / (lam + tau)) * J.mm(J.t().mm(_p) - tau * uk) - yhat.view(-1, 1)

        def proj(_x):
            # (c*n, 1).
            x_aug = y_targets.view(-1, 1) * n * _x

            x_aug_projected = modelbased.solvers.utils.torch_proj(x_aug, -1, 0)

            x_projected = x_aug_projected * y_targets.view(-1, 1) / n

            return x_projected

        def dual_to_primal(_p):
            return (1 / (lam + tau)) * (tau * uk - J.t().mm(_p))

        # (c*n, 1).
        p = proj(torch.empty(c * n, 1, device=uk.device))

        params = modelbased.utils.misc.Params(
            max_iter=5000,
            eps=1e-6,
            beta=0.3,
            gamma=1e-2,
            tau=1e-2,
            sigmamin=1e-6)

        if not stopping_condition:
            stopcond = None
        else:
            def stopcond(_p):
                _linloss = P(dual_to_primal(_p), proximal_term=False)
                return stopping_condition(uk, _linloss)

        p_new, losses = modelbased.solvers.projected_gradient.armijo(p, D, gradD, proj, params,
                                                                     tensor_type='pytorch', stopping_condition=stopcond)

        # Get primal solution.
        u = dual_to_primal(p_new)

        if verbose:
            logger.info("D(p0): {}".format(D(p)))
            logger.info("D(p*): {}".format(D(p_new)))

            logger.info("P(uk): {}".format(P(uk)))
            logger.info("P(u*): {}".format(P(u)))

        # Loss of the linearized sub-problem without the proximal term.
        linloss = P(u, proximal_term=False)

        return u.detach(), linloss

    def run(self, loader, **kwargs):
        raise NotImplementedError()
