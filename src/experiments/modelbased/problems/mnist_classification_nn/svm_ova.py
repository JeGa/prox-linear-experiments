import torch
from torch.nn import functional as F
import logging

import modelbased.data.mnist as mnist_data
import modelbased.data.utils
import modelbased.problems.mnist_classification_nn.nn as mnist_nn
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.utils.trainrun
import modelbased.utils.evaluate
import modelbased.solvers.utils
import modelbased.solvers.projected_gradient
import modelbased.solvers.prox_descent_damping

import modelbased.solvers.prox_descent_linesearch

logger = logging.getLogger(__name__)


def sql2norm(x):
    return (x ** 2).sum().item()


class SVM_OVA:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Using device: {}.".format(device))

        self.net = mnist_nn.SimpleConvNetMNIST(F.relu, device)

    @staticmethod
    def L(v, y_targets):
        """
        Apply the one-versus-all SVM loss.

        :param v: shape = (n, c).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.

        :return: L(v), shape = (n, 1).
        """
        _max = torch.max(torch.zeros(1, device=v.device), 1 - v * y_targets)

        return _max.sum(1, keepdim=True)

    @classmethod
    def h(cls, a, y_targets, n=None):
        """
        :param a: shape = (n, c).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param n: The factor the sum is multiplied with (1/N).

        :return: h(a) scalar.
        """
        if not n:
            n = a.size(0)

        return (1 / n) * cls.L(a, y_targets).sum().item()

    def c(self, u, x, set_param=False):
        """
        :param u: shape (d, 1).
        :param x: shape = (n, channels, ysize, xsize).
        :param set_param: If True, set the network parameters, if False just evaluate and keep the old parameters.

        :return: c(u), shape = (n, c).
        """
        return self.net.f(u, x, set_param)

    @staticmethod
    def reg(u, lam):
        """
        Squared l2-norm regularizer.

        :param u: shape = (d, 1).
        :param lam: Scalar regularizer weight factor.

        :return: lam * 0.5 * norm(u)**2 as scalar.
        """
        return lam * 0.5 * sql2norm(u)

    def loss(self, u, x, y_targets, lam):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, channels, ysize, xsize).
        :param y_targets: shape = (n, c) in one-hot encoding with "hot" = 1, else -1.
        :param lam: Scalar regularizer weight factor.

        :return: L(u)
        """
        return self.h(self.c(u, x), y_targets) + self.reg(u, lam)

    def solve_linearized_subproblem(self, uk, tau, x, y_targets, lam, verbose=True, stopping_condition=None):
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
                return _linloss + tau * 0.5 * sql2norm(_u - uk)
            else:
                return _linloss

        # Dual problem.
        def D(_p):
            dloss = (y_targets * _p.view(n, c)).sum() + (
                    1 / (2 * (lam + tau)) * sql2norm(-J.t().mm(_p) + tau * uk) - yhat.view(1, -1).mm(_p))

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

    def _predict(self, u, x):
        x = x.to(self.net.device)

        y = self.net.f(u, x)

        return y.argmax(1)

    def predict(self, x):
        return self._predict(self.net.params, x)

    def run(self, loader):
        raise NotImplementedError()


class ProxDescentFixed(SVM_OVA):
    def run(self, trainloader):
        num_epochs = 2
        lam = 0.0
        tau = 0.1

        def step_fun(x, yt):
            u = self.net.params

            u_new, _ = self.solve_linearized_subproblem(u, tau, x, yt, lam, verbose=False)

            loss = self.loss(u_new, x, yt, lam)

            self.net.params = u_new

            return [loss]

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        total_losses = modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                     interval_fun=interval_fun, interval=1)

        results = {
            'name': modelbased.utils.misc.append_time('prox_descent_fixed'),
            'type': 'train',

            'description': {
                'loss_function': 'SVM one-versus-all',
                'forward_model': 'shallow conv net',
                'optimization': 'ProxDescent with fixed proximal weight, '
                                'projected dual ascent with armijo on the subproblems.'
            },

            'loss': total_losses,

            'parameters': None,

            'info': {
                'epochs': num_epochs,
                'lambda': lam
            },

            'model_parameters': self.net.params.numpy().tolist()
        }

        return results


class ProxDescentLinesearch(SVM_OVA):
    def run(self, trainloader):
        params = modelbased.utils.misc.Params(
            max_iter=20,
            eps=1e-12,
            proximal_weight=0.1,
            gamma=0.6,
            delta=0.5,
            eta_max=2)

        num_epochs = 1
        lam = 0.01

        proxdescent = modelbased.solvers.prox_descent_linesearch.ProxDescentLinesearch(params,
                                                                                       tensor_type='pytorch',
                                                                                       verbose=True)

        def step_fun(x, yt):
            u_init = self.net.params

            def loss(u):
                return self.loss(u, x, yt, lam)

            def subsolver(u, tau):
                def stopcond(uk, linloss):
                    if linloss - loss(uk) < 0:
                        return True
                    else:
                        return False

                return self.solve_linearized_subproblem(u, tau, x, yt, lam, verbose=False,
                                                        stopping_condition=None)  # TODO stopcond.

            u_new, losses = proxdescent.run(u_init, loss, subsolver)

            self.net.params = u_new

            return losses

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        total_losses = modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                     interval_fun=interval_fun, interval=1)

        results = {
            'name': modelbased.utils.misc.append_time('prox_descent_linesearch'),
            'type': 'train',

            'description': {
                'loss_function': 'SVM one-versus-all',
                'forward_model': 'shallow conv net',
                'optimization': 'ProxDescent with linesearch, '
                                'projected dual ascent with armijo on the subproblems.'
            },

            'loss': total_losses,

            'parameters': params.__dict__,

            'info': {
                'epochs': num_epochs,
                'lambda': lam
            },

            'model_parameters': self.net.params.numpy().tolist()
        }

        return results


class ProxDescentDamping(SVM_OVA):
    def run(self, trainloader):
        params = modelbased.utils.misc.Params(
            max_iter=10,
            eps=1e-6,
            mu_min=1,
            tau=5,
            sigma=0.7)

        num_epochs = 1
        lam = 0.0

        proxdescent = modelbased.solvers.prox_descent_damping.ProxDescentDamping(params, tensor_type='pytorch',
                                                                                 verbose=True)

        def step_fun(x, yt):
            u_init = self.net.params

            def loss(u):
                return self.loss(u, x, yt, lam)

            def subsolver(u, tau):
                return self.solve_linearized_subproblem(u, tau, x, yt, lam, verbose=False)

            u_new, losses = proxdescent.run(u_init, loss, subsolver)

            self.net.params = u_new

            return losses

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        total_losses = modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                     interval_fun=interval_fun, interval=1)

        results = {
            'name': modelbased.utils.misc.append_time('prox_descent_damping'),
            'type': 'train',

            'description': {
                'loss_function': 'SVM one-versus-all',
                'forward_model': 'shallow conv net',
                'optimization': 'ProxDescent with damping, '
                                'projected dual ascent with armijo on the subproblems.'
            },

            'loss': total_losses,

            'parameters': params.__dict__,

            'info': {
                'epochs': num_epochs,
                'lambda': lam
            },

            'model_parameters': self.net.params.numpy().tolist()
        }

        return results


def data(train_samples, batchsize):
    def transform(x):
        return modelbased.utils.misc.one_hot(x, 10, hot=1, off=-1)

    trainloader, testloader, _, _, _ = mnist_data.load('datasets/mnist',
                                                       train_samples, train_samples,
                                                       batchsize, batchsize,
                                                       one_hot_encoding=transform)
    return trainloader, testloader


def train():
    train_samples = None
    batchsize = 10

    trainloader, testloader = data(train_samples, batchsize)

    classificator = ProxDescentFixed()
    train_results = classificator.run(trainloader)

    modelbased.utils.misc.plot_loss(train_results)
    modelbased.utils.yaml.write(train_results)

    evaluate(train_results, train_results['name'], trainloader, testloader)


def restore(train_results):
    params = torch.Tensor(train_results['model_parameters'])

    classificator = SVM_OVA()
    classificator.net.params = params

    return classificator


def evaluate(train_results, name, trainloader, testloader):
    classificator = restore(train_results)

    test_results = train_results

    test_results['name'] = name
    test_results['type'] = 'test'

    # On training data.
    modelbased.utils.evaluate.image_grid(name, classificator, trainloader)
    correct, num_samples = modelbased.utils.evaluate.zero_one(classificator, trainloader)

    test_results['zero_one_train'] = "{}/{}".format(correct, num_samples)

    # On test data.
    modelbased.utils.evaluate.image_grid(name, classificator, testloader)
    correct, num_samples = modelbased.utils.evaluate.zero_one(classificator, testloader)

    test_results['zero_one_test'] = "{}/{}".format(correct, num_samples)

    modelbased.utils.yaml.write(test_results)


def evaluate_from_file(name):
    train_results = modelbased.utils.yaml.load(name)

    train_samples = 50
    batchsize = 10
    trainloader, testloader = data(train_samples, batchsize)

    evaluate(train_results, modelbased.utils.misc.append_time(train_results['name']), trainloader, testloader)
