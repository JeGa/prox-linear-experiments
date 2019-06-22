import torch
from torch.nn import functional as F
import torch.random
import logging

import modelbased.data.spirals
import modelbased.data.utils
import modelbased.problems.spirals_classification.nn as spirals_nn
import modelbased.utils.misc
import modelbased.utils.trainrun
import modelbased.utils.yaml
import modelbased.utils.results
import modelbased.solvers.gradient_descent

logger = logging.getLogger(__name__)


class LogisticRegression:
    def __init__(self, m=2):
        """
        :param m: Sample dimension.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using device: {}.".format(device))

        layers = [m, 12, 12, 12, 2]
        self.net = spirals_nn.SimpleFullyConnected(layers, F.relu, device)

    @staticmethod
    def L_binary_cross_entropy_per_class(v, y_targets):
        """
        Logistic loss applied per sample and per output class.
        This is the same as pytorchs BCEWithLogitsLoss/binary_cross_entropy_with_logits.

        :param v: shape = (n, 2).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: L(v), shape = (n, 1).
        """
        r = torch.log(1 + torch.exp(-v))
        return 0.5 * (y_targets * r + (1 - y_targets) * (v + r)).sum(1)

    @staticmethod
    def L_binary_cross_entropy(v, y_targets):
        """
        Logistic loss applied per sample.

        :param v: shape = (n, 1).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: L(v), shape = (n, 1).
        """
        # TODO
        # torch.log(1 + torch.exp(-v))[y_targets == 1].unsqueeze(1)
        raise NotImplementedError()

    @classmethod
    def h(cls, a, y_targets):
        """
        :param a: shape = (n, 2).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: h(a) torch tensor.
        """
        n = a.size(0)

        return (1 / n) * cls.L_binary_cross_entropy_per_class(a, y_targets).sum()

    def c(self, u, x):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, m).

        :return: c(u) torch tensor, shape = (n, 2).
        """
        return self.net.f(u, x)

    def loss(self, u, x, y_targets):
        """
        :param u: shape = (d, 1).
        :param x: shape = (n, m).
        :param y_targets: shape = (n, 2) in one-hot encoding with "hot" = 1, else 0.

        :return: L(u) torch tensor.
        """
        return self.h(self.c(u, x), y_targets)

    def grad(self, u, x, yt):
        with self.net.param_eval(u):
            self.net.zero_grad()
            loss = self.h(self.net.forward(x), yt)
            loss.backward()

        return self.net.gradient

    def run(self, loader, **kwargs):
        raise NotImplementedError()


class batchG(LogisticRegression):
    def run(self, trainloader, **kwargs):
        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size  # This is the same as data_size in batch setting.

        x, yt = next(iter(trainloader))

        def f(_u):
            return self.loss(_u, x, yt).item()

        def G(_u):
            return self.grad(_u, x, yt)

        u0 = self.net.params

        batch_loss = [f(u0)]

        params = modelbased.utils.misc.Params(max_iter=num_epochs, sigma=1.5)

        u, batch_loss_noinit = modelbased.solvers.gradient_descent.fixed_stepsize(u0, f, G, params, verbose=True)
        batch_loss += batch_loss_noinit

        self.net.params = u

        # TODO: Move to base class.
        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('full_batch_gradient_descent'),
            type='train',
            description={
                'loss_function': 'Binary cross-entropy (per class).',
                'forward_model': 'Shallow fully connected net.',
                'optimization': 'Full batch gradient descent with fixed step size.'
            },
            train_dataset={
                'name': kwargs['data_name'],
                'size': data_size
            },
            loss={
                'batch': [batch_loss, [i for i in range(0, len(batch_loss) * data_size, data_size)]]
            },
            parameters={**vars(params), 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


class SG(LogisticRegression):
    def run(self, trainloader, **kwargs):
        params = modelbased.utils.misc.Params(max_iter=1, sigma=0.12)  # Only one gradient step per mini-batch.

        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size
        mini_batch_all_loss = []  # Loss per mini-batch step over all samples.
        mini_batch_loss = []  # Loss per mini-batch over mini-batch samples.

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size)

        # Initial loss.
        init_loss = self.loss(self.net.params, x_all, y_all).item()
        mini_batch_all_loss.append(init_loss)
        mini_batch_loss.append(init_loss)

        def step_fun(x, yt):
            u = self.net.params

            def f(_u):
                return self.loss(_u, x, yt).item()

            def G(_u):
                return self.grad(_u, x, yt)

            u, _losses = modelbased.solvers.gradient_descent.fixed_stepsize(u, f, G, params, verbose=False)

            self.net.params = u

            mini_batch_all_loss.append(self.loss(u, x_all, y_all).item())

            return _losses

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        mini_batch_loss += modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                         interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('stochastic_gradient'),
            type='train',
            description={
                'loss_function': 'Binary cross-entropy (per class).',
                'forward_model': 'Shallow fully connected net.',
                'optimization': 'Stochastic gradient descent with fixed step size.'
            },
            train_dataset={
                'name': kwargs['data_name'],
                'size': data_size
            },
            loss={
                # Loss per mini-batch step over mini-batch samples.
                'mini_batch': [
                    mini_batch_loss, [i for i in range(0, len(mini_batch_loss) * batch_size, batch_size)]],
                # Loss per mini-batch over all samples.
                'mini_batch_all': [
                    mini_batch_all_loss, [i for i in range(0, len(mini_batch_all_loss) * batch_size, batch_size)]]
            },
            parameters={**vars(params), 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


def batch(data_name, u_init=None):
    loader, data_size = modelbased.data.spirals.load(data_name)
    classificator = batchG()

    if u_init is not None:
        classificator.net.params = u_init

    u_init = classificator.net.params

    train_results = classificator.run(loader, data_name=data_name, data_size=data_size, num_epochs=5)

    return u_init, train_results


def stochastic(data_name, u_init=None):
    loader, data_size = modelbased.data.spirals.load(data_name, batch_size=50)
    classificator = SG()

    if u_init is not None:
        classificator.net.params = u_init

    u_init = classificator.net.params

    train_results = classificator.run(loader, data_name=data_name, data_size=data_size, num_epochs=5)

    return u_init, train_results


def run():
    torch.random.manual_seed(1234)

    u_init, batch_train_results = batch('datasets/binary-spirals/5000')
    u_init, stochastic_train_results = stochastic('datasets/binary-spirals/5000', u_init)

    modelbased.utils.misc.plot_losses([batch_train_results, stochastic_train_results],
                                      [['batch'], ['mini_batch_all']])

    modelbased.utils.yaml.write_result(batch_train_results)
    modelbased.utils.yaml.write_result(stochastic_train_results)