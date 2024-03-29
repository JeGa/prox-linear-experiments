import torch
import torch.random
import logging

import modelbased.data.spirals
import modelbased.data.utils
import modelbased.utils.misc
import modelbased.utils.trainrun
import modelbased.utils.yaml
import modelbased.utils.results
import modelbased.solvers.gradient_descent
import modelbased.problems.spirals_classification.logistic_regression_nn as logreg

logger = logging.getLogger(__name__)


class BatchGradient(logreg.LogisticRegression):
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

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('full-batch-gradient-descent'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'full batch gradient descent with fixed step size.'
            },
            train_dataset={
                'name': kwargs['data_name'],
                'size': data_size
            },
            loss={'batch': [batch_loss, [i for i in range(0, len(batch_loss) * data_size, data_size)]]},
            parameters={**vars(params), 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


class StochasticGradient(logreg.LogisticRegression):
    def run(self, trainloader, **kwargs):
        params = modelbased.utils.misc.Params(max_iter=1, sigma=0.12)  # Only one gradient step per mini-batch.

        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size

        mini_batch_loss = []  # Loss per mini-batch over mini-batch samples.
        batch_loss = []  # Loss per mini-batch step over all samples.

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size)

        # Initial loss.
        init_loss = self.loss(self.net.params, x_all, y_all).item()
        batch_loss.append(init_loss)
        mini_batch_loss.append(init_loss)

        def step_fun(x, yt):
            u = self.net.params

            def f(_u):
                return self.loss(_u, x, yt).item()

            def G(_u):
                return self.grad(_u, x, yt)

            u, _losses = modelbased.solvers.gradient_descent.fixed_stepsize(u, f, G, params, verbose=False)

            mini_batch_loss.extend(_losses)

            self.net.params = u

            batch_loss.append(self.loss(u, x_all, y_all).item())

            return False

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                      interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('stochastic-gradient'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'stochastic gradient descent with fixed step size.'
            },
            train_dataset={
                'name': kwargs['data_name'],
                'size': data_size
            },
            loss={
                # Loss per mini-batch step over mini-batch samples.
                'mini-batch': [
                    mini_batch_loss, [i for i in range(0, len(mini_batch_loss) * batch_size, batch_size)]],
                # Loss per mini-batch over all samples.
                'batch': [
                    batch_loss, [i for i in range(0, len(batch_loss) * batch_size, batch_size)]]
            },
            parameters={**vars(params), 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


def batch(data_name, u_init=None):
    loader, data_size = modelbased.data.spirals.load(data_name)
    classificator = BatchGradient()

    if u_init is not None:
        classificator.net.params = u_init

    u_init = classificator.net.params

    train_results = classificator.run(loader, data_name=data_name, data_size=data_size, num_epochs=5)

    return u_init, train_results


def stochastic(data_name, u_init=None):
    loader, data_size = modelbased.data.spirals.load(data_name, batch_size=50)
    classificator = StochasticGradient()

    if u_init is not None:
        classificator.net.params = u_init

    u_init = classificator.net.params

    train_results = classificator.run(loader, data_name=data_name, data_size=data_size, num_epochs=5)

    return u_init, train_results


def run():
    torch.random.manual_seed(1234)

    u_init, batch_train_results = batch('datasets/binary-spirals/5000')
    u_init, stochastic_train_results = stochastic('datasets/binary-spirals/5000', u_init)

    modelbased.utils.yaml.write_result(batch_train_results)
    modelbased.utils.yaml.write_result(stochastic_train_results)
