import logging

import modelbased.problems.mnist_classification.svmova_nn as svmova_nn
import modelbased.utils.misc
import modelbased.solvers.gradient_descent
import modelbased.data.utils
import modelbased.utils.results
import modelbased.utils.trainrun
import modelbased.utils.evaluate

logger = logging.getLogger(__name__)


class SGD(svmova_nn.SVM_OVA):
    def run(self, trainloader, **kwargs):
        lam = kwargs['lam']
        tau = kwargs['step_size']
        num_epochs = kwargs['num_epochs']
        num_samples = kwargs['num_samples']
        data_size = kwargs['data_size']
        evaluate = kwargs['evaluate']
        batch_size = trainloader.batch_size

        params = modelbased.utils.misc.Params(max_iter=1, sigma=tau)  # Only one gradient step per mini-batch.

        seen_samples = 0

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size, self.net.device)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        init_missclassifications = modelbased.utils.evaluate.zero_one_batch(self, x_all, y_all)[0]

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.
        missclassifications = [init_missclassifications]

        def step_fun(x, yt):
            nonlocal seen_samples

            u = self.net.params

            def f(_u):
                return self.loss(_u, x, yt, lam)

            def G(_u):
                return self.grad(_u, x, yt, lam)

            u_new, _ = modelbased.solvers.gradient_descent.fixed_stepsize(u, f, G, params, verbose=False)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_losses.append(self.loss(u_new, x, yt, lam))

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            # Missclassifications.
            if evaluate:
                missclassifications.append(modelbased.utils.evaluate.zero_one_batch(self, x_all, y_all)[0])

            seen_samples += batch_size
            if seen_samples == num_samples:
                return True

            return False

        def interval_fun(epoch, iteration, batch_iteration):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, mini_batch_losses[-1]))

        modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                      interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-sg-fixed'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'stochastic gradient descent with fixed step size.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))],
                'missclassifications': [missclassifications, list(range(len(missclassifications)))]
            },
            parameters={'tau': tau, 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results
