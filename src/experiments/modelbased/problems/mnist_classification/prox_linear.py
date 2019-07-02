import torch
import logging
import click

import modelbased.data.mnist as mnist_data
import modelbased.data.utils
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.utils.trainrun
import modelbased.utils.evaluate
import modelbased.utils.results
import modelbased.solvers.utils
import modelbased.solvers.projected_gradient
import modelbased.solvers.prox_descent_damping
import modelbased.solvers.prox_descent_linesearch
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


class FixedStepsize(SVM_OVA_ProxLinear):
    def run(self, trainloader, **kwargs):
        # Number of proximal steps per subproblem / modelfunction.
        sub_iterations = 1

        lam = kwargs['lam']
        tau = kwargs['step_size']
        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size, self.net.device)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.
        moreau_grad = []  # Norm of gradient of Moreau envelope of model functions per mini-batch step.

        def step_fun(x, yt):
            # We could also instantiate the ProxLinearFixed class, but this makes no sense here, since we just need to
            # call the solve_subproblem method.
            u = self.net.params

            u_new = None
            for i in range(sub_iterations):
                u_new, _ = self.solve_subproblem(u, tau, x, yt, lam, verbose=False)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_loss = self.loss(u_new, x, yt, lam)

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            # Norm of gradient of Moreau envelope of model functions.
            moreau_grad.append(tau * torch.norm(u_new - u, p=2).item())

            return [mini_batch_loss]

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        mini_batch_losses += modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                           interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-prox-linear-fixed'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'prox-linear with fixed proximal weight, '
                                       'projected dual ascent with armijo on the subproblems.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))],
                'moreau-grad': [moreau_grad, list(range(len(moreau_grad)))]
            },
            parameters={'tau': tau, 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


class Linesearch(SVM_OVA_ProxLinear):
    def run(self, trainloader, **kwargs):
        lam = 0

        # Number of proximal steps per subproblem / modelfunction.
        sub_iterations = 1

        params = modelbased.utils.misc.Params(
            max_iter=sub_iterations,
            eps=1e-12,
            proximal_weight=0.1,
            gamma=0.5,
            delta=0.5,
            eta_max=3)

        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.

        proxdescent = modelbased.solvers.prox_descent_linesearch.ProxDescentLinesearch(
            params, tensor_type='pytorch', verbose=True)

        def step_fun(x, yt):
            u = self.net.params

            def loss(u):
                return self.loss(u, x, yt, lam)

            def subsolver(u, tau):
                # TODO stopcond.
                # def stopcond(uk, linloss):  # TODO.
                #     if linloss - loss(uk) < 0:
                #         return True
                #     else:
                #         return False
                return self.solve_subproblem(u, tau, x, yt, lam, verbose=False, stopping_condition=None)

            u_new, _ = proxdescent.run(u, loss, subsolver)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_loss = self.loss(u_new, x, yt, lam)

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            return [mini_batch_loss]

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        mini_batch_losses += modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                           interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-prox-linear-linesearch'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'prox-linear with linesearch, '
                                       'projected dual ascent with armijo on the subproblems.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))]
            },
            parameters={**vars(params), 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


class Damping(SVM_OVA_ProxLinear):
    def run(self, trainloader, **kwargs):
        lam = 0.0

        # Number of proximal steps per subproblem / modelfunction.
        sub_iterations = 1

        params = modelbased.utils.misc.Params(
            max_iter=sub_iterations,
            eps=1e-6,
            mu_min=1,
            tau=5,
            sigma=0.7)

        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.
        moreau_grad = []  # Norm of gradient of Moreau envelope of model functions per mini-batch step.

        proxdescent = modelbased.solvers.prox_descent_damping.ProxDescentDamping(params,
                                                                                 tensor_type='pytorch',
                                                                                 verbose=True)

        def step_fun(x, yt):
            u = self.net.params

            accepted_tau = None

            def callback(u_new, tau):
                nonlocal accepted_tau
                accepted_tau = tau

            def loss(u):
                return self.loss(u, x, yt, lam)

            def subsolver(u, tau):
                return self.solve_subproblem(u, tau, x, yt, lam, verbose=False)

            u_new, losses = proxdescent.run(u, loss, subsolver, callback=callback)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_loss = self.loss(u_new, x, yt, lam)

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            # Norm of gradient of Moreau envelope of model functions.
            moreau_grad.append(accepted_tau * torch.norm(u_new - u, p=2).item())

            return [mini_batch_loss]

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        mini_batch_losses += modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                           interval_fun=interval_fun, interval=1)
        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-prox-linear-damping'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'prox-linear with damping, '
                                       'projected dual ascent with armijo on the subproblems.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))],
                'moreau-grad': [moreau_grad, list(range(len(moreau_grad)))]
            },
            parameters={**vars(params), 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results


def data(train_samples, test_samples, train_batchsize, test_batchsize):
    """
    :param train_samples: Number of train samples from complete data set.
    :param test_samples: Number of test samples from complete data set.
    :param train_batchsize: Batchsize for train loader.
    :param test_batchsize: Batchsize for test loader.

    :return: Tuple with
        trainloader, testloader: Pytorch data loader, with targets in one-hot encoding.
        train_size, test_size: Number of samples in the respective data set.
    """

    def transform(x):
        return modelbased.utils.misc.one_hot(x, 10, hot=1, off=-1)

    trainloader, testloader, _, _, train_size, test_size, _ = mnist_data.load('datasets/mnist',
                                                                              train_samples, test_samples,
                                                                              train_batchsize, test_batchsize,
                                                                              target_transform_function=transform)

    return trainloader, testloader, train_size, test_size


def restore(train_results):
    """
    :param train_results: Results object.

    :return: Instance of SVM_OVA with model parameters loaded from train_results.
    """
    params = torch.Tensor(train_results.model_parameters)

    classificator = svmova_nn.SVM_OVA()
    classificator.net.params = params

    return classificator


def evaluate(train_results, name, trainloader, testloader):
    """
    Evaluate the model saved in train_results.

    :param train_results: Results object.
    :param name: Name of the new Results object.
    :param trainloader: Training data loader.
    :param testloader:  Test data loader
    """
    classificator = restore(train_results)

    test_results = train_results

    test_results.name = name
    test_results.type = 'test'

    # On training data.
    modelbased.utils.evaluate.image_grid(name, classificator, trainloader)
    correct, num_samples = modelbased.utils.evaluate.zero_one(classificator, trainloader)

    print(correct, num_samples)
    # test_results['zero_one_train'] = "{}/{}".format(correct, num_samples)

    # On test data.
    modelbased.utils.evaluate.image_grid(name, classificator, testloader)
    correct, num_samples = modelbased.utils.evaluate.zero_one(classificator, testloader)

    print(correct, num_samples)
    # test_results['zero_one_test'] = "{}/{}".format(correct, num_samples)
    # modelbased.utils.yaml.write(test_results)


def evaluate_from_file(name):
    """
    Evaluate the model in the given result yaml file.

    :param name: Path of yaml result file.
    """
    train_results = modelbased.utils.results.Results(**modelbased.utils.yaml.load(name))

    train_samples = 50
    batchsize = 10
    trainloader, testloader, _, _ = data(train_samples, train_samples, batchsize, batchsize)

    evaluate(train_results, modelbased.utils.misc.append_time(train_results.name), trainloader, testloader)


@click.command()
@click.option('--train-samples', required=True, type=int, help='Set to -1 to use all training data.')
@click.option('--num-epochs', required=True, type=int)
@click.option('--batch-size', required=True, type=int)
@click.option('--lam', required=True, type=int)
@click.option('--step-size', required=True, type=int)
def argrun_fixed(train_samples, num_epochs, batch_size, lam, step_size):
    """
    Train MNIST classificator using prox-linear with fixed step size.
    """
    if train_samples == -1:
        train_samples = None

    trainloader, testloader, train_size, test_size = data(train_samples, train_samples, batch_size, batch_size)

    classificator = FixedStepsize()
    results = classificator.run(trainloader, data_size=train_size, num_epochs=num_epochs, lam=lam, step_size=step_size)
    modelbased.utils.yaml.write_result(results)


def run():
    argrun_fixed.callback(train_samples=10, num_epochs=1, batch_size=10, lam=0, step_size=15)
    # evaluate_from_file('modelbased/results/data/train/mnist-classification-prox-linear-fixed_01-07-19_10:29:18.yml')
