import click
import torch
import logging

import modelbased.data.utils
import modelbased.utils.trainrun
import modelbased.utils.results
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.problems.mnist_classification.misc
import modelbased.problems.mnist_classification.prox_linear as prox_linear

logger = logging.getLogger(__name__)


class FixedStepsize(prox_linear.SVM_OVA_ProxLinear):
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


@click.command()
@click.option('--train-samples', required=True, type=int, help='Set to -1 to use all training data.')
@click.option('--num-epochs', required=True, type=int)
@click.option('--batch-size', required=True, type=int)
@click.option('--lam', required=True, type=int)
@click.option('--step-size', required=True, type=int)
def argrun(train_samples, num_epochs, batch_size, lam, step_size):
    """
    Train MNIST classificator using prox-linear with fixed step size.
    """
    if train_samples == -1:
        train_samples = None

    trainloader, testloader, train_size, test_size = modelbased.problems.mnist_classification.misc.data(train_samples,
                                                                                                        train_samples,
                                                                                                        batch_size,
                                                                                                        batch_size)

    classificator = FixedStepsize()
    results = classificator.run(trainloader, data_size=train_size, num_epochs=num_epochs, lam=lam, step_size=step_size)
    modelbased.utils.yaml.write_result(results)
