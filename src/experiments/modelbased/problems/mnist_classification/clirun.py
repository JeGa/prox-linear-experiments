import click
import numpy as np
import torch

import modelbased.problems.mnist_classification.misc
import modelbased.problems.mnist_classification.prox_linear_fixed as plfixed
import modelbased.problems.mnist_classification.sgd as sgdfixed
import modelbased.utils.yaml


def seed():
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)


@click.command('mnist-cls-pl-fixed')
@click.option('--train-samples', required=True, type=int, help='Set to -1 to use all training data.')
@click.option('--num-epochs', required=True, type=int)
@click.option("--num-samples", required=True, type=int)
@click.option('--batch-size', required=True, type=int)
@click.option('--lam', required=True, type=float)
@click.option('--step-size', required=True, type=float)
@click.option('--evaluate', is_flag=True)
def pl_fixed(train_samples, num_samples, num_epochs, batch_size, lam, step_size, evaluate):
    """
    Train MNIST classificator using prox-linear with fixed step size.
    """
    if train_samples == -1:
        train_samples = None

    seed()

    trainloader, testloader, train_size, test_size = modelbased.problems.mnist_classification.misc.data(train_samples,
                                                                                                        train_samples,
                                                                                                        batch_size,
                                                                                                        batch_size)

    classificator = plfixed.FixedStepsize()
    results = classificator.run(trainloader, data_size=train_size, num_epochs=num_epochs, num_samples=num_samples,
                                lam=lam, step_size=step_size, evaluate=evaluate)
    modelbased.utils.yaml.write_result(results)


@click.command('mnist-cls-sgd-fixed')
@click.option('--train-samples', required=True, type=int, help='Set to -1 to use all training data.')
@click.option('--num-epochs', required=True, type=int)
@click.option("--num-samples", required=True, type=int)
@click.option('--batch-size', required=True, type=int)
@click.option('--lam', required=True, type=float)
@click.option('--step-size', required=True, type=float)
@click.option('--evaluate', is_flag=True)
def sgd_fixed(train_samples, num_samples, num_epochs, batch_size, lam, step_size, evaluate):
    """
    Train MNIST classificator using sgd with fixed step size.
    """
    if train_samples == -1:
        train_samples = None

    seed()

    trainloader, testloader, train_size, test_size = modelbased.problems.mnist_classification.misc.data(train_samples,
                                                                                                        train_samples,
                                                                                                        batch_size,
                                                                                                        batch_size)

    classificator = sgdfixed.SGD()
    results = classificator.run(trainloader, data_size=train_size, num_epochs=num_epochs, num_samples=num_samples,
                                lam=lam, step_size=step_size, evaluate=evaluate)
    modelbased.utils.yaml.write_result(results)
