import click

import modelbased.problems.mnist_classification.misc
import problems.mnist_classification.prox_linear_fixed as plfixed
import modelbased.utils.yaml


@click.command('mnist-cls-pl-fixed')
@click.option('--train-samples', required=True, type=int, help='Set to -1 to use all training data.')
@click.option('--num-epochs', required=True, type=int)
@click.option('--batch-size', required=True, type=int)
@click.option('--lam', required=True, type=int)
@click.option('--step-size', required=True, type=int)
def fixed(train_samples, num_epochs, batch_size, lam, step_size):
    """
    Train MNIST classificator using prox-linear with fixed step size.
    """
    if train_samples == -1:
        train_samples = None

    trainloader, testloader, train_size, test_size = modelbased.problems.mnist_classification.misc.data(train_samples,
                                                                                                        train_samples,
                                                                                                        batch_size,
                                                                                                        batch_size)

    classificator = plfixed.FixedStepsize()
    results = classificator.run(trainloader, data_size=train_size, num_epochs=num_epochs, lam=lam, step_size=step_size)
    modelbased.utils.yaml.write_result(results)
