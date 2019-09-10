import torch

import modelbased.problems.mnist_classification.svmova_nn as svmova
import modelbased.problems.mnist_classification.misc as misc
import modelbased.utils.results
import modelbased.utils.evaluate
import modelbased.utils.misc
import modelbased.utils.yaml


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

    # TODO: Change to new zero_one method signature.

    # On training data.
    # correct, num_samples = modelbased.utils.evaluate.zero_one(classificator, trainloader)
    # test_results['zero_one_train'] = "{}/{}".format(correct, num_samples)

    # On test data.
    # correct, num_samples = modelbased.utils.evaluate.zero_one(classificator, testloader)
    # test_results['zero_one_test'] = "{}/{}".format(correct, num_samples)

    # modelbased.utils.yaml.write(test_results)


def evaluate_from_file(name, train_samples=50, batch_size=10):
    """
    Evaluate the model in the given result yaml file.

    :param name: Path of yaml result file.
    """
    train_results = modelbased.utils.results.Results(**modelbased.utils.yaml.load(name))

    trainloader, testloader, _, _ = misc.data(train_samples, train_samples, batch_size, batch_size)

    evaluate(train_results, modelbased.utils.misc.append_time(train_results.name), trainloader, testloader)


def restore(train_results):
    """
    :param train_results: Results object.

    :return: Instance of SVM_OVA with model parameters loaded from train_results.
    """
    params = torch.Tensor(train_results.model_parameters)

    classificator = svmova.SVM_OVA()
    classificator.net.params = params

    return classificator
