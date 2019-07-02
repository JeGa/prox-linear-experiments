import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

import modelbased.utils.misc
import modelbased.utils.evaluate
import modelbased.utils.results
import modelbased.utils.yaml
import modelbased.data.mnist as mnist_data
import modelbased.problems.mnist_classification.svmova_nn as svmova_nn


def imshow_grid(x, num_images=16):
    plt.figure()
    plt.imshow(np.transpose(torchvision.utils.make_grid(x[:num_images], padding=5, normalize=True), (1, 2, 0)))
    plt.show()


def sql2norm(x):
    return (x ** 2).sum().item()


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
