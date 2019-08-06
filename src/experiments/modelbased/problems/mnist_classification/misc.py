import matplotlib.pyplot as plt
import numpy as np
import torchvision

import modelbased.data.mnist as mnist_data
import modelbased.utils.misc


def image_grid(x, file):
    plt.figure()
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    plt.imshow(np.transpose(torchvision.utils.make_grid(x, padding=2, pad_value=255, normalize=True),
                            (1, 2, 0)))
    plt.savefig(file, bbox_inches='tight', pad_inches=0)


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
