import datetime
import os
import torch
from matplotlib import pyplot as plt


class Params:
    def __init__(self, **params):
        self.__dict__.update(params)


def plot_losses(losses, filename):
    folder = 'modelbased/results/plots'

    plt.figure()
    plt.plot(range(len(losses)), losses, linewidth=0.4)

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)
    plt.title(filename)

    filepath = os.path.join(folder, filename + '.pdf')

    plt.savefig(filepath, bbox_inches='tight')


def append_time(name):
    return name + '_' + datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S')


def one_hot(x, num_classes, hot=1, off=0):
    """
    :param x: Torch tensor with label for one sample.
    :param num_classes: Integer with total number of classes.
    :param hot: Scalar indicating hot class.
    :param off: Scalar indicating all not hot classes.

    :return: Torch tensor of shape (classes) with one-hot encoding.
    """
    onehot = off * torch.ones(num_classes)
    onehot[x] = hot

    return onehot
