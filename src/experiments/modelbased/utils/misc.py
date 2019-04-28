import datetime
import os
import torch
import torchvision.utils
import numpy as np
import math
from matplotlib import pyplot as plt

import modelbased.utils.global_config as cfg


class Params:
    def __init__(self, **params):
        self.__dict__.update(params)


def make_folders():
    for item in cfg.folders.values():
        if not os.path.exists(item):
            os.mkdir(item)


def plot_losses(filename, results):
    filename = append_time(filename)

    parameters_text = ''
    for key, value in results['parameters'].items():
        parameters_text += os.linesep + "{}={}".format(key, value)

    loss = results['loss']

    markevery = len(loss) / 10

    plt.figure()

    plt.plot(range(len(loss)), loss, linewidth=0.4, marker='s', markevery=markevery, markerfacecolor='none')

    plt.text(0.05, 0, parameters_text,
             horizontalalignment='left', verticalalignment='top', transform=plt.gcf().transFigure)

    plt.title(filename)
    plt.subplots_adjust(bottom=0.15)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    filepath = os.path.join(cfg.folders['plots'], filename + '.pdf')

    plt.savefig(filepath, bbox_inches='tight')


def plot_grid(x, y, yt, nrow=6):
    """
    Plot the given x images in a grid with the corresponding predicted and ground truth labels.

    :param x: Torch tensor with shape = (n, channels, ysize, xsize).
    :param y: Torch tensor with shape = (c).
    :param yt: Torch tensor with shape = (c).
    :param nrow: Number of images per row.
    """
    n = x.size(0)

    if n < nrow:
        nrow = n

    img_data = np.transpose(torchvision.utils.make_grid(x, nrow=nrow, padding=0, normalize=True), (1, 2, 0))

    plt.figure()

    plt.imshow(img_data)
    plt.tight_layout()

    grid_y = math.ceil(n / nrow)

    for i in range(1, grid_y + 1):
        for j in range(1, nrow + 1):
            plt.text(j * 28 - 4, i * 28 - 7, str(yt[(i - 1) * nrow + j - 1].item()), size=16, color='red')
            plt.text(j * 28 - 4, i * 28 - 1, str(y[(i - 1) * nrow + j - 1].item()), size=16, color='yellow')

            if (i - 1) * nrow + j == n:
                break

    filename = append_time('mnist_results')
    plt.savefig(os.path.join(cfg.folders['plots'], filename + '.pdf'))


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
