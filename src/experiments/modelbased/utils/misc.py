import datetime
import torch
import torchvision.utils
import numpy as np
import math
from matplotlib import pyplot as plt

import modelbased.utils.global_config as cfg


class Params:
    def __init__(self, **params):
        self.__dict__.update(params)


import os


def make_folders():
    for item in cfg.folders.values():
        if not os.path.exists(item):
            os.mkdir(item)


# TODO: Use evaltool Plotter ... or just delete plot function.
def plot_losses(results, loss_keys):
    """
    Plots the losses and the additional information from the Result instances in the results parameter and saves them in
    pdf format.

    The additional information is appended as text at the end of the pdf file. This can be used for parameter,
    algorithm and other information.

    :param results: A list of Result instances.
    :param loss_keys: A list of lists with the loss keys to plot for the respective objects in elements.
        Should be the same length as results list. For example:

            results = [r1, r2],
            loss_keys = [['mini_batch', 'batch'], ['mini_batch']], # respective loss keys for r1 and r2.
    """
    if len(results) != len(loss_keys):
        raise ValueError("results and loss_keys need to be of the same size.")

    # Get max length, required for markevery.
    max_loss_length = 0
    for i, r in enumerate(results):
        for _, time_axis in r.loss.values():
            loss_length = time_axis[-1]
            if loss_length > max_loss_length:
                max_loss_length = loss_length

    num_marker = 20
    marker_dist = max_loss_length // num_marker

    # Now plot.
    plt.figure()
    ax = plt.subplot(111)

    filename = append_time('LOSS')
    plt.title(filename)

    text_pos = 0

    for i, r in enumerate(results):
        # Plot losses.
        for loss_key, loss_tuple in r.loss.items():
            if loss_key not in loss_keys[i]:
                continue

            loss_values, time_axis = loss_tuple

            step_per_point = time_axis[1]
            markevery = int(max(1.0, marker_dist // step_per_point))

            plt.plot(time_axis, loss_values, label=r.name + ' ' + loss_key,
                     linewidth=0.4, marker='s', markevery=markevery, markerfacecolor='none')

            plt.text(0, text_pos, r.info_text(),
                     horizontalalignment='left', verticalalignment='top', transform=plt.gcf().transFigure)

            text_pos -= 0.35

    plt.subplots_adjust(bottom=0.25)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

    plt.xlabel('Accessed data points')
    plt.ylabel('Objective')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    filepath = os.path.join(cfg.folders['plots'], filename + '.pdf')
    plt.savefig(filepath, bbox_inches='tight')


def plot_grid(filename, x, y, yt, nrow=6):
    """
    Plot the given x images in a grid with the corresponding predicted and ground truth labels.

    :param filename: Filename of the saved pdf.
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
