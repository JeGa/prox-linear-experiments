import os
from matplotlib import pyplot as plt
import datetime


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
