# TODO: Should move to plot scripts.

import matplotlib.pylab as plt
import numpy as np
import os
from collections import namedtuple

from modelbased.problems.simple_regression.robust_exp import RobustRegression
import modelbased.data.noise_from_model
import modelbased.utils.global_config as cfg

params = {
    'backend': 'pgf',
    'pgf.texsystem': 'lualatex',
    'text.latex.preamble': [r"\usepackage{lmodern}"],
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 8,
    'font.family': 'lmodern'
}

plt.rcParams.update(params)


def plot(name, model_parameters, u_init, data, size, margins):
    y_predict = RobustRegression.f(np.array(model_parameters), data.x)
    y_init = RobustRegression.f(u_init, data.x)

    plot_predictions(name, data.x, data.y, data.y_targets, y_predict, y_init, size, margins)


def plot_data(name, x, y, y_targets, size, margins):
    """
    All parameters have the shape (N, 1).

    :param name: Filename.
    :param x: Input samples.
    :param y: True output samples.
    :param y_targets: Noisy output samples.
    :param size: Tuple with x,y size of the plot.
    :param margins: Dict with plot margins.
    """
    plt.figure()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    # True.
    plt.plot(x, y, label=r'$h(\hat{x})$', c='black')

    # Samples.
    plt.scatter(x, y_targets, marker='x', label='$(p,q)$', linewidth=0.6, s=25)

    plt.minorticks_on()
    plt.gcf().set_size_inches(size[0], size[1], forward=True)
    plt.subplots_adjust(**margins)

    plt.legend()

    plt.savefig(os.path.join(cfg.folders['plots'], name + '.pdf'))


def plot_predictions(name, x, y, y_targets, y_predict, y_init, size, margins):
    """
    All parameters have the shape (N, 1).

    :param name: Filename.
    :param x: Input samples.
    :param y: True output samples.
    :param y_targets: Noisy output samples.
    :param y_predict: Predicted output after training.
    :param y_init: Predicted output before training.
    :param size: Tuple with x,y size of the plot.
    :param margins: Dict with plot margins.
    """
    plt.figure()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    # True.
    plt.plot(x, y, label=r'$h(\hat{x})$', c='black')

    # Samples.
    plt.scatter(x, y_targets, marker='x', label='$(p,q)$', linewidth=0.6, s=25)

    # Predictions after training.
    plt.plot(x, y_predict, label=r'$h(\bar{x})$', c='darkorange')

    # Predictions before training.
    plt.plot(x, y_init, label=r'$h(x^1)$', c='green')

    plt.minorticks_on()
    plt.gcf().set_size_inches(size[0], size[1], forward=True)
    plt.subplots_adjust(**margins)

    plt.legend()

    plt.savefig(os.path.join(cfg.folders['plots'], name + '.pdf'))


def generate_data(samples, P, seed, scale):
    """
    Sample random parameters of an exponential model and generate noisy training data, i.e. an Laplacian error is added
    to the outputs.

    :param samples: Number of samples to generate.
    :param P: Number of parameters for one variable of the exponential model. The total amount of parameters is 2P.
    :param seed: Seed for random generator.
    :param scale: Scale factor for noise.

    :return: Input samples (samples, 1), noisy output samples (samples, 1), true output samples (samples, 1).
    """
    # Generate some noisy data.
    a = 2 * np.random.random((P, 1))
    b = np.random.random((P, 1))
    u = np.concatenate((a, b), axis=0)

    def fun(_x):
        return RobustRegression.f(u, _x)

    x, y_noisy, y = modelbased.data.noise_from_model.generate(samples, fun, scale, noise='laplacian', seed=seed)

    return namedtuple('Data', 'x y y_targets seed scale')(x, y, y_noisy, seed, scale)
