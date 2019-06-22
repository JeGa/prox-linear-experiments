import matplotlib.pylab as plt
import numpy as np
from collections import namedtuple

from modelbased.problems.simple_regression.robust_exp import RobustRegression
import modelbased.data.noise_from_model


def plot_regression(x, y, y_targets, y_predict, y_init):
    """
    All parameters have the shape (N, 1).

    :param x: Input samples.
    :param y: True output samples.
    :param y_targets: Noisy output samples.
    :param y_predict: Predicted output after training.
    :param y_init: Predicted output before training.
    """
    plt.plot(x, y, label='true')
    plt.scatter(x, y_targets, marker='x', label='f + e')
    plt.plot(x, y_predict, label='predict')
    plt.plot(x, y_init, label='init')

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    plt.legend()
    plt.show()


def generate_data(samples, P, seed):
    """
    Sample random parameters of an exponential model and generate noisy training data, i.e. an Laplacian error is added
    to the outputs.

    :param samples: Number of samples to generate.
    :param P: Number of parameters for one variable of the exponential model. The total amount of parameters is 2P.

    :return: Input samples (samples, 1), noisy output samples (samples, 1), true output samples (samples, 1).
    """
    # Generate some noisy data.
    a = 2 * np.random.random((P, 1))
    b = np.random.random((P, 1))
    u = np.concatenate((a, b), axis=0)

    def fun(_x):
        return RobustRegression.f(u, _x)

    x, y_noisy, y = modelbased.data.noise_from_model.generate(samples, fun, noise='laplacian', seed=seed)

    return namedtuple('Data', 'x y y_targets seed')(x, y, y_noisy, seed)
