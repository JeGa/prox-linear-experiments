import matplotlib.pylab as plt
import numpy as np

from .robust_nn import RobustRegression
import modelbased.data.noise_from_model


def plot(x, y, y_noisy, y_predict, y_init):
    plt.plot(x, y, label='true')
    plt.scatter(x, y_noisy, marker='x', label='f + e')
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

    return x, y_noisy, y
