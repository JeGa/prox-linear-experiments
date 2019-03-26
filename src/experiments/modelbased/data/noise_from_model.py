import numpy as np
import matplotlib.pyplot as plt


def generate(N, f, xmin=-2, xmax=2, noise='laplacian', seed=999):
    x = np.expand_dims(np.linspace(xmin, xmax, N), -1)
    y = f(x)

    np.random.seed(seed)

    if noise == 'gaussian':
        y_noisy = y + np.expand_dims(np.random.normal(0, 0.5, N), -1)
    elif noise == 'laplacian':
        y_noisy = y + np.expand_dims(np.random.laplace(0, 3, N), -1)
    else:
        raise ValueError('Given noise not supported.')

    return x, y_noisy, y


def plot(x, y_noisy, y):
    plt.scatter(x, y_noisy, marker='x')
    plt.plot(x, y)
    plt.show()


def sinus():
    def f(x):
        return np.sin(x)

    plot(*generate(100, f, 0, 2 * np.pi))
