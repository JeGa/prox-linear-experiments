import numpy as np


def generate(N, f, scale=3, xmin=-2, xmax=2, noise='laplacian', seed=999):
    x = np.expand_dims(np.linspace(xmin, xmax, N), -1)
    y = f(x)

    np.random.seed(seed)

    if noise == 'gaussian':
        y_noisy = y + np.expand_dims(np.random.normal(0, scale, N), -1)
    elif noise == 'laplacian':
        y_noisy = y + np.expand_dims(np.random.laplace(0, scale, N), -1)
    else:
        raise ValueError('Given noise not supported.')

    return x, y_noisy, y
