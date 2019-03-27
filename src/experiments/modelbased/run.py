import numpy as np
import matplotlib.pyplot as plt

import data.noise_from_model
import problems.robust_regression as robreg


def run_robust_regression():
    N = 300
    P_gen = 10
    P_model = 20

    # Generate some noisy data.
    a = 2 * np.random.random((P_gen, 1))
    b = np.random.random((P_gen, 1))
    u = np.concatenate((a, b), axis=0)

    # index = np.random.binomial(1, 0.8, (P_gen, 1))
    # a[index] = 0

    def fun(x):
        return robreg.LaplaceNoise1d.f(u, x)

    x, y_noisy, y = data.noise_from_model.generate(N, fun)

    Reg = robreg.LaplaceNoise1d(x, y_noisy)

    u_init = 0.1 * np.ones((2 * P_model, 1))
    y_predict = Reg.run(u_init)

    y_init = Reg.f(u_init, x)

    plot(x, y, y_noisy, y_predict, y_init)


def plot(x, y, y_noisy, y_predict, y_init):
    plt.plot(x, y, label='true')
    plt.scatter(x, y_noisy, marker='x')
    plt.plot(x, y_predict, label='predict')
    plt.plot(x, y_init, label='init')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_robust_regression()
