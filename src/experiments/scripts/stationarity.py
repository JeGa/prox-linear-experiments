import numpy as np
from scripts.plot import *


def f(_x):
    return _x ** 2 + np.abs(_x)


def g(_x):
    return np.abs(_x ** 2 - 1)


def moreau(_x, lam, fun, prox):
    xp = prox(_x, lam)
    return fun(xp) + (1 / (2 * lam)) * (xp - _x) ** 2


def prox_f(_x, lam):
    def proxi(xi):
        lam_div = (1 / lam)

        if xi > lam:
            return (lam_div * xi - 1) / (lam_div + 2)
        elif xi < -lam:
            return (lam_div * xi + 1) / (lam_div + 2)
        else:
            return 0

    return np.array([proxi(xi) for xi in _x])


def prox_g(_x, lam):
    def proxi(xi):
        lam_div = (1 / lam)

        if xi > (2 + lam_div) / lam_div or xi < -(2 + lam_div) / (lam_div):
            return (lam_div * xi) / (2 + lam_div)

        if np.abs(xi) < (lam_div - 2) / lam_div:
            return (lam_div * xi) / (lam_div - 2)

        if np.abs(xi - 1) <= 2 * lam:
            return 1

        if np.abs(xi + 1) <= 2 * lam:
            return -1

    A = np.array([proxi(xi) for xi in _x])
    return A


def plot_g():
    y_g = g(x)

    y_m = moreau(x, 0.1, g, prox_g)

    plt.figure()

    plt.plot(x, y_g)
    plt.plot(x, y_m)

    plt.show()


def plot_f():
    y_f = f(x)

    y_m = moreau(x, 1, f, prox_f)

    plt.figure()

    plt.plot(x, y_f)
    plt.plot(x, y_m)

    plt.show()


def plot_multiple(fun, prox, lambda_list, function_label, filename):
    y_fun = fun(x)

    y_fun_entry = PlotEntry(x, y_fun, '-', 0.8, function_label)

    m = []
    for i in lambda_list:
        m.append(PlotEntry(x, moreau(x, i, fun, prox), '--', 0.6, r'env$_{' + str(i) + 'f}(x)$'))

    # plt.gcf().set_size_inches(3, 2.5, forward=True)
    # plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.15)
    plot([y_fun_entry] + m, filename)


x = np.linspace(-2, 2, num=300)


# plot_multiple(f, prox_f, [0.1, 0.5, 1, 1.5, 2, 2.5], '$f(x) = x^2 + |x|$', 'moreau-example-1')
# plot_multiple(g, prox_g, [0.05, 0.2, 0.4, 0.5], '$f(x) = |x^2 - 1|$', 'moreau-example-2')


def plot_slides():
    y_fun_entry = PlotEntry(x, g(x), '-', 0.8, '$f(x) = |x^2 - 1|$')
    y_moreau_entry = PlotEntry(x, moreau(x, 0.4, g, prox_g), '--', 0.8, 'env$_f(x)$')

    def ticks(x):
        y = g(np.array(x))
        labels = ['$x^' + str(i + 1) + '$' for i in range(len(x))]

        def fun():
            plt.gca().set_xticks(x)
            plt.gca().set_xticklabels(labels)
            plt.scatter(x, y, marker='o', c='red', linewidth=1.25, s=10, zorder=3)

        return fun

    plot([y_fun_entry], 'moreau-example-2-slides-1')
    plot([y_fun_entry], 'moreau-example-2-slides-2', custom=ticks([0.2]))
    plot([y_fun_entry], 'moreau-example-2-slides-3', custom=ticks([0.2, 0.4, 0.6, 0.8, 0.96]))
    plot([y_fun_entry, y_moreau_entry], 'moreau-example-2-slides-4')


plot_slides()
