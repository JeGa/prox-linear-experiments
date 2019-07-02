import matplotlib.pyplot as plt

params = {
    'backend': 'pgf',
    'pgf.texsystem': 'lualatex',
    'text.latex.preamble': [r"\usepackage{lmodern}"],
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 7,
    'font.family': 'lmodern'}

plt.rcParams.update(params)

import numpy as np
import pathlib
import modelbased.utils.global_config as cfg

plotfolder = pathlib.Path(cfg.folders['plots'])

x = np.linspace(-2, 2, num=300)


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


def plot_multiple(fun, prox, lambda_list, function_label, filename):
    y_fun = fun(x)

    m = []
    for i in lambda_list:
        m.append([moreau(x, i, fun, prox), i])

    plt.figure()
    plt.xlabel('x')
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.1)

    plt.plot(x, y_fun, label=function_label, linewidth=0.8)

    for i in m:
        plt.plot(x, i[0], '--', label='$\lambda=$' + str(i[1]), linewidth=0.6)

    plt.legend()

    plt.gcf().set_size_inches(3.1, 2.3, forward=True)
    plt.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.17)

    plt.savefig(plotfolder / (filename + '.pdf'))


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


plot_multiple(f, prox_f, [0.1, 0.5, 1, 1.5, 2, 2.5], '$f(x) = x^2 + |x|$', 'moreau-example-1')
plot_multiple(g, prox_g, [0.05, 0.2, 0.4, 0.5], '$f(x) = |x^2 - 1|$', 'moreau-example-2')
