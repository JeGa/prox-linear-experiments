import numpy as np
from scripts.plot import *


def f(_x):
    return np.abs(_x ** 2 - 1)


def sub_prox_linear(_xk, _x, lam):
    m = model_prox_linear(_xk, _x)

    return m + 0.5 * lam * (_x - _xk) ** 2


def model_prox_linear(_xk, _x):
    return np.abs(_xk ** 2 - 1 + 2 * _xk * (_x - _xk))


def model_gradient(_xk, _x):
    return f(_xk) + (_xk ** 2 - 1) / (np.abs(_xk ** 2 - 1)) * (_x - _xk)


def isnparray(x):
    return type(x) is np.ndarray


# =============================================================================
# Models.

def models_slides():
    x = np.linspace(-2, 2, num=300)
    xk = 0.5

    y_fun_entry = PlotEntry(x, f(x), '-', 0.8, r'$f(x) = |x^2-1|$')

    y_model_pl_entry = PlotEntry(x, model_prox_linear(xk, x), '-', 0.8, r'$\phi_{0.5}^1(x)$')

    y_sub_pl_entry = PlotEntry(x, sub_prox_linear(xk, x, 0.8), '--', 0.8,
                               r'$\phi_{0.5}^1(x) + \frac{1}{2\sigma} \|x - 0.5\|^2$')

    y_model_gd_entry = PlotEntry(x, model_gradient(xk, x), '--', 0.8, r'$\phi_{0.5}^2(x)$')

    def annotate():
        plt.annotate(xy=(1.25, 0), xytext=(1.8, 0), s='',
                     arrowprops={'arrowstyle': '->'})

    plot([y_fun_entry], 'models-slides-1')
    plot([y_fun_entry, y_model_pl_entry], 'models-slides-2')
    plot([y_fun_entry, y_model_pl_entry, y_sub_pl_entry], 'models-slides-3')
    plot([y_fun_entry, y_model_pl_entry, y_model_gd_entry], 'models-slides-4', custom=annotate)


models_slides()


# =============================================================================
# One-sided vs. two-sided.


def squared(_x, _xk, tau):
    return tau * 0.5 * (_x - _xk) ** 2


def models_slides_bound():
    x = np.linspace(-1.75, 1.75, num=300)
    xk = 0.5

    y_fun = f(x)

    y_fun_entry = PlotEntry(x, y_fun, '-', 0.8, r'$f(x) = |x^2-1|$')
    y_error_above_entry = PlotEntry(x, y_fun + squared(x, xk, 3), '--', 0.8, r'$f(x) + \frac{\tau}{2} \|x - 0.5\|^2$')
    y_model_pl_entry = PlotEntry(x, model_prox_linear(xk, x), '-', 0.8, r'$\phi_{0.5}^1(x)$')

    plot([y_fun_entry, y_error_above_entry, y_model_pl_entry], 'models-slides-bound')


models_slides_bound()
