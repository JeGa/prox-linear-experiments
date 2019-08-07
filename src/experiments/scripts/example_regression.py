import matplotlib.pyplot as plt
import pathlib
import numpy as np

import modelbased.utils.global_config as cfg
import modelbased.problems.simple_regression.misc

params = {
    'backend': 'pgf',
    'pgf.texsystem': 'lualatex',
    'text.latex.preamble': [r"\usepackage{lmodern}"],
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 7,
    'font.family': 'lmodern'
}

plt.rcParams.update(params)

plotfolder = pathlib.Path(cfg.folders['plots'])


def plot(x_gt, y_gt, samples, outliers, size, filename):
    plt.figure()

    plt.gcf().set_size_inches(size, size, forward=True)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
                    labelleft=False)

    plt.plot(x_gt, y_gt, label=r'$h(\hat{x})$', c='black', zorder=-1)
    plt.scatter(samples[:, 0], samples[:, 1], marker='x', label='$(p,q)$', linewidth=1.25, s=60)
    plt.scatter(outliers[:, 0], outliers[:, 1], marker='x', label=r'$(\bar{p},\bar{q})$', linewidth=1.25, s=60)

    plt.savefig(plotfolder / (filename + '.pdf'))


def robust_regression_example():
    samples = np.array(
        [[-0.75, 12],
         [-0.6, 8],
         [-0.25, 5],
         [0, 6],
         [0.75, 3],
         [1.5, 2],
         [1.75, 0.5]]
    )

    outliers = np.array(
        [[-0.5, 0],
         [1, 10]]
    )

    seed = 4444
    np.random.seed(seed)

    data = modelbased.problems.simple_regression.misc.generate_data(samples=200, P=10, seed=seed, scale=1)

    plot(data.x[50:], data.y[50:], samples, outliers, 1.1, 'regression-example')
    plot(data.x[50:], data.y[50:], samples, outliers, 2.2, 'regression-example-large')


robust_regression_example()
