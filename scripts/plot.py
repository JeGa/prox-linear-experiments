import matplotlib.pyplot as plt
import collections
import pathlib
import modelbased.utils.global_config as cfg

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

attributes = ('x', 'y', 'style', 'width', 'label', 'c')
PlotEntry = collections.namedtuple('PlotEntry', attributes, defaults=(None,) * len(attributes))


def plot(functions, filename, custom=None):
    plt.figure()
    plt.xlabel('x')
    # plt.minorticks_on()
    # plt.grid(which='major', linestyle='-', linewidth=0.1)

    plt.tick_params(axis='y', which='both', bottom=False, top=False, left=False, right=False)

    for i in functions:
        plt.plot(i.x, i.y, linestyle=i.style, label=i.label, linewidth=i.width, c=i.c)

    if custom:
        custom()

    plt.legend(loc='upper center')

    plt.gcf().set_size_inches(2.8, 2.2, forward=True)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.15)

    plt.savefig(plotfolder / (filename + '.pdf'))
