import matplotlib.pyplot as plt
import pathlib
import yaml
import scipy.ndimage

import evaltool
import modelbased.utils.global_config as cfg

params = {
    'backend': 'pgf',
    'pgf.texsystem': 'lualatex',
    'text.latex.preamble': [r"\usepackage{lmodern}"],
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 8,
    'font.family': 'lmodern'}

plt.rcParams.update(params)


def plot(files, size, margins, callback=None, subfolder=None):
    trainfolder = pathlib.Path(cfg.folders['data_train'])
    plotfolder = pathlib.Path(cfg.folders['plots'])

    trainfolders = [trainfolder]
    if subfolder:
        for sf in subfolder:
            trainfolders.append(trainfolder / sf)

    for plot_name, plots in files.items():
        settings = plots[-1]
        plotter = evaltool.Plotter()

        for line in plots[0:-1]:
            # Find file.
            filepath = None
            for sf in trainfolders:
                path = sf / line.file
                if path.exists():
                    filepath = path
                    break

            if not filepath:
                raise FileNotFoundError()

            with filepath.open('r') as f:
                yaml_data = yaml.safe_load(f)

            linelabel = line.label

            if callback:
                y_loss, x_loss, label = callback(yaml_data, line.loss)

                if label:
                    linelabel = label
            else:
                loss = yaml_data['loss'][line.loss]
                y_loss = loss[0]
                x_loss = loss[1]

            plotter.add(y_loss, x_loss, linelabel, text='')

        print('Plot: {}.'.format(plot_name))

        plotter.style('fancy', title=settings.title, xlabel=settings.xlabel, ylabel=settings.ylabel)
        plotter.legend(on=True)
        plotter.save_plot(plotfolder / plot_name, fileformat='.pdf', size=size, margins=margins)


def split(yaml_data, lineloss, x_from, x_to):
    loss = yaml_data['loss'][lineloss]
    y_loss = loss[0][x_from:x_to]
    x_loss = loss[1][x_from:x_to]

    return y_loss, x_loss


def smooth(y_loss, sigma):
    return scipy.ndimage.gaussian_filter1d(y_loss, sigma)


def compare_iterations_cb(yaml_data, lineloss, min_iterations, max_iterations, labels, smooth_factor):
    label = []

    if 'sigma' in labels:
        label += [r'$\sigma^{-1} = ' + str(yaml_data['parameters']['tau']) + '$']

    if 'lambda' in labels:
        label += [r'$\lambda = ' + str(yaml_data['parameters']['lambda']) + '$']

    # Build label string.
    if len(label) == 0:
        label_text = None
    elif len(label) == 1:
        label_text = label[0]
    else:
        label_text = label[0]
        for i in label[1:]:
            label_text += ', ' + i

    y_loss, x_loss = split(yaml_data, lineloss, min_iterations, max_iterations)

    if smooth_factor:
        y_loss = smooth(y_loss, smooth_factor)

    return y_loss, list(range(len(y_loss))), label_text


def compare_iterations(min_iterations, max_iterations, label=[], smooth_factor=None):
    def fun(yaml_data, lineloss):
        return compare_iterations_cb(yaml_data, lineloss,
                                     min_iterations, max_iterations, label, smooth_factor)

    return fun


def compare_samples_cb(yaml_data, lineloss, max_data_points):
    batch_size = yaml_data['parameters']['batch_size']

    label = r'batch-size $=' + str(batch_size) + '$'

    loss = yaml_data['loss'][lineloss]
    y_loss = loss[0]

    # Change time axis.
    x_loss = [i for i in range(0, len(y_loss) * batch_size, batch_size)]

    # Split.
    steps = (max_data_points // batch_size) + 1

    y_loss = y_loss[0:steps]
    x_loss = x_loss[0:steps]

    return y_loss, x_loss, label


def compare_samples(max_data_points):
    def fun(yaml_data, lineloss):
        return compare_samples_cb(yaml_data, lineloss, max_data_points)

    return fun
