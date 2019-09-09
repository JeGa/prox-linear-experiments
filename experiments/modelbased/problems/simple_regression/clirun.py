import click
import numpy as np

import modelbased.problems.simple_regression.misc as misc
import modelbased.problems.simple_regression.prox_linear as pl


def plot(seed, scale):
    np.random.seed(seed)

    data = misc.generate_data(samples=200, P=10, seed=seed, scale=scale)

    # Number of parameters of the prediction function.
    P_model = 20
    u_init = np.random.randn(P_model, 1)

    # results_fixed = FixedStepsize.run(u_init, data)
    # results_linesearch = Linesearch.run(u_init, data)
    results_damping = pl.Damping.run(u_init, data)

    # modelbased.utils.yaml.write_result(results_fixed)
    # modelbased.utils.yaml.write_result(results_linesearch)
    # modelbased.utils.yaml.write_result(results_damping)

    # Plots for documentation.

    # modelbased.problems.simple_regression.misc.plot_regression(
    #    'fixed', results_fixed.model_parameters, u_init, data)
    # modelbased.problems.simple_regression.misc.plot_regression(
    #    'linesearch', results_linesearch.model_parameters, u_init, data)
    misc.plot('damping-{}-{}'.format(seed, scale), results_damping.model_parameters, u_init, data,
              (3, 3), {'left': 0.22, 'right': 0.98, 'top': 0.95, 'bottom': 0.08})

    # For slides.
    size = (3.5, 3)
    margins = {'left': 0.12, 'right': 0.98, 'top': 0.95, 'bottom': 0.08}

    misc.plot('damping-{}-{}-slides'.format(seed, scale), results_damping.model_parameters, u_init, data, size, margins)
    misc.plot_data('damping-{}-{}-data-slides'.format(seed, scale), data.x, data.y, data.y_targets, size, margins)


@click.command('robreg-pl')
def run():
    # Tests are run with:
    # 1) seed=4444, scale=3, P_model=20.
    # 2) seed=4445, scale=15, P_model=20.
    # plot(4444, 3)
    plot(4445, 15)
