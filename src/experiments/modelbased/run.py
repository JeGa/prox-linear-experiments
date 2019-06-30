import matplotlib

matplotlib.use('Agg')
import logging

import problems.mnist_classification.prox_linear as mnist_classification
import problems.simple_regression.prox_linear as robust_regression_prox_linear
import problems.simple_regression.gradient_descent as robust_regression_gradient_descent
import problems.spirals_classification.gradient_descent as spirals_classification

import modelbased.utils.misc

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")
    modelbased.utils.misc.make_folders()

    run = 'mnist-cls'

    if run == 'mnist-cls':
        mnist_classification.run()
    elif run == 'spirals-cls':
        spirals_classification.run()
    elif run == 'exp-reg':
        robust_regression_prox_linear.run()
    elif run == 'exp-reg-gd':
        robust_regression_gradient_descent.run()
