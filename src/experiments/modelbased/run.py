import matplotlib
matplotlib.use('Agg')
import logging

# from problems.simple_robust_regression_exp.l1norm import run as run_reg
# import problems.mnist_classification_nn.svm_ova as svm_ova_proxdescent

import problems.spirals_classification.logistic_regression_nn as logistic_regression_nn

import modelbased.utils.misc

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")

    modelbased.utils.misc.make_folders()

    # svm_ova_proxdescent.evaluate_from_file('prox_descent_fixed_03-05-19_18:57:32')
    # svm_ova_proxdescent.train()
    logistic_regression_nn.run()
