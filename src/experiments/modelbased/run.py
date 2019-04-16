import matplotlib

# TODO matplotlib.use('Agg')

import logging

from problems.simple_robust_regression_exp.robust_regression import run as run_reg
from problems.mnist_classification_nn.classification_l1norm_noreg import run as run_cls
from problems.spirals_classification.linear_svm import run as run_spirals

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")

    # run_spirals()
    # run_cls()
    run_reg()
