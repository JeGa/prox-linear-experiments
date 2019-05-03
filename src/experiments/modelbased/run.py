import matplotlib

matplotlib.use('Agg')

import logging

from problems.simple_robust_regression_exp.l1norm import run as run_reg
from problems.mnist_classification_nn.svm_ova import run as run_cls_svm

import modelbased.utils.misc

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")

    modelbased.utils.misc.make_folders()

    # run_spirals()
    run_cls_svm()
    # run_reg()
