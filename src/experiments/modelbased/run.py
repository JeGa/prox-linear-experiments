import matplotlib

matplotlib.use('Agg')

import logging

from problems.simple_robust_regression_exp.l1norm import run as run_reg
from problems.mnist_classification_nn.l1norm_noreg import run as run_cls_l1
from problems.mnist_classification_nn.svm_ova import run as run_cls_svm
from problems.spirals_classification.linear_svm import run as run_spirals

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")

    # run_spirals()
    run_cls_svm()
    # run_reg()
