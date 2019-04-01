import logging

from problems.simple_robust_regression_exp.robust_regression import run as run_reg
from problems.mnist_classification_nn.mnist_classification import run as run_cls

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_cls()
