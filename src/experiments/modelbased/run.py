import matplotlib

# matplotlib.use('Agg')
import logging

# import problems.mnist_classification_nn.svm_ova as svm_ova_proxdescent
import problems.simple_regression.prox_linear as robust_regression
import problems.spirals_classification.gradient_descent as logistic_regression

import modelbased.utils.misc

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")
    modelbased.utils.misc.make_folders()

    # svm_ova_proxdescent.evaluate_from_file('prox_descent_fixed_03-05-19_18:57:32')
    # svm_ova_proxdescent.train()

    run = None

    if run == 'mnist-cls':
        pass
    elif run == 'spirals-cls':
        logistic_regression.run()
    elif run == 'exp-reg':
        robust_regression.run()
