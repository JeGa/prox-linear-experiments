import collections
from plotgen import *

PlotLine = collections.namedtuple('PlotLine', 'file loss label')
PlotSettings = collections.namedtuple('PlotSettings', 'title xlabel ylabel')

moreau_gradnorm_label = r'$\| \nabla \mathrm{env}_{f_{x^k}/\sigma}(x^k) \|$'
batch_loss_label = r'$f(x^k)$'
mini_batch_loss_label = r'$\tilde{f}(x^k)$'

files_robreg = {
    # Scale 15.
    'robust-regression-loss-scale15': [
        PlotLine('robust-regression-prox-linear-fixed_30-06-19_13:48:05.yml', 'batch', 'Fixed'),
        PlotLine('robust-regression-prox-linear-linesearch_30-06-19_13:48:35.yml', 'batch', 'Linesearch'),
        PlotLine('robust-regression-prox-linear-damping_30-06-19_13:49:11.yml', 'batch', 'Damping'),
        PlotSettings('', '$k$', batch_loss_label)
    ],
    'robust-regression-moreau-grad-scale15': [
        PlotLine('robust-regression-prox-linear-fixed_30-06-19_13:48:05.yml', 'moreau-grad', 'Fixed'),
        PlotLine('robust-regression-prox-linear-damping_30-06-19_13:49:11.yml', 'moreau-grad', 'Damping'),
        PlotSettings('', '$k$', moreau_gradnorm_label)
    ],
    # Scale 3.
    'robust-regression-loss-scale3': [
        PlotLine('robust-regression-prox-linear-fixed_30-06-19_13:41:55.yml', 'batch', 'Fixed'),
        PlotLine('robust-regression-prox-linear-linesearch_30-06-19_13:42:19.yml', 'batch', 'Linesearch'),
        PlotLine('robust-regression-prox-linear-damping_30-06-19_13:42:47.yml', 'batch', 'Damping'),
        PlotSettings('', '$k$', batch_loss_label)
    ],
    'robust-regression-moreau-grad-scale3': [
        PlotLine('robust-regression-prox-linear-fixed_30-06-19_13:41:55.yml', 'moreau-grad', 'Fixed'),
        PlotLine('robust-regression-prox-linear-damping_30-06-19_13:42:47.yml', 'moreau-grad', 'Damping'),
        PlotSettings('', '$k$', moreau_gradnorm_label)
    ],
}

files_gdsg = {
    'gd-vs-sg': [
        PlotLine('stochastic-gradient_01-07-19_09:51:05.yml', 'batch', 'SG'),
        PlotLine('full-batch-gradient-descent_01-07-19_09:51:02.yml', 'batch', 'Full batch'),
        PlotSettings('Full batch gradient vs. SG', 'Accessed data points', batch_loss_label)
    ]
}

# =============================================================================

files_mnist_1_1 = {
    # Batch-size = 10.
    'mnist-cls-batch-loss-fixed-step-sizes-bs10': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:36:05.yml', 'batch', r'$\sigma^{-1} = 0.1$'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:23:04.yml', 'batch', r'$\sigma^{-1} = 0.5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:18:36.yml', 'batch', r'$\sigma^{-1} = 1$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:15:56.yml', 'batch', r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:19:29.yml', 'batch', r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:20:34.yml', 'batch', r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:20:46.yml', 'batch', r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:48:07.yml', 'batch', r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', batch_loss_label)
    ],
    # Batch-size = 10.
    'mnist-cls-mini-batch-loss-fixed-step-sizes-bs10': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:36:05.yml', 'mini-batch', r'$\sigma^{-1} = 0.1$'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:23:04.yml', 'mini-batch', r'$\sigma^{-1} = 0.5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:18:36.yml', 'mini-batch', r'$\sigma^{-1} = 1$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:15:56.yml', 'mini-batch', r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:19:29.yml', 'mini-batch', r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:20:34.yml', 'mini-batch', r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:20:46.yml', 'mini-batch', r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:48:07.yml', 'mini-batch', r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', mini_batch_loss_label)
    ]
}

files_mnist_1_2 = {
    # Batch-size = 1.
    'mnist-cls-batch-loss-fixed-step-sizes-bs1': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_18:11:36.yml', 'batch', r'$\sigma^{-1} = 0.1$'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:47:37.yml', 'batch', r'$\sigma^{-1} = 0.5$'),
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_04:00:49.yml', 'batch', r'$\sigma^{-1} = 1$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:51:01.yml', 'batch', r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:51:56.yml', 'batch', r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:53:06.yml', 'batch', r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:55:43.yml', 'batch', r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_17:18:38.yml', 'batch', r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', batch_loss_label)
    ],
    # Batch-size = 1.
    'mnist-cls-mini-batch-loss-fixed-step-sizes-bs1': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_18:11:36.yml', 'mini-batch', r'$\sigma^{-1} = 0.1$'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:47:37.yml', 'mini-batch', r'$\sigma^{-1} = 0.5$'),
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_04:00:49.yml', 'mini-batch', r'$\sigma^{-1} = 1$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:51:01.yml', 'mini-batch', r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:51:56.yml', 'mini-batch', r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:53:06.yml', 'mini-batch', r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:55:43.yml', 'mini-batch', r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_17:18:38.yml', 'mini-batch', r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', mini_batch_loss_label)
    ]
}

files_mnist_1_3 = {
    # Batch-size = 10.
    'mnist-cls-moreau-grad-fixed-step-sizes-bs10': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:18:36.yml', 'moreau-grad', r'$\sigma^{-1} = 1$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:15:56.yml', 'moreau-grad', r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:19:29.yml', 'moreau-grad', r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:20:34.yml', 'moreau-grad', r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:20:46.yml', 'moreau-grad', r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_14:48:07.yml', 'moreau-grad', r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', moreau_gradnorm_label)
    ],
    # Batch-size = 1.
    'mnist-cls-moreau-grad-fixed-step-sizes-bs1': [
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_04:00:49.yml', 'moreau-grad', r'$\sigma^{-1} = 1$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:51:01.yml', 'moreau-grad', r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:51:56.yml', 'moreau-grad', r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:53:06.yml', 'moreau-grad', r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_16:55:43.yml', 'moreau-grad', r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_17:18:38.yml', 'moreau-grad', r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', moreau_gradnorm_label)
    ],
}

# =============================================================================

files_mnist_2_1 = {
    # Step size = 1.
    'mnist-cls-batch-loss-fixed-batch-sizes-tau1': [
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_04:00:49.yml', 'batch', 'batch-size = 1'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:43:38.yml', 'batch', 'batch-size = 2'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:35:07.yml', 'batch', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:34:24.yml', 'batch', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:38:11.yml', 'batch', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:56:18.yml', 'batch', 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:16:46.yml', 'batch', 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_21:49:27.yml', 'batch', 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', batch_loss_label)
    ],
    # Step size = 1.
    'mnist-cls-mini-batch-loss-fixed-batch-sizes-tau1': [
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_04:00:49.yml', 'mini-batch', 'batch-size = 1'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:43:38.yml', 'mini-batch', 'batch-size = 2'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:35:07.yml', 'mini-batch', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:34:24.yml', 'mini-batch', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:38:11.yml', 'mini-batch', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:56:18.yml', 'mini-batch', 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:16:46.yml', 'mini-batch', 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_21:49:27.yml', 'mini-batch', 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', mini_batch_loss_label)
    ]
}

files_mnist_2_2 = {
    # Step size = 20.
    'mnist-cls-batch-loss-fixed-batch-sizes-tau20': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:21:42.yml', 'batch', 'batch-size = 1'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:11:46.yml', 'batch', 'batch-size = 2'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:07:48.yml', 'batch', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:16:17.yml', 'batch', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:27:04.yml', 'batch', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:33:49.yml', 'batch', 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:41:38.yml', 'batch', 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:44:42.yml', 'batch', 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', batch_loss_label)
    ],
    # Step size = 20.
    'mnist-cls-mini-batch-loss-fixed-batch-sizes-tau20': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:21:42.yml', 'mini-batch', 'batch-size = 1'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:11:46.yml', 'mini-batch', 'batch-size = 2'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:07:48.yml', 'mini-batch', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:16:17.yml', 'mini-batch', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:27:04.yml', 'mini-batch', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:33:49.yml', 'mini-batch', 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:41:38.yml', 'mini-batch', 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:44:42.yml', 'mini-batch', 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', mini_batch_loss_label)
    ]
}

files_mnist_2_3 = {
    # Step size = 1.
    'mnist-cls-moreau-grad-fixed-batch-sizes-tau1': [
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_04:00:49.yml', 'moreau-grad', 'batch-size = 1'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:43:38.yml', 'moreau-grad', 'batch-size = 2'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:35:07.yml', 'moreau-grad', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:34:24.yml', 'moreau-grad', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:38:11.yml', 'moreau-grad', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_19:56:18.yml', 'moreau-grad', 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:16:46.yml', 'moreau-grad', 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_21:49:27.yml', 'moreau-grad', 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', moreau_gradnorm_label)
    ],
    # Step size = 20.
    'mnist-cls-moreau-grad-fixed-batch-sizes-tau20': [
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:21:42.yml', 'moreau-grad', 'batch-size = 1'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:11:46.yml', 'moreau-grad', 'batch-size = 2'),
        # PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:07:48.yml', 'moreau-grad', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:16:17.yml', 'moreau-grad', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:27:04.yml', 'moreau-grad', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:33:49.yml', 'moreau-grad', 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:41:38.yml', 'moreau-grad', 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_05-07-19_20:44:42.yml', 'moreau-grad', 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', moreau_gradnorm_label)
    ]
}

# =============================================================================

files_mnist_3_11 = {
    'mnist-cls-missclass-fixed-step-sizes-bs1': [
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_18:35:14.yml', 'missclassifications',
        #         r'$\sigma^{-1} = 1'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:19:15.yml', 'missclassifications',
                 r'$\sigma^{-1} = 5$'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:22:02.yml', 'missclassifications',
                 r'$\sigma^{-1} = 10$'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:27:26.yml', 'missclassifications',
                 r'$\sigma^{-1} = 15$'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:36:08.yml', 'missclassifications',
                 r'$\sigma^{-1} = 20$'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:44:59.yml', 'missclassifications',
                 r'$\sigma^{-1} = 50$'),
        PlotSettings('', '$k$', 'Missclassifications')
    ]
}

files_mnist_3_12 = {
    'mnist-cls-missclass-fixed-batch-sizes-tau5': [
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:29:24.yml', 'missclassifications',
        #         'batch-size = 1'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:57:18.yml', 'missclassifications',
                 'batch-size = 1'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:37:43.yml', 'missclassifications',
                 'batch-size = 5'),
        # PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'missclassifications',
        #         'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:54:40.yml', 'missclassifications',
                 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:07:26.yml', 'missclassifications',
                 'batch-size = 50'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:12:30.yml', 'missclassifications',
                 'batch-size = 100'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:18:45.yml', 'missclassifications',
                 'batch-size = 200'),
        PlotSettings('', 'Accessed data points', 'Missclassifications')
    ]
}

files_mnist_3_2 = {
    # Batch size = 10.
    'mnist-cls-batch-loss-fixed-lambda': [
        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_17:15:11.yml', 'batch',
        #         r'$\sigma^{-1} = 0.01$, $\lambda = 0$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_17:05:43.yml', 'batch',
        #         r'$\sigma^{-1} = 0.01$, $\lambda = 0$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_17:04:57.yml', 'batch',
        #         r'$\sigma^{-1} = 0.1$, $\lambda = 0$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_17:02:12.yml', 'batch',
        #         r'$\sigma^{-1} = 0.1$, $\lambda = 0$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:53:13.yml', 'batch',
        #         r'$\sigma^{-1} = 0.5$, $\lambda = 0$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:40:09.yml', 'batch',
        #         r'$\sigma^{-1} = 0.5$, $\lambda = 0$'),

        PlotLine('mnist-classification-prox-linear-fixed_08-07-19_11:41:22.yml', 'batch',
                 r'$\sigma^{-1} = 5$, $\lambda = 0$'),

        # =====================================================================

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:39:32.yml', 'batch',
        #         r'$\sigma^{-1} = 5$, $\lambda = 0.01$'),

        PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:54:56.yml', 'batch',
                 r'$\sigma^{-1} = 5$, $\lambda = 0.1$'),

        PlotLine('mnist-classification-prox-linear-fixed_07-07-19_17:52:33.yml', 'batch',
                 r'$\sigma^{-1} = 5$, $\lambda = 1$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:40:21.yml', 'batch',
        #          r'$\sigma^{-1} = 10$, $\lambda = 0.01$'),
        #
        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:47:25.yml', 'batch',
        #         r'$\sigma^{-1} = 10$, $\lambda = 0.1$'),

        # PlotLine('mnist-classification-prox-linear-fixed_07-07-19_18:31:31.yml', 'batch',
        #         r'$\sigma^{-1} = 10$, $\lambda = 1$'),

        PlotSettings('', '$k$', batch_loss_label)
    ],
    'mnist-cls-missclass-loss-fixed-lambda': [
        PlotLine('mnist-classification-prox-linear-fixed_08-07-19_11:41:22.yml', 'missclassifications',
                 r'$\sigma^{-1} = 5$, $\lambda = 0$'),

        PlotLine('mnist-classification-prox-linear-fixed_07-07-19_16:54:56.yml', 'missclassifications',
                 r'$\sigma^{-1} = 5$, $\lambda = 0.1$'),

        PlotLine('mnist-classification-prox-linear-fixed_07-07-19_17:52:33.yml', 'missclassifications',
                 r'$\sigma^{-1} = 5$, $\lambda = 1$'),

        PlotSettings('', '$k$', 'Missclassifications')
    ]
}

# =============================================================================

files_mnist_sgd_test1 = {
    # Batch size = 10.
    'test': [
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:47:16.yml', 'batch', r'$\sigma = 0.01$'),
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:48.yml', 'batch', r'$\sigma = 0.05$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_23:00:49.yml', 'batch', r'$\sigma = 0.1$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:51.yml', 'batch', r'$\sigma = 0.5$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:52.yml', 'batch', r'$\sigma = 2.0$'),
        PlotSettings('', '$k$', batch_loss_label)
    ]
}

files_mnist_sgd_test2 = {
    # Batch size = 1.
    'test2': [
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:05.yml', 'batch', r'$\sigma = 0.01$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:47.yml', 'batch', r'$\sigma = 0.05$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:42.yml', 'batch', r'$\sigma = 0.1$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:39.yml', 'batch', r'$\sigma = 1.0$'),
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:57:22.yml', 'batch', r'$\sigma = 2.0$'),
        PlotSettings('', '$k$', batch_loss_label)
    ]
}

# =============================================================================

files_mnist_sgd_pl_compare_11 = {
    'mnist-cls-batch-loss-sg-pl-bs10-1': [
        # SG.
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:47:16.yml', 'batch', r'SG $\sigma_{SG} = 0.01$'),
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:48.yml', 'batch', r'SG $\sigma = 0.05$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'batch',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', batch_loss_label)
    ]
}

files_mnist_sgd_pl_compare_12 = {
    'mnist-cls-batch-loss-sg-pl-bs10-2': [
        # SG.
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:47:16.yml', 'mini-batch', r'SG$\sigma_{SG} = 0.01$'),
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:48.yml', 'mini-batch', r'SG $\sigma = 0.05$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'mini-batch',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', mini_batch_loss_label)
    ]
}

files_mnist_sgd_pl_compare_31 = {
    'mnist-cls-batch-loss-sg-pl-bs10-3': [
        # SG.
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:47:16.yml', 'missclassifications',
        #         r'SG$\sigma_{SG} = 0.01$'),
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:48.yml', 'missclassifications',
                 r'SG $\sigma = 0.05$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'missclassifications',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', 'Missclassifications')
    ]
}

files_mnist_sgd_pl_compare_21 = {
    # Batch size = 1.
    'mnist-cls-batch-loss-sg-pl-bs1-1': [
        # SG.
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:05.yml', 'batch', r'SG $\sigma = 0.01$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:57:18.yml', 'batch',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', batch_loss_label)
    ]
}

files_mnist_sgd_pl_compare_22 = {
    # Batch size = 1.
    'mnist-cls-batch-loss-sg-pl-bs1-2': [
        # SG.
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:05.yml', 'mini-batch', r'SG $\sigma = 0.01$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:57:18.yml', 'mini-batch',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', mini_batch_loss_label)
    ]
}

files_mnist_sgd_pl_compare_32 = {
    # Batch size = 1.
    'mnist-cls-batch-loss-sg-pl-bs1-3': [
        # SG.
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:55:05.yml', 'missclassifications',
                 r'SG $\sigma = 0.01$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:57:18.yml', 'missclassifications',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', 'Missclassifications')
    ]
}

# =============================================================================


files_mnist_sgd_pl_compare_31_slides = {
    'mnist-cls-batch-loss-sg-pl-bs10-3-slides': [
        # SG.
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:47:16.yml', 'missclassifications',
        #         r'SG$\sigma_{SG} = 0.01$'),
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:48.yml', 'missclassifications',
                 r'SG $\sigma = 0.05$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'missclassifications',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', 'Missclassifications')
    ]
}

files_mnist_sgd_pl_compare_11_slides = {
    'mnist-cls-batch-loss-sg-pl-bs10-1-slides': [
        # SG.
        # PlotLine('mnist-classification-sg-fixed_11-07-19_22:47:16.yml', 'batch', r'SG $\sigma_{SG} = 0.01$'),
        PlotLine('mnist-classification-sg-fixed_11-07-19_22:45:48.yml', 'batch', r'SG $\sigma = 0.05$'),

        # PL.
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'batch',
                 r'prox-linear $\sigma^{-1} = 5$'),

        PlotSettings('', '$k$', batch_loss_label)
    ]
}

# =============================================================================

files_mnist_subiter = {
    'subiter': [
        # Step size = 5, subiter = 3.
        PlotLine('mnist-classification-prox-linear-fixed_09-07-19_22:19:22.yml', 'batch', 'batch-size = 1*'),
        PlotLine('mnist-classification-prox-linear-fixed_09-07-19_22:33:22.yml', 'batch', 'batch-size = 5*'),
        PlotLine('mnist-classification-prox-linear-fixed_09-07-19_22:48:04.yml', 'batch', 'batch-size = 10*'),
        PlotLine('mnist-classification-prox-linear-fixed_09-07-19_23:05:33.yml', 'batch', 'batch-size = 20*'),
        PlotLine('mnist-classification-prox-linear-fixed_09-07-19_23:23:54.yml', 'batch', 'batch-size = 50*'),
        # PlotLine('mnist-classification-prox-linear-fixed_09-07-19_23:48:26.yml', 'batch', 'batch-size = 100*'),
        # PlotLine('mnist-classification-prox-linear-fixed_10-07-19_00:03:55.yml', 'batch', 'batch-size = 200*'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:57:18.yml', 'batch', 'batch-size = 1'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:37:43.yml', 'batch', 'batch-size = 5'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:45:38.yml', 'batch', 'batch-size = 10'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_15:54:40.yml', 'batch', 'batch-size = 20'),
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_16:07:26.yml', 'batch', 'batch-size = 50'),
        PlotSettings('', 'Accessed data points', batch_loss_label)
    ]
}

# =============================================================================

files_mnist_test = {
    'test': [
        PlotLine('mnist-classification-prox-linear-fixed_06-07-19_14:23:13.yml', 'batch', '1'),
        PlotSettings('', '$k$', batch_loss_label)
    ]
}

page_margin = {'left': 0.19, 'right': 0.98, 'top': 0.95, 'bottom': 0.16}
page_size = (3, 2.2)


def test():
    plot(files_mnist_test, page_size, page_margin)


def robreg():
    plot(files_robreg, page_size, {'left': 0.22, 'right': 0.98, 'top': 0.95, 'bottom': 0.16}, subfolder=['robreg'])


def gdvssg():
    plot(files_gdsg, (3.9, 2.9), {'left': 0.16, 'right': 0.95, 'top': 0.9, 'bottom': 0.15}, subfolder=['gd_vs_sg'])


def page1():
    plot(files_mnist_1_1, page_size, page_margin, compare_iterations(5, 500, label=('sigma',)), ['bs=10'])
    plot(files_mnist_1_2, page_size, page_margin, compare_iterations(4, 2500, label=('sigma',), smooth_factor=5),
         ['bs=1'])
    plot(files_mnist_1_3, page_size, page_margin, compare_iterations(4, 2500, label=('sigma',), smooth_factor=5),
         ['bs=1', 'bs=10'])


def page2():
    plot(files_mnist_2_1, page_size, page_margin, compare_samples(1200), ['tau=1'])
    plot(files_mnist_2_2, page_size, page_margin, compare_samples(1200), ['tau=20'])
    plot(files_mnist_2_3, page_size, page_margin, compare_samples(1200), ['tau=20', 'tau=1'])


def page3_missclass():
    plot(files_mnist_3_11, page_size, page_margin, compare_iterations(0, 1500, label=('sigma',)), ['missclass_bs=1'])
    plot(files_mnist_3_12, page_size, page_margin, compare_samples(4000), ['missclass_tau=5'])


def page3_lambda():
    plot(files_mnist_3_2, page_size, page_margin, compare_iterations(0, 300, label=('lambda',)), ['lambdav2'])


def page4_compare():
    plot(files_mnist_sgd_pl_compare_11, page_size, page_margin, compare_iterations(0, 500),
         ['sgd-bs=10', 'missclass_tau=5'])
    plot(files_mnist_sgd_pl_compare_12, page_size, page_margin, compare_iterations(0, 500),
         ['sgd-bs=10', 'missclass_tau=5'])
    plot(files_mnist_sgd_pl_compare_31, page_size, page_margin, compare_iterations(0, 500),
         ['sgd-bs=10', 'missclass_tau=5'])

    plot(files_mnist_sgd_pl_compare_21, page_size, page_margin, compare_iterations(0, 500),
         ['sgd-bs=1', 'missclass_tau=5'])
    plot(files_mnist_sgd_pl_compare_22, page_size, page_margin, compare_iterations(0, 500),
         ['sgd-bs=1', 'missclass_tau=5'])
    plot(files_mnist_sgd_pl_compare_32, page_size, page_margin, compare_iterations(0, 500),
         ['sgd-bs=1', 'missclass_tau=5'])


def slides():
    _page_margin = {'left': 0.19, 'right': 0.98, 'top': 0.95, 'bottom': 0.16}
    _page_size = (2, 2.2)

    plot(files_mnist_sgd_pl_compare_11_slides, _page_size, _page_margin, compare_iterations(0, 500),
         ['sgd-bs=10', 'missclass_tau=5'])
    plot(files_mnist_sgd_pl_compare_31_slides, _page_size, _page_margin, compare_iterations(0, 500),
         ['sgd-bs=10', 'missclass_tau=5'])


# robreg()

# page1()
# page2()
# page3_missclass()
# page3_lambda()
# page4_compare()
slides()

# plot(files_mnist_subiter, page_size, page_margin, compare_samples(1000), ['missclass_tau=5'])
# plot(files_mnist_sgd, page_size, page_margin, compare_iterations(0, 2000), ['sgd-bs=10'])
# plot(files_mnist_sgd2, page_size, page_margin, compare_iterations(0, 2000), ['sgd-bs=1'])
