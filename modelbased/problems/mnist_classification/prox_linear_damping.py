import torch
import logging

import modelbased.data.utils
import modelbased.utils.trainrun
import modelbased.utils.results
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.solvers.prox_descent_damping
import modelbased.problems.mnist_classification.misc
import modelbased.problems.mnist_classification.prox_linear as prox_linear

logger = logging.getLogger(__name__)


class Damping(prox_linear.SVM_OVA_ProxLinear):
    def run(self, trainloader, **kwargs):
        lam = 0.0

        # Number of proximal steps per subproblem / modelfunction.
        sub_iterations = 1

        params = modelbased.utils.misc.Params(
            max_iter=sub_iterations,
            eps=1e-6,
            mu_min=1,
            tau=5,
            sigma=0.7)

        num_epochs = kwargs['num_epochs']
        num_samples = kwargs['num_samples']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size

        seen_samples = 0

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.
        moreau_grad = []  # Norm of gradient of Moreau envelope of model functions per mini-batch step.

        proxdescent = modelbased.solvers.prox_descent_damping.ProxDescentDamping(params,
                                                                                 tensor_type='pytorch',
                                                                                 verbose=True)

        def step_fun(x, yt):
            nonlocal seen_samples

            u = self.net.params

            accepted_tau = None

            def callback(u_new, tau):
                nonlocal accepted_tau
                accepted_tau = tau

            def loss(_u):
                return self.loss(_u, x, yt, lam)

            def subsolver(_u, tau):
                return self.solve_subproblem(_u, tau, x, yt, lam, verbose=False)

            u_new, losses = proxdescent.run(u, loss, subsolver, callback=callback)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_loss = self.loss(u_new, x, yt, lam)
            mini_batch_losses.append(mini_batch_loss)

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            # Norm of gradient of Moreau envelope of model functions.
            moreau_grad.append(accepted_tau * torch.norm(u_new - u, p=2).item())

            seen_samples += batch_size
            if seen_samples == num_samples:
                return True

            return False

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                      interval_fun=interval_fun, interval=1)
        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-prox-linear-damping'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'prox-linear with damping, '
                                       'projected dual ascent with armijo on the subproblems.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))],
                'moreau-grad': [moreau_grad, list(range(len(moreau_grad)))]
            },
            parameters={**vars(params), 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results
