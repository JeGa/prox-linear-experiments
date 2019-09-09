import logging
import torch

import modelbased.data.utils
import modelbased.problems.mnist_classification.prox_linear as prox_linear
import modelbased.utils.misc
import modelbased.utils.results
import modelbased.utils.trainrun
import modelbased.utils.evaluate

logger = logging.getLogger(__name__)


class FixedStepsize(prox_linear.SVM_OVA_ProxLinear):
    def run(self, trainloader, **kwargs):
        # Number of proximal steps per subproblem / modelfunction.
        sub_iterations = 1

        lam = kwargs['lam']
        tau = kwargs['step_size']
        num_epochs = kwargs['num_epochs']
        num_samples = kwargs['num_samples']
        data_size = kwargs['data_size']
        evaluate = kwargs['evaluate']
        batch_size = trainloader.batch_size

        seen_samples = 0

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size, self.net.device)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        init_missclassifications = modelbased.utils.evaluate.zero_one_batch(self, x_all, y_all)[0]

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.
        moreau_grad = []  # Norm of gradient of Moreau envelope of model functions per mini-batch step.
        missclassifications = [init_missclassifications]

        def step_fun(x, yt):
            nonlocal seen_samples

            # We could also instantiate the ProxLinearFixed class, but this makes no sense here, since we just need to
            # call the solve_subproblem method.
            u = self.net.params

            u_new = None
            for i in range(sub_iterations):
                u_new, _ = self.solve_subproblem(u, tau, x, yt, lam, verbose=False)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_loss = self.loss(u_new, x, yt, lam)

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            # Norm of gradient of Moreau envelope of model functions.
            moreau_grad.append(tau * torch.norm(u_new - u, p=2).item())

            mini_batch_losses.append(mini_batch_loss)  # TODO: Move up.

            # Missclassifications.
            if evaluate:
                missclassifications.append(modelbased.utils.evaluate.zero_one_batch(self, x_all, y_all)[0])

            seen_samples += batch_size
            if seen_samples == num_samples:
                return True

            return False

        def interval_fun(epoch, iteration, batch_iteration):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, mini_batch_losses[-1]))

        modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                      interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-prox-linear-fixed'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'prox-linear with fixed proximal weight, '
                                       'projected dual ascent with armijo on the subproblems.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))],
                'moreau-grad': [moreau_grad, list(range(len(moreau_grad)))],
                'missclassifications': [missclassifications, list(range(len(missclassifications)))]
            },
            parameters={'tau': tau, 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results
