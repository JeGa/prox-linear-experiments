import logging

import modelbased.data.utils
import modelbased.utils.trainrun
import modelbased.utils.results
import modelbased.utils.misc
import modelbased.utils.yaml
import modelbased.solvers.prox_descent_linesearch
import modelbased.problems.mnist_classification.misc
import modelbased.problems.mnist_classification.prox_linear as prox_linear

logger = logging.getLogger(__name__)


class Linesearch(prox_linear.SVM_OVA_ProxLinear):
    def run(self, trainloader, **kwargs):
        lam = 0

        # Number of proximal steps per subproblem / modelfunction.
        sub_iterations = 1

        params = modelbased.utils.misc.Params(
            max_iter=sub_iterations,
            eps=1e-12,
            proximal_weight=0.1,
            gamma=0.5,
            delta=0.5,
            eta_max=3)

        num_epochs = kwargs['num_epochs']
        data_size = kwargs['data_size']
        batch_size = trainloader.batch_size

        x_all, y_all = modelbased.data.utils.get_samples(trainloader, data_size)

        init_loss = self.loss(self.net.params, x_all, y_all, lam)

        mini_batch_losses = [init_loss]  # Loss per mini-batch step over mini-batch samples.
        batch_losses = [init_loss]  # Loss per mini-batch step over all samples.

        proxdescent = modelbased.solvers.prox_descent_linesearch.ProxDescentLinesearch(
            params, tensor_type='pytorch', verbose=True)

        def step_fun(x, yt):
            u = self.net.params

            def loss(u):
                return self.loss(u, x, yt, lam)

            def subsolver(u, tau):
                # TODO stopcond.
                # def stopcond(uk, linloss):  # TODO.
                #     if linloss - loss(uk) < 0:
                #         return True
                #     else:
                #         return False
                return self.solve_subproblem(u, tau, x, yt, lam, verbose=False, stopping_condition=None)

            u_new, _ = proxdescent.run(u, loss, subsolver)

            self.net.params = u_new

            # Mini-batch loss.
            mini_batch_loss = self.loss(u_new, x, yt, lam)

            # Batch loss.
            batch_losses.append(self.loss(u_new, x_all, y_all, lam))

            return [mini_batch_loss]

        def interval_fun(epoch, iteration, batch_iteration, _total_losses):
            logger.info("[{}:{}/{}:{}/{}] Loss={:.6f}.".format(iteration, batch_iteration, len(trainloader),
                                                               epoch, num_epochs, _total_losses[-1]))

        mini_batch_losses += modelbased.utils.trainrun.run(num_epochs, trainloader, step_fun, self.net.device,
                                                           interval_fun=interval_fun, interval=1)

        results = modelbased.utils.results.Results(
            name=modelbased.utils.misc.append_time('mnist-classification-prox-linear-linesearch'),
            type='train',
            description={
                **self.description(),
                'optimization method': 'prox-linear with linesearch, '
                                       'projected dual ascent with armijo on the subproblems.'
            },
            train_dataset={
                'name': 'MNIST',
                'size': data_size
            },
            loss={
                'mini-batch': [mini_batch_losses, list(range(len(mini_batch_losses)))],
                'batch': [batch_losses, list(range(len(batch_losses)))]
            },
            parameters={**vars(params), 'lambda': lam, 'num_epochs': num_epochs, 'batch_size': batch_size},
            info=None,
            model_parameters=self.net.params.cpu().numpy().tolist()
        )

        return results