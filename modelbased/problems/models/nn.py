import torch
import torch.nn.functional as F

import scipy.optimize
import numpy as np


class BaseNetwork(torch.nn.Module):
    def __init__(self, device):
        super(BaseNetwork, self).__init__()

        self.current_u = None
        self.device = device

    def numparams(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def params(self):
        parameters = torch.empty(self.numparams(), device=self.device)

        def f(from_index, to_index, p):
            parameters[from_index:to_index] = p.view(-1)

        self._iter_params(f)

        return parameters.unsqueeze(1).detach()

    @params.setter
    def params(self, u):
        u = u.squeeze()

        with torch.no_grad():
            def f(from_index, to_index, p):
                p.copy_(u[from_index:to_index].view(p.shape))

            self._iter_params(f)

    def param_eval(self, u):
        """
        Use this in a 'with' statement to use the network with the given parameters and reset the parameters on exit
        to the old ones.

        For example:

        ::

            with self.param_eval(u):
                return self.forward(x)

        :param u: shape (d, 1) where d is the number of parameters.

        :return: self.
        """
        self.current_u = self.params
        self.params = u
        return self

    def __enter__(self):
        """
        Required to use the 'with' statement.
        """
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_u is None:
            raise UserWarning("No current_u parameter set.")
        else:
            self.params = self.current_u

    @property
    def gradient(self):
        g = torch.empty(self.numparams(), device=self.device)

        def f(from_index, to_index, p):
            g[from_index:to_index] = p.grad.view(-1)

        self._iter_params(f)

        return g.unsqueeze(1)

    def _iter_params(self, f):
        from_index = 0

        for p in self.parameters():
            size = p.numel()

            f(from_index, from_index + size, p)

            from_index += size

    def f(self, u, x, set_param=False):
        if set_param:
            self.params = u
            return self.forward(x)
        else:
            with self.param_eval(u):
                return self.forward(x)

    def Jacobian(self, u, x, retain_graph=False):
        """
        :param u: shape = (d, 1).
        :param x: Training input, shape depends on network architecture.

        :param retain_graph: If False, does not retain graph after call.

        :return: Jacobian with shape (c*n, d) in row-major order, where c is the output dimension and n is the number of
            samples.
        """
        with self.param_eval(u):
            self.zero_grad()

            y = self.forward(x)

            ysize = y.numel()
            J = torch.empty(ysize, self.numparams(), device=self.device)

            for i, yi in enumerate(y):  # Samples.
                for j, c in enumerate(yi):  # Classes.

                    # Free the graph at the last backward call.
                    if i == y.size(0) - 1 and j == yi.size(0) - 1:
                        c.backward(retain_graph=retain_graph)
                    else:
                        c.backward(retain_graph=True)

                    J[i * y.size(1) + j] = self.gradient.squeeze()

                    self.zero_grad()

            return J.detach()

    # def _check_grad(self, x): # TODO
    #     err = []
    #
    #     for i in range(x.shape[0]):
    #         for j in range(10):
    #             def f(u):
    #                 with torch.no_grad():
    #                     u = torch.as_tensor(u)
    #                     u = u.unsqueeze(1)
    #
    #                     fi = self.f(u, x)[i, j].squeeze().item()
    #
    #                 return fi
    #
    #             def g(u):
    #                 u = torch.as_tensor(u)
    #                 u = u.unsqueeze(1)
    #
    #                 Ji = self.Jacobian(u, x)[i + j].squeeze().detach().numpy()
    #
    #                 return Ji
    #
    #             u0 = self.params.squeeze().detach().numpy()
    #
    #             err.append(scipy.optimize.check_grad(f, g, u0))
    #             print(err)
    #
    #     print(np.mean(err))

    def forward(self, x):
        raise NotImplementedError()
