import torch
import torch.nn.functional as F


class SimpleConvNetMNIST(torch.nn.Module):
    def __init__(self, h):
        super(SimpleConvNetMNIST, self).__init__()

        self.output_dim = 10

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

        self.h = h

        self.current_u = None

    def forward(self, x):
        x = self.h(F.max_pool2d(self.conv1(x), 2))
        x = self.h(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)

        x = self.h(self.fc1(x))
        x = self.fc2(x)

        return x

    def numparams(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def params(self):
        parameters = torch.empty(self.numparams())

        def f(from_index, to_index, p):
            parameters[from_index:to_index] = p.view(-1)

        self._iter_params(f)

        return parameters

    @params.setter
    def params(self, u):
        with torch.no_grad():
            def f(from_index, to_index, p):
                p.copy_(u[from_index:to_index].view(p.shape))

            self._iter_params(f)

    def no_param_set(self, u):
        self.current_u = self.params
        self.params = u
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_u is None:
            raise UserWarning("No current_u parameter set.")
        else:
            self.params = self.current_u

    @property
    def gradients(self):
        g = torch.empty(self.numparams())

        def f(from_index, to_index, p):
            g[from_index:to_index] = p.grad.view(-1)

        self._iter_params(f)

        return g

    def _iter_params(self, f):
        from_index = 0

        for p in self.parameters():
            size = p.numel()

            f(from_index, from_index + size, p)

            from_index += size
