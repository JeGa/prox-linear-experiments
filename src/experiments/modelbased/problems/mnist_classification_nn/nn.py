import torch
import torch.nn.functional as F


class BaseNetwork(torch.nn.Module):
    def numparams(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def params(self):
        parameters = torch.empty(self.numparams())

        def f(from_index, to_index, p):
            parameters[from_index:to_index] = p.view(-1)

        self._iter_params(f)

        return parameters.unsqueeze(1)

    @params.setter
    def params(self, u):
        u = u.squeeze()
        with torch.no_grad():
            def f(from_index, to_index, p):
                p.copy_(u[from_index:to_index].view(p.shape))

            self._iter_params(f)

    def param_eval(self, u):
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
    def gradient(self):
        g = torch.empty(self.numparams())

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


class SimpleConvNetMNIST(BaseNetwork):
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

# TODO
# class SimpleFFNet(BaseNetwork):
#     def __init__(self, layers, h, criterion):
#         super(SimpleFFNet, self).__init__()
#
#         self.fclayers = []
#
#         for i in range(len(layers) - 2):
#             submodule = torch.nn.Linear(layers[i], layers[i + 1])
#
#             self.fclayers.append(submodule)
#             self.add_module('fc' + str(i), submodule)
#
#         self.linear_layer = torch.nn.Linear(layers[-2], layers[-1])
#
#         self.output_dim = layers[-1]
#
#         self.h = h
#         self.criterion = criterion
#         self.layer_size = layers
#
#         self.z = []
#         self.a = []
#
#     def forward(self, x):
#         self.z = []
#         self.a = []
#
#         for i in range(len(self.fclayers)):
#             if i == 0:
#                 zi = self.fclayers[i](x)
#             else:
#                 zi = self.fclayers[i](self.a[-1])
#             ai = self.h(zi)
#
#             self.z.append(zi)
#             self.a.append(ai)
#
#         zL = self.linear_layer(self.a[-1])
#         self.z.append(zL)
#
#         return zL
