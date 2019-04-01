import logging


def fixed_stepsize(f, net, params):
    """
    :param f: Function evaluating the net and loss, calling zero_grad, for example:

        def f():
            self.net.zero_grad()
            return self.loss(self.net(x), yt)

    :param params: Object with the following members:

        max_iter
        sigma
    """
    losses = []

    for i in range(params.max_iter):
        L = f()

        L.backward()

        for p in net.parameters():
            if p.grad is None:
                continue

            p.data.add_(-params.sigma, p.grad.data)

        loss = f().item()
        losses.append(loss)

        logging.info("Iteration {}/{}, Loss = {}.".format(i, params.max_iter, loss))

    return losses
