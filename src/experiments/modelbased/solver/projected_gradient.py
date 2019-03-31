import numpy as np


def fixed_stepsize(x0, f, G, proj, params):
    """
    :param params: Object with the following members:

        max_iter
        sigma
    """
    losses = []

    x = x0

    for i in range(params.max_iter):
        grad = G(x)

        x = proj(x - params.sigma * grad)

        losses.append(f(x).squeeze())

    return x, losses


def armijo(x0, f, G, proj, params):
    """
    :param params: Object with the following members:

        max_iter
        eps
        beta
        gamma
        tau
        sigmamin
    """
    losses = []
    stop = False

    x = x0

    for i in range(params.max_iter):
        grad = G(x)
        dtau = x - proj(x - params.tau * grad)

        opt = np.sqrt((dtau ** 2)).sum() / params.tau
        if opt <= params.eps:
            break

        dk = -dtau

        # Armijo.
        i = 0
        while True:
            sigma = params.beta ** i

            if sigma < params.sigmamin:
                stop = True
                break

            if f(x + sigma * dk) - f(x) <= params.gamma * sigma * grad.T.dot(dk):
                x = x + sigma * dk
                break

            i += 1

        if stop:
            break

        loss = f(x).squeeze()
        losses.append(loss)

    return x, losses
