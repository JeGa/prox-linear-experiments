import logging


def fixed_stepsize(x0, f, G, params):
    """
    :param params: Object with the following members:

        max_iter
        sigma
    """
    losses = []

    x = x0

    for i in range(params.max_iter):
        grad = G(x)

        x = x - params.sigma * grad

        loss = f(x)
        losses.append(loss)

        logging.info("Iteration {}/{}, Loss = {}.".format(i, params.max_iter, loss))

    return x, losses


def armijo(x0, f, G, params):
    """
    :param params: Object with the following members:

        max_iter
        beta
        gamma
    """
    losses = []

    x = x0

    for i in range(params.max_iter):
        grad = G(x)
        current_loss = f(x)

        j = 0
        while True:
            sigma = params.beta ** j

            x_new = x - sigma * grad
            new_loss = f(x_new)

            if new_loss - current_loss <= -sigma * params.gamma * (grad ** 2).sum():
                break

            j += 1

        x = x_new

        loss = f(x)
        losses.append(loss)

        logging.info("Iteration {}/{}, Loss = {}.".format(i, params.max_iter, loss))

    return x, losses
