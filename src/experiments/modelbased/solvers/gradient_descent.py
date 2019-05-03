import logging

logger = logging.getLogger(__name__)


def fixed_stepsize(x0, f, G, params, verbose=False):
    """
    :param x0: Initial guess.
    :param f: Function to minimize.
    :param G: Function evaluating the gradient of f.

    :param params: Object with the following members:

        max_iter
        sigma

    :param verbose: If True, logs information as INFO.
    """
    losses = []

    x = x0

    for i in range(params.max_iter):
        grad = G(x)

        x = x - params.sigma * grad

        loss = f(x)
        losses.append(loss)

        if verbose:
            logger.info("Iteration {}/{}, Loss = {}.".format(i, params.max_iter, loss))

    return x, losses


def armijo(x0, f, G, params, verbose=False):
    """
    :param x0: Initial guess.
    :param f: Function to minimize.
    :param G: Function evaluating the gradient of f.

    :param params: Object with the following members:

        max_iter
        beta
        gamma

    :param verbose: If True, logs information as INFO.
    """
    losses = []

    x = x0

    terminate = False

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

            if sigma <= 1e-8:
                terminate = True
                break

            j += 1

        # No stepsize found with sufficiently large decrease.
        if terminate:
            if verbose:
                logger.info("Terminating since no suitable stepsize is found.")
            break

        x = x_new

        loss = f(x)
        losses.append(loss)

        if verbose:
            logger.info("Iteration {}/{}, Loss = {}, sigma = {}.".format(i, params.max_iter, loss, sigma))

    return x, losses
