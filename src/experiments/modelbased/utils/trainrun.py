def run(epochs, loader, step_fun, epoch_fun=None, interval_fun=None, interval=None):
    total_losses = []

    iteration = 0

    for i in range(epochs):
        for x, yt in loader:
            total_losses += step_fun(x, yt)

            if interval and interval_fun:
                if iteration % interval == 0:
                    interval_fun(i, iteration, total_losses)

            iteration += 1

        if epoch_fun:
            epoch_fun()

    return total_losses
