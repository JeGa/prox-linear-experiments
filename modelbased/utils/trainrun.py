# TODO: Remove the losses return value.
# TODO: Change usages.


def run(epochs, loader, step_fun, device, epoch_fun=None, interval_fun=None, interval=None):
    iteration = 0

    for i in range(epochs):
        batch_iteration = 0
        for x, yt in loader:
            x, yt = x.to(device), yt.to(device)

            stop = step_fun(x, yt)

            if interval and interval_fun:
                if iteration % interval == 0:
                    interval_fun(i, iteration, batch_iteration)

            if stop:
                break

            iteration += 1
            batch_iteration += 1

        if epoch_fun:
            epoch_fun()
