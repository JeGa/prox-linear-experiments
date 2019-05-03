def run(epochs, loader, step_fun, device, epoch_fun=None, interval_fun=None, interval=None):
    total_losses = []

    iteration = 0

    for i in range(epochs):

        batch_iteration = 0
        for x, yt in loader:
            x, yt = x.to(device), yt.to(device)

            loss = step_fun(x, yt)

            if loss:
                total_losses += loss

            if interval and interval_fun:
                if iteration % interval == 0:
                    interval_fun(i, iteration, batch_iteration, total_losses)

            iteration += 1
            batch_iteration += 1

        if epoch_fun:
            epoch_fun()

    return total_losses
