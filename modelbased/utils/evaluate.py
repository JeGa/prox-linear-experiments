import logging

import modelbased.data.utils
import modelbased.utils.misc
import modelbased.utils.trainrun

logger = logging.getLogger(__name__)


def image_grid(filename, classificator, dataloader):
    # Predict with some training data.
    x, yt = modelbased.data.utils.get_samples(dataloader, 36)

    yt_predict = yt.argmax(1)
    y_predict = classificator.predict(x)

    modelbased.utils.misc.plot_grid(filename, x, y_predict, yt_predict)


def zero_one(classificator, x, yt):
    yt_predict = yt.argmax(1)
    y_predict = classificator.predict(x)

    return (yt_predict != y_predict).sum().item()


def zero_one_loader(classificator, dataloader):
    logging.info("Evaluating zero-one loss.")

    wrong = 0
    num_samples = 0

    def step_fun(x, yt):
        nonlocal wrong, num_samples

        wrong += zero_one(classificator, x, yt)
        num_samples += yt.size(0)

        return False

    def interval_fun(_, iteration, batch_iteration, _total_losses):
        logger.info("[{}:{}/{}] Evaluating zero-one loss: {}/{}.".format(iteration, batch_iteration, len(dataloader),
                                                                         wrong, num_samples))

    modelbased.utils.trainrun.run(1, dataloader, step_fun, classificator.net.device,
                                  interval_fun=interval_fun, interval=1)

    return wrong, num_samples


def zero_one_batch(classificator, x_all, y_all):
    wrong = zero_one(classificator, x_all, y_all)
    samples = x_all.size(0)

    return wrong, samples
