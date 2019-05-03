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


def zero_one(classificator, dataloader):
    logging.info("Evaluating zero-one loss.")

    correct = 0
    num_samples = 0

    def step_fun(x, yt):
        nonlocal correct, num_samples

        yt_predict = yt.argmax(1)
        y_predict = classificator.predict(x)

        correct += (yt_predict == y_predict).sum().item()
        num_samples += yt.size(0)

        return None

    def interval_fun(_, iteration, batch_iteration, _total_losses):
        logger.info("[{}:{}/{}] Evaluating zero-one loss: {}/{}.".format(iteration, batch_iteration, len(dataloader),
                                                                         correct, num_samples))

    modelbased.utils.trainrun.run(1, dataloader, step_fun, classificator.net.device,
                                  interval_fun=interval_fun, interval=1)

    return correct, num_samples
