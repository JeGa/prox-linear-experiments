import pathlib
import modelbased.problems.mnist_classification.misc
import modelbased.utils.global_config as cfg

plotfolder = pathlib.Path(cfg.folders['plots'])


def get_images(num):
    trainloader, testloader, train_size, test_size = modelbased.problems.mnist_classification.misc.data(num, num, num,
                                                                                                        num)
    x, _ = next(iter(trainloader))

    return x


modelbased.problems.mnist_classification.misc.image_grid(
    get_images(8), plotfolder / 'imgrid.pdf', nrow=8,
    size=None, margin=None)

modelbased.problems.mnist_classification.misc.image_grid(
    get_images(9), plotfolder / 'mnist-example.pdf', nrow=3, size=(1.1, 1.1),
    margin={'left': 0.02, 'right': 0.98, 'top': 0.98, 'bottom': 0.02})
