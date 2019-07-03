import matplotlib

matplotlib.use('Agg')
import logging
import click

import modelbased.problems.mnist_classification.clirun as mnist_cls_clirun
import modelbased.utils.misc


@click.group()
def cli():
    pass


cli.add_command(mnist_cls_clirun.fixed)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(funcName)s - %(message)s")
    modelbased.utils.misc.make_folders()

    cli()
