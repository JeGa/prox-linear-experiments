import yaml
import os

import modelbased.utils.global_config as cfg


def write(results):
    """
    Dumps the complete results dict into a yaml file.


    :param results: Dictionary to save in yaml format.

        Needs to have a 'name' entry which is used for the filename.
    """
    filepath = os.path.join(cfg.folders['data'], results['name'] + '.yml')

    with open(filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
