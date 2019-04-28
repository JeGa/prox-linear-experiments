import yaml
import os

import modelbased.utils.global_config as cfg
import modelbased.utils.misc


def write(filename, results):
    filename = modelbased.utils.misc.append_time(filename)
    filepath = os.path.join(cfg.folders['data'], filename + '.yml')

    with open(filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
