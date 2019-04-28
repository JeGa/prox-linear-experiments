import yaml
import os

import modelbased.utils.global_config as cfg


def write(filename, losses, params):
    filepath = os.path.join(cfg.folders['data'], filename + '.yml')

    with open(filepath, 'w') as f:
        to_write = []

        paramsdict = {'parameters': params.__dict__}
        lossdict = {'data': losses}

        to_write.append(paramsdict)
        to_write.append(lossdict)

        yaml.dump(to_write, f, default_flow_style=False)
