import yaml
import os

import modelbased.utils.global_config as cfg
import modelbased.utils.results


def write_result(results):
    """
    Dumps the complete results object into a yaml file.

    :param results: Object of type modelbased.utils.results.Results.
    :return: The filepath of the saved yaml file.
    """
    if not isinstance(results, modelbased.utils.results.Results):
        raise ValueError("Function needs Results object.")

    filepath = os.path.join(cfg.folders['data_' + results.type], results.name + '.yml')

    with open(filepath, 'w') as f:
        yaml.dump(vars(results), f, default_flow_style=False)

    return filepath


def load(name):
    """
    :param name: Name with .yml extension.
    :return: Loaded dict.
    """
    filepath = os.path.join(name)

    with open(filepath, 'r') as f:
        results = yaml.safe_load(f)

    return results
