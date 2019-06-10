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


# TODO: Remove this and use the new Results class.
def write(results):
    """
    Dumps the complete results dict into a yaml file.


    :param results: Dictionary to save in yaml format.

        Needs to have a 'name' entry which is used for the filename.
        Needs to have a 'type' entry which is used to determine the directory.
    """
    filepath = os.path.join(cfg.folders['data_' + results['type']], results['name'] + '.yml')

    with open(filepath, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)


def load(name):
    """
    :param name: Name with .yml extension.
    :return: Loaded dict.
    """
    filepath = os.path.join(name)

    with open(filepath, 'r') as f:
        results = yaml.safe_load(f)

    return results
