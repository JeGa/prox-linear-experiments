import yaml
import os


def write(filename, losses, params):
    folder = 'modelbased/results/data'

    filepath = os.path.join(folder, filename)

    with open(filepath, 'w') as f:
        to_write = []

        paramsdict = {'parameters': params.__dict__}
        lossdict = {'data': losses}

        to_write.append(paramsdict)
        to_write.append(lossdict)

        yaml.dump(to_write, f, default_flow_style=False)
