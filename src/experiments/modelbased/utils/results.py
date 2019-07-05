import os


def format_dict(_dict):
    if not _dict:
        return ''

    dict_items = iter(_dict.items())

    key, value = next(dict_items)

    formatted_text = "{}={}".format(key, value)

    for key, value in dict_items:
        formatted_text += os.linesep + "{}={}".format(key, value)

    return formatted_text


# TODO: All methods should use this format.
class Results:
    __slots__ = (
        'name',
        'type',  # train or test.
        'description',  # A dict describing the algorithm.
        'train_dataset',  # A dict with information about the training data set.

        'parameters',  # Parameters of the optimization algorithm.
        'info',  # A dict with any additional information.

        'loss',  # Contains a dict with the different losses.
        'model_parameters'  # Parameters of the model.
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def __dict__(self):
        return {var: getattr(self, var) for var in self.__slots__ if hasattr(self, var)}
