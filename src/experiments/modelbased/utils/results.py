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

        'loss',  # Contains a dict with the different losses.
        'parameters',  # Parameters of the optimization algorithm.
        'info',  # A dict with any additional information.

        'model_parameters'  # Parameters of the model.
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def __dict__(self):
        return {var: getattr(self, var) for var in self.__slots__ if hasattr(self, var)}

    # TODO
    def info_text(self):
        """
        Returns a string containing all relevant information about the used methods, etc.
        Does not contain any model parameters or losses.

        :return: string.
        """
        return self.name + os.linesep + "mode=" + self.type + os.linesep + \
               format_dict(self.description) + os.linesep + format_dict(self.train_dataset) + os.linesep + \
               format_dict(self.parameters) + os.linesep + format_dict(self.info)
