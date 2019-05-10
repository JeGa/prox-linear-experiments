import pandas as pd
import torch.utils.data as data
import os


class BinarySpirals(data.Dataset):
    def __init__(self, folder, transform=None, target_transform=None):
        self.files = {'X_train': os.path.join(folder, 'binary_spirals_X_train'),
                      'y_train': os.path.join(folder, 'binary_spirals_y_train')}

        self.X_train = None
        self.y_train = None
        self.transform = transform
        self.target_transform = target_transform

        self.read_csv()

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, item):
        x = self.X_train[item, :]
        y = self.y_train[item, :]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def read_csv(self):
        self.X_train = pd.read_csv(self.files['X_train'], header=None).values.astype('float32').reshape(-1, 2)
        self.y_train = pd.read_csv(self.files['y_train'], header=None).values.astype('float32').reshape(-1, 2)


def load(folder, batch_size=None):
    """
    :param folder: Location of spirals data.
    :param batch_size: Batch size for the data loader, if None, batch size is the size of the data set (batch mode).

    :return: dataloader.
    """
    dataset = BinarySpirals(folder=folder)

    if batch_size:
        dataloader = data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    else:
        dataloader = data.DataLoader(dataset, sampler=data.sampler.SequentialSampler(dataset), batch_size=len(dataset))

    return dataloader, len(dataset)
