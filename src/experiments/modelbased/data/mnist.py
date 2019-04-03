import torch
import torch.utils.data as data
import torchvision.transforms as tf
import torchvision.datasets as datasets
import logging


def one_hot(x, num_classes):
    """
    :param x: Torch tensor with label for one sample.
    :param num_classes: Integer with total number of classes.

    :return: Torch tensor of shape (classes) with one-hot encoding.
    """
    onehot = torch.zeros(num_classes)
    onehot[x] = 1

    return onehot


def load(folder, training_samples=None, test_samples=None,
         training_batch_size=None, test_batch_size=None,
         one_hot_encoding=False):
    input_transform = tf.Compose([tf.ToTensor(), tf.Normalize([0.5], [0.5])])

    if one_hot_encoding:
        target_transform = tf.Lambda(lambda x: one_hot(x, 10))
    else:
        target_transform = None

    trainset = datasets.MNIST(root=folder, train=True, download=True,
                              transform=input_transform, target_transform=target_transform)
    testset = datasets.MNIST(root=folder, train=False, download=True,
                             transform=input_transform, target_transform=target_transform)

    if training_samples:
        trainset = data.Subset(trainset, range(training_samples))

    if test_samples:
        testset = data.Subset(testset, range(test_samples))

    if training_batch_size is not None and test_batch_size is not None:
        if training_batch_size > len(trainset):
            logging.warning("Training batch size larger than dataset.")
        if test_batch_size > len(testset):
            logging.warning("Test batch size larger than dataset.")

        trainloader = data.DataLoader(trainset, shuffle=True, batch_size=training_batch_size)
        testloader = data.DataLoader(testset, shuffle=True, batch_size=test_batch_size)
    else:
        training_sampler = data.sampler.SequentialSampler(trainset)
        test_sampler = data.sampler.SequentialSampler(testset)

        training_batch_size = len(trainset)
        test_batch_size = len(testset)

        trainloader = data.DataLoader(trainset, sampler=training_sampler, batch_size=training_batch_size)
        testloader = data.DataLoader(testset, sampler=test_sampler, batch_size=test_batch_size)

    classes = 10

    return trainloader, testloader, training_batch_size, test_batch_size, classes
