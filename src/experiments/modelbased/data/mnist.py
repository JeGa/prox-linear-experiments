import torch.utils.data as data
import torchvision.transforms as tf
import torchvision.datasets as datasets
import logging


def load(folder, training_samples=None, test_samples=None,
         training_batch_size=None, test_batch_size=None,
         target_transform_function=None):
    """
    :param folder: Location of mnist data.
    :param training_samples: If only a subset of the training data is required.
    :param test_samples: If only a subset of the test data is required.

    :param training_batch_size: Batch size for training data.
    :param test_batch_size: Batch size for test data.

    If one batch size or both are None, the batch size is the length of the respective data set (batch mode).

    :param target_transform_function: Function which, if not None, is applied to the targets.
        ``For example one_hot_encoding=lambda x: utils.misc.one_hot(x, 10)``.

    :return: trainloader, testloader, training_batch_size, test_batch_size, classes.
    """
    input_transform = tf.Compose([tf.ToTensor(), tf.Normalize([0.5], [0.5])])

    if target_transform_function:
        target_transform = tf.Lambda(target_transform_function)
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
