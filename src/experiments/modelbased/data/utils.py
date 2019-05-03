import torch


def get_samples(dataloader, num_samples):
    """
    Extracts num_samples from the torch data loader.
    This is required because it only allows to get batches of samples and not a fixed size.

    :param dataloader: torch data loader.
    :param num_samples: Number of samples to extract.

    :return: (x, yt), input and ground truth samples with batch size = num_samples.
    """
    data_x = ()
    data_yt = ()

    samples = 0
    for x, yt in dataloader:
        data_x += (x,)
        data_yt += (yt,)

        samples += x.size(0)

        if samples >= num_samples:
            break

    if samples > num_samples:
        samples = num_samples

    x = torch.cat(data_x, 0)[0:samples]
    yt = torch.cat(data_yt, 0)[0:samples]

    return x, yt
