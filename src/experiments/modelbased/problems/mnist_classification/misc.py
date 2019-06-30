import numpy as np
import torchvision
import matplotlib.pyplot as plt


def imshow_grid(x, num_images=16):
    plt.figure()
    plt.imshow(np.transpose(torchvision.utils.make_grid(x[:num_images], padding=5, normalize=True), (1, 2, 0)))
    plt.show()


def sql2norm(x):
    return (x ** 2).sum().item()
