import torch
from typing import List
import torch.nn as nn


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "mnist":
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    else:
        raise Exception("Unknown dataset")


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

_MNIST_MEAN = [0.5]
_MNIST_STDDEV = [0.5]

_DEFAULT_MEAN = [0.5, 0.5, 0.5]
_DEFAULT_STDDEV = [0.5, 0.5, 0.5]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NormalizeLayer(torch.nn.Module):

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = nn.Parameter(torch.tensor(means), requires_grad=False)
        self.sds = nn.Parameter(torch.tensor(sds), requires_grad=False)

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2).contiguous()
        return (input - means) / sds
