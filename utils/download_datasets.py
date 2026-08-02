"""Download datasets used in the notes."""

import torchvision.datasets as datasets

import dnnlpy

ROOT = dnnlpy.get_data_root()


class DatasetDownloadError(RuntimeError):
    pass


try:
    ds = datasets.MNIST(ROOT, download=True)
except Exception as err:
    raise DatasetDownloadError('Error downloading MNIST dataset.') from err

try:
    ds = datasets.Caltech101(ROOT, download=True)
except Exception as err:
    raise DatasetDownloadError('Error downloading Caltech101 dataset.') from err
