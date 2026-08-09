"""Download datasets used in the notes."""

import torchvision.datasets as datasets

import dnnlpy

ROOT = dnnlpy.get_data_root()
DATASETS = [
    datasets.MNIST,
    datasets.Caltech101,
]


def main():
    for dataset in DATASETS:
        try:
            dataset(ROOT, download=True)
        except Exception as err:
            message = f'Error downloading {dataset.__name__} dataset.'
            raise RuntimeError(message) from err


if __name__ == '__main__':
    main()
