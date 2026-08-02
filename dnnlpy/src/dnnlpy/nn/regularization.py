import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

__all__ = [
    'Dropout',
    'Dropout1d',
    'Dropout2d',
    'Dropout3d',
]


class Dropout(nn.Module):
    """Randomly zero elements of the input tensor."""

    def __init__(self, p: float = 0.5, inplace: bool = False, *, fast: bool = False):
        """Initialize the Dropout module.

        Args:
            p (float, default: 0.5): Probability of an element to be zeroed. Default: 0.5.
            inplace (bool, default: False): If set to True, will do this operation in-place.
                Default: False.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise AssertionError(f'`p` must be between 0 and 1, but got {p}.')

        self.p = p
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.dropout(x, self.p, self.training, self.inplace)
        return dF.dropout(x, self.p, self.training, self.inplace)

    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'


class Dropout1d(Dropout):
    """Randomly zero whole channels in 2D or 3D inputs."""

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.dropout1d(x, self.p, self.training, self.inplace)
        return dF.dropout1d(x, self.p, self.training, self.inplace)


class Dropout2d(Dropout):
    """Randomly zero whole channels in 3D or 4D inputs."""

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.dropout2d(x, self.p, self.training, self.inplace)
        return dF.dropout2d(x, self.p, self.training, self.inplace)


class Dropout3d(Dropout):
    """Randomly zero whole channels in 4D or 5D inputs."""

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.dropout3d(x, self.p, self.training, self.inplace)
        return dF.dropout3d(x, self.p, self.training, self.inplace)
