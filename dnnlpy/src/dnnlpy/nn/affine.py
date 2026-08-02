import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

__all__ = [
    'Bilinear',
    'Flatten',
    'Identity',
    'Linear',
    'Unflatten',
]


class Identity(nn.Module):
    """A module that returns the input as is."""

    def __init__(self, *, fast: bool = False):
        """Initialize the Identity module."""
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        """Return the input as is."""
        return x


class Flatten(nn.Module):
    """Flatten a contiguous range of dimensions into one dimension."""

    def __init__(
        self,
        start_dim: int = 1,
        end_dim: int = -1,
        *,
        fast: bool = False,
    ):
        """Initialize the Flatten module.

        Args:
            start_dim (int, default: 1): First dimension to flatten.
            end_dim (int, default: -1): Last dimension to flatten.
            fast (bool, default: False): Preserved for API consistency; has no effect.
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'


class Unflatten(nn.Module):
    """Unflatten a dimension into a desired shape."""

    def __init__(
        self,
        dim: int,
        unflattened_size: tuple[int, ...],
        *,
        fast: bool = False,
    ):
        """Initialize the Unflatten module.

        Args:
            dim (int): Dimension to unflatten.
            unflattened_size (tuple[int, ...]): New shape of the unflattened dimension.
            fast (bool, default: False): Preserved for API consistency; has no effect.
        """
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        return x.unflatten(dim=self.dim, sizes=self.unflattened_size)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, unflattened_size={self.unflattened_size}'


class Linear(nn.Module):
    """Apply an affine transformation to the incoming data."""

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize the weight and optional bias parameters.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, default: True): Whether to learn an additive bias.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast = fast

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.linear(x, weight=self.weight, bias=self.bias)
        return dF.linear(x, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}'
        )


class Bilinear(nn.Module):
    """Apply a bilinear transformation to two incoming tensors."""

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in1_features: int,
        in2_features: int,
        out_features: int,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize the weight and optional bias parameters.

        Args:
            in1_features (int): Size of each first input sample.
            in2_features (int): Size of each second input sample.
            out_features (int): Size of each output sample.
            bias (bool, default: True): Whether to learn an additive bias.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.fast = fast

        self.weight = nn.Parameter(
            torch.empty(out_features, in1_features, in2_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in1_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        if self.fast:
            return F.bilinear(input1, input2, weight=self.weight, bias=self.bias)
        return dF.bilinear(input1, input2, weight=self.weight, bias=self.bias)

    def extra_repr(self) -> str:
        return (
            f'in1_features={self.in1_features}, '
            f'in2_features={self.in2_features}, '
            f'out_features={self.out_features}, '
            f'bias={self.bias is not None}'
        )
