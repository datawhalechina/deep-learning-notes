import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF
from .common_types import Size2D

__all__ = [
    'Fold',
    'Unfold',
]


class Fold(nn.Module):
    """Combine sliding local blocks into a batched image tensor."""

    def __init__(
        self,
        output_size: Size2D,
        kernel_size: Size2D,
        dilation: Size2D = 1,
        padding: Size2D = 0,
        stride: Size2D = 1,
        *,
        fast: bool = False,
    ):
        """Initialize a fold operation.

        Args:
            output_size (Size2D): Spatial size of the output image.
            kernel_size (Size2D): Size of each sliding block.
            dilation (Size2D, default: 1): Spacing between kernel elements.
            padding (Size2D, default: 0): Padding used when the blocks were extracted.
            stride (Size2D, default: 1): Stride used when the blocks were extracted.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.fold(
                x,
                output_size=self.output_size,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
        return dF.fold(
            x,
            output_size=self.output_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

    def extra_repr(self) -> str:
        return (
            f'output_size={self.output_size}, '
            f'kernel_size={self.kernel_size}, '
            f'dilation={self.dilation}, '
            f'padding={self.padding}, '
            f'stride={self.stride}'
        )


class Unfold(nn.Module):
    """Extract sliding local blocks from a batched image tensor."""

    def __init__(
        self,
        kernel_size: Size2D,
        dilation: Size2D = 1,
        padding: Size2D = 0,
        stride: Size2D = 1,
        *,
        fast: bool = False,
    ):
        """Initialize an unfold operation.

        Args:
            kernel_size (Size2D): Size of each sliding block.
            dilation (Size2D, default: 1): Spacing between kernel elements.
            padding (Size2D, default: 0): Implicit zero padding on both sides.
            stride (Size2D, default: 1): Stride of each sliding block.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.unfold(
                x,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )
        return dF.unfold(
            x,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )

    def extra_repr(self) -> str:
        return (
            f'kernel_size={self.kernel_size}, '
            f'dilation={self.dilation}, '
            f'padding={self.padding}, '
            f'stride={self.stride}'
        )
