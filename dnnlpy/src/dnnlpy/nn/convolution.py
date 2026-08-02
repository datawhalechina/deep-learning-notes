import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF
from .common_types import (
    Padding1D,
    Padding2D,
    Padding3D,
    PaddingMode,
    PaddingND,
    Size1D,
    Size2D,
    Size3D,
    SizeND,
    TupleND,
)

__all__ = [
    'Conv1d',
    'Conv2d',
    'Conv3d',
]


def _as_tuple(value: SizeND, ndim: int, name: str) -> TupleND:
    """Convert a size value to a tuple of length `ndim`."""
    if isinstance(value, int):
        return (value,) * ndim

    if len(value) != ndim:
        raise AssertionError(f'`{name}` must be an int or a tuple of {ndim} ints.')

    return tuple(value)


def _padding_tuple(
    kernel_size: TupleND,
    padding: PaddingND,
    dilation: TupleND,
) -> TupleND:
    """Pad the input tensor according to the specified padding and kernel size."""
    ndim = len(kernel_size)

    if not isinstance(padding, str):
        padding = _as_tuple(padding, ndim, 'padding')

        # (depth, height, width) -> (left, right, top, bottom, front, back)
        return tuple(pad for val in reversed(padding) for pad in (val, val))

    if padding == 'valid':
        return (0,) * (2 * ndim)

    pad_list = []
    for k, d in zip(reversed(kernel_size), reversed(dilation), strict=True):
        pad = d * (k - 1)
        pad_list.extend((pad // 2, pad - pad // 2))

    return tuple(pad_list)


class _ConvNd(nn.Module):
    """Base class for both convolution and transposed convolution modules."""

    weight: Tensor
    bias: Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: TupleND,
        stride: TupleND,
        padding: TupleND,
        padding_mode: str,
        dilation: TupleND,
        groups: int,
        bias: bool,
        transposed: bool,
        output_padding: TupleND,
        *,
        fast: bool = False,
    ):
        """Initialize a ND convolution module.

        Do not instantiate this class directly. Use one of the subclasses instead.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (TupleND): Size of the convolving kernel. If a single integer
                is provided, then the kernel will be square (i.e., the same size in
                all dimensions).
            stride (TupleND, default: 1): Stride of the convolution.
            padding (TupleND, default: 0): Padding added to both sides of the input.
                String values `'valid'` and `'same'` are supported.
            padding_mode (str, default: 'zeros'): Padding mode. Supported values are
                `'zeros'`, `'reflect'`, `'replicate'`, and `'circular'`.
            dilation (TupleND, default: 1): Spacing between kernel elements.
            groups (int, default: 1): Number of blocked connections from input channels
                to output channels.
            bias (bool, default: True): If `True`, learn an additive bias.
            transposed (bool, default: False): If `True`, will use a transposed convolution
                (also known as a deconvolution).
            output_padding (TupleND, default: 0): Additional size added to one side of each
                dimension in the output shape. Only used for transposed convolutions.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups
        self.transposed = transposed
        self.output_padding = output_padding
        self.fast = fast

        if groups <= 0:
            raise AssertionError('`groups` must be a positive integer.')
        if in_channels % groups != 0:
            raise AssertionError('`in_channels` must be divisible by `groups`.')
        if out_channels % groups != 0:
            raise AssertionError('`out_channels` must be divisible by `groups`.')

        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise AssertionError(
                    f'Invalid padding string `{padding}`, should be one of '
                    f'{valid_padding_strings}.'
                )
            if padding == 'same' and any(s != 1 for s in stride):
                raise AssertionError(
                    'padding=`same` is not supported for strided convolutions.'
                )

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise AssertionError(
                f'`padding_mode` must be one of {valid_padding_modes}, but got '
                f'padding_mode=`{padding_mode}`.'
            )

        if transposed:
            size = (in_channels, out_channels // groups, *kernel_size)
            self.weight = nn.Parameter(torch.empty(size))
        else:
            size = (out_channels, in_channels // groups, *kernel_size)
            self.weight = nn.Parameter(torch.empty(size))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias)  # simplified version

    def extra_repr(self) -> str:
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv1d(_ConvNd):
    """Apply a 1D convolution over an input signal composed of several input planes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size1D,
        stride: Size1D = 1,
        padding: Padding1D = 0,
        padding_mode: PaddingMode = 'zeros',
        dilation: Size1D = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a 1D convolution module.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (Size1D, default: required): Size of the convolving kernel.
                If a single integer is provided, then the kernel will be square (i.e.,
                the same size in all dimensions).
            stride (Size1D, default: 1): Stride of the convolution.
            padding (Padding1D, default: 0): Padding added to both sides of the input.
                String values `'valid'` and `'same'` are supported.
            padding_mode (PaddingMode, default: 'zeros'): Padding mode. Supported values
                are `'zeros'`, `'reflect'`, `'replicate'`, and `'circular'`.
            dilation (Size1D, default: 1): Spacing between kernel elements.
            groups (int, default: 1): Number of blocked connections from input channels
                to output channels.
            bias (bool, default: True): If `True`, learn an additive bias.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        kernel_size_1d = _as_tuple(kernel_size, 1, 'kernel_size')
        stride_1d = _as_tuple(stride, 1, 'stride')
        dilation_1d = _as_tuple(dilation, 1, 'dilation')
        padding_1d = _padding_tuple(kernel_size_1d, padding, dilation_1d)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_1d,
            stride=stride_1d,
            padding=padding_1d,
            padding_mode=padding_mode,
            dilation=dilation_1d,
            groups=groups,
            bias=bias,
            transposed=False,
            output_padding=(0,),
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            x = F.pad(x, self.padding, mode=self.padding_mode)
            _conv_padding = (0,) * len(self.padding)
        else:
            _conv_padding = self.padding

        if self.fast:
            return F.conv1d(
                x,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=_conv_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return dF.conv1d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=_conv_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv2d(_ConvNd):
    """Applies a 2D convolution over an input signal composed of several input planes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size2D,
        stride: Size2D = 1,
        padding: Padding2D = 0,
        padding_mode: PaddingMode = 'zeros',
        dilation: Size2D = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a 2D convolution module.

        Args:
        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (Size2D, default: required): Size of the convolving kernel.
                If a single integer is provided, then the kernel will be square (i.e.,
                the same size in all dimensions).
            stride (Size2D, default: 1): Stride of the convolution.
            padding (Padding2D, default: 0): Padding added to both sides of the input.
                String values `'valid'` and `'same'` are supported.
            padding_mode (PaddingMode, default: 'zeros'): Padding mode. Supported values
                are `'zeros'`, `'reflect'`, `'replicate'`, and `'circular'`.
            dilation (Size2D, default: 1): Spacing between kernel elements.
            groups (int, default: 1): Number of blocked connections from input channels
                to output channels.
            bias (bool, default: True): If `True`, learn an additive bias.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        kernel_size_2d = _as_tuple(kernel_size, 2, 'kernel_size')
        stride_2d = _as_tuple(stride, 2, 'stride')
        dilation_2d = _as_tuple(dilation, 2, 'dilation')
        padding_2d = _padding_tuple(kernel_size_2d, padding, dilation_2d)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_2d,
            stride=stride_2d,
            padding=padding_2d,
            padding_mode=padding_mode,
            dilation=dilation_2d,
            groups=groups,
            bias=bias,
            transposed=False,
            output_padding=(0, 0),
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            x = F.pad(x, self.padding, mode=self.padding_mode)
            _conv_padding = (0,) * len(self.padding)
        else:
            _conv_padding = self.padding

        if self.fast:
            return F.conv2d(
                x,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=_conv_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return dF.conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=_conv_padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv3d(_ConvNd):
    """Applies a 3D convolution over an input signal composed of several input planes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Size3D,
        stride: Size3D = 1,
        padding: Padding3D = 0,
        padding_mode: PaddingMode = 'zeros',
        dilation: Size3D = 1,
        groups: int = 1,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a 3D convolution module.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (Size3D, default: required): Size of the convolving kernel.
                If a single integer is provided, then the kernel will be square (i.e.,
                the same size in all dimensions).
            stride (Size3D, default: 1): Stride of the convolution.
            padding (Padding3D, default: 0): Padding added to both sides of the input.
                String values `'valid'` and `'same'` are supported.
            padding_mode (PaddingMode, default: 'zeros'): Padding mode. Supported values
                are `'zeros'`, `'reflect'`, `'replicate'`, and `'circular'`.
            dilation (Size3D, default: 1): Spacing between kernel elements.
            groups (int, default: 1): Number of blocked connections from input channels
                to output channels.
            bias (bool, default: True): If `True`, learn an additive bias.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        kernel_size_3d = _as_tuple(kernel_size, 3, 'kernel_size')
        stride_3d = _as_tuple(stride, 3, 'stride')
        dilation_3d = _as_tuple(dilation, 3, 'dilation')
        padding_3d = _padding_tuple(kernel_size_3d, padding, dilation_3d)

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_3d,
            stride=stride_3d,
            padding=padding_3d,
            padding_mode=padding_mode,
            dilation=dilation_3d,
            groups=groups,
            bias=bias,
            transposed=False,
            output_padding=(0, 0, 0),
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != 'zeros':
            x = F.pad(x, self.padding, mode=self.padding_mode)
            _conv_padding = (0,) * len(self.padding)
        else:
            _conv_padding = self.padding

        if self.fast:
            return F.conv3d(
                x,
                weight=self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=_conv_padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        return dF.conv3d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=_conv_padding,
            dilation=self.dilation,
            groups=self.groups,
        )
