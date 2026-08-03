import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..common_types import TupleND

__all__ = [
    'conv1d',
    'conv2d',
    'conv3d',
]


def _conv_nd(
    x: Tensor,
    weight: Tensor,  # (C_out, G_in, *kernel_size)
    bias: Tensor | None,  # (C_out,)
    stride: TupleND,
    padding: TupleND,
    padding_mode: str,
    dilation: TupleND,
    groups: int,
    ndim: int,
) -> Tensor:
    """Apply a ND convolution over an input image.

    Args:
        x (Tensor): Input tensor with shape `(N, C_in, *)`.
        weight (Tensor): Convolution kernel with shape `(C_out, C_in // groups, *)`.
        bias (Tensor | None): Optional additive bias with shape `(C_out,)`.
        stride (tuple[int, ...]): Stride of the sliding window.
        padding (tuple[int, ...]): Explicit padding in `F.pad` order.
        padding_mode (str): Padding mode. Supported values are `'zeros'`, `'reflect'`,
            `'replicate'`, and `'circular'`.
        dilation (tuple[int, ...]): Spacing between kernel elements.
        groups (int): Number of blocked connections from input channels to output channels.
        ndim (int): Number of spatial dimensions.

    Returns:
        Tensor: Output tensor with shape `(N, C_out, *)`.
    """
    kernel_size = tuple(weight.shape[2:])

    if any(value != 0 for value in padding):
        if padding_mode != 'zeros':
            x = F.pad(x, padding, mode=padding_mode)
        else:
            x = F.pad(x, padding)

    # Calculate output dimensions
    output_size = []
    for l, k, s, d in zip(x.shape[2:], kernel_size, stride, dilation, strict=True):
        output_size.append((l - d * (k - 1) - 1) // s + 1)
    output_dim = tuple(output_size)

    if any(size <= 0 for size in output_size):
        raise RuntimeError('Calculated output size is too small.')

    # Calculate effective kernel size considering dilation
    eff_kernel_size = []
    for k, d in zip(kernel_size, dilation, strict=True):
        eff_kernel_size.append(d * (k - 1) + 1)
    eff_kernel_size = tuple(eff_kernel_size)

    windows = x
    unfold_dims = tuple(range(2, ndim + 2))
    for dim, k, s in zip(unfold_dims, eff_kernel_size, stride, strict=True):
        # Assume x.shape = (N, C, H, W)
        # 1) Step 1: (N, C, H, W) -> (N, C, H_out, W, kH)
        # 2) Step 2: (N, C, H_out, W, kH) -> (N, C, H_out, W_out, kH, kW)
        windows = windows.unfold(dim, k, s)

    # Suppose dilation = (2, 2), then index = [:, :, :, :, ::2, ::2]
    index = [slice(None)] * (2 + ndim)
    for d in dilation:
        index.append(slice(None, None, d))
    # (N, C, H_out, W_out, kH, kW)
    windows = windows[tuple(index)]

    B, C_in = windows.shape[:2]
    G_in = C_in // groups
    C_out = weight.shape[0]
    G_out = C_out // groups

    L = math.prod(output_dim)  # L = H_out * W_out
    K = G_in * math.prod(kernel_size)  # K = G_in * kH * kW

    # (N, C, H_out, W_out, kH, kW) -> (N, groups, G_in, H_out, W_out, kH, kW)
    windows = windows.reshape(B, groups, G_in, *output_dim, *kernel_size)

    # (N, groups, G_in, H_out, W_out, kH, kW) -> (N, groups, H_out, W_out, G_in, kH, kW)
    output_axes = tuple(range(3, 3 + ndim))
    kernel_axes = tuple(range(3 + ndim, 3 + 2 * ndim))
    windows = windows.permute(0, 1, *output_axes, 2, *kernel_axes)

    # (N, groups, H_out, W_out, G_in, kH, kW) -> (N, groups, H_out * W_out, G_in * kH * kW)
    windows = windows.reshape(B, groups, L, K)

    # weight: (C_out, G_in, kH, kW)
    # (C_out, G_in, kH, kW) -> (groups, G_out, G_in * kH * kW)
    kernels = weight.reshape(groups, G_out, K)

    # y: (N, groups, G_out, H_out * W_out)
    y = torch.einsum('nglk,gok->ngol', windows, kernels)

    # (N, groups, G_out, H_out * W_out) -> (N, C_out, H_out, W_out)
    y = y.reshape(B, C_out, *output_dim)

    if bias is not None:
        # bias: (C_out,) -> (1, C_out, 1, 1)
        y = y + bias.reshape(1, -1, *((1,) * ndim))

    return y


def conv1d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: TupleND = (1,),
    padding: TupleND = (0, 0),
    padding_mode: str = 'zeros',
    dilation: TupleND = (1,),
    groups: int = 1,
) -> Tensor:
    """Apply a 1D convolution over an input signal.

    Args:
        x (Tensor): Input tensor with shape `(N, C_in, L_in)`.
        weight (Tensor): Convolution kernel with shape `(C_out, C_in // groups, K)`.
        bias (Tensor | None, default: None): Optional additive bias with shape `(C_out,)`.
        stride (tuple[int], default: (1,)): Stride of the sliding window.
        padding (tuple[int, int], default: (0, 0)): Explicit padding in `F.pad` order,
            as `(left, right)`.
        padding_mode (str, default: 'zeros'): Padding mode. Supported values are `'zeros'`,
            `'reflect'`, `'replicate'`, and `'circular'`.
        dilation (tuple[int], default: (1,)): Spacing between kernel elements.
        groups (int, default: 1): Number of blocked connections from input channels to
            output channels.

    Returns:
        Tensor: Output tensor with shape `(N, C_out, L_out)`.
    """
    return _conv_nd(
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        ndim=1,
    )


def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: TupleND = (1, 1),
    padding: TupleND = (0, 0, 0, 0),
    padding_mode: str = 'zeros',
    dilation: TupleND = (1, 1),
    groups: int = 1,
) -> Tensor:
    """Apply a 2D convolution over an input image.

    Args:
        x (Tensor): Input tensor with shape `(N, C_in, H_in, W_in)`.
        weight (Tensor): Convolution kernel with shape `(C_out, C_in // groups, K_h, K_w)`.
        bias (Tensor | None, default: None): Optional additive bias with shape `(C_out,)`.
        stride (tuple[int, int], default: (1, 1)): Stride of the sliding window.
        padding (tuple[int, int, int, int], default: (0, 0, 0, 0)): Explicit padding in
            `F.pad` order, as `(left, right, top, bottom)`.
        padding_mode (str, default: 'zeros'): Padding mode. Supported values are `'zeros'`,
            `'reflect'`, `'replicate'`, and `'circular'`.
        dilation (tuple[int, int], default: (1, 1)): Spacing between kernel elements.
        groups (int, default: 1): Number of blocked connections from input channels to
            output channels.

    Returns:
        Tensor: Output tensor with shape `(N, C_out, H_out, W_out)`.
    """
    return _conv_nd(
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        ndim=2,
    )


def conv3d(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: TupleND = (1, 1, 1),
    padding: TupleND = (0, 0, 0, 0, 0, 0),
    padding_mode: str = 'zeros',
    dilation: TupleND = (1, 1, 1),
    groups: int = 1,
) -> Tensor:
    """Apply a 3D convolution over an input volume.

    Args:
        x (Tensor): Input tensor with shape `(N, C_in, D_in, H_in, W_in)`.
        weight (Tensor): Convolution kernel with shape `(C_out, C_in // groups, K_d, K_h, K_w)`.
        bias (Tensor | None, default: None): Optional additive bias with shape `(C_out,)`.
        stride (tuple[int, int, int], default: (1, 1, 1)): Stride of the sliding window.
        padding (tuple[int, int, int, int, int, int], default: (0, 0, 0, 0, 0, 0)): Explicit
            padding in `F.pad` order, as `(left, right, top, bottom, front, back)`.
        padding_mode (str, default: 'zeros'): Padding mode. Supported values are `'zeros'`,
            `'reflect'`, `'replicate'`, and `'circular'`.
        dilation (tuple[int, int, int], default: (1, 1, 1)): Spacing between kernel elements.
        groups (int, default: 1): Number of blocked connections from input channels to output
            channels.

    Returns:
        Tensor: Output tensor with shape `(N, C_out, D_out, H_out, W_out)`.
    """
    return _conv_nd(
        x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        ndim=3,
    )
