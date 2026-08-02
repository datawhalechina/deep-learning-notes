import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..common_types import Size2D, Tuple2D

__all__ = ['fold', 'unfold']


def _as_tuple(value: Size2D) -> Tuple2D:
    """Convert an integer or a tuple of two integers into a tuple of two integers."""
    if isinstance(value, int):
        return (value, value)
    return value


def _count_sliding_blocks(
    spatial_size: Tuple2D,
    kernel_size: Tuple2D,
    dilation: Tuple2D,
    padding: Tuple2D,
    stride: Tuple2D,
) -> Tuple2D:
    """Calculate the number of sliding blocks along each spatial dimension."""
    output_size = []

    z = zip(spatial_size, kernel_size, dilation, padding, stride, strict=True)
    for l, k, d, p, s in z:
        kernel = d * (k - 1) + 1
        length = (l + 2 * p - kernel) // s + 1
        if length <= 0:
            raise RuntimeError('Calculated output size is too small.')
        output_size.append(length)

    return tuple(output_size)


def fold(
    x: Tensor,
    output_size: Size2D,
    kernel_size: Size2D,
    dilation: Size2D = 1,
    padding: Size2D = 0,
    stride: Size2D = 1,
) -> Tensor:
    """Combine sliding local blocks into a batched image tensor.

    Args:
        x (Tensor): Input tensor with shape `(N, C * prod(kernel_size), L)`.
        output_size (int | tuple[int, int]): Spatial size of the output image.
        kernel_size (int | tuple[int, int]): Size of each sliding block.
        dilation (int | tuple[int, int], default: 1): Spacing between kernel elements.
        padding (int | tuple[int, int], default: 0): Padding used when blocks were extracted.
        stride (int | tuple[int, int], default: 1): Stride used when blocks were extracted.

    Returns:
        Tensor: Tensor with shape `(N, C, *output_size)`.
    """
    output_size = _as_tuple(output_size)
    kernel_size = _as_tuple(kernel_size)
    dilation = _as_tuple(dilation)
    padding = _as_tuple(padding)
    stride = _as_tuple(stride)

    height, width = _count_sliding_blocks(
        output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    batch_size = x.size(0)
    channels = x.size(1) // math.prod(kernel_size)
    patches = x.reshape(batch_size, channels, *kernel_size, height, width)

    padded_output_size = (
        output_size[0] + 2 * padding[0],
        output_size[1] + 2 * padding[1],
    )

    output = torch.zeros(batch_size, channels, *padded_output_size)
    output = output.to(device=x.device, dtype=x.dtype)

    for i in range(kernel_size[0]):
        h_start = i * dilation[0]
        h_stop = h_start + height * stride[0]
        for j in range(kernel_size[1]):
            w_start = j * dilation[1]
            w_stop = w_start + width * stride[1]
            output[
                :, :, h_start : h_stop : stride[0], w_start : w_stop : stride[1]
            ] += patches[:, :, i, j]

    h_start = padding[0]
    h_stop = h_start + output_size[0]
    w_start = padding[1]
    w_stop = w_start + output_size[1]
    return output[:, :, h_start:h_stop, w_start:w_stop]


def unfold(
    x: Tensor,
    kernel_size: Size2D,
    dilation: Size2D = 1,
    padding: Size2D = 0,
    stride: Size2D = 1,
) -> Tensor:
    """Extract sliding local blocks from a batched image tensor.

    Args:
        x (Tensor): Input tensor with shape `(N, C, H, W)`.
        kernel_size (int | tuple[int, int]): Size of the sliding block.
        dilation (int | tuple[int, int], default: 1): Spacing between kernel elements.
        padding (int | tuple[int, int], default: 0): Implicit zero padding on both sides.
        stride (int | tuple[int, int], default: 1): Stride of the sliding block.

    Returns:
        Tensor: Tensor with shape `(N, C * prod(kernel_size), L)`.
    """
    kernel_size = _as_tuple(kernel_size)
    dilation = _as_tuple(dilation)
    padding = _as_tuple(padding)
    stride = _as_tuple(stride)

    if any(value != 0 for value in padding):
        x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]))

    eff_kernel_size = []
    for k, d in zip(kernel_size, dilation, strict=True):
        eff_kernel_size.append(d * (k - 1) + 1)
    eff_kernel_size = tuple(eff_kernel_size)

    windows = x.unfold(2, eff_kernel_size[0], stride[0])
    windows = windows.unfold(3, eff_kernel_size[1], stride[1])
    windows = windows[..., :: dilation[0], :: dilation[1]]

    batch_size, channels = windows.shape[:2]
    windows = windows.permute(0, 1, 4, 5, 2, 3)
    windows = windows.reshape(batch_size, channels * math.prod(kernel_size), -1)

    return windows
