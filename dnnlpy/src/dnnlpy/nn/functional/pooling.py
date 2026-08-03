import itertools as it
import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..common_types import SizeND, TupleND

__all__ = [
    'adaptive_avg_pool1d',
    'adaptive_avg_pool2d',
    'adaptive_avg_pool3d',
    'adaptive_max_pool1d',
    'adaptive_max_pool2d',
    'adaptive_max_pool3d',
    'avg_pool1d',
    'avg_pool2d',
    'avg_pool3d',
    'max_pool1d',
    'max_pool2d',
    'max_pool3d',
]


def _as_tuple(value: SizeND, ndim: int, name: str) -> TupleND:
    """Convert a size value to a tuple of length `ndim`."""
    if isinstance(value, int):
        return (value,) * ndim

    if len(value) != ndim:
        raise AssertionError(f'`{name}` must be an int or a tuple of {ndim} ints.')

    return tuple(value)


def _check_input(x: Tensor, ndim: int) -> None:
    """Check that the input tensor has the expected number of dimensions."""
    if x.ndim not in (ndim + 1, ndim + 2):
        raise AssertionError(
            f'Expected {ndim + 1}D or {ndim + 2}D input, '
            f'but got shape {tuple(x.shape)}.'
        )


def _pool_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    ceil_mode: bool,
) -> int:
    """Calculate the output size of a pooling operation.

    Args:
        input_size (int): Size of the input dimension.
        kernel_size (int): Size of the pooling window.
        stride (int): Stride of the pooling window.
        padding (int): Padding added to both sides.
        dilation (int): Spacing between pooling-window elements.
        ceil_mode (bool): If set to True, use ceil instead of floor to compute the
            output shape.

    Returns:
        int: Calculated output size.
    """
    effective_kernel = dilation * (kernel_size - 1) + 1
    numerator = input_size + 2 * padding - effective_kernel

    if ceil_mode:
        output_size = math.ceil(numerator / stride) + 1
        if (output_size - 1) * stride >= input_size + padding:
            output_size -= 1
    else:
        output_size = numerator // stride + 1

    if output_size <= 0:
        raise RuntimeError('Calculated output size is too small.')

    return output_size


def _adaptive_pool_output_size(
    input_size: TupleND,
    output_size: SizeND,
    ndim: int,
) -> TupleND:
    """Calculate the output size of an adaptive pooling operation.

    Args:
        input_size (TupleND): Size of the input dimensions.
        output_size (SizeND): Target output size.
        ndim (int): Number of spatial dimensions.

    Returns:
        TupleND: Calculated output size.
    """
    if isinstance(output_size, int):
        output_size = (output_size,) * ndim
    else:
        if len(output_size) != ndim:
            raise AssertionError(
                f'`output_size` must be an int or a tuple of {ndim} values.'
            )

    return tuple(
        l_in if l_out is None else l_out
        for l_in, l_out in zip(input_size, output_size, strict=True)
    )


def _pool_windows(
    x: Tensor,
    kernel_size: TupleND,
    stride: TupleND,
    padding: TupleND,
    dilation: TupleND,
    ceil_mode: bool,
    pad_value: float,
) -> tuple[Tensor, TupleND]:
    """Create a tensor of pooling windows from the input tensor.

    Args:
        x (Tensor): Input tensor of shape `(N, C, *)`.
        kernel_size (TupleND): Size of the pooling window.
        stride (TupleND): Stride of the pooling window.
        padding (TupleND): Symmetric padding added to each spatial dimension.
        dilation (TupleND): Spacing between pooling-window elements.
        ceil_mode (bool): If set to True, use ceil instead of floor to compute the
            output shape.
        pad_value (float): Value to use for padding.

    Returns:
        tuple[Tensor, TupleND]: A tuple containing the tensor of pooling windows and
            the extra padding applied to the input tensor.
    """
    ndim = len(kernel_size)
    _check_input(x, ndim)

    output_size = []
    z = zip(x.shape[-ndim:], kernel_size, stride, padding, dilation, strict=True)
    for l, k, s, p, g in z:
        output_size.append(_pool_output_size(l, k, s, p, g, ceil_mode=ceil_mode))
    output_size = tuple(output_size)

    eff_kernel_size = []
    for k, g in zip(kernel_size, dilation, strict=True):
        eff_kernel_size.append(g * (k - 1) + 1)
    eff_kernel_size = tuple(eff_kernel_size)

    extra_padding = []
    z = zip(x.shape[-ndim:], output_size, stride, eff_kernel_size, padding, strict=True)
    for l, o, s, eff, p in z:
        extra_padding.append(max(0, (o - 1) * s + eff - (l + 2 * p)))
    extra_padding = tuple(extra_padding)

    pad = []
    for left, extra in zip(reversed(padding), reversed(extra_padding), strict=True):
        pad.extend((left, left + extra))
    pad = tuple(pad)

    if any(pad):
        x = F.pad(x, pad, value=pad_value)

    windows = x
    spatial_dim = x.ndim - ndim
    z = zip(range(spatial_dim, x.ndim), eff_kernel_size, stride, strict=True)
    for dim, size, step in z:
        windows = windows.unfold(dim, size, step)

    index = [slice(None)] * (windows.ndim - ndim)
    index.extend(slice(None, None, gap) for gap in dilation)
    return windows[tuple(index)], extra_padding


def _avg_pool_nd(
    x: Tensor,
    kernel_size: TupleND,
    stride: TupleND,
    padding: TupleND,
    ceil_mode: bool,
    count_include_pad: bool,
    divisor_override: int | None,
) -> Tensor:
    """Apply an N-dimensional average pooling operation.

    Args:
        x (Tensor): Input tensor of shape `(N, C, *)`.
        kernel_size (TupleND): Size of the pooling window.
        stride (TupleND): Stride of the pooling window.
        padding (TupleND): Implicit zero padding to be added on both sides.
        ceil_mode (bool): If set to True, use ceil instead of floor to compute the
            output shape.
        count_include_pad (bool): If set to True, include zero padding in the averaging
            calculation.
        divisor_override (int | None): If specified, use this value as the divisor
            instead of the pooling region size.

    Returns:
        Tensor: Result of the average pooling operation.
    """
    ndim = len(kernel_size)
    windows, extra_padding = _pool_windows(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=(1,) * ndim,
        ceil_mode=ceil_mode,
        pad_value=0.0,
    )
    total = windows.sum(dim=tuple(range(-ndim, 0)))

    if divisor_override is not None:
        if divisor_override == 0:
            raise AssertionError('`divisor_override` must not be zero.')
        return total / divisor_override

    leading_dims = (1,) * (x.ndim - ndim)
    following_dims = []
    for length, pad in zip(x.shape[-ndim:], padding, strict=True):
        following_dims.append(length + 2 * pad)
    following_dims = tuple(following_dims)

    if count_include_pad:
        mask = x.new_ones(*leading_dims, *following_dims)

        mask_padding = []
        for extra in reversed(extra_padding):
            mask_padding.extend((0, extra))
        mask_padding = tuple(mask_padding)

    else:
        mask = x.new_ones(*leading_dims, *x.shape[-ndim:])

        mask_padding = []
        z = zip(reversed(padding), reversed(extra_padding), strict=True)
        for pad, extra in z:
            mask_padding.extend((pad, pad + extra))
        mask_padding = tuple(mask_padding)

    if any(mask_padding):
        mask = F.pad(mask, mask_padding)

    spatial_dim = mask.ndim - ndim
    z = zip(range(spatial_dim, mask.ndim), kernel_size, stride, strict=True)
    for dim, size, step in z:
        mask = mask.unfold(dim, size, step)

    divisor = mask.sum(dim=tuple(range(-ndim, 0)))
    return total / divisor


def _max_pool_nd(
    x: Tensor,
    kernel_size: TupleND,
    stride: TupleND,
    padding: TupleND,
    dilation: TupleND,
    ceil_mode: bool,
) -> Tensor:
    """Apply an N-dimensional max pooling operation.

    Args:
        x (Tensor): Input tensor of shape `(N, C, *)`.
        kernel_size (TupleND): Size of the pooling window.
        stride (TupleND): Stride of the pooling window.
        padding (TupleND): Implicit negative infinity padding added to both sides.
        dilation (TupleND): Spacing between pooling-window elements.
        ceil_mode (bool): If set to True, use ceil instead of floor to compute the
            output shape.

    Returns:
        Tensor: Result of the max pooling operation.
    """
    ndim = len(kernel_size)
    windows, _ = _pool_windows(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        pad_value=-math.inf,
    )
    windows = windows.flatten(start_dim=windows.ndim - ndim)
    return windows.max(dim=-1).values


def _adaptive_pool_nd(
    x: Tensor,
    output_size: SizeND,
    mode: str,
    ndim: int,
) -> Tensor:
    """Apply an N-dimensional adaptive pooling operation.

    Args:
        x (Tensor): Input tensor of shape `(N, C, *)`.
        output_size (SizeND): Target output size.
        mode (str): Pooling mode, either `'avg'` or `'max'`.
        ndim (int): Number of spatial dimensions.

    Returns:
        Tensor: Result of the adaptive pooling operation.
    """
    _check_input(x, ndim)

    input_size = tuple(x.shape[-ndim:])
    output_size = _adaptive_pool_output_size(input_size, output_size, ndim)

    if any(size <= 0 for size in output_size):
        raise AssertionError('`output_size` values must be greater than zero.')

    pooled = []
    for output_idx in it.product(*(range(size) for size in output_size)):
        slices = [slice(None)] * (x.ndim - ndim)
        z = zip(output_idx, x.shape[-ndim:], output_size, strict=True)

        for idx, l_in, l_out in z:
            start = math.floor(idx * l_in / l_out)
            end = math.ceil((idx + 1) * l_in / l_out)
            slices.append(slice(start, end))

        region = x[tuple(slices)]
        dims = tuple(range(-ndim, 0))

        if mode == 'avg':
            pooled.append(region.mean(dim=dims))
        elif mode == 'max':
            region = region.flatten(start_dim=region.ndim - ndim)
            pooled.append(region.max(dim=-1).values)
        else:
            raise NotImplementedError(f'Pooling mode `{mode}` is not implemented.')

    pooled = torch.stack(pooled, dim=-1)
    pooled = pooled.reshape(*x.shape[:-ndim], *output_size)
    return pooled


def avg_pool1d(
    x: Tensor,
    kernel_size: SizeND,
    stride: SizeND | None = None,
    padding: SizeND = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """Apply a 1D average pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, L)` or `(C, L)`.
        kernel_size (Size1D): Size of the pooling window.
        stride (Size1D, optional): Stride of the pooling window. If `None`, use
            `kernel_size`.
        padding (Size1D, default: 0): Implicit zero padding added to both sides.
        ceil_mode (bool, default: False): If set to True, use ceil instead of floor
            to compute the output shape.
        count_include_pad (bool, default: True): If set to True, include zero padding
            in the averaging calculation.

    Returns:
        Tensor: Average-pooled tensor with shape `(N, C, L_out)` or `(C, L_out)`.
    """
    kernel_size = _as_tuple(kernel_size, 1, 'kernel_size')
    stride = kernel_size if stride is None else _as_tuple(stride, 1, 'stride')
    padding = _as_tuple(padding, 1, 'padding')

    return _avg_pool_nd(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=None,
    )


def avg_pool2d(
    x: Tensor,
    kernel_size: SizeND,
    stride: SizeND | None = None,
    padding: SizeND = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Apply a 2D average pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, H, W)` or `(C, H, W)`.
        kernel_size (Size2D): Size of the pooling window.
        stride (Size2D, optional): Stride of the pooling window. If `None`, use
            `kernel_size`.
        padding (Size2D, default: 0): Implicit zero padding added to both sides.
        ceil_mode (bool, default: False): If set to True, use ceil instead of floor
            to compute the output shape.
        count_include_pad (bool, default: True): If set to True, include zero padding
            in the averaging calculation.
        divisor_override (int, optional): If specified, use this value as the divisor
            instead of the pooling region size.

    Returns:
        Tensor: Average-pooled tensor with shape `(N, C, H_out, W_out)` or
            `(C, H_out, W_out)`.
    """
    kernel_size = _as_tuple(kernel_size, 2, 'kernel_size')
    stride = kernel_size if stride is None else _as_tuple(stride, 2, 'stride')
    padding = _as_tuple(padding, 2, 'padding')

    return _avg_pool_nd(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


def avg_pool3d(
    x: Tensor,
    kernel_size: SizeND,
    stride: SizeND | None = None,
    padding: SizeND = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int | None = None,
) -> Tensor:
    """Apply a 3D average pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        kernel_size (Size3D): Size of the pooling window.
        stride (Size3D, optional): Stride of the pooling window. If `None`, use
            `kernel_size`.
        padding (Size3D, default: 0): Implicit zero padding added to both sides.
        ceil_mode (bool, default: False): If set to True, use ceil instead of floor
            to compute the output shape.
        count_include_pad (bool, default: True): If set to True, include zero padding
            in the averaging calculation.
        divisor_override (int, optional): If specified, use this value as the divisor
            instead of the pooling region size.

    Returns:
        Tensor: Average-pooled tensor with shape `(N, C, D_out, H_out, W_out)` or
            `(C, D_out, H_out, W_out)`.
    """
    kernel_size = _as_tuple(kernel_size, 3, 'kernel_size')
    stride = kernel_size if stride is None else _as_tuple(stride, 3, 'stride')
    padding = _as_tuple(padding, 3, 'padding')

    return _avg_pool_nd(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )


def max_pool1d(
    x: Tensor,
    kernel_size: SizeND,
    stride: SizeND | None = None,
    padding: SizeND = 0,
    dilation: SizeND = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """Apply a 1D max pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, L)` or `(C, L)`.
        kernel_size (Size1D): Size of the pooling window.
        stride (Size1D, optional): Stride of the pooling window. If `None`, use
            `kernel_size`.
        padding (Size1D, default: 0): Implicit negative infinity padding added to
            both sides.
        dilation (Size1D, default: 1): Spacing between kernel elements.
        ceil_mode (bool, default: False): If set to True, use ceil instead of floor
            to compute the output shape.

    Returns:
        Tensor: Max-pooled tensor with shape `(N, C, L_out)` or `(C, L_out)`.
    """
    kernel_size = _as_tuple(kernel_size, 1, 'kernel_size')
    stride = kernel_size if stride is None else _as_tuple(stride, 1, 'stride')
    padding = _as_tuple(padding, 1, 'padding')
    dilation = _as_tuple(dilation, 1, 'dilation')

    return _max_pool_nd(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def max_pool2d(
    x: Tensor,
    kernel_size: SizeND,
    stride: SizeND | None = None,
    padding: SizeND = 0,
    dilation: SizeND = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """Apply a 2D max pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, H, W)` or `(C, H, W)`.
        kernel_size (Size2D): Size of the pooling window.
        stride (Size2D, optional): Stride of the pooling window. If `None`, use
            `kernel_size`.
        padding (Size2D, default: 0): Implicit negative infinity padding added to
            both sides.
        dilation (Size2D, default: 1): Spacing between kernel elements.
        ceil_mode (bool, default: False): If set to True, use ceil instead of floor
            to compute the output shape.

    Returns:
        Tensor: Max-pooled tensor with shape `(N, C, H_out, W_out)` or
            `(C, H_out, W_out)`.
    """
    kernel_size = _as_tuple(kernel_size, 2, 'kernel_size')
    stride = kernel_size if stride is None else _as_tuple(stride, 2, 'stride')
    padding = _as_tuple(padding, 2, 'padding')
    dilation = _as_tuple(dilation, 2, 'dilation')

    return _max_pool_nd(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def max_pool3d(
    x: Tensor,
    kernel_size: SizeND,
    stride: SizeND | None = None,
    padding: SizeND = 0,
    dilation: SizeND = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """Apply a 3D max pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        kernel_size (Size3D): Size of the pooling window.
        stride (Size3D, optional): Stride of the pooling window. If `None`, use
            `kernel_size`.
        padding (Size3D, default: 0): Implicit negative infinity padding added to
            both sides.
        dilation (Size3D, default: 1): Spacing between kernel elements.
        ceil_mode (bool, default: False): If set to True, use ceil instead of floor
            to compute the output shape.

    Returns:
        Tensor: Max-pooled tensor with shape `(N, C, D_out, H_out, W_out)` or
            `(C, D_out, H_out, W_out)`.
    """
    kernel_size = _as_tuple(kernel_size, 3, 'kernel_size')
    stride = kernel_size if stride is None else _as_tuple(stride, 3, 'stride')
    padding = _as_tuple(padding, 3, 'padding')
    dilation = _as_tuple(dilation, 3, 'dilation')

    return _max_pool_nd(
        x,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )


def adaptive_avg_pool1d(x: Tensor, output_size: SizeND) -> Tensor:
    """Apply a 1D adaptive average pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, L)` or `(C, L)`.
        output_size (Size1D): Target output size. An integer is interpreted as
            `(output_size,)`.

    Returns:
        Tensor: Average-pooled tensor with shape `(N, C, L_out)` or `(C, L_out)`.
    """
    return _adaptive_pool_nd(x, output_size, mode='avg', ndim=1)


def adaptive_avg_pool2d(x: Tensor, output_size: SizeND) -> Tensor:
    """Apply a 2D adaptive average pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, H, W)` or `(C, H, W)`.
        output_size (Size2D): Target output size. An integer is interpreted as
            `(output_size, output_size)`.

    Returns:
        Tensor: Average-pooled tensor with shape `(N, C, H_out, W_out)` or
            `(C, H_out, W_out)`.
    """
    return _adaptive_pool_nd(x, output_size, mode='avg', ndim=2)


def adaptive_avg_pool3d(x: Tensor, output_size: SizeND) -> Tensor:
    """Apply a 3D adaptive average pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        output_size (Size3D): Target output size. An integer is interpreted as
            `(output_size, output_size, output_size)`.

    Returns:
        Tensor: Average-pooled tensor with shape `(N, C, D_out, H_out, W_out)` or
            `(C, D_out, H_out, W_out)`.
    """
    return _adaptive_pool_nd(x, output_size, mode='avg', ndim=3)


def adaptive_max_pool1d(x: Tensor, output_size: SizeND) -> Tensor:
    """Apply a 1D adaptive max pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, L)` or `(C, L)`.
        output_size (Size1D): Target output size. An integer is interpreted as
            `(output_size,)`.

    Returns:
        Tensor: Max-pooled tensor with shape `(N, C, L_out)` or `(C, L_out)`.
    """
    return _adaptive_pool_nd(x, output_size, mode='max', ndim=1)


def adaptive_max_pool2d(x: Tensor, output_size: SizeND) -> Tensor:
    """Apply a 2D adaptive max pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, H, W)` or `(C, H, W)`.
        output_size (Size2D): Target output size. An integer is interpreted as
            `(output_size, output_size)`.

    Returns:
        Tensor: Max-pooled tensor with shape `(N, C, H_out, W_out)` or
            `(C, H_out, W_out)`.
    """
    return _adaptive_pool_nd(x, output_size, mode='max', ndim=2)


def adaptive_max_pool3d(x: Tensor, output_size: SizeND) -> Tensor:
    """Apply a 3D adaptive max pooling operation.

    Args:
        x (Tensor): Input tensor with shape `(N, C, D, H, W)` or `(C, D, H, W)`.
        output_size (Size3D): Target output size. An integer is interpreted as
            `(output_size, output_size, output_size)`.

    Returns:
        Tensor: Max-pooled tensor with shape `(N, C, D_out, H_out, W_out)` or
            `(C, D_out, H_out, W_out)`.
    """
    return _adaptive_pool_nd(x, output_size, mode='max', ndim=3)
