import torch
from torch import Tensor

__all__ = [
    'dropout',
    'dropout1d',
    'dropout2d',
    'dropout3d',
]


def _dropout_nd(
    x: Tensor,
    p: float,
    training: bool,
    inplace: bool,
    mask_shape: tuple[int, ...],
) -> Tensor:
    """Randomly zero elements of the input tensor with a given mask shape.

    Args:
        x (Tensor): Input tensor with arbitrary shape.
        p (float, default: 0.5): Probability of an element to be zeroed. Default: 0.5.
        training (bool, default: True): Apply dropout if True. Default: True.
        inplace (bool, default: False): If set to True, will do this operation in-place.
            Default: False.
        mask_shape (tuple[int, ...]): Shape of the dropout mask.

    Returns:
        Tensor: Output tensor with the same shape as input.
    """
    if not 0.0 <= p <= 1.0:
        raise AssertionError(f'`p` must be between 0 and 1, but got {p}.')

    if not training or p == 0.0:
        return x

    if p == 1.0:
        if inplace:
            return x.zero_()
        return torch.zeros_like(x)

    keep = 1.0 - p
    mask = torch.rand(mask_shape, device=x.device, dtype=x.dtype) < keep

    if inplace:
        return x.mul_(mask).div_(keep)

    return x * mask / keep


def dropout(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    """Randomly zero elements of the input tensor.

    Args:
        x (Tensor): Input tensor with arbitrary shape.
        p (float, default: 0.5): Probability of an element to be zeroed. Default: 0.5.
        training (bool, default: True): Apply dropout if True. Default: True.
        inplace (bool, default: False): If set to True, will do this operation in-place.
            Default: False.

    Returns:
        Tensor: Output tensor with the same shape as input.
    """
    return _dropout_nd(x, p, training, inplace, x.shape)


def dropout1d(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    """Randomly zero whole channels in 2D or 3D inputs.

    Args:
        x (Tensor): Input tensor of shape (N, C, L) or (C, L).
        p (float, default: 0.5): Probability of an element to be zeroed. Default: 0.5.
        training (bool, default: True): Apply dropout if True. Default: True.
        inplace (bool, default: False): If set to True, will do this operation in-place.
            Default: False.

    Returns:
        Tensor: Output tensor with the same shape as input.
    """
    if x.ndim not in (2, 3):
        raise AssertionError(
            f'`F.dropout1d` expected 2D or 3D input, got {x.ndim}D input.'
        )

    if x.ndim == 3:
        mask_shape = (x.size(0), x.size(1), 1)
    else:
        mask_shape = (x.size(0), 1)

    return _dropout_nd(x, p, training, inplace, mask_shape)


def dropout2d(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    """Randomly zero whole channels in 3D or 4D inputs.

    Args:
        x (Tensor): Input tensor of shape (N, C, H, W) or (C, H, W).
        p (float, default: 0.5): Probability of an element to be zeroed. Default: 0.5.
        training (bool, default: True): Apply dropout if True. Default: True.
        inplace (bool, default: False): If set to True, will do this operation in-place.
            Default: False.

    Returns:
        Tensor: Output tensor with the same shape as input.
    """
    if x.ndim not in (3, 4):
        raise AssertionError(
            f'`F.dropout2d` expected 3D or 4D input, got {x.ndim}D input.'
        )

    if x.ndim == 4:
        mask_shape = (x.size(0), x.size(1), 1, 1)
    else:
        mask_shape = (x.size(0), 1, 1)

    return _dropout_nd(x, p, training, inplace, mask_shape)


def dropout3d(
    x: Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> Tensor:
    """Randomly zero whole channels in 4D or 5D inputs.

    Args:
        x (Tensor): Input tensor of shape (N, C, D, H, W) or (C, D, H, W).
        p (float, default: 0.5): Probability of an element to be zeroed. Default: 0.5.
        training (bool, default: True): Apply dropout if True. Default: True.
        inplace (bool, default: False): If set to True, will do this operation in-place.
            Default: False.

    Returns:
        Tensor: Output tensor with the same shape as input.
    """
    if x.ndim not in (4, 5):
        raise AssertionError(
            f'`F.dropout3d` expected 4D or 5D input, got {x.ndim}D input.'
        )

    if x.ndim == 5:
        mask_shape = (x.size(0), x.size(1), 1, 1, 1)
    else:
        mask_shape = (x.size(0), 1, 1, 1)

    return _dropout_nd(x, p, training, inplace, mask_shape)
