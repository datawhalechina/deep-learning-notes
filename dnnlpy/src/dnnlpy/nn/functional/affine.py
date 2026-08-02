import torch
from torch import Tensor

__all__ = [
    'bilinear',
    'linear',
]


def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Apply a linear transformation to the incoming data.

    Args:
        x (Tensor): Input tensor with shape `(*, in_features)`.
        weight (Tensor): Weight tensor with shape `(out_features, in_features)`.
        bias (Tensor | None, default: None): Optional bias tensor with shape `(out_features,)`.

    Returns:
        Tensor: Transformed tensor with shape `(*, out_features)`.
    """
    if x.ndim == 2 and bias is not None:
        return torch.addmm(bias, x, weight.T)
    elif bias is not None:
        return x @ weight.T + bias
    else:
        return x @ weight.T


def bilinear(
    x1: Tensor,
    x2: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
) -> Tensor:
    """Apply a bilinear transformation to two incoming tensors.

    Args:
        x1 (Tensor): First input tensor with shape `(*, in1_features)`.
        x2 (Tensor): Second input tensor with shape `(*, in2_features)`.
        weight (Tensor): Weight tensor with shape`(out_features, in1_features, in2_features)`.
        bias (Tensor | None, default: None): Optional bias tensor with shape `(out_features,)`.

    Returns:
        Tensor: Transformed tensor with shape `(*, out_features)`.
    """
    y = torch.einsum('...i,oij,...j->...o', x1, weight, x2)
    if bias is not None:
        y = y + bias
    return y
