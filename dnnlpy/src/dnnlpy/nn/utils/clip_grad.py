# pyright: reportPrivateImportUsage=false

from collections.abc import Iterable

import torch
from torch import Tensor, linalg

type TensorOrTensors = Tensor | Iterable[Tensor]

__all__ = ['clip_grad_norm_', 'clip_grad_value_']


@torch.no_grad()
def clip_grad_norm_(
    params: TensorOrTensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> Tensor:
    """Clip gradient norm in place and return the norm before clipping.

    Args:
        params (Tensor | Iterable[Tensor]): Parameters whose gradients are
            clipped.
        max_norm (float): Maximum permitted gradient norm.
        norm_type (float, default: 2.0): Order of the vector norm. May be `inf`.
        error_if_nonfinite (bool, default: False): Whether to raise when the total
            norm is non-finite.
        foreach (bool | None, default: None): Whether to use PyTorch's faster
            foreach tensor operations. `None` uses the simple implementation.

    Returns:
        The total gradient norm before clipping.
    """
    if isinstance(params, Tensor):
        params = [params]
    else:
        params = list(params)

    grads = [param.grad for param in params if param.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    norm_type = float(norm_type)
    if foreach:
        grad_norms = torch._foreach_norm(grads, norm_type)
    else:
        grad_norms = [linalg.vector_norm(grad, norm_type) for grad in grads]

    device = grads[0].device
    grad_norms = [grad_norm.to(device) for grad_norm in grad_norms]
    total_norm = linalg.vector_norm(torch.stack(grad_norms), norm_type)

    if error_if_nonfinite and not total_norm.isfinite():
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from `parameters` '
            'is non-finite, so it cannot be clipped.'
        )

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef = clip_coef.clamp(max=1.0)

    if foreach:
        torch._foreach_mul_(grads, clip_coef.to(device))
    else:
        for grad in grads:
            grad.mul_(clip_coef.to(grad.device))

    return total_norm


@torch.no_grad()
def clip_grad_value_(
    parameters: TensorOrTensors,
    clip_value: float,
    foreach: bool | None = None,
) -> None:
    """Clamp gradients in place to `[-clip_value, clip_value]`.

    Args:
        parameters (Tensor | Iterable[Tensor]): Parameters whose gradients are
            clipped.
        clip_value (float): Maximum absolute gradient value.
        foreach (bool | None, default: None): Whether to use PyTorch's faster
            foreach tensor operations. `None` uses the simple implementation.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    clip_value = float(clip_value)

    if foreach:
        torch._foreach_clamp_min_(grads, -clip_value)
        torch._foreach_clamp_max_(grads, clip_value)
    else:
        for grad in grads:
            grad.clamp_(min=-clip_value, max=clip_value)
