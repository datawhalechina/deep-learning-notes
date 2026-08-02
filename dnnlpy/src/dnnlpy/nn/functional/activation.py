import math

import torch
from torch import Tensor

__all__ = [
    'celu',
    'elu',
    'gelu',
    'glu',
    'hardshrink',
    'hardsigmoid',
    'hardswish',
    'hardtanh',
    'leaky_relu',
    'log_sigmoid',
    'log_softmax',
    'mish',
    'prelu',
    'relu',
    'relu6',
    'rrelu',
    'selu',
    'sigmoid',
    'silu',
    'softmax',
    'softmin',
    'softplus',
    'softshrink',
    'softsign',
    'swiglu',
    'tanh',
    'tanhshrink',
    'threshold',
]


def _split_gated_input(x: Tensor, dim: int) -> tuple[Tensor, ...]:
    """Split an input into equally sized gate and value tensors."""
    if x.size(dim) % 2 != 0:
        raise AssertionError('The size of the split dimension must be even.')
    return x.chunk(2, dim=dim)


def celu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Apply the continuously differentiable exponential linear unit element-wise."""
    y = torch.where(x > 0, x, alpha * (x / alpha).expm1())
    if inplace:
        return x.copy_(y)
    return y


def elu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """Apply the exponential linear unit function element-wise."""
    y = torch.where(x > 0, x, alpha * x.expm1())
    if inplace:
        return x.copy_(y)
    return y


def gelu(x: Tensor, approximate: str = 'none') -> Tensor:
    """Apply the Gaussian Error Linear Unit function element-wise."""
    if approximate == 'tanh':
        scale = math.sqrt(2 / math.pi)
        return 0.5 * x * (1.0 + tanh(scale * (x + 0.044715 * x.pow(3))))
    else:
        return 0.5 * x * (1.0 + (x / math.sqrt(2)).erf())


def glu(x: Tensor, dim: int = -1) -> Tensor:
    """Apply the gated linear unit along the specified dimension."""
    value, gate = _split_gated_input(x, dim)
    return value * sigmoid(gate)


def hardshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    """Apply the hard shrinkage function element-wise."""
    return torch.where((x > lambd) | (x < -lambd), x, x.new_zeros(()))


def hardsigmoid(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the hard sigmoid function element-wise."""
    y = relu6(x + 3) / 6
    if inplace:
        return x.copy_(y)
    return y


def hardswish(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the hard swish function element-wise."""
    y = x * hardsigmoid(x)
    if inplace:
        return x.copy_(y)
    return y


def hardtanh(
    x: Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> Tensor:
    """Apply the hard hyperbolic tangent function element-wise."""
    if inplace:
        return x.clamp_(min=min_val, max=max_val)
    return x.clamp(min=min_val, max=max_val)


def leaky_relu(
    x: Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> Tensor:
    """Apply the leaky rectified linear unit function element-wise."""
    y = torch.where(x >= 0, x, x * negative_slope)
    if inplace:
        return x.copy_(y)
    return y


def log_softmax(x: Tensor, dim: int) -> Tensor:
    """Apply the log-softmax function along the specified dimension."""
    max_x = x.max(dim=dim, keepdim=True).values
    log_sum_exp = (x - max_x).logsumexp(dim=dim, keepdim=True)
    return x - max_x - log_sum_exp


def log_sigmoid(x: Tensor) -> Tensor:
    """Apply the log-sigmoid function element-wise."""
    return -softplus(-x)


def mish(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the mish function element-wise."""
    y = x * tanh(softplus(x))
    if inplace:
        return x.copy_(y)
    return y


def prelu(x: Tensor, weight: Tensor) -> Tensor:
    """Apply the parametric rectified linear unit function element-wise."""
    if weight.numel() == 1:
        negative_slope = weight.reshape(())
    else:
        if x.ndim < 2:
            raise AssertionError(
                'Expected input with at least 2 dimensions when '
                '`weight` has more than one element.'
            )
        negative_slope = weight.reshape(1, weight.numel(), *([1] * (x.ndim - 2)))

    return torch.where(x >= 0, x, x * negative_slope)


def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the rectified linear unit function element-wise."""
    if inplace:
        return x.clamp_(min=0)
    return x.clamp(min=0)


def relu6(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the rectified linear unit 6 function element-wise."""
    if inplace:
        return x.clamp_(min=0, max=6)
    return x.clamp(min=0, max=6)


def rrelu(
    x: Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = False,
) -> Tensor:
    """Apply the randomized leaky rectified linear unit function element-wise."""
    if training:
        negative_slope = lower + (upper - lower) * torch.rand_like(x)
    else:
        negative_slope = (lower + upper) / 2

    y = torch.where(x >= 0, x, x * negative_slope)
    if inplace:
        return x.copy_(y)
    return y


def selu(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the scaled exponential linear unit function element-wise."""
    scale = 1.0507009873554805
    alpha = 1.6732632423543772
    y = scale * elu(x, alpha=alpha)
    if inplace:
        return x.copy_(y)
    return y


def sigmoid(x: Tensor) -> Tensor:
    """Apply the sigmoid function element-wise."""
    nonneg = x >= 0
    exp_term = torch.where(nonneg, -x, x).exp()
    return torch.where(nonneg, 1 / (1 + exp_term), exp_term / (1 + exp_term))


def silu(x: Tensor, inplace: bool = False) -> Tensor:
    """Apply the sigmoid linear unit function element-wise."""
    if inplace:
        return x.mul_(sigmoid(x))
    return x * sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Apply the softmax function along the specified dimension."""
    max_x = x.max(dim=dim, keepdim=True).values
    exp_x = (x - max_x).exp()
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def softmin(x: Tensor, dim: int) -> Tensor:
    """Apply the softmin function along the specified dimension."""
    return softmax(-x, dim=dim)


def softplus(x: Tensor, beta: float = 1.0, threshold: float = 20.0) -> Tensor:
    """Apply the softplus function element-wise."""
    y = beta * x
    softplus = y.clamp(min=0) + (-y.abs()).exp().log1p()
    return torch.where(y > threshold, x, softplus / beta)


def softshrink(x: Tensor, lambd: float = 0.5) -> Tensor:
    """Apply the soft shrinkage function element-wise."""
    return torch.where(
        x > lambd,
        x - lambd,
        torch.where(x < -lambd, x + lambd, x.new_zeros(())),
    )


def softsign(x: Tensor) -> Tensor:
    """Apply the softsign function element-wise."""
    return x / (1 + x.abs())


def swiglu(x: Tensor, dim: int = -1) -> Tensor:
    """Apply the SwiGLU activation along the specified dimension."""
    gate, value = _split_gated_input(x, dim)
    return silu(gate) * value


def tanh(x: Tensor) -> Tensor:
    """Apply the hyperbolic tangent function element-wise."""
    return x.tanh()


def tanhshrink(x: Tensor) -> Tensor:
    """Apply the tanh shrinkage function element-wise."""
    return x - tanh(x)


def threshold(
    x: Tensor, threshold: float, value: float, inplace: bool = False
) -> Tensor:
    """Apply a threshold to each element of the input tensor."""
    y = torch.where(x > threshold, x, x.new_full((), value))
    if inplace:
        return x.copy_(y)
    return y
