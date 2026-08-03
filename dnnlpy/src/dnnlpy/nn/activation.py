from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

type Approx = Literal['none', 'tanh']

__all__ = [
    'CELU',
    'ELU',
    'GELU',
    'GLU',
    'SELU',
    'HardShrink',
    'HardSigmoid',
    'HardSwish',
    'HardTanh',
    'LeakyReLU',
    'LogSigmoid',
    'LogSoftmax',
    'Mish',
    'PReLU',
    'RReLU',
    'ReLU',
    'ReLU6',
    'SiLU',
    'Sigmoid',
    'SoftShrink',
    'SoftSign',
    'Softmax',
    'Softmin',
    'Softplus',
    'SwiGLU',
    'Tanh',
    'TanhShrink',
    'Threshold',
]


class CELU(nn.Module):
    """Apply the continuously differentiable exponential linear unit element-wise."""

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the CELU activation function.

        Args:
            alpha (float, default: 1.0): Multiplicative factor for negative inputs.
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.celu(x, alpha=self.alpha, inplace=self.inplace)
        return dF.celu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace = ', inplace=True' if self.inplace else ''
        return f'alpha={self.alpha}{inplace}'


class ELU(nn.Module):
    """Apply the exponential linear unit function element-wise."""

    def __init__(
        self,
        alpha: float = 1.0,
        inplace: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the ELU activation function.

        Args:
            alpha (float, default: 1.0): Multiplicative factor for negative inputs.
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.elu(x, alpha=self.alpha, inplace=self.inplace)
        return dF.elu(x, alpha=self.alpha, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace = ', inplace=True' if self.inplace else ''
        return f'alpha={self.alpha}{inplace}'


class GELU(nn.Module):
    """Apply the Gaussian Error Linear Unit function element-wise."""

    def __init__(self, approximate: Approx = 'none', *, fast: bool = False):
        """Initialize the GELU activation function.

        Args:
            approximate (Literal['none', 'tanh'], default: 'none'): The approximation
                method to use. Options are 'none' for the exact GELU function or 'tanh'
                for the tanh approximation.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.approximate = approximate
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.gelu(x, approximate=self.approximate)
        return dF.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f'approximate={self.approximate!r}'


class GLU(nn.Module):
    """Apply the gated linear unit along a specified dimension."""

    def __init__(self, dim: int = -1, *, fast: bool = False):
        """Initialize the gated linear unit.

        Args:
            dim (int, default: -1): Dimension along which to split the input.
            fast (bool, default: False): If set to True, use the implementation
                from :func:`torch.nn.functional.glu`.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if x.size(self.dim) % 2 != 0:
            raise AssertionError('The size of the split dimension must be even.')
        if self.fast:
            return F.glu(x, dim=self.dim)
        return dF.glu(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class HardShrink(nn.Module):
    """Apply the hard shrinkage function element-wise."""

    def __init__(self, lambd: float = 0.5, *, fast: bool = False):
        """Initialize the Hardshrink activation function.

        Args:
            lambd (float, default: 0.5): Lambda value for the shrinkage threshold.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.lambd = lambd
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.hardshrink(x, lambd=self.lambd)
        return dF.hardshrink(x, lambd=self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)


class HardSigmoid(nn.Module):
    """Apply the hard sigmoid function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the Hardsigmoid activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.hardsigmoid(x, inplace=self.inplace)
        return dF.hardsigmoid(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class HardSwish(nn.Module):
    """Apply the hard swish function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the Hardswish activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.hardswish(x, inplace=self.inplace)
        return dF.hardswish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class HardTanh(nn.Module):
    """Apply the hard hyperbolic tangent function element-wise."""

    def __init__(
        self,
        min_val: float = -1.0,
        max_val: float = 1.0,
        inplace: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the Hardtanh activation function.

        Args:
            min_val (float, default: -1.0): Minimum output value.
            max_val (float, default: 1.0): Maximum output value.
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.hardtanh(
                x,
                min_val=self.min_val,
                max_val=self.max_val,
                inplace=self.inplace,
            )
        return dF.hardtanh(
            x,
            min_val=self.min_val,
            max_val=self.max_val,
            inplace=self.inplace,
        )

    def extra_repr(self) -> str:
        inplace = ', inplace=True' if self.inplace else ''
        return f'min_val={self.min_val}, max_val={self.max_val}{inplace}'


class LeakyReLU(nn.Module):
    """Apply the leaky rectified linear unit function element-wise."""

    def __init__(
        self,
        negative_slope: float = 0.01,
        inplace: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the LeakyReLU activation function.

        Args:
            negative_slope (float, default: 0.01): Slope used for negative inputs.
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.leaky_relu(
                x,
                negative_slope=self.negative_slope,
                inplace=self.inplace,
            )
        return dF.leaky_relu(
            x,
            negative_slope=self.negative_slope,
            inplace=self.inplace,
        )

    def extra_repr(self) -> str:
        inplace = ', inplace=True' if self.inplace else ''
        return f'negative_slope={self.negative_slope}{inplace}'


class LogSigmoid(nn.Module):
    """Apply the log-sigmoid function element-wise."""

    def __init__(self, *, fast: bool = False):
        """Initialize the LogSigmoid activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.logsigmoid(x)
        return dF.log_sigmoid(x)


class LogSoftmax(nn.Module):
    """Apply the log-softmax function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        """Initialize the log-softmax activation function.

        Args:
            dim (int): Dimension along which log-softmax will be computed.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.log_softmax(x, dim=self.dim)
        return dF.log_softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Mish(nn.Module):
    """Apply the mish function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the Mish activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.mish(x, inplace=self.inplace)
        return dF.mish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class PReLU(nn.Module):
    """Apply the parametric rectified linear unit function element-wise."""

    def __init__(
        self,
        num_parameters: int = 1,
        init: float = 0.25,
        *,
        fast: bool = False,
    ):
        """Initialize the PReLU activation function.

        Args:
            num_parameters (int, default: 1): Number of learnable slope parameters.
            init (float, default: 0.25): Initial value of the learnable slopes.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.num_parameters = num_parameters
        self.init = init
        self.fast = fast
        self.weight = nn.Parameter(torch.empty(num_parameters))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, self.init)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.prelu(x, weight=self.weight)
        return dF.prelu(x, weight=self.weight)

    def extra_repr(self) -> str:
        return f'num_parameters={self.num_parameters}'


class ReLU(nn.Module):
    """Apply the rectified linear unit function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the ReLU activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.relu(x, inplace=self.inplace)
        return dF.relu(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class ReLU6(nn.Module):
    """Apply the rectified linear unit 6 function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the ReLU6 activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.relu6(x, inplace=self.inplace)
        return dF.relu6(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class RReLU(nn.Module):
    """Apply the randomized leaky rectified linear unit function element-wise."""

    def __init__(
        self,
        lower: float = 1.0 / 8,
        upper: float = 1.0 / 3,
        inplace: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the RReLU activation function.

        Args:
            lower (float, default: 1/8): Lower bound for the randomized slope.
            upper (float, default: 1/3): Upper bound for the randomized slope.
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.rrelu(
                x,
                lower=self.lower,
                upper=self.upper,
                training=self.training,
                inplace=self.inplace,
            )
        return dF.rrelu(
            x,
            lower=self.lower,
            upper=self.upper,
            training=self.training,
            inplace=self.inplace,
        )

    def extra_repr(self) -> str:
        inplace = ', inplace=True' if self.inplace else ''
        return f'lower={self.lower}, upper={self.upper}{inplace}'


class SELU(nn.Module):
    """Apply the scaled exponential linear unit function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the SELU activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.selu(x, inplace=self.inplace)
        return dF.selu(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class Sigmoid(nn.Module):
    """Apply the sigmoid function element-wise."""

    def __init__(self, *, fast: bool = False):
        """Initialize the sigmoid activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.sigmoid(x)
        return dF.sigmoid(x)


class SiLU(nn.Module):
    """Apply the sigmoid linear unit function element-wise."""

    def __init__(self, inplace: bool = False, *, fast: bool = False):
        """Initialize the SiLU activation function.

        Args:
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.silu(x, inplace=self.inplace)
        return dF.silu(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        return 'inplace=True' if self.inplace else ''


class Softmax(nn.Module):
    """Apply the softmax function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        """Initialize the softmax activation function.

        Args:
            dim (int): Dimension along which softmax will be computed.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softmax(x, dim=self.dim)
        return dF.softmax(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Softmin(nn.Module):
    """Apply the softmin function along a specified dimension."""

    def __init__(self, dim: int, *, fast: bool = False):
        """Initialize the softmin activation function.

        Args:
            dim (int): Dimension along which softmin will be computed.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softmin(x, dim=self.dim)
        return dF.softmin(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Softplus(nn.Module):
    """Apply the softplus function element-wise."""

    def __init__(
        self,
        beta: float = 1.0,
        threshold: float = 20.0,
        *,
        fast: bool = False,
    ):
        """Initialize the Softplus activation function.

        Args:
            beta (float, default: 1.0): Beta value for the softplus formulation.
            threshold (float, default: 20.0): Values above this revert to a linear
                function for numerical stability.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softplus(x, beta=self.beta, threshold=self.threshold)
        return dF.softplus(x, beta=self.beta, threshold=self.threshold)

    def extra_repr(self) -> str:
        return f'beta={self.beta}, threshold={self.threshold}'


class SoftShrink(nn.Module):
    """Apply the soft shrinkage function element-wise."""

    def __init__(self, lambd: float = 0.5, *, fast: bool = False):
        """Initialize the Softshrink activation function.

        Args:
            lambd (float, default: 0.5): Lambda value for the shrinkage threshold.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.lambd = lambd
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softshrink(x, lambd=self.lambd)
        return dF.softshrink(x, self.lambd)

    def extra_repr(self) -> str:
        return str(self.lambd)


class SoftSign(nn.Module):
    """Apply the softsign function element-wise."""

    def __init__(self, *, fast: bool = False):
        """Initialize the Softsign activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.softsign(x)
        return dF.softsign(x)


class SwiGLU(nn.Module):
    """Apply the SwiGLU activation along a specified dimension."""

    def __init__(self, dim: int = -1, *, fast: bool = False):
        """Initialize the SwiGLU activation.

        Args:
            dim (int, default: -1): Dimension along which to split the input.
            fast (bool, default: False): If set to True, use the fast SiLU
                implementation from :func:`torch.nn.functional.silu`.
        """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if x.size(self.dim) % 2 != 0:
            raise AssertionError('The size of the split dimension must be even.')

        if self.fast:
            gate, value = x.chunk(2, dim=self.dim)
            return F.silu(gate) * value

        return dF.swiglu(x, dim=self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Tanh(nn.Module):
    """Apply the hyperbolic tangent function element-wise."""

    def __init__(self, *, fast: bool = False):
        """Initialize the hyperbolic tangent activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.tanh(x)
        return dF.tanh(x)


class TanhShrink(nn.Module):
    """Apply the tanh shrinkage function element-wise."""

    def __init__(self, *, fast: bool = False):
        """Initialize the Tanhshrink activation function.

        Args:
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.tanhshrink(x)
        return dF.tanhshrink(x)


class Threshold(nn.Module):
    """Apply a threshold to each element of the input tensor."""

    def __init__(
        self,
        threshold: float,
        value: float,
        inplace: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the Threshold activation function.

        Args:
            threshold (float): Values at or below this threshold are replaced.
            value (float): Replacement value for thresholded elements.
            inplace (bool, default: False): Whether to perform the operation in-place.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.threshold(
                x,
                threshold=self.threshold,
                value=self.value,
                inplace=self.inplace,
            )
        return dF.threshold(
            x,
            threshold=self.threshold,
            value=self.value,
            inplace=self.inplace,
        )

    def extra_repr(self) -> str:
        inplace = ', inplace=True' if self.inplace else ''
        return f'threshold={self.threshold}, value={self.value}{inplace}'
