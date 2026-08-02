from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    'Module',
    'Optimizer',
    'Parameter',
]


class Parameter(np.ndarray):
    """Trainable NumPy array that stores an optional gradient."""

    __array_priority__ = 1000  # type: ignore[assignment]

    grad: np.ndarray | None

    def __new__(cls, data: Any, dtype: npt.DTypeLike = np.float32):
        """Create a parameter from array-like data."""
        obj = np.asarray(data, dtype=dtype).view(cls)
        obj.grad = None
        return obj

    def __array_finalize__(self, obj: Any):
        """Preserve gradient state when NumPy creates parameter views."""
        if obj is None:
            return
        self.grad = getattr(obj, 'grad', None)

    def __array_wrap__(self, out_arr: Any, context=None, return_scalar=False):
        """Return plain arrays from ufunc results instead of parameters."""
        return np.asarray(out_arr)

    @property
    def data(self) -> np.ndarray:  # type: ignore[override]
        """Return this parameter as a plain NumPy array."""
        return np.asarray(self)


class Module(ABC):
    """Base class for NumPy MLP layers with manual backpropagation."""

    def __init__(self):
        """Initialize the layer context storage."""
        self.ctx = None

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Allow the layer to be called like a function."""
        return self.forward(*args, **kwargs)

    def extra_repr(self) -> str:
        """Return extra lines displayed inside `repr(module)`."""
        return ''

    def __repr__(self) -> str:
        """Return a compact module representation."""
        extra = self.extra_repr()

        if extra:
            lines = extra.split('\n')
            lines = ['    ' + line for line in lines]
            return f'{self.__class__.__name__}(\n' + '\n'.join(lines) + '\n)'

        return f'{self.__class__.__name__}()'

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Compute layer outputs from input values."""

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Propagate output gradients back to this layer's inputs."""

    def save_to_context(self, *args: Any):
        """Save any objects needed for backward pass in the layer context."""
        if isinstance(args, tuple) and len(args) == 1:
            self.ctx = args[0]
        else:
            self.ctx = args

    def load_from_context(self) -> Any:
        """Load saved objects from the layer context."""
        if self.ctx is None:
            raise AssertionError('No context found. Must call forward before backward.')
        return self.ctx

    def parameters(self) -> Iterator[Parameter]:
        """Yield trainable parameters from this module and child modules."""
        for value in self.__dict__.values():
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()


class Optimizer(ABC):
    """Base class for optimizers that update model parameters."""

    def __init__(self, params: Iterable[Parameter]):
        """Initialize the optimizer with the parameters to update."""
        self.params = list(params)

    @abstractmethod
    def step(self):
        """Update all parameters owned by this optimizer."""

    def zero_grad(self, set_to_none: bool = True):
        """Clear stored parameter gradients.

        Args:
            set_to_none (bool, default: True): If `True`, replace existing
                gradients with `None`. Otherwise, zero gradients in place.
        """
        for p in self.params:
            if p.grad is None:
                continue

            if set_to_none:
                p.grad = None
            else:
                p.grad.fill(0)
