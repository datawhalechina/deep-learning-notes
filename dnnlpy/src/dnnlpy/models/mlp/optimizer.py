from collections.abc import Iterable
from typing import override

from .base import Optimizer, Parameter

__all__ = ['SGD']


class SGD(Optimizer):
    """Stochastic gradient descent optimizer for MLP parameters."""

    def __init__(self, params: Iterable[Parameter], lr: float = 0.1):
        """Initialize SGD.

        Args:
            params (Iterable[Parameter]): Parameters to update.
            lr (float, default: 0.1): Learning rate.
        """
        super().__init__(params)
        self.lr = lr

    @override
    def step(self):
        """Apply one in-place SGD update to parameters with gradients."""
        for p in self.params:
            if p.grad is None:
                continue

            p -= self.lr * p.grad
