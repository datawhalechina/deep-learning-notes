from typing import override

import numpy as np

from .base import Module

__all__ = [
    'ReLU',
    'Sigmoid',
    'Softmax',
    'Tanh',
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply the logistic sigmoid elementwise."""
    return 1 / (1 + np.exp(-x))


class Sigmoid(Module):
    """Elementwise sigmoid activation layer."""

    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid and cache the output for backpropagation."""
        self.ctx = sigmoid(x)
        return self.ctx

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Return gradients through the sigmoid nonlinearity."""
        assert self.ctx is not None, 'Must call forward before backward.'
        return grad * self.ctx * (1 - self.ctx)


def tanh(x: np.ndarray) -> np.ndarray:
    """Apply hyperbolic tangent elementwise."""
    return np.tanh(x)


class Tanh(Module):
    """Elementwise hyperbolic tangent activation layer."""

    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply tanh and cache the output for backpropagation."""
        self.ctx = tanh(x)
        return self.ctx

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Return gradients through the tanh nonlinearity."""
        assert self.ctx is not None, 'Must call forward before backward.'
        return grad * (1 - np.square(self.ctx))


def relu(x: np.ndarray) -> np.ndarray:
    """Apply rectified linear activation elementwise."""
    return np.maximum(0, x)


class ReLU(Module):
    """Elementwise rectified linear activation layer."""

    def __init__(self):
        super().__init__()

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU and cache the input for backpropagation."""
        self.ctx = x
        return relu(x)

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Return gradients through the ReLU nonlinearity."""
        assert self.ctx is not None, 'Must call forward before backward.'
        return grad * (self.ctx > 0)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return row-wise softmax probabilities for batched logits."""
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class Softmax(Module):
    """Row-wise softmax activation layer."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax and cache probabilities for backpropagation."""
        self.ctx = softmax(x)
        return self.ctx

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Return gradients through the softmax Jacobian."""
        assert self.ctx is not None, 'Must call forward before backward.'
        dot = np.sum(grad * self.ctx, axis=self.dim, keepdims=True)
        return self.ctx * (grad - dot)

    def extra_repr(self) -> str:
        """Return extra string representation of the layer."""
        return f'dim={self.dim}'
