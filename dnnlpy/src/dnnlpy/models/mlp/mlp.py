from typing import override

import numpy as np

from .activation import ReLU
from .base import Module
from .layer import Linear

__all__ = ['MLP']


class MLP(Module):
    """Two-layer MLP classifier with a ReLU hidden layer."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        """Initialize the MLP.

        Args:
            input_dim (int): Number of input features per sample.
            hidden_dim (int): Number of hidden units.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, num_classes)

    @override
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return class logits for a batch of inputs."""
        h = self.fc1(x)
        a = self.relu(h)
        logits = self.fc2(a)
        return logits

    @override
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate gradients through both linear layers and ReLU."""
        grad = self.fc2.backward(grad)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad)
        return grad
