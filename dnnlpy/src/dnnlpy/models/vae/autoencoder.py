import math

import torch.nn as nn
from torch import Tensor

import dnnlpy.nn as dnn

__all__ = ['AutoEncoder']


class AutoEncoder(nn.Module):
    """A fully connected autoencoder for small image tensors."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
        """Initialize encoder and decoder networks.

        Args:
            input_shape (tuple[int, int, int]): Per-sample input shape, excluding batch size.
            hidden_dim (int, default: 256): Width of the hidden fully connected layer.
            latent_dim (int, default: 32): Size of the latent representation.
        """
        super().__init__()
        self.latent_dim = latent_dim

        input_dim = math.prod(input_shape)
        self.encoder = nn.Sequential(
            dnn.Flatten(),  # 28x28 -> 784
            dnn.Linear(input_dim, hidden_dim),
            dnn.ReLU(),
            dnn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            dnn.Linear(latent_dim, hidden_dim),
            dnn.ReLU(),
            dnn.Linear(hidden_dim, input_dim),
            dnn.Sigmoid(),
            dnn.Unflatten(1, input_shape),  # 784 -> 28x28
        )

    def encode(self, x: Tensor) -> Tensor:
        """Encode inputs into latent vectors."""
        z = self.encoder(x)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors back to input-shaped tensors."""
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct inputs through the encoder and decoder."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat
