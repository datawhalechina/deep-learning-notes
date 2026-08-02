import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

type LossFn = Literal['mse', 'bce']

__all__ = ['VAE']


class VAE(nn.Module):
    """A fully connected variational autoencoder for small image tensors."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_dim: int = 256,
        latent_dim: int = 32,
    ):
        """Initialize encoder, latent heads, and decoder.

        Args:
            input_shape (tuple[int, int, int]): Per-sample input shape, excluding batch size.
            hidden_dim (int, default: 256): Width of the hidden fully connected layer.
            latent_dim (int, default: 32): Size of the Gaussian latent distribution.
        """
        super().__init__()
        self.latent_dim = latent_dim

        input_dim = math.prod(input_shape)
        self.encoder = nn.Sequential(
            dnn.Flatten(),
            dnn.Linear(input_dim, hidden_dim),
            dnn.ReLU(),
        )
        self.fc_mu = dnn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = dnn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            dnn.Linear(latent_dim, hidden_dim),
            dnn.ReLU(),
            dnn.Linear(hidden_dim, input_dim),
            dnn.Sigmoid(),
            dnn.Unflatten(1, input_shape),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode inputs into latent mean and log-variance tensors."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample latent vectors with the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = torch.addcmul(mu, std, eps)
        return latent

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vectors back to input-shaped tensors."""
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Reconstruct inputs and return reconstruction, mean, and log-variance."""
        mu, logvar = self.encode(x)
        latent = self.reparameterize(mu, logvar)
        x_hat = self.decode(latent)
        return x_hat, mu, logvar

    @staticmethod
    def loss(
        x_hat: Tensor,
        x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        loss_fn: LossFn = 'bce',
        beta: float = 1.0,
        normalize: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute VAE total, reconstruction, and KL losses per batch item.

        Args:
            x_hat (Tensor): Reconstructed samples.
            x (Tensor): Target samples.
            mu (Tensor): Latent Gaussian means.
            logvar (Tensor): Latent Gaussian log-variances.
            loss_fn (Literal['mse', 'bce'], default: 'bce'): Reconstruction loss type.
            beta (float, default: 1.0): Weight applied to the KL divergence term.
            normalize (bool, default: True): Whether to normalize losses by batch size.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple of `(loss, recon_loss, kl_loss)`
            normalized by batch size.
        """
        if loss_fn == 'mse':
            recon_loss = dF.mse_loss(x_hat, x, reduction='sum')
        elif loss_fn == 'bce':
            recon_loss = dF.bce_loss(x_hat, x, reduction='sum')
        else:
            raise NotImplementedError(f'Unsupported loss function: {loss_fn}.')

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        if normalize:
            B = x.size(0)
            return loss / B, recon_loss / B, kl_loss / B

        return loss, recon_loss, kl_loss
