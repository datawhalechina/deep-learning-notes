import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ['SinusoidalTimestepEmbedding']


class SinusoidalTimestepEmbedding(nn.Module):
    """Create sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, embedding_dim: int, max_period: int = 10000):
        """Initialize timestep embedding parameters.

        Args:
            embedding_dim (int): Size of each timestep embedding.
            max_period (int, default: 10000): Controls the minimum sinusoidal frequency.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return torch.zeros(
                timesteps.size(0),
                self.embedding_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )

        scale = -math.log(self.max_period) / max(half_dim - 1, 1)
        emb = torch.arange(half_dim, device=timesteps.device) * scale
        emb = timesteps.unsqueeze(1) * emb.exp().unsqueeze(0)
        emb = torch.concat([emb.sin(), emb.cos()], dim=-1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb
