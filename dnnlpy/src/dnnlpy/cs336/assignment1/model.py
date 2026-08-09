"""Transformer components used by CS336 Assignment 1."""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

__all__ = [
    'MultiheadSelfAttention',
    'RotaryPositionalEmbedding',
    'SwiGLU',
    'TransformerBlock',
    'TransformerLM',
]


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """Create a SwiGLU feed-forward network.

        Args:
            embed_dim (int): Dimension of the input token embeddings.
            hidden_dim (int): Dimension of the hidden projections.
            bias (bool, default: True): Whether to use bias terms in linear layers.
            dropout (float, default: 0.0): Dropout probability for the output.
        """
        super().__init__()
        self.fc1 = dnn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc2 = dnn.Linear(hidden_dim, embed_dim, bias=bias)
        self.fc3 = dnn.Linear(embed_dim, hidden_dim, bias=bias)
        self.dropout = dnn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc2(dF.silu(self.fc1(x)) * self.fc3(x))
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional embeddings to adjacent feature pairs."""

    sin: Tensor
    cos: Tensor

    def __init__(self, embed_dim: int, theta: float = 10000.0, max_seq_len: int = 2048):
        """Create a rotary positional embedding.

        Args:
            embed_dim (int): Dimension of each query or key head.
            theta (float): Base used to construct the rotation frequencies.
            max_seq_len (int): Maximum supported sequence length.
        """
        super().__init__()
        if embed_dim % 2 != 0:
            raise AssertionError('`embed_dim` must be even.')
        if theta <= 0:
            raise AssertionError('`theta` must be positive.')
        if max_seq_len <= 0:
            raise AssertionError('`max_seq_len` must be positive.')

        self.embed_dim = embed_dim
        self.theta = theta
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.arange(0, embed_dim, 2, dtype=torch.float32)
        frequencies = torch.pow(theta, -frequencies / embed_dim)
        angles = positions[:, None] * frequencies[None, :]

        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_pos: Tensor | None = None) -> Tensor:
        """Rotate the final dimension of a query or key tensor."""
        T = x.size(-2)

        if token_pos is None:
            token_pos = torch.arange(T, device=x.device)
        else:
            token_pos = token_pos.to(device=x.device, dtype=torch.long)

        if token_pos.min() < 0 or token_pos.max() >= self.max_seq_len:
            raise AssertionError('Token positions must be within `max_seq_len`.')

        cos = self.cos[token_pos].to(dtype=x.dtype)
        sin = self.sin[token_pos].to(dtype=x.dtype)

        while cos.ndim < x.ndim - 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odds = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odds * sin
        output[..., 1::2] = x_even * sin + x_odds * cos
        return output


class MultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        bias: bool = True,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding | None = None,
        *,
        fast: bool = False,
    ):
        """Create a causal self-attention block.

        Args:
            embed_dim (int, default: 128): Dimension of the input token embeddings.
            num_heads (int, default: 4): Number of attention heads.
            bias (bool, default: True): Whether to use bias terms in linear layers.
            dropout (float, default: 0.0): Dropout probability for attention weights
                and the projected attention output.
            rope (RotaryPositionalEmbedding, optional): Optional rotary positional
                embedding to apply to the queries and keys.
            fast (bool, default: False): Whether to use PyTorch scaled-dot-product
                attention (could be faster and memory-efficient on some hardware).
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise AssertionError('`embed_dim` must be divisible by `num_heads`.')
        if rope is not None and rope.embed_dim != embed_dim // num_heads:
            raise AssertionError(
                '`rope.embed_dim` must be equal to `embed_dim // num_heads`.'
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.fast = fast

        self.q_proj = dnn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = dnn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = dnn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = dnn.Linear(embed_dim, embed_dim, bias=bias)

        self.rope = rope
        self.resid_dropout = dnn.Dropout(dropout)

    def split_heads(self, x: Tensor) -> Tensor:
        """Split the final dimension of a tensor into multiple attention heads."""
        *B, T, _ = x.size()  # (B, seq_len, embed_dim)
        x = x.reshape(*B, T, self.num_heads, self.head_dim)
        x = x.transpose(-3, -2)  # (B, num_heads, T, head_dim)
        return x

    def forward(self, x: Tensor, token_pos: Tensor | None = None) -> Tensor:
        """Compute causal self-attention for a batch of token embeddings."""
        *B, T, _ = x.size()

        query = self.split_heads(self.q_proj(x))
        key = self.split_heads(self.k_proj(x))
        value = self.split_heads(self.v_proj(x))

        if self.rope is not None:
            query = self.rope(query, token_pos)
            key = self.rope(key, token_pos)

        if self.fast:
            x = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )  # fmt: off
        else:
            x, _ = dF.scaled_dot_product_attention(
                query, key, value,
                is_causal=True,
                dropout=self.dropout,
                training=self.training,
                need_weights=False,
            )  # fmt: off

        x = x.transpose(-3, -2)
        x = x.reshape(*B, T, self.embed_dim)
        x = self.out_proj(x)
        x = self.resid_dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-RMSNorm Transformer decoder block."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
        rope: RotaryPositionalEmbedding | None = None,
        *,
        fast: bool = False,
    ):
        """Create a Transformer decoder block.

        Args:
            embed_dim (int, default: 128): Dimension of the input token embeddings.
            num_heads (int, default: 4): Number of attention heads.
            hidden_dim (int, default: 512): Dimension of the SwiGLU hidden layers.
            bias (bool, default: True): Whether to use bias terms in linear layers.
            dropout (float, default: 0.0): Dropout probability for attention and SwiGLU.
            rope (RotaryPositionalEmbedding, optional): Optional rotary positional
                embedding to apply to the queries and keys.
            fast (bool, default: False): Whether to use PyTorch scaled-dot-product
                attention (could be faster and memory-efficient on some hardware).
        """
        super().__init__()
        self.fast = fast

        self.norm1 = dnn.RMSNorm(embed_dim, eps=1e-5)
        self.attn = MultiheadSelfAttention(
            embed_dim,
            num_heads,
            bias=bias,
            dropout=dropout,
            rope=rope,
            fast=fast,
        )
        self.norm2 = dnn.RMSNorm(embed_dim, eps=1e-5)
        self.mlp = SwiGLU(embed_dim, hidden_dim, bias=bias, dropout=dropout)

    def forward(self, x: Tensor, token_pos: Tensor | None = None) -> Tensor:
        """Transform a batch of token embeddings."""
        x = x + self.attn(self.norm1(x), token_pos)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
        weight_tying: bool = False,
        rope_theta: float = 10000.0,
        *,
        fast: bool = False,
    ):
        """Create a decoder-only Transformer language model.

        Args:
            vocab_size (int): Number of vocabulary entries.
            block_size (int): Maximum context window length.
            embed_dim (int, default: 128): Token embedding dimension.
            num_layers (int, default: 4): Number of Transformer decoder blocks.
            num_heads (int, default: 4): Number of attention heads in each block.
            hidden_dim (int, default: 512): Hidden dimension of each SwiGLU network.
            bias (bool, default: True): Whether to use bias terms in linear layers.
            dropout (float, default: 0.0): Dropout probability for embeddings, attention,
                and feed-forward layers.
            weight_tying (bool, default: False): Whether to share token embedding weights
                with the language-model output head.
            rope_theta (float, default: 10000.0): Base used for rotary embeddings.
            fast (bool, default: False): Whether to use PyTorch scaled-dot-product attention.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.weight_tying = weight_tying
        self.fast = fast

        self.token_embed = dnn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = dnn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    bias=bias,
                    dropout=dropout,
                    rope=RotaryPositionalEmbedding(
                        embed_dim // num_heads,
                        theta=rope_theta,
                        max_seq_len=block_size,
                    ),
                    fast=fast,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = dnn.RMSNorm(embed_dim, eps=1e-5)
        self.lm_head = dnn.Linear(embed_dim, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = cast(nn.Parameter, self.token_embed.weight)
            assert self.lm_head.weight is self.token_embed.weight

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute logits for a batch of input sequences."""
        if input_ids.ndim != 2:
            raise AssertionError('`input_ids` must have shape (B, T).')

        T = input_ids.size(1)
        if T > self.block_size:
            raise AssertionError(
                f'Sequence length {T} exceeds block_size {self.block_size}.'
            )

        x = self.embed_dropout(self.token_embed(input_ids))
        x = self.blocks(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x
