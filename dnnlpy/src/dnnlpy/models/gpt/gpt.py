from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

from .utils import sample_next_token

__all__ = [
    'MiniGPT',
    'MiniGPTBlock',
    'MiniGPTCausalSelfAttention',
    'MiniGPTMLP',
]


class MiniGPTCausalSelfAttention(nn.Module):
    """Causal self-attention block for MiniGPT token representations."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        bias: bool = True,
        dropout: float = 0.0,
        *,
        fast: bool = False,
    ):
        """Create a causal self-attention block.

        A causal self-attention block computes attention scores for each token in the input
        sequence, allowing each token to attend only to previous tokens (including itself).

        Args:
            embed_dim (int, default: 128): Dimension of the input token embeddings.
            num_heads (int, default: 4): Number of attention heads.
            bias (bool, default: True): Whether to use bias terms in the attention layers.
            dropout (float, default: 0.0): Dropout probability for attention weights and
                the projected attention output.
            fast (bool, default: False): Whether to use the fast attention implementation.
        """
        super().__init__()
        self.fast = fast

        self.attn = dnn.MultiheadAttention(
            embed_dim,
            num_heads,
            bias=bias,
            dropout=dropout,
            fast=fast,
        )
        self.resid_dropout = dnn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_output, _ = self.attn(x, x, x, is_causal=True, need_weights=False)
        return self.resid_dropout(attn_output)


class MiniGPTMLP(nn.Module):
    """Feed-forward MLP block used inside a MiniGPT decoder block."""

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """Create a feed-forward MLP block.

        Args:
            embed_dim (int, default: 128): Dimension of the input token embeddings.
            hidden_dim (int, default: 512): Dimension of the hidden layer in the MLP.
            bias (bool, default: True): Whether to use bias terms in the linear layers.
            dropout (float, default: 0.0): Dropout probability for the output of the MLP.
        """
        super().__init__()
        self.net = nn.Sequential(
            dnn.Linear(embed_dim, hidden_dim, bias=bias),
            dnn.GELU(),
            dnn.Linear(hidden_dim, embed_dim, bias=bias),
            dnn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class MiniGPTBlock(nn.Module):
    """Pre-LayerNorm Transformer decoder block for MiniGPT."""

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
        *,
        fast: bool = False,
    ):
        """Create a Transformer decoder block with pre-layer normalization.

        Args:
            embed_dim (int, default: 128): Dimension of the input token embeddings.
            num_heads (int, default: 4): Number of attention heads in self-attention layer.
            hidden_dim (int, default: 512): Dimension of the hidden layer in feed-forward MLP.
            bias (bool, default: True): Whether to use bias terms in the linear layers.
            dropout (float, default: 0.0): Dropout probability for attention and MLP outputs.
            fast (bool, default: False): Whether self-attention uses its fast implementation.
        """
        super().__init__()
        self.fast = fast

        self.norm1 = dnn.LayerNorm(embed_dim, bias=bias)
        self.attn = MiniGPTCausalSelfAttention(
            embed_dim,
            num_heads,
            bias=bias,
            dropout=dropout,
            fast=fast,
        )
        self.norm2 = dnn.LayerNorm(embed_dim, bias=bias)
        self.mlp = MiniGPTMLP(embed_dim, hidden_dim, bias=bias, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    """A tutorial-sized GPT-style language model with a small number of parameters."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,  # or context window
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        bias: bool = True,
        dropout: float = 0.0,
        weight_tying: bool = True,
        *,
        fast: bool = False,
    ):
        """Create a tiny GPT-style language model.

        Args:
            vocab_size (int): Number of vocabulary entries.
            block_size (int): Maximum context window length.
            embed_dim (int, default: 128): Token and position embedding dimension.
            num_layers (int, default: 4): Number of Transformer decoder blocks.
            num_heads (int, default: 4): Number of attention heads in each block.
            hidden_dim (int, default: 512): Hidden dimension of each feed-forward MLP.
            bias (bool, default: True): Whether to use bias terms in linear layers.
            dropout (float, default: 0.0): Dropout probability for embeddings, attention,
                and feed-forward layers.
            weight_tying (bool, default: True): Whether to share token embedding weights
                with the language-model output head.
            fast (bool, default: False): Whether attention layers use their fast
                implementation. Other MiniGPT modules are unaffected.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.weight_tying = weight_tying
        self.fast = fast

        self.token_embed = dnn.Embedding(vocab_size, embed_dim)
        self.pos_embed = dnn.Embedding(block_size, embed_dim)
        self.embed_dropout = dnn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                MiniGPTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    bias=bias,
                    dropout=dropout,
                    fast=fast,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = dnn.LayerNorm(embed_dim, bias=bias)
        self.lm_head = dnn.Linear(embed_dim, vocab_size, bias=False)

        if weight_tying:
            self.lm_head.weight = cast(nn.Parameter, self.token_embed.weight)
            assert self.lm_head.weight is self.token_embed.weight

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the model parameters."""
        for module in self.modules():
            if isinstance(module, dnn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, dnn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Perform depth-scaled initialization for the attention and MLP layers in each block.
        if len(self.blocks) > 0:
            residual_std = 0.02 / (2 * len(self.blocks)) ** 0.5
            for block in self.blocks:
                block = cast(MiniGPTBlock, block)
                nn.init.normal_(
                    block.attn.attn.out_proj.weight,
                    mean=0.0,
                    std=residual_std,
                )
                nn.init.normal_(
                    block.mlp.net[2].weight,  # type: ignore[assignment]
                    mean=0.0,
                    std=residual_std,
                )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Compute the logits for a batch of input sequences."""
        if input_ids.ndim != 2:
            raise AssertionError('`input_ids` must have shape (B, T).')

        T = input_ids.size(1)
        if T > self.block_size:
            raise AssertionError(
                f'Sequence length {T} exceeds block_size {self.block_size}.'
            )

        x_tok = self.token_embed(input_ids)
        if self.pos_embed is None:
            x = self.embed_dropout(x_tok)
        else:
            pos = torch.arange(T, device=input_ids.device)
            x_pos = self.pos_embed(pos)
            x = self.embed_dropout(x_tok + x_pos)

        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def loss(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor:
        """Compute the cross-entropy loss for a batch of input sequences.

        As a language model, the loss is computed by predicting the next token in the
        sequence. If `targets` is not provided, it is assumed that the targets are the
        input sequence shifted by one position.

        Args:
            input_ids (Tensor): Input token ids of shape (B, T).
            targets (Tensor | None, default: None): Target token ids of shape (B, T).

        Returns:
            Tensor: The computed cross-entropy loss.
        """
        if targets is not None and targets.ndim != 2:
            raise AssertionError('`targets` must have shape (B, T).')

        logits = self(input_ids)

        if targets is None:
            logits = logits[:, :-1, :]
            targets = input_ids[:, 1:]

        return dF.cross_entropy_loss(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

    def generate(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        greedy: bool = False,
    ) -> Tensor:
        """Generate new token ids autoregressively.

        Args:
            logits (Tensor): Logits tensor of shape (B, T, V) from the model.
            temperature (float, default: 1.0): Sampling temperature.
            top_k (int | None, default: None): If specified, use top-k sampling.
            top_p (float | None, default: None): If specified, use top-p sampling.
            greedy (bool, default: False): If True, use greedy decoding.

        Returns:
            Tensor: The next token ids sampled from the logits, shape (B,).
        """
        next_token = sample_next_token(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy,
        )
        return next_token
