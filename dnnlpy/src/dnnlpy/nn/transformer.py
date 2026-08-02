from collections.abc import Callable
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF
from .affine import Linear
from .attention import MultiheadAttention
from .normalization import LayerNorm
from .regularization import Dropout

type Activation = Callable[[Tensor], Tensor]

__all__ = [
    'LearnablePositionalEmbedding',
    'SinusoidalPositionalEncoding',
    'Transformer',
    'TransformerDecoder',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerEncoderLayer',
]


def _get_activation_fn(act_fn: str | Activation, *, fast: bool = False) -> Activation:
    """Resolve a transformer activation name or callable."""
    if act_fn == 'relu':
        return F.relu if fast else dF.relu  # type: ignore
    if act_fn == 'gelu':
        return F.gelu if fast else dF.gelu
    if callable(act_fn):
        return act_fn
    raise TypeError('The activation function must be `relu`, `gelu`, or a callable.')


def _get_activation_name(act_fn: Activation) -> str:
    """Return a concise name for an activation callable."""
    return getattr(act_fn, '__name__', act_fn.__class__.__name__)


def _clone_module(module: nn.Module, num_layers: int) -> nn.ModuleList:
    """Deep-copy a module into a `ModuleList`."""
    return nn.ModuleList(deepcopy(module) for _ in range(num_layers))


class LearnablePositionalEmbedding(nn.Module):
    """Add learnable position embeddings to batch-first sequences."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        """Precompute learnable position embeddings.

        Args:
            embed_dim (int): Embedding dimension of each token.
            max_len (int, default: 5000): Maximum supported sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pe = nn.Embedding(max_len, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to `x`."""
        if x.size(1) > self.max_len:
            raise AssertionError(f'Sequence length {x.size(1)} exceeds {self.max_len}.')

        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pe(positions)
        x = x + pos_emb.unsqueeze(0)
        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, max_len={self.max_len}'


class SinusoidalPositionalEncoding(nn.Module):
    """Add fixed sinusoidal position encodings to batch-first sequences."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        """Precompute sinusoidal encodings.

        Args:
            embed_dim (int): Embedding dimension of each token.
            max_len (int, default: 5000): Maximum supported sequence length.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        exp_term = torch.arange(0, embed_dim, 2) / embed_dim
        div_term = torch.pow(10000.0, exp_term)

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term[: pe[:, 1::2].size(1)])

        # Add a batch dimension for broadcasting
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encodings to `x`."""
        if x.size(1) > self.max_len:
            raise AssertionError(f'Sequence length {x.size(1)} exceeds {self.max_len}.')

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # type: ignore
        return x

    def extra_repr(self) -> str:
        return f'embed_dim={self.embed_dim}, max_len={self.max_len}'


class TransformerEncoderLayer(nn.Module):
    """A batch-first Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        bias: bool = True,
        dropout: float = 0.1,
        activation: str | Activation = 'relu',
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize self-attention, feed-forward, and normalization blocks.

        Args:
            d_model (int): Token embedding dimension.
            num_heads (int): Number of attention heads.
            dim_feedforward (int, default: 2048): Hidden dimension of the feed-forward block.
            bias (bool, default: True): Whether linear and layer norm modules include bias.
            dropout (float, default: 0.1): Dropout probability.
            activation (str | Activation, default: 'relu'): Feed-forward activation.
            layer_norm_eps (float, default: 1e-5): Epsilon for layer normalization.
            norm_first (bool, default: False): If `True`, use pre-normalization.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.norm_first = norm_first
        self.fast = fast

        self.self_attn = MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
            fast=fast,
        )

        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, fast=fast)
        self.dropout = Dropout(dropout, fast=fast)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, fast=fast)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.dropout1 = Dropout(dropout, fast=fast)
        self.dropout2 = Dropout(dropout, fast=fast)

        self.activation = _get_activation_fn(activation, fast=fast)

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Pass source tokens through the encoder layer."""
        if self.norm_first:  # Pre-LN
            x = src + self._sa_block(
                self.norm1(src),
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
            return x
        else:  # Post-LN
            x = self.norm1(
                src + self._sa_block(
                    src,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )
            )  # fmt: skip
            x = self.norm2(x + self._ff_block(x))
            return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        """Apply self-attention and dropout."""
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )  # fmt: skip
        x = self.dropout1(x)
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        """Apply the feed-forward block and dropout."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def extra_repr(self) -> str:
        return (
            f'd_model={self.self_attn.embed_dim}, '
            f'num_heads={self.self_attn.num_heads}, '
            f'dim_feedforward={self.linear1.out_features}, '
            f'dropout={self.dropout.p}, '
            f'activation={_get_activation_name(self.activation)!r}, '
            f'layer_norm_eps={self.norm1.eps}, '
            f'norm_first={self.norm_first}, '
            f'bias={self.self_attn.bias}'
        )


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        """Initialize an encoder stack.

        Args:
            encoder_layer (TransformerEncoderLayer): Prototype encoder layer to clone.
            num_layers (int): Number of encoder layers.
            norm (nn.Module | None, default: None): Optional final normalization module.
        """
        super().__init__()
        self.layers = _clone_module(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Pass source tokens through all encoder layers."""
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def extra_repr(self) -> str:
        return f'num_layers={self.num_layers}'


class TransformerDecoderLayer(nn.Module):
    """A batch-first Transformer decoder layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        bias: bool = True,
        dropout: float = 0.1,
        activation: str | Activation = 'relu',
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize self-attention, cross-attention, and feed-forward blocks.

        Args:
            d_model (int): Token embedding dimension.
            num_heads (int): Number of attention heads.
            dim_feedforward (int, default: 2048): Hidden dimension of the feed-forward block.
            bias (bool, default: True): Whether linear and layer norm modules include bias.
            dropout (float, default: 0.1): Dropout probability.
            activation (str | Activation, default: 'relu'): Feed-forward activation.
            layer_norm_eps (float, default: 1e-5): Epsilon for layer normalization.
            norm_first (bool, default: False): If `True`, use pre-normalization.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.norm_first = norm_first
        self.fast = fast

        self.self_attn = MultiheadAttention(
            d_model,
            num_heads,
            bias=bias,
            dropout=dropout,
            fast=fast,
        )
        self.mha_attn = MultiheadAttention(
            d_model,
            num_heads,
            bias=bias,
            dropout=dropout,
            fast=fast,
        )

        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, fast=fast)
        self.dropout = Dropout(dropout, fast=fast)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, fast=fast)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.dropout1 = Dropout(dropout, fast=fast)
        self.dropout2 = Dropout(dropout, fast=fast)
        self.dropout3 = Dropout(dropout, fast=fast)

        self.activation = _get_activation_fn(activation, fast=fast)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass target tokens through the decoder layer."""
        if self.norm_first:
            x = tgt + self._sa_block(
                self.norm1(tgt),
                tgt_mask,
                tgt_key_padding_mask,
                tgt_is_causal,
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            return x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                tgt + self._sa_block(
                    tgt,
                    tgt_mask,
                    tgt_key_padding_mask,
                    tgt_is_causal,
                )
            )  # fmt: skip
            x = self.norm2(
                x + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )  # fmt: skip
            return self.norm3(x + self._ff_block(x))

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        """Apply masked self-attention and dropout."""
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )  # fmt: skip
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool,
    ) -> Tensor:
        """Apply encoder-decoder cross-attention and dropout."""
        x, _ = self.mha_attn(
            x,
            memory,
            memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout2(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Apply the feed-forward block and dropout."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return self.dropout3(x)

    def extra_repr(self) -> str:
        return (
            f'd_model={self.self_attn.embed_dim}, '
            f'num_heads={self.self_attn.num_heads}, '
            f'dim_feedforward={self.linear1.out_features}, '
            f'dropout={self.dropout.p}, '
            f'activation={_get_activation_name(self.activation)!r}, '
            f'layer_norm_eps={self.norm1.eps}, '
            f'norm_first={self.norm_first}, '
            f'bias={self.self_attn.bias}'
        )


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers."""

    def __init__(
        self,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        norm: nn.Module | None = None,
    ):
        """Initialize a decoder stack.

        Args:
            decoder_layer (TransformerDecoderLayer): Prototype decoder layer to clone.
            num_layers (int): Number of decoder layers.
            norm (nn.Module | None, default: None): Optional final normalization module.
        """
        super().__init__()
        self.layers = _clone_module(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Pass target tokens and encoder memory through all decoder layers."""
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def extra_repr(self) -> str:
        return f'num_layers={self.num_layers}'


class Transformer(nn.Module):
    """A batch-first encoder-decoder Transformer."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        bias: bool = True,
        dropout: float = 0.1,
        activation: str | Activation = 'relu',
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize a full encoder-decoder Transformer.

        Args:
            d_model (int, default: 512): Token embedding dimension.
            num_heads (int, default: 8): Number of attention heads.
            num_encoder_layers (int, default: 6): Number of encoder layers.
            num_decoder_layers (int, default: 6): Number of decoder layers.
            dim_feedforward (int, default: 2048): Hidden dimension of feed-forward blocks.
            bias (bool, default: True): Whether linear and layer norm modules include bias.
            dropout (float, default: 0.1): Dropout probability.
            activation (str | Activation, default: 'relu'): Feed-forward activation.
            layer_norm_eps (float, default: 1e-5): Epsilon for layer normalization.
            norm_first (bool, default: False): If `True`, use pre-normalization inside layers.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.fast = fast

        encoder_layer = TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward,
            bias=bias,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            fast=fast,
        )
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            norm=encoder_norm,
        )

        decoder_layer = TransformerDecoderLayer(
            d_model,
            num_heads,
            dim_feedforward,
            bias=bias,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            norm_first=norm_first,
            fast=fast,
        )
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, fast=fast)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            norm=decoder_norm,
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool = False,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        """Encode `src` and decode `tgt` against the encoder memory."""
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise AssertionError(
                'The feature number of `src` and `tgt` must be equal to `d_model`.'
            )
        if src.size(0) != tgt.size(0):
            raise AssertionError('The batch size of `src` and `tgt` must be equal.')

        memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'num_heads={self.num_heads}, '
            f'num_encoder_layers={self.encoder.num_layers}, '
            f'num_decoder_layers={self.decoder.num_layers}'
        )
