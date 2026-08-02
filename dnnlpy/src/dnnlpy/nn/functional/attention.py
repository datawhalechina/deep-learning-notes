import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device

from .. import functional as dF

type AttentionOutput = tuple[Tensor, Tensor | None]

__all__ = [
    'basic_attention',
    'generate_causal_mask',
    'multi_head_attention',
    'scaled_dot_product_attention',
]


def basic_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    need_weights: bool = True,
) -> AttentionOutput:
    """Compute unscaled dot-product attention.

    Args:
        query (Tensor): Query tensor.
        key (Tensor): Key tensor with the same final dimension as `query`.
        value (Tensor): Value tensor with the same sequence length as `key`.
        need_weights (bool, default: True): Whether to return attention weights.

    Returns:
        AttentionOutput: Tuple of output tensor and optional attention weights.
    """
    scores = query @ key.transpose(-2, -1)
    attn_weights = dF.softmax(scores, dim=-1)
    output = attn_weights @ value

    if need_weights:
        return output, attn_weights

    return output, None


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    dropout: float = 0.0,
    training: bool = True,
    need_weights: bool = True,
    scale: float | None = None,
) -> AttentionOutput:
    """Compute scaled dot-product attention.

    Boolean masks use the same convention as `torch.nn.Transformer`:
    `True` always means the element is masked out and `False` means the
    element participates in attention. Float masks are additive attention biases.

    Args:
        query (Tensor): Query tensor of shape `(batch, target_len, embed_dim)`.
        key (Tensor): Key tensor of shape `(batch, source_len, key_dim)`.
        value (Tensor): Value tensor of shape `(batch, source_len, value_dim)`.
        attn_mask (Tensor | None, default: None): Optional attention mask.
        is_causal (bool, default: False): Whether to apply a causal mask.
        dropout (float, default: 0.0): Dropout probability for attention weights.
        training (bool, default: True): Whether dropout is active.
        need_weights (bool, default: True): Whether to return attention weights.
        scale (float | None, default: None): Optional scaling factor for attention scores.
            If `None`, defaults to `1 / sqrt(embed_dim)`.

    Returns:
        AttentionOutput: Tuple of output tensor and optional attention weights.
    """
    if query.size(-1) != key.size(-1):
        raise AssertionError(
            '`query` and `key` must have the same embedding dimension.'
        )
    if key.size(-2) != value.size(-2):
        raise AssertionError('`key` and `value` must have the same sequence length.')
    if is_causal and attn_mask is not None:
        raise AssertionError('`attn_mask` and `is_causal` cannot both be set.')
    if not 0.0 <= dropout <= 1.0:
        raise AssertionError('`dropout` must be between 0 and 1.')

    tgt_len = query.size(-2)
    src_len = key.size(-2)

    scale_factor = (1.0 / math.sqrt(query.size(-1))) if scale is None else scale
    scores = query @ key.transpose(-2, -1)
    scores = scores * scale_factor

    if is_causal:
        causal_mask = torch.ones(
            (tgt_len, src_len),
            dtype=torch.bool,
            device=query.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, -math.inf)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(attn_mask, -math.inf)
        else:
            scores = scores + attn_mask.to(device=query.device, dtype=query.dtype)

    attn_weights = dF.softmax(scores, dim=-1)
    attn_weights = dF.dropout(attn_weights, p=dropout, training=training)
    output = attn_weights @ value

    if need_weights:
        return output, attn_weights

    return output, None


def multi_head_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    out_proj_weight: Tensor,
    q_proj_bias: Tensor | None = None,
    k_proj_bias: Tensor | None = None,
    v_proj_bias: Tensor | None = None,
    out_proj_bias: Tensor | None = None,
    attn_mask: Tensor | None = None,
    is_causal: bool = False,
    dropout: float = 0.0,
    training: bool = True,
    need_weights: bool = False,
    *,
    fast: bool = False,
) -> AttentionOutput:
    """Compute batch-first multi-head attention from explicit projection weights.

    Projection weights are right-multiplied, so their shapes are
    `(input_dim, embed_dim)`. Boolean masks use `True` to mask out
    positions.

    Args:
        query (Tensor): Query tensor of shape `(batch, target_len, embed_dim)`.
        key (Tensor): Key tensor of shape `(batch, source_len, key_dim)`.
        value (Tensor): Value tensor of shape `(batch, source_len, value_dim)`.
        num_heads (int): Number of attention heads.
        q_proj_weight (Tensor): Query projection weight.
        k_proj_weight (Tensor): Key projection weight.
        v_proj_weight (Tensor): Value projection weight.
        out_proj_weight (Tensor): Output projection weight.
        q_proj_bias (Tensor | None, default: None): Optional query projection bias.
        k_proj_bias (Tensor | None, default: None): Optional key projection bias.
        v_proj_bias (Tensor | None, default: None): Optional value projection bias.
        out_proj_bias (Tensor | None, default: None): Optional output projection bias.
        attn_mask (Tensor | None, default: None): Optional attention mask.
        is_causal (bool, default: False): Whether to apply a causal mask.
        dropout (float, default: 0.0): Dropout probability for attention weights.
        training (bool, default: True): Whether dropout is active.
        need_weights (bool, default: False): Whether to return per-head attention weights.

    Returns:
        AttentionOutput: Tuple of output tensor and optional per-head attention weights.
    """
    batch_size, tgt_len, embed_dim = query.size()
    src_len = key.size(1)

    if embed_dim % num_heads != 0:
        raise AssertionError('`embed_dim` must be divisible by `num_heads`.')
    if key.size(0) != batch_size or value.size(0) != batch_size:
        raise AssertionError('`query`, `key`, and `value` batch sizes must match.')
    if key.size(1) != value.size(1):
        raise AssertionError('`key` and `value` must have the same sequence length.')
    if not 0.0 <= dropout <= 1.0:
        raise AssertionError('`dropout` must be between 0 and 1.')

    head_dim = embed_dim // num_heads

    q = query @ q_proj_weight
    k = key @ k_proj_weight
    v = value @ v_proj_weight

    if q_proj_bias is not None:
        q = q + q_proj_bias
    if k_proj_bias is not None:
        k = k + k_proj_bias
    if v_proj_bias is not None:
        v = v + v_proj_bias

    q = q.view(batch_size, tgt_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, src_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, src_len, num_heads, head_dim).transpose(1, 2)

    if fast:
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.bitwise_not()

        head_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout if training else 0.0,
            is_causal=is_causal,
        )  # fmt: skip
        attn_weights = None
    else:
        head_output, attn_weights = scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout=dropout,
            training=training,
            need_weights=need_weights,
        )  # fmt: skip

    output = head_output.transpose(1, 2)
    # Using `reshape` instead of `view` to avoid issues with non-contiguous tensors.
    output = output.reshape(batch_size, tgt_len, embed_dim)
    output = output @ out_proj_weight

    if out_proj_bias is not None:
        output = output + out_proj_bias

    if need_weights:
        return output, attn_weights

    return output, None


def generate_causal_mask(
    sz: int,
    device: Device = None,
) -> Tensor:
    """Generate an upper-triangular causal attention mask.

    Args:
        sz (int): Height and width of the square mask.
        device (Device, default: None): Optional output device.

    Returns:
        Tensor: A square mask where positions above the diagonal are masked.
    """
    mask = torch.full((sz, sz), -torch.inf, device=device)
    mask = mask.triu(diagonal=1)
    return mask
