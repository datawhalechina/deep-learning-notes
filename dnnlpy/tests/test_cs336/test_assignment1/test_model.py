from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

from .adapters import (
    run_embedding,
    run_linear,
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_rmsnorm,
    run_rope,
    run_scaled_dot_product_attention,
    run_silu,
    run_swiglu,
    run_transformer_block,
    run_transformer_lm,
)
from .conftest import SnapShot

type StateDict = tuple[dict[str, Tensor], dict[str, Any]]

__all__ = [
    'test_3d_scaled_dot_product_attention',
    'test_4d_scaled_dot_product_attention',
    'test_embedding',
    'test_linear',
    'test_multihead_self_attention',
    'test_multihead_self_attention_with_rope',
    'test_rmsnorm',
    'test_rope',
    'test_silu',
    'test_swiglu',
    'test_transformer_block',
    'test_transformer_lm',
    'test_transformer_lm_truncated_input',
]


def test_linear(
    snapshot: SnapShot,
    ts_state_dict: StateDict,
    embeddings: Tensor,
) -> None:
    """Test the linear layer with the provided embeddings."""
    weight = ts_state_dict[0]['blocks.0.mlp.fc1.weight']

    actual = run_linear(embeddings, weight)
    expected = F.linear(embeddings, weight)

    snapshot.assert_close(actual)
    assert_close(actual, expected)


def test_embedding(
    snapshot: SnapShot,
    ts_state_dict: StateDict,
    token_ids: Tensor,
) -> None:
    """Test the embedding layer with the provided token IDs."""
    weight = ts_state_dict[0]['token_embed.weight']

    actual = run_embedding(token_ids, weight)
    excepted = F.embedding(token_ids, weight)

    snapshot.assert_close(actual)
    assert_close(actual, excepted)


def test_swiglu(
    snapshot: SnapShot,
    ts_state_dict: StateDict,
    embeddings: Tensor,
) -> None:
    """Test the SwiGLU activation function with the provided embeddings."""
    fc1_weight, fc2_weight, fc3_weight = [
        ts_state_dict[0][f'blocks.0.mlp.{name}.weight']
        for name in ['fc1', 'fc2', 'fc3']
    ]

    actual_output = run_swiglu(
        x=embeddings,
        w1_weight=fc1_weight,
        w2_weight=fc2_weight,
        w3_weight=fc3_weight,
    )
    snapshot.assert_close(actual_output, atol=1e-5)


def test_3d_scaled_dot_product_attention(
    snapshot: SnapShot,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
) -> None:
    """Test scaled dot product attention with 3D input tensors (batch_size=1)."""
    actual = run_scaled_dot_product_attention(query, key, value, attn_mask)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask)

    snapshot.assert_close(actual, atol=1e-5)
    assert_close(actual, expected)


def test_4d_scaled_dot_product_attention(
    snapshot: SnapShot,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor,
) -> None:
    """Test scaled dot product attention with 4D input tensors (batch_size>1)."""
    n_heads = 2
    batch_size = query.size(0) // n_heads

    query = query.reshape(batch_size, n_heads, *query.shape[1:])
    key = key.reshape(batch_size, n_heads, *key.shape[1:])
    value = value.reshape(batch_size, n_heads, *value.shape[1:])
    attn_mask = attn_mask.reshape(batch_size, n_heads, *attn_mask.shape[1:])

    actual = run_scaled_dot_product_attention(query, key, value, attn_mask)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask)

    snapshot.assert_close(actual, atol=1e-5)
    assert_close(actual, expected)


def test_multihead_self_attention(
    snapshot: SnapShot,
    embeddings: Tensor,
    d_model: int,
    n_heads: int,
    ts_state_dict: StateDict,
) -> None:
    """Test multi-head self-attention against the reference implementation."""
    d = ts_state_dict[0]
    q_proj_weight = d['blocks.0.attn.q_proj.weight']
    k_proj_weight = d['blocks.0.attn.k_proj.weight']
    v_proj_weight = d['blocks.0.attn.v_proj.weight']
    o_proj_weight = d['blocks.0.attn.out_proj.weight']

    actual_output = run_multihead_self_attention(
        x=embeddings,
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
    )
    snapshot.assert_close(actual_output, atol=1e-5)


def test_multihead_self_attention_with_rope(
    snapshot: SnapShot,
    embeddings: Tensor,
    pos_ids: Tensor,
    d_model: int,
    n_heads: int,
    ts_state_dict: StateDict,
    theta: float,
    n_keys: int,
) -> None:
    """Test rotary-position multi-head self-attention against the reference."""
    d = ts_state_dict[0]
    q_proj_weight = d['blocks.0.attn.q_proj.weight']
    k_proj_weight = d['blocks.0.attn.k_proj.weight']
    v_proj_weight = d['blocks.0.attn.v_proj.weight']
    o_proj_weight = d['blocks.0.attn.out_proj.weight']

    pos_ids = pos_ids.unsqueeze(0)  # Add batch dimension
    output = run_multihead_self_attention_with_rope(
        x=embeddings,
        token_pos=pos_ids,
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
    )
    snapshot.assert_close(output, atol=1e-5)


def test_transformer_lm(
    snapshot: SnapShot,
    token_ids: Tensor,
    vocab_size: int,
    n_keys: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    theta: float,
    ts_state_dict: StateDict,
) -> None:
    """Test the Transformer language model against the reference implementation."""
    state_dict = ts_state_dict[0]

    output = run_transformer_lm(
        x=token_ids,
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
    )
    snapshot.assert_close(output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    snapshot: SnapShot,
    vocab_size: int,
    n_keys: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    theta: float,
    ts_state_dict: StateDict,
    token_ids: Tensor,
) -> None:
    """Test Transformer language model output for truncated input sequences."""
    in_indices_truncated = token_ids[..., : token_ids.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        x=in_indices_truncated,
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=ts_state_dict[0],
    )
    snapshot.assert_close(truncated_actual_output, atol=1e-4)


def test_transformer_block(
    snapshot: SnapShot,
    ts_state_dict: StateDict,
    embeddings: Tensor,
    d_model: int,
    n_heads: int,
    d_ff: int,
    n_keys: int,
    theta: float,
) -> None:
    """Test a Transformer block against the reference implementation."""
    prefix = 'blocks.0.'
    block_state_dict = {
        name.removeprefix(prefix): param
        for name, param in ts_state_dict[0].items()
        if name.startswith(prefix)
    }

    output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_state_dict,
        x=embeddings,
    )
    snapshot.assert_close(output, atol=1e-4)


def test_rmsnorm(
    snapshot: SnapShot, ts_state_dict: StateDict, embeddings: Tensor
) -> None:
    """Test RMSNorm with the provided embeddings and state dictionary."""
    state_dict = ts_state_dict[0]
    normalized_shape = embeddings.size(-1)
    weight = state_dict['blocks.1.norm1.weight']

    actual = run_rmsnorm(embeddings, weight, eps=1e-5)
    expected = F.rms_norm(embeddings, (normalized_shape,), weight, eps=1e-5)

    snapshot.assert_close(actual, atol=1e-4)
    assert_close(actual, expected)


def test_rope(
    snapshot: SnapShot,
    embeddings: Tensor,
    d_model: int,
    theta: float,
    n_queries: int,
    pos_ids: Tensor,
) -> None:
    """Test rotary positional embeddings against the reference implementation."""
    output = run_rope(
        x=embeddings,
        token_pos=pos_ids,
        embed_dim=d_model,
        theta=theta,
        max_seq_len=n_queries,
    )
    snapshot.assert_close(output, atol=1e-5)


def test_silu() -> None:
    """Test the SiLU activation function against PyTorch's implementation."""
    x = torch.tensor(
        [
            [0.2352, 0.9259, 0.5189, 0.4725, 0.9730],
            [0.7581, 0.9692, 0.2129, 0.9345, 0.0149],
        ]
    )

    actual = run_silu(x)
    expected = F.silu(x)

    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
