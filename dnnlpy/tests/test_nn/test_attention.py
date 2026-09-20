import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


def test_basic_attention(batch_size: int, src_len: int, tgt_len: int, d_model: int):
    options = {'dtype': torch.float64, 'requires_grad': True}

    q1 = torch.randn(batch_size, tgt_len, d_model, **options)
    k1 = torch.randn(batch_size, src_len, d_model, **options)
    v1 = torch.randn(batch_size, src_len, d_model, **options)

    q2 = _copy(q1)
    k2 = _copy(k1)
    v2 = _copy(v1)

    actual, actual_weights = dF.basic_attention(q1, k1, v1)
    expected_weights = F.softmax(q2 @ k2.transpose(-2, -1), dim=-1)
    expected = expected_weights @ v2

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert actual_weights is not None
    assert_close(actual_weights, expected_weights)
    assert_close(actual, expected)
    assert_close(q1.grad, q2.grad)
    assert_close(k1.grad, k2.grad)
    assert_close(v1.grad, v2.grad)


def test_scaled_dot_product_attention(
    batch_size: int, src_len: int, tgt_len: int, d_model: int
):
    options = {'dtype': torch.float64, 'requires_grad': True}

    q1 = torch.randn(batch_size, tgt_len, d_model, **options)
    k1 = torch.randn(batch_size, src_len, d_model, **options)
    v1 = torch.randn(batch_size, src_len, d_model, **options)

    q2 = _copy(q1)
    k2 = _copy(k1)
    v2 = _copy(v1)

    scale = 1 / math.sqrt(q1.size(-1))

    actual, actual_weights = dF.scaled_dot_product_attention(q1, k1, v1)
    expected = F.scaled_dot_product_attention(q2, k2, v2)
    expected_weights = F.softmax((q2 @ k2.transpose(-2, -1)) * scale, dim=-1)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert actual_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights)
    assert_close(q1.grad, q2.grad, rtol=1e-5, atol=1e-6)
    assert_close(k1.grad, k2.grad, rtol=1e-5, atol=1e-6)
    assert_close(v1.grad, v2.grad, rtol=1e-5, atol=1e-6)


def test_scaled_dot_product_attention_boolean_mask(
    batch_size: int, src_len: int, tgt_len: int, num_heads: int, head_dim: int
):
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)

    attn_mask = torch.zeros(1, tgt_len, src_len, dtype=torch.bool)
    attn_mask[..., ::2] = True

    actual, actual_weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=~attn_mask)

    assert actual_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    masked_weights = actual_weights.masked_select(attn_mask.expand_as(actual_weights))
    assert_close(masked_weights, torch.zeros_like(masked_weights))


def test_scaled_dot_product_attention_supports_additive_mask(
    batch_size: int, src_len: int, tgt_len: int, d_model: int
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, d_model)
    value = torch.randn(batch_size, src_len, d_model)

    attn_mask = torch.zeros(tgt_len, src_len)
    attn_mask[:, -1] = -torch.inf

    actual, actual_weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    assert actual_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights[..., -1], torch.zeros_like(actual_weights[..., -1]))


def test_scaled_dot_product_attention_supports_causal_mode(
    batch_size: int, src_len: int, tgt_len: int, num_heads: int, head_dim: int
):
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)

    actual, actual_weights = dF.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    expected = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    forbidden = torch.ones(tgt_len, src_len, dtype=torch.bool).triu(diagonal=1)

    assert actual_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    masked_weights = actual_weights[..., forbidden]
    assert_close(masked_weights, torch.zeros_like(masked_weights))


def test_generate_causal_mask_masks_future_positions():
    actual = dF.generate_causal_mask(4)
    expected = torch.tensor(
        [
            [0.0, -torch.inf, -torch.inf, -torch.inf],
            [0.0, 0.0, -torch.inf, -torch.inf],
            [0.0, 0.0, 0.0, -torch.inf],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_scaled_dot_product_attention_causal_uses_boolean_mask(
    batch_size: int, src_len: int, tgt_len: int, num_heads: int, head_dim: int
):
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)
    mask = torch.ones(tgt_len, src_len, dtype=torch.bool).triu(diagonal=1)

    actual_causal, actual_causal_weights = dF.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    actual_masked, actual_masked_weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=mask
    )

    assert actual_causal_weights is not None
    assert actual_masked_weights is not None
    assert_close(actual_causal, actual_masked)
    assert_close(actual_causal_weights, actual_masked_weights)


def test_scaled_dot_product_attention_respects_scale_and_training_dropout_flag(
    batch_size: int, src_len: int, tgt_len: int, num_heads: int, head_dim: int
):
    query = torch.randn(batch_size, num_heads, tgt_len, head_dim)
    key = torch.randn(batch_size, num_heads, src_len, head_dim)
    value = torch.randn(batch_size, num_heads, src_len, head_dim)

    actual, _ = dF.scaled_dot_product_attention(
        query, key, value,
        dropout=0.8,
        training=False,
        scale=0.25,
    )  # fmt: skip
    expected = F.scaled_dot_product_attention(
        query, key, value,
        dropout_p=0.0,
        scale=0.25,
    )  # fmt: skip

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_multi_head_attention_matches_explicit_cross_attention_with_bias(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
    head_dim: int,
    key_dim: int,
    value_dim: int,
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)
    W_Q = torch.randn(d_model, d_model)
    W_K = torch.randn(key_dim, d_model)
    W_V = torch.randn(value_dim, d_model)
    W_O = torch.randn(d_model, d_model)
    b_Q = torch.randn(d_model)
    b_K = torch.randn(d_model)
    b_V = torch.randn(d_model)
    b_O = torch.randn(d_model)

    actual, actual_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads,
        q_proj_weight=W_Q,
        k_proj_weight=W_K,
        v_proj_weight=W_V,
        out_proj_weight=W_O,
        q_proj_bias=b_Q,
        k_proj_bias=b_K,
        v_proj_bias=b_V,
        out_proj_bias=b_O,
        need_weights=True,
    )

    Q = ((query @ W_Q) + b_Q).view(batch_size, tgt_len, num_heads, head_dim)
    K = ((key @ W_K) + b_K).view(batch_size, src_len, num_heads, head_dim)
    V = ((value @ W_V) + b_V).view(batch_size, src_len, num_heads, head_dim)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    actual_head, actual_head_weights = dF.scaled_dot_product_attention(Q, K, V)
    expected = actual_head.transpose(1, 2).reshape(batch_size, tgt_len, d_model)
    expected = (expected @ W_O) + b_O

    assert actual_weights is not None
    assert actual_head_weights is not None
    assert_close(actual_weights, actual_head_weights)
    assert_close(actual, expected)


def test_multi_head_attention_matches_torch_cross_attention_with_bias(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)

    q_weight = torch.randn(d_model, d_model)
    k_weight = torch.randn(key_dim, d_model)
    v_weight = torch.randn(value_dim, d_model)
    out_weight = torch.randn(d_model, d_model)

    q_bias = torch.randn(d_model)
    k_bias = torch.randn(d_model)
    v_bias = torch.randn(d_model)
    out_bias = torch.randn(d_model)

    actual, actual_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        q_proj_bias=q_bias,
        k_proj_bias=k_bias,
        v_proj_bias=v_bias,
        out_proj_bias=out_bias,
        need_weights=True,
    )
    expected, expected_weights = F.multi_head_attention_forward(
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=None,
        in_proj_bias=torch.concat([q_bias, k_bias, v_bias]),
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=out_weight.T,
        out_proj_bias=out_bias,
        training=True,
        need_weights=True,
        use_separate_proj_weight=True,
        q_proj_weight=q_weight.T,
        k_proj_weight=k_weight.T,
        v_proj_weight=v_weight.T,
        average_attn_weights=False,
    )
    expected = expected.transpose(0, 1)

    assert actual_weights is not None
    assert expected_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)


def test_multi_head_attention_gradients_match_torch(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
):
    custom_tensors = [
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in [
            (batch_size, tgt_len, d_model),
            (batch_size, src_len, key_dim),
            (batch_size, src_len, value_dim),
            (d_model, d_model),
            (key_dim, d_model),
            (value_dim, d_model),
            (d_model, d_model),
        ]
    ]
    reference_tensors = [_copy(tensor) for tensor in custom_tensors]

    q1, k1, v1, q_weight1, k_weight1, v_weight1, out_weight1 = custom_tensors
    q2, k2, v2, q_weight2, k_weight2, v_weight2, out_weight2 = reference_tensors  # fmt: skip

    actual, _ = dF.multi_head_attention(
        q1, k1, v1,
        num_heads=num_heads,
        q_proj_weight=q_weight1,
        k_proj_weight=k_weight1,
        v_proj_weight=v_weight1,
        out_proj_weight=out_weight1,
    )  # fmt: skip
    expected, _ = F.multi_head_attention_forward(
        q2.transpose(0, 1),
        k2.transpose(0, 1),
        v2.transpose(0, 1),
        embed_dim_to_check=d_model,
        num_heads=num_heads,
        in_proj_weight=None,
        in_proj_bias=None,
        bias_k=None,
        bias_v=None,
        add_zero_attn=False,
        dropout_p=0.0,
        out_proj_weight=out_weight2.T,
        out_proj_bias=None,
        training=True,
        need_weights=False,
        use_separate_proj_weight=True,
        q_proj_weight=q_weight2.T,
        k_proj_weight=k_weight2.T,
        v_proj_weight=v_weight2.T,
    )
    expected = expected.transpose(0, 1)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for custom_grad, reference_grad in zip(custom_tensors, reference_tensors):
        assert_close(custom_grad.grad, reference_grad.grad, rtol=1e-5, atol=1e-6)


def test_fast_multi_head_attention_matches_slow_boolean_mask(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)

    q_weight = torch.randn(d_model, d_model)
    k_weight = torch.randn(key_dim, d_model)
    v_weight = torch.randn(value_dim, d_model)
    out_weight = torch.randn(d_model, d_model)

    attn_mask = torch.zeros(tgt_len, src_len, dtype=torch.bool)
    attn_mask[:, -1] = True

    actual_slow, actual_slow_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        attn_mask=attn_mask,
    )
    actual_fast, actual_fast_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        attn_mask=attn_mask,
        fast=True,
    )

    assert actual_slow_weights is None
    assert actual_fast_weights is None
    assert_close(actual_fast, actual_slow, rtol=1e-5, atol=1e-6)


def test_fast_multi_head_attention_respects_training_dropout_flag(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)

    q_weight = torch.randn(d_model, d_model)
    k_weight = torch.randn(key_dim, d_model)
    v_weight = torch.randn(value_dim, d_model)
    out_weight = torch.randn(d_model, d_model)

    actual_slow, _ = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        dropout=0.8,
        training=False,
    )
    actual_fast, actual_fast_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        dropout=0.8,
        training=False,
        fast=True,
    )

    assert actual_fast_weights is None
    assert_close(actual_fast, actual_slow, rtol=1e-5, atol=1e-6)


def test_fast_multi_head_attention_returns_no_weights(
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key = torch.randn(batch_size, src_len, key_dim)
    value = torch.randn(batch_size, src_len, value_dim)

    q_weight = torch.randn(d_model, d_model)
    k_weight = torch.randn(key_dim, d_model)
    v_weight = torch.randn(value_dim, d_model)
    out_weight = torch.randn(d_model, d_model)

    actual_slow, actual_slow_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        need_weights=True,
    )
    actual_fast, actual_fast_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        q_proj_weight=q_weight,
        k_proj_weight=k_weight,
        v_proj_weight=v_weight,
        out_proj_weight=out_weight,
        need_weights=True,
        fast=True,
    )

    assert actual_slow_weights is not None
    assert_close(actual_fast, actual_slow, rtol=1e-5, atol=2e-6)
    assert actual_fast_weights is None


def test_fast_multihead_attention_module_rejects_weight_return(
    batch_size: int, tgt_len: int, d_model: int, num_heads: int
):
    query = torch.randn(batch_size, tgt_len, d_model)
    custom = dnn.MultiheadAttention(d_model, num_heads, fast=True)

    with pytest.raises(AssertionError, match='need_weights=True'):
        custom(query, query, query, need_weights=True)


def test_multihead_attention_module_matches_torch(
    batch_size: int, tgt_len: int, d_model: int, num_heads: int
):
    query = torch.randn(batch_size, tgt_len, d_model)
    key_padding_mask = torch.zeros(batch_size, tgt_len, dtype=torch.bool)
    key_padding_mask[:, -1] = True

    custom = dnn.MultiheadAttention(d_model, num_heads)
    reference = nn.MultiheadAttention(d_model, num_heads)

    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.concat(
                [
                    custom.q_proj.weight,
                    custom.k_proj.weight,
                    custom.v_proj.weight,
                ],
                dim=0,
            )
        )

        assert custom.q_proj.bias is not None
        assert custom.k_proj.bias is not None
        assert custom.v_proj.bias is not None

        reference.in_proj_bias.copy_(
            torch.concat(
                [
                    custom.q_proj.bias,
                    custom.k_proj.bias,
                    custom.v_proj.bias,
                ],
                dim=0,
            )
        )

        assert custom.out_proj.bias is not None
        reference.out_proj.weight.copy_(custom.out_proj.weight)
        reference.out_proj.bias.copy_(custom.out_proj.bias)

    actual, actual_weights = custom(
        query,
        query,
        query,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    expected, expected_weights = reference(
        query.transpose(0, 1),
        query.transpose(0, 1),
        query.transpose(0, 1),
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    expected = expected.transpose(0, 1)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)


def test_multihead_attention_module_matches_torch_without_bias(
    batch_size: int, tgt_len: int, d_model: int, num_heads: int
):
    query = torch.randn(batch_size, tgt_len, d_model)

    custom = dnn.MultiheadAttention(d_model, num_heads, bias=False)
    reference = nn.MultiheadAttention(d_model, num_heads, bias=False)

    assert custom.bias is False
    assert custom.q_proj.bias is None
    assert custom.k_proj.bias is None
    assert custom.v_proj.bias is None
    assert custom.out_proj.bias is None

    with torch.no_grad():
        reference.in_proj_weight.copy_(
            torch.concat(
                [
                    custom.q_proj.weight,
                    custom.k_proj.weight,
                    custom.v_proj.weight,
                ],
                dim=0,
            )
        )
        reference.out_proj.weight.copy_(custom.out_proj.weight)

    actual, actual_weights = custom(
        query,
        query,
        query,
        need_weights=True,
    )
    expected, expected_weights = reference(
        query.transpose(0, 1),
        query.transpose(0, 1),
        query.transpose(0, 1),
        need_weights=True,
    )
    expected = expected.transpose(0, 1)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)


def test_sinusoidal_positional_encoding(batch_size: int, src_len: int, d_model: int):
    x = torch.zeros(batch_size, src_len, d_model)
    custom = dnn.SinusoidalPositionalEncoding(d_model, max_len=src_len)

    actual = custom(x)
    expected = custom.pos_embed.expand_as(actual)

    assert actual.shape == x.shape
    assert_close(actual, expected)
