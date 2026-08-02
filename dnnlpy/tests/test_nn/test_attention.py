import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

BATCH_SIZE = 4
SRC_LEN = 8
TGT_LEN = 4
D_MODEL = 6
NUM_HEADS = 2
HEAD_DIM = D_MODEL // NUM_HEADS
KEY_DIM = 6
VALUE_DIM = 8


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


def test_basic_attention():
    options = {'dtype': torch.float64, 'requires_grad': True}

    q1 = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL, **options)
    k1 = torch.randn(BATCH_SIZE, SRC_LEN, D_MODEL, **options)
    v1 = torch.randn(BATCH_SIZE, SRC_LEN, D_MODEL, **options)

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


def test_scaled_dot_product_attention():
    options = {'dtype': torch.float64, 'requires_grad': True}

    q1 = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL, **options)
    k1 = torch.randn(BATCH_SIZE, SRC_LEN, D_MODEL, **options)
    v1 = torch.randn(BATCH_SIZE, SRC_LEN, D_MODEL, **options)

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


def test_scaled_dot_product_attention_boolean_mask():
    query = torch.randn(BATCH_SIZE, NUM_HEADS, TGT_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)

    attn_mask = torch.tensor(
        [
            [
                [True, True, False, True, False, False, True, False],
                [False, True, True, True, False, False, False, True],
                [True, False, True, False, True, False, True, False],
                [False, False, True, False, True, True, False, True],
            ]
        ]
    )

    actual, actual_weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=~attn_mask)

    assert actual_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    masked_weights = actual_weights.masked_select(attn_mask.expand_as(actual_weights))
    assert_close(masked_weights, torch.zeros_like(masked_weights))


def test_scaled_dot_product_attention_supports_additive_mask():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key = torch.randn(BATCH_SIZE, SRC_LEN, D_MODEL)
    value = torch.randn(BATCH_SIZE, SRC_LEN, D_MODEL)

    attn_mask = torch.zeros(TGT_LEN, SRC_LEN)
    attn_mask[:, -1] = -torch.inf

    actual, actual_weights = dF.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask
    )
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)

    assert actual_weights is not None
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert_close(actual_weights[..., -1], torch.zeros_like(actual_weights[..., -1]))


def test_scaled_dot_product_attention_supports_causal_mode():
    query = torch.randn(BATCH_SIZE, NUM_HEADS, TGT_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)

    actual, actual_weights = dF.scaled_dot_product_attention(
        query, key, value, is_causal=True
    )
    expected = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    forbidden = torch.ones(TGT_LEN, SRC_LEN, dtype=torch.bool).triu(diagonal=1)

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


def test_scaled_dot_product_attention_causal_uses_boolean_mask():
    query = torch.randn(BATCH_SIZE, NUM_HEADS, TGT_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)
    mask = torch.ones(TGT_LEN, SRC_LEN, dtype=torch.bool).triu(diagonal=1)

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


def test_scaled_dot_product_attention_respects_scale_and_training_dropout_flag():
    query = torch.randn(BATCH_SIZE, NUM_HEADS, TGT_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SRC_LEN, HEAD_DIM)

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


def test_multi_head_attention_matches_explicit_cross_attention_with_bias():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key = torch.randn(BATCH_SIZE, SRC_LEN, KEY_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, VALUE_DIM)
    W_Q = torch.randn(D_MODEL, D_MODEL)
    W_K = torch.randn(KEY_DIM, D_MODEL)
    W_V = torch.randn(VALUE_DIM, D_MODEL)
    W_O = torch.randn(D_MODEL, D_MODEL)
    b_Q = torch.randn(D_MODEL)
    b_K = torch.randn(D_MODEL)
    b_V = torch.randn(D_MODEL)
    b_O = torch.randn(D_MODEL)

    actual, actual_weights = dF.multi_head_attention(
        query,
        key,
        value,
        NUM_HEADS,
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

    Q = ((query @ W_Q) + b_Q).view(BATCH_SIZE, TGT_LEN, NUM_HEADS, HEAD_DIM)
    K = ((key @ W_K) + b_K).view(BATCH_SIZE, SRC_LEN, NUM_HEADS, HEAD_DIM)
    V = ((value @ W_V) + b_V).view(BATCH_SIZE, SRC_LEN, NUM_HEADS, HEAD_DIM)

    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)

    actual_head, actual_head_weights = dF.scaled_dot_product_attention(Q, K, V)
    expected = actual_head.transpose(1, 2).reshape(BATCH_SIZE, TGT_LEN, D_MODEL)
    expected = (expected @ W_O) + b_O

    assert actual_weights is not None
    assert actual_head_weights is not None
    assert_close(actual_weights, actual_head_weights)
    assert_close(actual, expected)


def test_multi_head_attention_matches_torch_cross_attention_with_bias():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key = torch.randn(BATCH_SIZE, SRC_LEN, KEY_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, VALUE_DIM)

    q_weight = torch.randn(D_MODEL, D_MODEL)
    k_weight = torch.randn(KEY_DIM, D_MODEL)
    v_weight = torch.randn(VALUE_DIM, D_MODEL)
    out_weight = torch.randn(D_MODEL, D_MODEL)

    q_bias = torch.randn(D_MODEL)
    k_bias = torch.randn(D_MODEL)
    v_bias = torch.randn(D_MODEL)
    out_bias = torch.randn(D_MODEL)

    actual, actual_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=NUM_HEADS,
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
        embed_dim_to_check=D_MODEL,
        num_heads=NUM_HEADS,
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


def test_multi_head_attention_gradients_match_torch():
    custom_tensors = [
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in [
            (BATCH_SIZE, TGT_LEN, D_MODEL),
            (BATCH_SIZE, SRC_LEN, KEY_DIM),
            (BATCH_SIZE, SRC_LEN, VALUE_DIM),
            (D_MODEL, D_MODEL),
            (KEY_DIM, D_MODEL),
            (VALUE_DIM, D_MODEL),
            (D_MODEL, D_MODEL),
        ]
    ]
    reference_tensors = [_copy(tensor) for tensor in custom_tensors]

    q1, k1, v1, q_weight1, k_weight1, v_weight1, out_weight1 = custom_tensors
    q2, k2, v2, q_weight2, k_weight2, v_weight2, out_weight2 = reference_tensors  # fmt: skip

    actual, _ = dF.multi_head_attention(
        q1, k1, v1,
        num_heads=NUM_HEADS,
        q_proj_weight=q_weight1,
        k_proj_weight=k_weight1,
        v_proj_weight=v_weight1,
        out_proj_weight=out_weight1,
    )  # fmt: skip
    expected, _ = F.multi_head_attention_forward(
        q2.transpose(0, 1),
        k2.transpose(0, 1),
        v2.transpose(0, 1),
        embed_dim_to_check=D_MODEL,
        num_heads=NUM_HEADS,
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


def test_fast_multi_head_attention_matches_slow_boolean_mask():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key = torch.randn(BATCH_SIZE, SRC_LEN, KEY_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, VALUE_DIM)

    q_weight = torch.randn(D_MODEL, D_MODEL)
    k_weight = torch.randn(KEY_DIM, D_MODEL)
    v_weight = torch.randn(VALUE_DIM, D_MODEL)
    out_weight = torch.randn(D_MODEL, D_MODEL)

    attn_mask = torch.zeros(TGT_LEN, SRC_LEN, dtype=torch.bool)
    attn_mask[:, -1] = True

    actual_slow, actual_slow_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=NUM_HEADS,
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
        num_heads=NUM_HEADS,
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


def test_fast_multi_head_attention_respects_training_dropout_flag():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key = torch.randn(BATCH_SIZE, SRC_LEN, KEY_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, VALUE_DIM)

    q_weight = torch.randn(D_MODEL, D_MODEL)
    k_weight = torch.randn(KEY_DIM, D_MODEL)
    v_weight = torch.randn(VALUE_DIM, D_MODEL)
    out_weight = torch.randn(D_MODEL, D_MODEL)

    actual_slow, _ = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=NUM_HEADS,
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
        num_heads=NUM_HEADS,
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


def test_fast_multi_head_attention_returns_no_weights():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key = torch.randn(BATCH_SIZE, SRC_LEN, KEY_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, VALUE_DIM)

    q_weight = torch.randn(D_MODEL, D_MODEL)
    k_weight = torch.randn(KEY_DIM, D_MODEL)
    v_weight = torch.randn(VALUE_DIM, D_MODEL)
    out_weight = torch.randn(D_MODEL, D_MODEL)

    actual_slow, actual_slow_weights = dF.multi_head_attention(
        query,
        key,
        value,
        num_heads=NUM_HEADS,
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
        num_heads=NUM_HEADS,
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


def test_fast_multihead_attention_module_rejects_weight_return():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    custom = dnn.MultiheadAttention(D_MODEL, NUM_HEADS, fast=True)

    with pytest.raises(AssertionError, match='need_weights=True'):
        custom(query, query, query, need_weights=True)


def test_multihead_attention_module_matches_torch():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)
    key_padding_mask = torch.tensor(
        [
            [False, False, True, False],
            [False, True, True, False],
            [False, False, False, True],
            [False, True, False, True],
        ]
    )

    custom = dnn.MultiheadAttention(D_MODEL, NUM_HEADS)
    reference = nn.MultiheadAttention(D_MODEL, NUM_HEADS)

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


def test_multihead_attention_module_matches_torch_without_bias():
    query = torch.randn(BATCH_SIZE, TGT_LEN, D_MODEL)

    custom = dnn.MultiheadAttention(D_MODEL, NUM_HEADS, bias=False)
    reference = nn.MultiheadAttention(D_MODEL, NUM_HEADS, bias=False)

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


def test_sinusoidal_positional_encoding():
    x = torch.zeros(BATCH_SIZE, SRC_LEN, D_MODEL)
    custom = dnn.SinusoidalPositionalEncoding(D_MODEL, max_len=SRC_LEN)

    actual = custom(x)
    expected = custom.pe.expand_as(actual)  # type: ignore

    assert actual.shape == x.shape
    assert_close(actual, expected)
