import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn.functional as dF

BATCH_SIZE = 4
TGT_LEN = 8
SRC_LEN = 4
EMBED_DIM = 6


@pytest.mark.parametrize('is_causal', [False, True])
def test_flash_attention_v1_forward_accepts_batch_input(is_causal: bool):
    query = torch.randn(BATCH_SIZE, TGT_LEN, EMBED_DIM)
    key = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)

    actual = dF.flash_attention_v1_forward(
        query, key, value, Br=2, Bc=3, is_causal=is_causal
    )
    expected = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)

    assert actual.shape == expected.shape
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_flash_attention_v1_forward_keeps_2d_input_compatible():
    query = torch.randn(TGT_LEN, EMBED_DIM)
    key = torch.randn(SRC_LEN, EMBED_DIM)
    value = torch.randn(SRC_LEN, EMBED_DIM)

    actual = dF.flash_attention_v1_forward(query, key, value, Br=3, Bc=2)
    expected = F.scaled_dot_product_attention(query, key, value)

    assert actual.shape == expected.shape
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('is_causal', [False, True])
def test_flash_attention_v1_backward_matches_autograd_for_batch_input(is_causal: bool):
    query = torch.randn(BATCH_SIZE, TGT_LEN, EMBED_DIM, requires_grad=True)
    key = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM, requires_grad=True)
    value = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM, requires_grad=True)
    dO = torch.randn(BATCH_SIZE, TGT_LEN, EMBED_DIM)

    expected = F.scaled_dot_product_attention(query, key, value, is_causal=is_causal)
    expected.backward(dO)

    actual_dq, actual_dk, actual_dv = dF.flash_attention_v1_backward(
        query.detach(),
        key.detach(),
        value.detach(),
        dO,
        Br=2,
        Bc=3,
        is_causal=is_causal,
    )

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None

    assert_close(actual_dq, query.grad, rtol=1e-5, atol=1e-6)
    assert_close(actual_dk, key.grad, rtol=1e-5, atol=1e-6)
    assert_close(actual_dv, value.grad, rtol=1e-5, atol=1e-6)


def test_flash_attention_v1_backward_rejects_dropout():
    query = torch.randn(BATCH_SIZE, TGT_LEN, EMBED_DIM)
    key = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)
    dO = torch.randn(2, 3, 4)

    with pytest.raises(NotImplementedError):
        dF.flash_attention_v1_backward(query, key, value, dO, Br=2, Bc=2, dropout=0.1)


@pytest.mark.parametrize(
    ('key', 'value', 'match'),
    [
        (torch.randn(2, 3), torch.randn(2, 3), 'same number of dimensions'),
        (
            torch.randn(2, SRC_LEN, EMBED_DIM - 1),
            torch.randn(2, SRC_LEN, EMBED_DIM),
            'same embedding dim',
        ),
        (
            torch.randn(2, SRC_LEN, EMBED_DIM),
            torch.randn(2, SRC_LEN + 1, EMBED_DIM),
            'same sequence length',
        ),
    ],
)
def test_flash_attention_v1_forward_rejects_invalid_tensors(
    key: Tensor, value: Tensor, match: str
):
    query = torch.randn(BATCH_SIZE, TGT_LEN, EMBED_DIM)

    with pytest.raises(AssertionError, match=match):
        dF.flash_attention_v1_forward(query, key, value, Br=2, Bc=2)


def test_flash_attention_v1_backward_rejects_invalid_gradient():
    query = torch.randn(BATCH_SIZE, TGT_LEN, EMBED_DIM)
    key = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)
    value = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)
    dO = torch.randn(BATCH_SIZE, SRC_LEN, EMBED_DIM)

    with pytest.raises(AssertionError, match='output shape'):
        dF.flash_attention_v1_backward(query, key, value, dO, Br=2, Bc=2)
