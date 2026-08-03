from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

from dnnlpy.models import gpt


def _make_model(**overrides: Any) -> gpt.MiniGPT:
    config = {
        'vocab_size': 17,
        'block_size': 6,
        'embed_dim': 8,
        'num_layers': 1,
        'num_heads': 2,
        'hidden_dim': 16,
        'dropout': 0.0,
    }
    config.update(overrides)
    return gpt.MiniGPT(**config)


def test_minigpt_forward_returns_vocab_logits_and_supports_backward():
    model = _make_model()
    input_ids = torch.randint(0, 17, (2, 5))

    logits = model(input_ids)
    logits.mean().backward()

    assert logits.shape == (2, 5, 17)
    assert model.token_embed.weight.grad is not None
    assert model.token_embed.weight.grad.isfinite().all()


def test_minigpt_causal_attention_hides_future_tokens():
    model = _make_model(block_size=4)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    changed_future = input_ids.clone()
    changed_future[:, -1] = 5

    actual = model(input_ids)
    changed = model(changed_future)

    assert_close(actual[:, :-1], changed[:, :-1])


def test_minigpt_attention_applies_dropout_to_projected_output():
    attention = gpt.MiniGPTCausalSelfAttention(
        embed_dim=8,
        num_heads=2,
        dropout=1.0,
    )
    attention.attn.dropout = 0.0
    inputs = torch.randn(2, 4, 8)

    attention.eval()
    output_without_dropout = attention(inputs)
    attention.train()
    output_with_dropout = attention(inputs)

    assert torch.count_nonzero(output_without_dropout) > 0
    assert torch.count_nonzero(output_with_dropout) == 0


def test_minigpt_scales_residual_projection_initialization_by_depth(monkeypatch):
    num_layers = 4
    model = _make_model(num_layers=num_layers, weight_tying=False)

    def fill_with_std(tensor: Tensor, mean: float = 0.0, std: float = 1.0):
        del mean
        with torch.no_grad():
            tensor.fill_(std)
        return tensor

    monkeypatch.setattr(torch.nn.init, 'normal_', fill_with_std)
    model.reset_parameters()

    expected_residual_std = 0.02 / (2 * num_layers) ** 0.5
    for block in model.blocks:
        block = cast(gpt.MiniGPTBlock, block)
        assert_close(
            block.attn.attn.out_proj.weight,
            torch.full_like(block.attn.attn.out_proj.weight, expected_residual_std),
        )
        assert_close(
            block.mlp.net[2].weight,
            torch.full_like(block.mlp.net[2].weight, expected_residual_std),  # type: ignore[arg-type]
        )
        assert_close(
            block.attn.attn.q_proj.weight,
            torch.full_like(block.attn.attn.q_proj.weight, 0.02),
        )


def test_minigpt_ties_token_embedding_and_lm_head_by_default():
    model = _make_model()

    assert model.weight_tying is True
    assert model.lm_head.weight is model.token_embed.weight


def test_minigpt_can_disable_weight_tying():
    model = _make_model(weight_tying=False)

    assert model.weight_tying is False
    assert model.lm_head.weight is not model.token_embed.weight


def test_minigpt_forward_validates_input_shape_and_block_size():
    model = _make_model(block_size=4)

    with pytest.raises(AssertionError, match=r'shape \(B, T\)'):
        model(torch.randint(0, 17, (5,)))

    with pytest.raises(AssertionError, match='exceeds block_size'):
        model(torch.randint(0, 17, (2, 5)))


def test_minigpt_loss_shifts_inputs_when_targets_are_omitted():
    torch.manual_seed(0)
    model = _make_model()
    input_ids = torch.randint(0, 17, (2, 5))

    actual = model.loss(input_ids)
    logits = model(input_ids)[:, :-1, :]
    expected = F.cross_entropy(
        logits.reshape(-1, 17),
        input_ids[:, 1:].reshape(-1),
    )

    assert actual.ndim == 0
    assert_close(actual, expected)


def test_minigpt_loss_uses_provided_targets_without_shifting():
    torch.manual_seed(0)
    model = _make_model()
    input_ids = torch.randint(0, 17, (2, 5))
    targets = torch.randint(0, 17, (2, 5))

    actual = model.loss(input_ids, targets)
    expected = F.cross_entropy(
        model(input_ids).reshape(-1, 17),
        targets.reshape(-1),
    )

    assert_close(actual, expected)


def test_minigpt_loss_rejects_non_matrix_targets():
    model = _make_model()
    input_ids = torch.randint(0, 17, (2, 5))

    with pytest.raises(AssertionError, match=r'targets.*shape \(B, T\)'):
        model.loss(input_ids, torch.randint(0, 17, (10,)))


def test_minigpt_generate_returns_greedy_next_token_from_last_position():
    model = _make_model(vocab_size=5, block_size=3)
    logits = torch.tensor(
        [
            [[0.0, 9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 7.0, 0.0, 0.0]],
            [[8.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 6.0, 0.0]],
        ]
    )

    next_token = model.generate(logits, greedy=True)

    assert next_token.tolist() == [[2], [3]]


def test_get_batch_returns_aligned_next_token_targets():
    token_ids = torch.arange(20)

    inputs, targets = gpt.get_batch(token_ids, block_size=4, batch_size=3)

    assert inputs.shape == targets.shape == (3, 4)
    assert torch.equal(targets, inputs + 1)


def test_top_k_and_top_p_sampling_mask_filtered_logits():
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))

    top_k = gpt.top_k_sampling(logits.clone(), top_k=2)
    top_p = gpt.top_p_sampling(logits.clone(), top_p=0.7)

    assert torch.equal(top_k.isfinite(), torch.tensor([[True, True, False]]))
    assert torch.equal(top_p.isfinite(), torch.tensor([[True, True, False]]))


def test_sample_next_token_validates_logits_and_temperature():
    with pytest.raises(AssertionError, match=r'logits.*shape \(B, V\)'):
        gpt.sample_next_token(torch.zeros(2, 3, 4))

    with pytest.raises(AssertionError, match='temperature.*positive'):
        gpt.sample_next_token(torch.zeros(2, 4), temperature=0.0)
