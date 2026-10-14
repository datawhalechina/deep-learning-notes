from pathlib import Path
from typing import cast

import pytest
import torch
from torch.testing import assert_close

import dnnlpy
import dnnlpy.cs224n.assignment3 as a3

SNAPSHOT_DIR = Path(__file__).parent / 'snapshots'

TEST_CONFIG = a3.GPT2Config(
    vocab_size=16,
    context_length=16,
    d_model=48,
    n_heads=2,
    n_layers=2,
    weight_tying=False,
    fast=False,
)


@pytest.fixture(scope='session')
def set_seed():
    dnnlpy.set_seed(42)


@pytest.fixture(scope='module')
def model(set_seed: None) -> a3.GPT2Model:
    model = a3.GPT2Model(config=TEST_CONFIG)
    state_dict = torch.load(SNAPSHOT_DIR / 'model_state_dict.pt')
    model.load_state_dict(state_dict)
    model.eval()
    return model


@pytest.fixture(scope='module')
def mlp(model: a3.GPT2Model) -> a3.GPT2MLP:
    return cast(a3.GPT2MLP, model.backbone[0].mlp)


@pytest.fixture(scope='module')
def attention(model: a3.GPT2Model) -> a3.GPT2CasualAttention:
    return cast(a3.GPT2CasualAttention, model.backbone[0].attn)


@pytest.fixture(scope='module')
def decoder_block(model: a3.GPT2Model) -> a3.GPT2Block:
    return cast(a3.GPT2Block, model.backbone[0])


def test_mlp(mlp: a3.GPT2MLP):
    """Test the forward pass of the GPT2MLP."""
    inputs = torch.load(SNAPSHOT_DIR / 'mlp_input.pt')

    actual = mlp(inputs)
    expected = torch.load(SNAPSHOT_DIR / 'mlp_output.pt')

    assert_close(actual, expected)


def test_attention(attention: a3.GPT2CasualAttention):
    """Test the forward pass of the GPT2SelfAttention."""
    inputs = torch.load(SNAPSHOT_DIR / 'attention_input.pt')

    actual = attention(inputs)
    expected = torch.load(SNAPSHOT_DIR / 'attention_output.pt')

    assert_close(actual, expected)


def test_decoder_block(decoder_block: a3.GPT2Block):
    """Test the forward pass of the GPT2Block."""
    inputs = torch.load(SNAPSHOT_DIR / 'decoder_block_input.pt')

    actual = decoder_block(inputs)
    expected = torch.load(SNAPSHOT_DIR / 'decoder_block_output.pt')

    assert_close(actual, expected)


def test_forward(model: a3.GPT2Model):
    """Test the forward pass of the GPT2Model."""
    inputs = torch.load(SNAPSHOT_DIR / 'forward_input.pt')

    actual = model(inputs)
    expected = torch.load(SNAPSHOT_DIR / 'forward_output.pt')

    assert_close(actual, expected)


def test_generate(model: a3.GPT2Model):
    """Test the generate method of the GPT2Model."""
    inputs = torch.load(SNAPSHOT_DIR / 'generate_input.pt')

    actual = model.generate(inputs, max_new_token=2, greedy=True)
    expected = torch.load(SNAPSHOT_DIR / 'generate_output.pt')

    assert_close(actual, expected)


def test_loss_on_batch(model: a3.GPT2Model):
    """Test the get_loss_on_batch method of the GPT2Model."""
    inputs = torch.load(SNAPSHOT_DIR / 'loss_on_batch_input.pt')

    x = inputs[:, :-1]
    y = inputs[:, 1:]

    actual = model.loss(x, y)
    expected = torch.load(SNAPSHOT_DIR / 'loss_on_batch_output.pt')

    assert_close(actual, expected)
