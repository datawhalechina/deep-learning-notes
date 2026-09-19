import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


@torch.no_grad()
def _copy_mha_to_torch(
    custom: dnn.MultiheadAttention,
    reference: nn.MultiheadAttention,
):
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
    if reference.in_proj_bias is not None:
        assert custom.q_proj.bias is not None
        assert custom.k_proj.bias is not None
        assert custom.v_proj.bias is not None

        reference.in_proj_bias.copy_(
            torch.concat(
                [
                    custom.q_proj.bias,
                    custom.k_proj.bias,
                    custom.v_proj.bias,
                ]
            )
        )
    reference.out_proj.weight.copy_(custom.out_proj.weight)
    if reference.out_proj.bias is not None:
        assert custom.out_proj.bias is not None
        reference.out_proj.bias.copy_(custom.out_proj.bias)


@torch.no_grad()
def _copy_encoder_layer_to_torch(
    custom: dnn.TransformerEncoderLayer,
    reference: nn.TransformerEncoderLayer,
):
    _copy_mha_to_torch(custom.self_attn, reference.self_attn)
    reference.linear1.load_state_dict(custom.linear1.state_dict())
    reference.linear2.load_state_dict(custom.linear2.state_dict())
    reference.norm1.load_state_dict(custom.norm1.state_dict())
    reference.norm2.load_state_dict(custom.norm2.state_dict())


@torch.no_grad()
def _copy_decoder_layer_to_torch(
    custom: dnn.TransformerDecoderLayer,
    reference: nn.TransformerDecoderLayer,
):
    _copy_mha_to_torch(custom.self_attn, reference.self_attn)
    _copy_mha_to_torch(custom.mha_attn, reference.multihead_attn)
    reference.linear1.load_state_dict(custom.linear1.state_dict())
    reference.linear2.load_state_dict(custom.linear2.state_dict())
    reference.norm1.load_state_dict(custom.norm1.state_dict())
    reference.norm2.load_state_dict(custom.norm2.state_dict())
    reference.norm3.load_state_dict(custom.norm3.state_dict())


@torch.no_grad()
def _copy_encoder_to_torch(
    custom: dnn.TransformerEncoder,
    reference: nn.TransformerEncoder,
):
    z = zip(custom.layers, reference.layers, strict=True)
    for custom_layer, reference_layer in z:
        _copy_encoder_layer_to_torch(custom_layer, reference_layer)  # type: ignore
    if custom.norm is not None and reference.norm is not None:
        reference.norm.load_state_dict(custom.norm.state_dict())


@torch.no_grad()
def _copy_decoder_to_torch(
    custom: dnn.TransformerDecoder,
    reference: nn.TransformerDecoder,
):
    z = zip(custom.layers, reference.layers, strict=True)
    for custom_layer, reference_layer in z:
        _copy_decoder_layer_to_torch(custom_layer, reference_layer)  # type: ignore
    if custom.norm is not None and reference.norm is not None:
        reference.norm.load_state_dict(custom.norm.state_dict())


@pytest.mark.parametrize('norm_first', [False, True])
def test_transformer_encoder_layer_matches_torch(
    norm_first: bool, batch_size: int, src_len: int, d_model: int, num_heads: int
):
    src = torch.randn(batch_size, src_len, d_model)
    src_mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    src_key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
        ]
    )
    custom = dnn.TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        norm_first=norm_first,
    )
    reference = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=norm_first,
    )
    _copy_encoder_layer_to_torch(custom, reference)

    actual = custom(
        src,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
    )
    expected = reference(
        src,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
    )

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_transformer_encoder_matches_torch_stack_with_norm(
    batch_size: int, src_len: int, d_model: int, num_heads: int
):
    src = torch.randn(batch_size, src_len, d_model)
    custom_layer = dnn.TransformerEncoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
    )
    custom_norm = dnn.LayerNorm(d_model)
    custom = dnn.TransformerEncoder(custom_layer, num_layers=2, norm=custom_norm)

    reference_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    )
    reference_norm = nn.LayerNorm(d_model)
    reference = nn.TransformerEncoder(
        reference_layer, num_layers=2, norm=reference_norm
    )
    _copy_encoder_to_torch(custom, reference)

    assert_close(custom(src), reference(src), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('norm_first', [False, True])
def test_transformer_decoder_layer_matches_torch(
    norm_first: bool,
    batch_size: int,
    src_len: int,
    tgt_len: int,
    d_model: int,
    num_heads: int,
):
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)
    tgt_mask = dF.generate_causal_mask(tgt_len)
    memory_key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
        ]
    )

    custom = dnn.TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        activation='gelu',
        norm_first=norm_first,
    )
    reference = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        activation='gelu',
        batch_first=True,
        norm_first=norm_first,
    )
    _copy_decoder_layer_to_torch(custom, reference)

    actual = custom(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )
    expected = reference(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_transformer_decoder_matches_torch_stack_with_norm(
    batch_size: int, src_len: int, tgt_len: int, d_model: int, num_heads: int
):
    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)

    custom_layer = dnn.TransformerDecoderLayer(
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=16,
        dropout=0.0,
    )
    custom_norm = dnn.LayerNorm(d_model)
    custom = dnn.TransformerDecoder(custom_layer, num_layers=2, norm=custom_norm)

    reference_layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    )
    reference_norm = nn.LayerNorm(d_model)
    reference = nn.TransformerDecoder(
        reference_layer, num_layers=2, norm=reference_norm
    )
    _copy_decoder_to_torch(custom, reference)

    actual = custom(tgt, memory)
    expected = reference(tgt, memory)

    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_transformer_matches_torch_batch_first_transformer(
    batch_size: int, src_len: int, tgt_len: int, d_model: int, num_heads: int
):
    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)
    src_key_padding_mask = torch.tensor(
        [
            [False, False, False, True],
            [False, True, False, True],
        ]
    )
    tgt_mask = dF.generate_causal_mask(tgt_len)

    custom = dnn.Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        norm_first=False,
    )
    reference = nn.Transformer(
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=False,
    )
    _copy_encoder_to_torch(custom.encoder, reference.encoder)
    _copy_decoder_to_torch(custom.decoder, reference.decoder)

    actual = custom(
        src,
        tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    expected = reference(
        src,
        tgt,
        tgt_mask=tgt_mask,
        src_key_padding_mask=src_key_padding_mask,
        memory_key_padding_mask=src_key_padding_mask,
    )
    assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_transformer_omits_batch_first_parameter(
    batch_size: int, src_len: int, tgt_len: int, d_model: int, num_heads: int
):
    with pytest.raises(TypeError):
        dnn.TransformerEncoderLayer(d_model, num_heads, batch_first=False)  # type: ignore[call-arg]

    src = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randn(batch_size, tgt_len, d_model)

    custom = dnn.Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )

    actual = custom(src, tgt)
    assert actual.shape == tgt.shape
