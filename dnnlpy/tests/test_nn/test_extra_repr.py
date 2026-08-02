import pytest
import torch
import torch.nn as nn

import dnnlpy.nn as dnn


def _assert_extra_repr(module: nn.Module, expected: str) -> None:
    assert type(module).extra_repr is not nn.Module.extra_repr
    assert module.extra_repr() == expected
    assert expected in repr(module)


@pytest.mark.parametrize(
    ('module', 'expected'),
    [
        (dnn.CELU(alpha=0.2, inplace=True), 'alpha=0.2, inplace=True'),
        (dnn.ELU(alpha=0.2, inplace=True), 'alpha=0.2, inplace=True'),
        (dnn.GELU(approximate='tanh'), "approximate='tanh'"),
        (dnn.HardSigmoid(inplace=True), 'inplace=True'),
        (dnn.HardSwish(inplace=True), 'inplace=True'),
        (
            dnn.HardTanh(min_val=-2.0, max_val=3.0, inplace=True),
            'min_val=-2.0, max_val=3.0, inplace=True',
        ),
        (
            dnn.LeakyReLU(negative_slope=0.2, inplace=True),
            'negative_slope=0.2, inplace=True',
        ),
        (dnn.Mish(inplace=True), 'inplace=True'),
        (dnn.ReLU(inplace=True), 'inplace=True'),
        (dnn.ReLU6(inplace=True), 'inplace=True'),
        (
            dnn.RReLU(lower=0.2, upper=0.4, inplace=True),
            'lower=0.2, upper=0.4, inplace=True',
        ),
        (dnn.SELU(inplace=True), 'inplace=True'),
        (dnn.SiLU(inplace=True), 'inplace=True'),
        (
            dnn.Threshold(threshold=0.2, value=0.3, inplace=True),
            'threshold=0.2, value=0.3, inplace=True',
        ),
    ],
)
def test_configurable_activation_extra_repr(module: nn.Module, expected: str):
    _assert_extra_repr(module, expected)


@pytest.mark.parametrize(
    ('module', 'expected'),
    [
        (
            dnn.BCELoss(reduction='sum', weight=torch.ones(2)),
            "reduction='sum', weight=True",
        ),
        (
            dnn.BCEWithLogitsLoss(
                reduction='none',
                weight=torch.ones(2),
                pos_weight=torch.ones(2),
            ),
            "reduction='none', weight=True, pos_weight=True",
        ),
        (
            dnn.NLLLoss(reduction='sum', weight=torch.ones(3), ignore_index=2),
            "reduction='sum', weight=True, ignore_index=2",
        ),
        (
            dnn.CrossEntropyLoss(
                reduction='sum',
                weight=torch.ones(3),
                ignore_index=2,
                label_smoothing=0.1,
            ),
            ("reduction='sum', weight=True, ignore_index=2, label_smoothing=0.1"),
        ),
        (dnn.MSELoss(reduction='sum'), "reduction='sum', weight=False"),
        (dnn.L1Loss(reduction='none'), "reduction='none', weight=False"),
        (
            dnn.SmoothL1Loss(reduction='sum', beta=0.5),
            "reduction='sum', beta=0.5",
        ),
        (dnn.HuberLoss(reduction='sum', delta=0.5), "reduction='sum', delta=0.5"),
        (
            dnn.KLDivLoss(reduction='batchmean', log_target=True),
            "reduction='batchmean', log_target=True",
        ),
    ],
)
def test_loss_extra_repr(module: nn.Module, expected: str):
    _assert_extra_repr(module, expected)


def test_multihead_attention_extra_repr_exposes_attention_configuration():
    module = dnn.MultiheadAttention(
        embed_dim=8,
        num_heads=2,
        bias=False,
        kdim=6,
        vdim=7,
        dropout=0.2,
    )

    _assert_extra_repr(
        module,
        'embed_dim=8, num_heads=2, dropout=0.2, bias=False, kdim=6, vdim=7',
    )


@pytest.mark.parametrize(
    'module',
    [
        dnn.LearnablePositionalEmbedding(embed_dim=8, max_len=128),
        dnn.SinusoidalPositionalEncoding(embed_dim=8, max_len=128),
    ],
)
def test_positional_encoding_extra_repr(module: nn.Module):
    _assert_extra_repr(module, 'embed_dim=8, max_len=128')


def test_transformer_layer_extra_repr_exposes_layer_configuration():
    encoder = dnn.TransformerEncoderLayer(
        d_model=8,
        num_heads=2,
        dim_feedforward=16,
        bias=False,
        dropout=0.2,
        activation='gelu',
        layer_norm_eps=1e-4,
        norm_first=True,
    )
    decoder = dnn.TransformerDecoderLayer(
        d_model=8,
        num_heads=2,
        dim_feedforward=16,
        bias=False,
        dropout=0.2,
        activation='gelu',
        layer_norm_eps=1e-4,
        norm_first=True,
    )
    expected = (
        'd_model=8, num_heads=2, dim_feedforward=16, dropout=0.2, '
        "activation='gelu', layer_norm_eps=0.0001, norm_first=True, bias=False"
    )

    _assert_extra_repr(encoder, expected)
    _assert_extra_repr(decoder, expected)


def test_transformer_container_extra_repr_exposes_stack_sizes():
    encoder_layer = dnn.TransformerEncoderLayer(8, 2, dim_feedforward=16)
    decoder_layer = dnn.TransformerDecoderLayer(8, 2, dim_feedforward=16)
    encoder = dnn.TransformerEncoder(encoder_layer, num_layers=2)
    decoder = dnn.TransformerDecoder(decoder_layer, num_layers=3)
    transformer = dnn.Transformer(
        d_model=8,
        num_heads=2,
        num_encoder_layers=2,
        num_decoder_layers=3,
        dim_feedforward=16,
    )

    _assert_extra_repr(encoder, 'num_layers=2')
    _assert_extra_repr(decoder, 'num_layers=3')
    _assert_extra_repr(
        transformer,
        'd_model=8, num_heads=2, num_encoder_layers=2, num_decoder_layers=3',
    )


def test_normalization_extra_repr_reports_disabled_bias():
    batch_norm = dnn.BatchNorm1d(4, affine=True, bias=False)
    instance_norm = dnn.InstanceNorm1d(4, affine=True, bias=False)

    _assert_extra_repr(
        batch_norm,
        (
            '4, eps=1e-05, momentum=0.1, affine=True, bias=False, '
            'track_running_stats=True'
        ),
    )
    _assert_extra_repr(
        instance_norm,
        (
            '4, eps=1e-05, momentum=0.1, affine=True, bias=False, '
            'track_running_stats=False'
        ),
    )
