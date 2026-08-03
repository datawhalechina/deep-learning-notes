# pyright: reportOptionalMemberAccess=false
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.batchnorm import _BatchNorm as ReferenceBatchNorm
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF
from dnnlpy.nn.normalization import (
    _BatchNorm as CustomBatchNorm,
    _InstanceNorm as CustomInstanceNorm,
)


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


@pytest.mark.parametrize(
    ('custom_fn', 'reference_fn', 'shape'),
    [
        (
            lambda x: dF.batch_norm(x, None, None, use_batch_stats=True),
            lambda x: F.batch_norm(x, None, None, training=True),
            (4, 3, 4, 5),
        ),
        (
            lambda x: dF.instance_norm(x, None, None, use_instance_stats=True),
            lambda x: F.instance_norm(x, None, None, use_input_stats=True),
            (2, 4, 5, 6),
        ),
        (
            lambda x: dF.layer_norm(x, (3, 4)),
            lambda x: F.layer_norm(x, (3, 4)),
            (2, 3, 4),
        ),
        (
            lambda x: dF.rms_norm(x, (3, 4)),
            lambda x: F.rms_norm(x, (3, 4)),
            (2, 3, 4),
        ),
        (
            lambda x: dF.group_norm(x, 2),
            lambda x: F.group_norm(x, 2),
            (2, 4, 5, 6),
        ),
        (
            lambda x: dF.local_response_norm(x, 3),
            lambda x: F.local_response_norm(x, 3),
            (2, 4, 5, 6),
        ),
    ],
    ids=['batch', 'instance', 'layer', 'rms', 'group', 'local-response'],
)
def test_normalization_function_gradients_match_torch(
    custom_fn: Callable[[Tensor], Tensor],
    reference_fn: Callable[[Tensor], Tensor],
    shape: tuple[int, ...],
):
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x2 = _copy(x1)

    actual = custom_fn(x1)
    expected = reference_fn(x2)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(actual, expected)
    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize(
    'shape',
    [
        (8, 4),
        (8, 4, 5),
        (8, 4, 5, 6),
        (8, 4, 3, 5, 6),
    ],
)
def test_batch_norm_function_matches_torch_training(shape: tuple[int, ...]):
    x = torch.randn(shape)
    weight = torch.randn(4)
    bias = torch.randn(4)

    actual_running_mean = torch.zeros(4)
    actual_running_var = torch.ones(4)
    expected_running_mean = actual_running_mean.clone()
    expected_running_var = actual_running_var.clone()

    actual = dF.batch_norm(
        x,
        actual_running_mean,
        actual_running_var,
        weight=weight,
        bias=bias,
        use_batch_stats=True,
        momentum=0.1,
    )
    expected = F.batch_norm(
        x,
        expected_running_mean,
        expected_running_var,
        weight=weight,
        bias=bias,
        training=True,
        momentum=0.1,
    )

    assert_close(actual, expected)
    assert_close(actual_running_mean, expected_running_mean)
    assert_close(actual_running_var, expected_running_var)


def test_batch_norm_function_matches_torch_eval_with_running_stats():
    x = torch.randn(8, 4, 5)
    weight = torch.randn(4)
    bias = torch.randn(4)

    running_mean = torch.randn(4)
    running_var = torch.rand(4) + 0.5

    actual = dF.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        use_batch_stats=False,
    )
    expected = F.batch_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        training=False,
    )

    assert_close(actual, expected)


def test_batch_norm_function_uses_batch_stats_when_running_stats_are_none():
    x = torch.randn(8, 4)
    weight = torch.randn(4)
    bias = torch.randn(4)

    actual = dF.batch_norm(
        x,
        None,
        None,
        weight=weight,
        bias=bias,
        use_batch_stats=True,
    )
    expected = F.batch_norm(
        x,
        None,
        None,
        weight=weight,
        bias=bias,
        training=True,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('custom_cls', 'reference_cls', 'shape', 'training', 'track_running_stats'),
    [
        (custom_cls, reference_cls, shape, training, track_running_stats)
        for custom_cls, reference_cls, shape in [
            (dnn.BatchNorm1d, nn.BatchNorm1d, (8, 4)),
            (dnn.BatchNorm1d, nn.BatchNorm1d, (8, 4, 5)),
            (dnn.BatchNorm2d, nn.BatchNorm2d, (8, 4, 5, 6)),
            (dnn.BatchNorm3d, nn.BatchNorm3d, (8, 4, 3, 5, 6)),
        ]
        for training in [True, False]
        for track_running_stats in [True, False]
    ],
)
def test_batch_norm_modules_match_torch_for_mode_and_tracking(
    custom_cls: type[CustomBatchNorm],
    reference_cls: type[ReferenceBatchNorm],
    shape: tuple[int, ...],
    training: bool,
    track_running_stats: bool,
):
    x = torch.randn(shape)

    custom = custom_cls(4, track_running_stats=track_running_stats)
    reference = reference_cls(4, track_running_stats=track_running_stats)
    reference.load_state_dict(custom.state_dict())

    if not training:
        custom.eval()
        reference.eval()
        if track_running_stats:
            custom.running_mean.copy_(torch.randn(4))
            custom.running_var.copy_(torch.rand(4) + 0.5)
            reference.load_state_dict(custom.state_dict())

    actual = custom(x)
    expected = reference(x)

    assert_close(actual, expected)

    if track_running_stats:
        assert_close(custom.running_mean, reference.running_mean)
        assert_close(custom.running_var, reference.running_var)
        assert_close(custom.num_batches_tracked, reference.num_batches_tracked)
    else:
        assert custom.running_mean is None
        assert custom.running_var is None
        assert custom.num_batches_tracked is None


def test_batch_norm_module_supports_no_affine():
    x = torch.randn(8, 4, 5, 6)

    custom = dnn.BatchNorm2d(4, affine=False)
    reference = nn.BatchNorm2d(4, affine=False)
    reference.load_state_dict(custom.state_dict())

    assert custom.weight is None
    assert custom.bias is None
    assert_close(custom(x), reference(x))


def test_batch_norm_module_supports_tracking_disabled_in_eval_mode():
    x = torch.randn(8, 4)
    shifted = torch.randn(8, 4) + 10

    custom = dnn.BatchNorm1d(4, track_running_stats=False)
    reference = nn.BatchNorm1d(4, track_running_stats=False)

    custom.eval()
    reference.eval()

    assert custom.running_mean is None
    assert custom.running_var is None
    assert custom.num_batches_tracked is None
    assert_close(custom(x), reference(x))
    assert_close(custom(shifted), reference(shifted))


def test_batch_norm_module_supports_cumulative_moving_average():
    custom = dnn.BatchNorm1d(4, momentum=None)
    reference = nn.BatchNorm1d(4, momentum=None)
    reference.load_state_dict(custom.state_dict())

    for _ in range(3):
        x = torch.randn(8, 4)
        assert_close(custom(x), reference(x))

    assert_close(custom.running_mean, reference.running_mean)
    assert_close(custom.running_var, reference.running_var)


def test_batch_norm_reset_parameters_restores_defaults():
    custom = dnn.BatchNorm1d(4)
    custom.weight.data.normal_()
    custom.bias.data.normal_()
    custom.running_mean.normal_()
    custom.running_var.normal_()
    custom.num_batches_tracked.add_(5)

    custom.reset_parameters()

    assert_close(custom.weight, torch.ones(4))
    assert_close(custom.bias, torch.zeros(4))
    assert_close(custom.running_mean, torch.zeros(4))
    assert_close(custom.running_var, torch.ones(4))
    assert_close(custom.num_batches_tracked, torch.tensor(0))


def test_batch_norm_module_supports_no_bias_extension():
    custom = dnn.BatchNorm1d(4, bias=False)
    x = torch.randn(8, 4)

    assert custom.bias is None
    assert custom(x).shape == x.shape


@pytest.mark.parametrize(
    ('custom', 'bad_shape'),
    [
        (dnn.BatchNorm1d(4), (8, 4, 5, 6)),
        (dnn.BatchNorm2d(4), (8, 4, 5)),
        (dnn.BatchNorm3d(4), (8, 4, 5, 6)),
    ],
)
def test_batch_norm_modules_reject_invalid_rank(
    custom: CustomBatchNorm,
    bad_shape: tuple[int, ...],
):
    with pytest.raises(AssertionError):
        custom(torch.randn(bad_shape))


def test_batch_norm_module_rejects_wrong_channel_count():
    custom = dnn.BatchNorm2d(4)

    with pytest.raises(AssertionError):
        custom(torch.randn(8, 3, 5, 6))


def test_batch_norm_function_rejects_single_value_per_channel_when_training():
    with pytest.raises(RuntimeError):
        dF.batch_norm(
            torch.randn(1, 4),
            torch.zeros(4),
            torch.ones(4),
            use_batch_stats=True,
        )


@pytest.mark.parametrize(
    ('shape', 'reference_fn'),
    [
        ((8, 4, 5), F.instance_norm),
        ((8, 4, 5, 6), F.instance_norm),
        ((8, 4, 3, 5, 6), F.instance_norm),
    ],
)
def test_instance_norm_function_matches_torch_training(
    shape: tuple[int, ...], reference_fn: Callable[..., Tensor]
):
    x = torch.randn(shape)
    weight = torch.randn(4)
    bias = torch.randn(4)
    actual_running_mean = torch.zeros(4)
    actual_running_var = torch.ones(4)
    expected_running_mean = actual_running_mean.clone()
    expected_running_var = actual_running_var.clone()

    actual = dF.instance_norm(
        x,
        actual_running_mean,
        actual_running_var,
        weight=weight,
        bias=bias,
        use_instance_stats=True,
        momentum=0.1,
    )
    expected = reference_fn(
        x,
        expected_running_mean,
        expected_running_var,
        weight=weight,
        bias=bias,
        use_input_stats=True,
        momentum=0.1,
    )

    assert_close(actual, expected)
    assert_close(actual_running_mean, expected_running_mean)
    assert_close(actual_running_var, expected_running_var)


def test_instance_norm_function_matches_torch_eval_with_running_stats():
    x = torch.randn(8, 4, 5, 6)
    weight = torch.randn(4)
    bias = torch.randn(4)
    running_mean = torch.randn(4)
    running_var = torch.rand(4) + 0.5

    actual = dF.instance_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        use_instance_stats=False,
    )
    expected = F.instance_norm(
        x,
        running_mean,
        running_var,
        weight=weight,
        bias=bias,
        use_input_stats=False,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('custom_cls', 'reference_cls', 'shape', 'training', 'track_running_stats'),
    [
        (custom_cls, reference_cls, shape, training, track_running_stats)
        for custom_cls, reference_cls, shape in [
            (dnn.InstanceNorm1d, nn.InstanceNorm1d, (8, 4, 5)),
            (dnn.InstanceNorm2d, nn.InstanceNorm2d, (8, 4, 5, 6)),
            (dnn.InstanceNorm3d, nn.InstanceNorm3d, (8, 4, 3, 5, 6)),
        ]
        for training in [True, False]
        for track_running_stats in [True, False]
    ],
)
def test_instance_norm_modules_match_torch_for_mode_and_tracking(
    custom_cls: type[CustomBatchNorm],
    reference_cls: type[ReferenceBatchNorm],
    shape: tuple[int, ...],
    training: bool,
    track_running_stats: bool,
):
    x = torch.randn(shape)
    custom = custom_cls(
        4,
        affine=True,
        track_running_stats=track_running_stats,
    )
    reference = reference_cls(
        4,
        affine=True,
        track_running_stats=track_running_stats,
    )
    reference.load_state_dict(custom.state_dict())

    if not training:
        custom.eval()
        reference.eval()
        if track_running_stats:
            custom.running_mean.copy_(torch.randn(4))
            custom.running_var.copy_(torch.rand(4) + 0.5)
            reference.load_state_dict(custom.state_dict())

    actual = custom(x)
    expected = reference(x)

    assert_close(actual, expected)

    if track_running_stats:
        assert_close(custom.running_mean, reference.running_mean)
        assert_close(custom.running_var, reference.running_var)
    else:
        assert custom.running_mean is None
        assert custom.running_var is None


def test_instance_norm_module_supports_no_affine():
    x = torch.randn(8, 4, 5, 6)

    custom = dnn.InstanceNorm2d(4, affine=False)
    reference = nn.InstanceNorm2d(4, affine=False)
    reference.load_state_dict(custom.state_dict())

    assert custom.weight is None
    assert custom.bias is None
    assert_close(custom(x), reference(x))


def test_instance_norm_reset_parameters_restores_defaults():
    custom = dnn.InstanceNorm1d(4, affine=True, track_running_stats=True)
    custom.weight.data.normal_()
    custom.bias.data.normal_()
    custom.running_mean.normal_()
    custom.running_var.normal_()

    custom.reset_parameters()

    assert_close(custom.weight, torch.ones(4))
    assert_close(custom.bias, torch.zeros(4))
    assert_close(custom.running_mean, torch.zeros(4))
    assert_close(custom.running_var, torch.ones(4))


def test_instance_norm_module_supports_no_bias_extension():
    x = torch.randn(8, 4, 5)

    custom = dnn.InstanceNorm1d(4, affine=True, bias=False)

    assert custom.bias is None
    assert custom(x).shape == x.shape


@pytest.mark.parametrize(
    ('custom', 'bad_shape'),
    [
        (dnn.InstanceNorm1d(4), (8, 4)),
        (dnn.InstanceNorm2d(4), (8, 4, 5)),
        (dnn.InstanceNorm3d(4), (8, 4, 5, 6)),
    ],
)
def test_instance_norm_modules_reject_invalid_rank(
    custom: CustomInstanceNorm,
    bad_shape: tuple[int, ...],
):
    with pytest.raises(AssertionError):
        custom(torch.randn(bad_shape))


def test_instance_norm_module_rejects_wrong_channel_count():
    custom = dnn.InstanceNorm2d(4)

    with pytest.raises(AssertionError):
        custom(torch.randn(8, 3, 5, 6))


def test_instance_norm_function_rejects_single_spatial_value_when_training():
    with pytest.raises(RuntimeError):
        dF.instance_norm(
            torch.randn(8, 4, 1),
            torch.zeros(4),
            torch.ones(4),
            use_instance_stats=True,
        )


def test_layer_norm_function_matches_torch():
    x = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4)
    bias = torch.randn(3, 4)

    actual = dF.layer_norm(x, (3, 4), weight=weight, bias=bias)
    expected = F.layer_norm(x, (3, 4), weight=weight, bias=bias)

    assert_close(actual, expected)


@pytest.mark.parametrize('normalized_shape', [4, (3, 4)])
def test_layer_norm_module_matches_torch(normalized_shape: int | tuple[int, ...]):
    x = torch.randn(2, 3, 4)

    custom = dnn.LayerNorm(normalized_shape)
    reference = nn.LayerNorm(normalized_shape)
    reference.load_state_dict(custom.state_dict())

    assert_close(custom(x), reference(x))


def test_layer_norm_module_supports_no_bias_extension():
    x = torch.randn(2, 3, 4)

    custom = dnn.LayerNorm(4, bias=False)

    assert custom.bias is None
    assert custom(x).shape == x.shape


def test_layer_norm_module_supports_no_affine():
    x = torch.randn(2, 3, 4)

    custom = dnn.LayerNorm((3, 4), elementwise_affine=False)
    reference = nn.LayerNorm((3, 4), elementwise_affine=False)

    assert custom.weight is None
    assert reference.bias is None
    assert_close(custom(x), reference(x))


def test_layer_norm_function_rejects_wrong_normalized_shape():
    with pytest.raises(AssertionError):
        dF.layer_norm(torch.randn(2, 3, 4), (2, 4))


def test_rms_norm_function_matches_torch():
    x = torch.randn(2, 3, 4)
    weight = torch.randn(3, 4)

    actual = dF.rms_norm(x, (3, 4), weight=weight)
    expected = F.rms_norm(x, (3, 4), weight=weight)

    assert_close(actual, expected)


def test_rms_norm_function_upcasts_before_squaring():
    x = torch.full((2, 4), 300.0, dtype=torch.float16)

    actual = dF.rms_norm(x, 4)

    assert actual.dtype == x.dtype
    assert torch.isfinite(actual).all()
    assert_close(actual, torch.ones_like(x))


@pytest.mark.parametrize('normalized_shape', [4, (3, 4)])
def test_rms_norm_module_matches_torch(normalized_shape: int | tuple[int, ...]):
    x = torch.randn(2, 3, 4)

    custom = dnn.RMSNorm(normalized_shape)
    reference = nn.RMSNorm(normalized_shape)
    reference.load_state_dict(custom.state_dict())

    assert_close(custom(x), reference(x))


def test_rms_norm_module_supports_no_affine():
    x = torch.randn(2, 3, 4)

    custom = dnn.RMSNorm((3, 4), elementwise_affine=False)
    reference = nn.RMSNorm((3, 4), elementwise_affine=False)

    assert custom.weight is None
    assert_close(custom(x), reference(x))


def test_rms_norm_reset_parameters_restores_defaults():
    custom = dnn.RMSNorm((3, 4))
    custom.weight.data.normal_()

    custom.reset_parameters()

    assert_close(custom.weight, torch.ones(3, 4))


def test_rms_norm_function_rejects_wrong_normalized_shape():
    with pytest.raises(AssertionError):
        dF.rms_norm(torch.randn(2, 3, 4), (2, 4))


@pytest.mark.parametrize('shape', [(2, 4), (2, 4, 5), (2, 4, 5, 6)])
def test_group_norm_function_matches_torch(shape: tuple[int, ...]):
    x = torch.randn(shape)
    weight = torch.randn(4)
    bias = torch.randn(4)

    actual = dF.group_norm(x, 2, weight=weight, bias=bias)
    expected = F.group_norm(x, 2, weight=weight, bias=bias)

    assert_close(actual, expected)


def test_group_norm_module_matches_torch():
    x = torch.randn(2, 4, 5, 6)

    custom = dnn.GroupNorm(2, 4)
    reference = nn.GroupNorm(2, 4)
    reference.load_state_dict(custom.state_dict())

    assert_close(custom(x), reference(x))


def test_group_norm_module_supports_no_affine():
    x = torch.randn(2, 4, 5, 6)

    custom = dnn.GroupNorm(2, 4, affine=False)
    reference = nn.GroupNorm(2, 4, affine=False)

    assert custom.weight is None
    assert custom.bias is None
    assert_close(custom(x), reference(x))


def test_group_norm_rejects_invalid_group_count():
    with pytest.raises(AssertionError):
        dF.group_norm(torch.randn(2, 4, 5), 0)


def test_group_norm_module_rejects_wrong_channel_count():
    custom = dnn.GroupNorm(2, 4)

    with pytest.raises(AssertionError):
        custom(torch.randn(2, 3, 5))


@pytest.mark.parametrize('shape', [(2, 4, 5), (2, 4, 5, 6)])
def test_local_response_norm_function_matches_torch(shape: tuple[int, ...]):
    x = torch.randn(shape)

    actual = dF.local_response_norm(x, 3, alpha=1e-3, beta=0.5, k=2.0)
    expected = F.local_response_norm(x, 3, alpha=1e-3, beta=0.5, k=2.0)

    assert_close(actual, expected)


def test_local_response_norm_module_matches_torch():
    x = torch.randn(2, 4, 5, 6)

    custom = dnn.LocalResponseNorm(3, alpha=1e-3, beta=0.5, k=2.0)
    reference = nn.LocalResponseNorm(3, alpha=1e-3, beta=0.5, k=2.0)

    assert_close(custom(x), reference(x))
