import inspect
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

type TupleND = tuple[int, ...]


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


@pytest.mark.parametrize(
    ('name', 'shape', 'kwargs'),
    [
        (
            'avg_pool1d',
            (2, 3, 9),
            {
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'ceil_mode': True,
                'count_include_pad': False,
            },
        ),
        (
            'avg_pool2d',
            (2, 3, 7, 8),
            {
                'kernel_size': (3, 2),
                'stride': (2, 1),
                'padding': (1, 0),
                'ceil_mode': True,
                'count_include_pad': False,
                'divisor_override': 5,
            },
        ),
        (
            'avg_pool3d',
            (2, 3, 6, 7, 8),
            {
                'kernel_size': (2, 3, 2),
                'stride': (2, 1, 2),
                'padding': (0, 1, 0),
                'ceil_mode': True,
                'count_include_pad': False,
            },
        ),
    ],
)
def test_avg_pool_matches_torch(name: str, shape: TupleND, kwargs: Any):
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x2 = _copy(x1)

    actual = getattr(dF, name)(x1, **kwargs)
    expected = getattr(F, name)(x2, **kwargs)
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize(
    ('name', 'shape', 'kwargs'),
    [
        (
            'max_pool1d',
            (2, 3, 11),
            {
                'kernel_size': 3,
                'stride': 2,
                'padding': 1,
                'dilation': 2,
                'ceil_mode': True,
            },
        ),
        (
            'max_pool2d',
            (2, 3, 8, 9),
            {
                'kernel_size': (3, 2),
                'stride': (2, 1),
                'padding': (1, 0),
                'dilation': (1, 2),
                'ceil_mode': True,
            },
        ),
        (
            'max_pool3d',
            (2, 3, 7, 8, 9),
            {
                'kernel_size': (2, 3, 2),
                'stride': (1, 2, 2),
                'padding': (0, 1, 0),
                'dilation': (2, 1, 1),
                'ceil_mode': True,
            },
        ),
    ],
)
def test_max_pool_matches_torch(name: str, shape: TupleND, kwargs: Any):
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x2 = _copy(x1)

    actual = getattr(dF, name)(x1, **kwargs)
    expected = getattr(F, name)(x2, **kwargs)
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize('mode', ['avg', 'max'])
@pytest.mark.parametrize(
    ('ndim', 'shape', 'output_size'),
    [
        (1, (2, 3, 8), 3),
        (2, (2, 3, 7, 8), (3, None)),
        (3, (2, 3, 6, 7, 8), (2, None, 3)),
    ],
)
def test_adaptive_pool_matches_torch(
    mode: str, ndim: int, shape: TupleND, output_size: int | TupleND
):
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x2 = _copy(x1)
    name = f'adaptive_{mode}_pool{ndim}d'

    actual = getattr(dF, name)(x1, output_size)
    expected = getattr(F, name)(x2, output_size)
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize(
    ('custom_cls', 'reference_cls', 'shape', 'kwargs'),
    [
        (
            dnn.AvgPool1d,
            nn.AvgPool1d,
            (2, 3, 9),
            {'kernel_size': 3, 'stride': 2, 'padding': 1},
        ),
        (
            dnn.AvgPool2d,
            nn.AvgPool2d,
            (2, 3, 7, 8),
            {
                'kernel_size': (3, 2),
                'stride': (2, 1),
                'padding': (1, 0),
                'ceil_mode': True,
                'count_include_pad': False,
                'divisor_override': 5,
            },
        ),
        (
            dnn.AvgPool3d,
            nn.AvgPool3d,
            (2, 3, 6, 7, 8),
            {'kernel_size': 2, 'stride': 1, 'padding': 1},
        ),
        (
            dnn.MaxPool1d,
            nn.MaxPool1d,
            (2, 3, 11),
            {'kernel_size': 3, 'stride': 2, 'padding': 1},
        ),
        (
            dnn.MaxPool2d,
            nn.MaxPool2d,
            (2, 3, 8, 9),
            {
                'kernel_size': (3, 2),
                'stride': (2, 1),
                'padding': (1, 0),
                'dilation': (1, 2),
                'ceil_mode': True,
            },
        ),
        (
            dnn.MaxPool3d,
            nn.MaxPool3d,
            (2, 3, 7, 8, 9),
            {'kernel_size': 2, 'stride': 1, 'padding': 1},
        ),
        (dnn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool1d, (2, 3, 8), {'output_size': 3}),
        (
            dnn.AdaptiveAvgPool2d,
            nn.AdaptiveAvgPool2d,
            (2, 3, 7, 8),
            {'output_size': (3, None)},
        ),
        (
            dnn.AdaptiveAvgPool3d,
            nn.AdaptiveAvgPool3d,
            (2, 3, 6, 7, 8),
            {'output_size': (2, None, 3)},
        ),
        (dnn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool1d, (2, 3, 8), {'output_size': 3}),
        (
            dnn.AdaptiveMaxPool2d,
            nn.AdaptiveMaxPool2d,
            (2, 3, 7, 8),
            {'output_size': (3, None)},
        ),
        (
            dnn.AdaptiveMaxPool3d,
            nn.AdaptiveMaxPool3d,
            (2, 3, 6, 7, 8),
            {'output_size': (2, None, 3)},
        ),
    ],
)
@pytest.mark.parametrize('fast', [False, True])
def test_pool_modules_match_torch(
    custom_cls: type[nn.Module],
    reference_cls: type[nn.Module],
    shape: TupleND,
    kwargs: Any,
    fast: bool,
):
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    x2 = _copy(x1)

    custom = custom_cls(**kwargs, fast=fast)
    reference = reference_cls(**kwargs)

    actual = custom(x1)
    expected = reference(x2)

    assert custom.fast is fast
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize(
    ('custom_fn', 'reference_fn', 'shape', 'args'),
    [
        (dF.avg_pool1d, F.avg_pool1d, (3, 8), (3,)),
        (dF.avg_pool2d, F.avg_pool2d, (3, 7, 8), (3,)),
        (dF.avg_pool3d, F.avg_pool3d, (3, 6, 7, 8), (3,)),
        (dF.max_pool1d, F.max_pool1d, (3, 8), (3,)),
        (dF.max_pool2d, F.max_pool2d, (3, 7, 8), (3,)),
        (dF.max_pool3d, F.max_pool3d, (3, 6, 7, 8), (3,)),
    ],
)
def test_pooling_supports_unbatched_input(
    custom_fn: Callable[..., Tensor],
    reference_fn: Callable[..., Tensor],
    shape: TupleND,
    args: Any,
):
    x = torch.randn(shape, dtype=torch.float64)
    actual = custom_fn(x, *args)
    expected = reference_fn(x, *args)

    assert_close(actual, expected)


def test_max_pooling_apis_omit_return_indices():
    names = [
        'max_pool1d',
        'max_pool2d',
        'max_pool3d',
        'adaptive_max_pool1d',
        'adaptive_max_pool2d',
        'adaptive_max_pool3d',
    ]
    for name in names:
        assert 'return_indices' not in inspect.signature(getattr(dF, name)).parameters

    class_names = [
        'MaxPool1d',
        'MaxPool2d',
        'MaxPool3d',
        'AdaptiveMaxPool1d',
        'AdaptiveMaxPool2d',
        'AdaptiveMaxPool3d',
    ]
    for name in class_names:
        assert 'return_indices' not in inspect.signature(getattr(dnn, name)).parameters


@pytest.mark.parametrize(
    ('custom_fn', 'reference_fn', 'size'),
    [
        (dF.max_pool1d, F.max_pool1d, 2),
        (dF.adaptive_max_pool1d, F.adaptive_max_pool1d, 1),
    ],
)
def test_max_pooling_tie_gradient_matches_torch(
    custom_fn: Callable[..., Tensor], reference_fn: Callable[..., Tensor], size: int
):
    x1 = torch.tensor([[[2.0, 2.0]]], dtype=torch.float64, requires_grad=True)
    x2 = _copy(x1)

    actual = custom_fn(x1, size)
    expected = reference_fn(x2, size)
    assert_close(actual, expected)

    grad = torch.ones_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(x1.grad, x2.grad)
