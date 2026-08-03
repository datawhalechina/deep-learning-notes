from typing import Any, Protocol

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.conv import _ConvNd as ReferenceConv
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF
from dnnlpy.nn.convolution import _ConvNd as CustomConv

type TupleND = tuple[int, ...]
type Size1D = int | tuple[int]
type Size2D = int | tuple[int, int]
type Size3D = int | tuple[int, int, int]
type SizeND = int | TupleND


class ConvFn(Protocol):
    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: SizeND = 1,
        padding: SizeND = 0,
        dilation: SizeND = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
    ) -> Tensor: ...


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


def _padding_tuple(padding: SizeND, ndim: int) -> TupleND:
    """Pad a convolution input tensor with the given padding."""
    if isinstance(padding, int):
        padding = (padding,) * ndim
    return tuple(pad for value in reversed(padding) for pad in (value, value))


def _as_tuple(value: SizeND, ndim: int) -> TupleND:
    """Convert a value to a tuple of the given length."""
    if isinstance(value, int):
        return (value,) * ndim
    return value


@torch.no_grad()
def _copy_conv_parameters(custom: CustomConv, reference: ReferenceConv):
    custom.weight.copy_(reference.weight)
    if custom.bias is not None and reference.bias is not None:
        custom.bias.copy_(reference.bias)


@pytest.mark.parametrize(
    ('custom_fn', 'reference_fn', 'shape', 'weight_shape', 'stride'),
    [
        (dF.conv1d, F.conv1d, (2, 3, 8), (4, 3, 3), (1,)),
        (dF.conv2d, F.conv2d, (2, 3, 6, 7), (4, 3, 3, 2), (1, 1)),
        (dF.conv3d, F.conv3d, (2, 3, 5, 6, 7), (4, 3, 3, 2, 3), (1, 1, 1)),
    ],
)
def test_convolution_function_gradients_match_torch(
    custom_fn: ConvFn,
    reference_fn: ConvFn,
    shape: TupleND,
    weight_shape: TupleND,
    stride: TupleND,
):
    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    w1 = torch.randn(weight_shape, dtype=torch.float64, requires_grad=True)
    b1 = torch.randn(weight_shape[0], dtype=torch.float64, requires_grad=True)

    x2 = _copy(x1)
    w2 = _copy(w1)
    b2 = _copy(b1)

    actual = custom_fn(x1, w1, bias=b1, stride=stride)
    expected = reference_fn(x2, w2, bias=b2, stride=stride)

    assert_close(actual, expected, atol=1e-10, rtol=1e-7)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(x1.grad, x2.grad, atol=1e-10, rtol=1e-7)
    assert_close(w1.grad, w2.grad, atol=1e-10, rtol=1e-7)
    assert_close(b1.grad, b2.grad, atol=1e-10, rtol=1e-7)


@pytest.mark.parametrize(
    ('stride', 'padding', 'dilation', 'groups'),
    [
        (1, 0, 1, 1),
        ((2,), (1,), (1,), 1),
        ((1,), (2,), (2,), 1),
        (1, 1, 1, 2),
    ],
)
def test_conv1d_matches_torch(
    stride: Size1D, padding: Size1D, dilation: Size1D, groups: int
):
    x = torch.randn(2, 4, 11)
    weight = torch.randn(6, 4 // groups, 3)
    bias = torch.randn(6)
    conv_pad = _padding_tuple(padding, ndim=1)

    actual = dF.conv1d(
        x,
        weight,
        bias=bias,
        stride=_as_tuple(stride, ndim=1),
        padding=conv_pad,
        dilation=_as_tuple(dilation, ndim=1),
        groups=groups,
    )
    expected = F.conv1d(
        F.pad(x, conv_pad),
        weight,
        bias=bias,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('stride', 'padding', 'dilation', 'groups'),
    [
        (1, 0, 1, 1),
        ((2, 1), (1, 2), 1, 1),
        ((1, 2), (2, 1), (2, 1), 1),
        (1, 1, 1, 2),
    ],
)
def test_conv2d_matches_torch(
    stride: Size2D, padding: Size2D, dilation: Size2D, groups: int
):
    x = torch.randn(2, 4, 8, 9)
    weight = torch.randn(6, 4 // groups, 3, 3)
    bias = torch.randn(6)
    conv_pad = _padding_tuple(padding, ndim=2)

    actual = dF.conv2d(
        x,
        weight,
        bias=bias,
        stride=_as_tuple(stride, ndim=2),
        padding=conv_pad,
        dilation=_as_tuple(dilation, ndim=2),
        groups=groups,
    )
    expected = F.conv2d(
        F.pad(x, conv_pad),
        weight,
        bias=bias,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ('stride', 'padding', 'dilation', 'groups'),
    [
        (1, 0, 1, 1),
        ((1, 2, 1), (1, 0, 2), 1, 1),
        ((1, 2, 1), (1, 2, 1), (1, 2, 1), 1),
        (1, 1, 1, 2),
    ],
)
def test_conv3d_matches_torch(
    stride: Size3D, padding: Size3D, dilation: Size3D, groups: int
):
    x = torch.randn(2, 4, 5, 7, 8)
    weight = torch.randn(6, 4 // groups, 3, 3, 3)
    bias = torch.randn(6)
    conv_pad = _padding_tuple(padding, ndim=3)

    actual = dF.conv3d(
        x,
        weight,
        bias=bias,
        stride=_as_tuple(stride, ndim=3),
        padding=conv_pad,
        dilation=_as_tuple(dilation, ndim=3),
        groups=groups,
    )
    expected = F.conv3d(
        F.pad(x, conv_pad),
        weight,
        bias=bias,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_conv2d_supports_no_bias():
    x = torch.randn(2, 3, 6, 7)
    weight = torch.randn(4, 3, 3, 2)
    conv_pad = _padding_tuple((1, 0), ndim=2)

    actual = dF.conv2d(x, weight, padding=conv_pad)
    expected = F.conv2d(F.pad(x, conv_pad), weight)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('conv_name', 'shape', 'weight_shape', 'padding'),
    [
        ('conv1d', (2, 3, 8), (4, 3, 3), (1,)),
        ('conv2d', (2, 3, 6, 7), (4, 3, 3, 2), (1, 2)),
        ('conv3d', (2, 3, 5, 6, 7), (4, 3, 3, 2, 3), (1, 1, 2)),
    ],
)
@pytest.mark.parametrize('padding_mode', ['reflect', 'replicate', 'circular'])
def test_convolution_supports_padding_mode(
    conv_name: str,
    shape: TupleND,
    weight_shape: TupleND,
    padding: TupleND,
    padding_mode: str,
):
    x = torch.randn(shape)
    weight = torch.randn(weight_shape)
    bias = torch.randn(weight_shape[0])

    ndim = len(shape) - 2
    conv_pad = _padding_tuple(padding, ndim=ndim)

    custom_fn = getattr(dF, conv_name)
    reference_fn = getattr(F, conv_name)

    actual = custom_fn(
        x,
        weight,
        bias=bias,
        padding=conv_pad,
        padding_mode=padding_mode,
    )
    y = F.pad(x, conv_pad, mode=padding_mode)
    expected = reference_fn(y, weight, bias=bias)

    assert_close(actual, expected)


@pytest.mark.parametrize(
    (
        'custom_cls',
        'reference_cls',
        'shape',
        'kwargs',
    ),
    [
        (
            dnn.Conv1d,
            nn.Conv1d,
            (2, 4, 11),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3,),
                'stride': (2,),
                'padding': (1,),
                'dilation': (1,),
            },
        ),
        (
            dnn.Conv2d,
            nn.Conv2d,
            (2, 4, 8, 9),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3),
                'stride': (2, 1),
                'padding': (1, 2),
                'dilation': (1, 1),
            },
        ),
        (
            dnn.Conv3d,
            nn.Conv3d,
            (2, 4, 5, 7, 8),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3, 3),
                'stride': (1, 2, 1),
                'padding': (1, 0, 2),
                'dilation': (1, 1, 1),
            },
        ),
        (
            dnn.Conv2d,
            nn.Conv2d,
            (2, 4, 8, 9),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3),
                'stride': (1, 1),
                'padding': (2, 1),
                'dilation': (2, 1),
            },
        ),
        (
            dnn.Conv3d,
            nn.Conv3d,
            (2, 4, 5, 7, 8),
            {
                'in_channels': 4,
                'out_channels': 6,
                'kernel_size': (3, 3, 3),
                'stride': (1, 1, 1),
                'padding': (2, 1, 1),
                'dilation': (2, 1, 1),
            },
        ),
    ],
)
def test_conv_module_matches_torch(
    custom_cls: type[CustomConv],
    reference_cls: type[ReferenceConv],
    shape: TupleND,
    kwargs: Any,
):
    x = torch.randn(shape)
    custom = custom_cls(**kwargs)
    reference = reference_cls(**kwargs)
    _copy_conv_parameters(custom, reference)

    actual = custom(x)
    expected = reference(x)

    assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ('custom_cls', 'reference_cls', 'shape', 'padding'),
    [
        (dnn.Conv1d, nn.Conv1d, (2, 3, 8), (1,)),
        (dnn.Conv2d, nn.Conv2d, (2, 3, 6, 7), (1, 2)),
        (dnn.Conv3d, nn.Conv3d, (2, 3, 5, 6, 7), (1, 1, 2)),
    ],
)
@pytest.mark.parametrize('padding_mode', ['reflect', 'replicate', 'circular'])
def test_conv_module_supports_padding_mode(
    custom_cls: type[CustomConv],
    reference_cls: type[ReferenceConv],
    shape: TupleND,
    padding: TupleND,
    padding_mode: str,
):
    x = torch.randn(shape)
    kwargs = {
        'in_channels': 3,
        'out_channels': 4,
        'kernel_size': (3,) * (len(shape) - 2),
        'padding': padding,
        'padding_mode': padding_mode,
    }

    custom = custom_cls(**kwargs)
    reference = reference_cls(**kwargs)
    _copy_conv_parameters(custom, reference)

    actual = custom(x)
    expected = reference(x)

    assert_close(actual, expected)
