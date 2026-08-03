import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF

type Size2D = int | tuple[int, int]


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


@pytest.mark.parametrize(
    ('kernel_size', 'dilation', 'padding', 'stride'),
    [
        (2, 1, 0, 1),
        ((2, 3), 1, (1, 0), (2, 1)),
        ((2, 3), (1, 2), (1, 0), (2, 1)),
        ((3, 2), (2, 1), (2, 1), (1, 2)),
    ],
)
def test_unfold_matches_torch(
    kernel_size: Size2D,
    dilation: Size2D,
    padding: Size2D,
    stride: Size2D,
):
    base = torch.randn(2, 3, 6, 7, dtype=torch.float64, requires_grad=True)
    x1 = _copy(base)
    x2 = _copy(base)

    actual = dF.unfold(
        x1,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    expected = F.unfold(
        x2,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert x1.grad is not None
    assert x2.grad is not None
    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize(
    ('output_size', 'kernel_size', 'dilation', 'padding', 'stride'),
    [
        ((6, 7), 2, 1, 0, 1),
        ((6, 7), (2, 3), 1, (1, 0), (2, 1)),
        ((6, 7), (2, 3), (1, 2), (1, 0), (2, 1)),
        ((6, 7), (3, 2), (2, 1), (2, 1), (1, 2)),
    ],
)
def test_fold_matches_torch(
    output_size: tuple[int, int],
    kernel_size: Size2D,
    dilation: Size2D,
    padding: Size2D,
    stride: Size2D,
):
    images = torch.randn(2, 3, *output_size, dtype=torch.float64)
    blocks = F.unfold(
        images,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    x1 = _copy(blocks)
    x2 = _copy(blocks)

    actual = dF.fold(
        x1,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    expected = F.fold(
        x2,
        output_size=output_size,
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )

    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert x1.grad is not None
    assert x2.grad is not None
    assert_close(x1.grad, x2.grad)


def test_fold_accumulates_overlapping_blocks():
    image = torch.ones(1, 1, 4, 4)
    unfolded = dF.unfold(image, kernel_size=3, stride=1)

    actual = dF.fold(unfolded, output_size=(4, 4), kernel_size=3, stride=1)
    expected = F.fold(unfolded, output_size=(4, 4), kernel_size=3, stride=1)

    assert_close(actual, expected)


@pytest.mark.parametrize('fast', [False, True])
def test_unfold_module_matches_torch_forward_and_backward(fast: bool):
    base = torch.randn(2, 3, 6, 7, dtype=torch.float64)
    x1 = _copy(base)
    x2 = _copy(base)

    custom = dnn.Unfold(
        kernel_size=(2, 3),
        dilation=(1, 2),
        padding=(1, 0),
        stride=(2, 1),
        fast=fast,
    )
    reference = nn.Unfold(
        kernel_size=(2, 3),
        dilation=(1, 2),
        padding=(1, 0),
        stride=(2, 1),
    )

    actual = custom(x1)
    expected = reference(x2)

    assert custom.fast is fast
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert x1.grad is not None
    assert x2.grad is not None
    assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize('fast', [False, True])
def test_fold_module_matches_torch_forward_and_backward(fast: bool):
    output_size = (6, 7)
    options = {
        'kernel_size': (2, 3),
        'dilation': (1, 2),
        'padding': (1, 0),
        'stride': (2, 1),
    }

    images = torch.randn(2, 3, *output_size, dtype=torch.float64)
    blocks = F.unfold(images, **options)

    x1 = _copy(blocks)
    x2 = _copy(blocks)

    custom = dnn.Fold(output_size, **options, fast=fast)
    reference = nn.Fold(output_size, **options)

    actual = custom(x1)
    expected = reference(x2)

    assert custom.fast is fast
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert x1.grad is not None
    assert x2.grad is not None
    assert_close(x1.grad, x2.grad)


def test_fold_and_unfold_module_extra_repr_match_torch():
    fold_options = {
        'output_size': (6, 7),
        'kernel_size': (2, 3),
        'dilation': (1, 2),
        'padding': (1, 0),
        'stride': (2, 1),
    }
    unfold_options = {
        'kernel_size': (2, 3),
        'dilation': (1, 2),
        'padding': (1, 0),
        'stride': (2, 1),
    }

    custom = dnn.Fold(**fold_options, fast=False)
    reference = nn.Fold(**fold_options)
    assert custom.extra_repr() == reference.extra_repr()

    custom = dnn.Unfold(**unfold_options, fast=False)
    reference = nn.Unfold(**unfold_options)
    assert custom.extra_repr() == reference.extra_repr()
