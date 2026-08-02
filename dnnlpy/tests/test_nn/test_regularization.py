from typing import Protocol

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


class DropoutFn(Protocol):
    def __call__(self, x: Tensor, p: float = 0.5, training: bool = False) -> Tensor: ...


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


@pytest.mark.parametrize(
    ('custom_fn', 'reference_fn', 'shape'),
    [
        (dF.dropout, F.dropout, (2, 3, 4)),
        (dF.dropout1d, F.dropout1d, (2, 3, 4)),
        (dF.dropout2d, F.dropout2d, (2, 3, 4, 5)),
        (dF.dropout3d, F.dropout3d, (2, 3, 4, 5, 6)),
    ],
)
def test_dropout_functions_return_input_when_not_training(
    custom_fn: DropoutFn, reference_fn: DropoutFn, shape: tuple[int, ...]
):
    x1 = torch.randn(shape, requires_grad=True)
    x2 = _copy(x1)

    actual = custom_fn(x1, p=0.5, training=False)
    expected = reference_fn(x2, p=0.5, training=False)

    assert actual is x1
    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(x1.grad, x2.grad)


def test_dropout_module_returns_input_in_eval_mode():
    x = torch.randn(2, 3, 4, 5)

    custom = dnn.Dropout2d(p=0.5)
    reference = nn.Dropout2d(p=0.5)

    custom.eval()
    reference.eval()

    actual = custom(x)
    expected = reference(x)

    assert actual is x
    assert_close(actual, expected)


@pytest.mark.parametrize(
    ('custom_fn', 'shape'),
    [
        (dF.dropout, (40, 50)),
        (dF.dropout1d, (8, 12, 10)),
        (dF.dropout2d, (8, 12, 5, 6)),
        (dF.dropout3d, (4, 6, 3, 4, 5)),
    ],
)
def test_dropout_functions_scale_surviving_values(
    custom_fn: DropoutFn, shape: tuple[int, ...]
):
    x = torch.ones(shape)
    p = 0.25

    actual = custom_fn(x, p=p, training=True)

    unique_values = actual.unique()
    expected_values = torch.tensor([0.0, 1.0 / (1.0 - p)], dtype=x.dtype)

    for value in unique_values:
        assert torch.any(torch.isclose(value, expected_values))


@pytest.mark.parametrize(
    ('custom_fn', 'shape', 'reduce_dims'),
    [
        (dF.dropout1d, (8, 12, 10), (2,)),
        (dF.dropout2d, (8, 12, 5, 6), (2, 3)),
        (dF.dropout3d, (4, 6, 3, 4, 5), (2, 3, 4)),
    ],
)
def test_dropout_nd_zeros_whole_channels(
    custom_fn: DropoutFn, shape: tuple[int, ...], reduce_dims: tuple[int, ...]
):
    x = torch.ones(shape)

    actual = custom_fn(x, p=0.5, training=True)
    channel_sums = actual.sum(dim=reduce_dims)
    channel_vars = actual.var(dim=reduce_dims, correction=0)

    assert_close(channel_vars, torch.zeros_like(channel_vars))

    for index in torch.nonzero(channel_sums == 0, as_tuple=False):
        slices = tuple(index.tolist()) + (slice(None),) * len(reduce_dims)
        assert_close(actual[slices], torch.zeros_like(actual[slices]))


@pytest.mark.parametrize(
    ('custom_fn', 'shape'),
    [
        (dF.dropout1d, (3, 10)),
        (dF.dropout2d, (3, 5, 6)),
        (dF.dropout3d, (3, 4, 5, 6)),
    ],
)
def test_dropout_nd_supports_unbatched_inputs(
    custom_fn: DropoutFn, shape: tuple[int, ...]
):
    x = torch.ones(shape)
    actual = custom_fn(x, p=0.5, training=True)

    assert actual.shape == x.shape


@pytest.mark.parametrize(
    ('custom_fn', 'reference_fn', 'shape'),
    [
        (dF.dropout, F.dropout, (2, 3)),
        (dF.dropout1d, F.dropout1d, (2, 3, 4)),
        (dF.dropout2d, F.dropout2d, (2, 3, 4, 5)),
        (dF.dropout3d, F.dropout3d, (2, 3, 4, 5, 6)),
    ],
)
def test_dropout_functions_match_torch_for_extreme_probabilities(
    custom_fn: DropoutFn, reference_fn: DropoutFn, shape: tuple[int, ...]
):
    x = torch.randn(shape)

    assert_close(custom_fn(x, p=0.0), reference_fn(x, p=0.0))
    assert_close(custom_fn(x, p=1.0), reference_fn(x, p=1.0))


def test_dropout_supports_inplace_operation():
    x = torch.ones(8, 10)
    actual = dF.dropout(x, p=1.0, training=True, inplace=True)

    assert actual is x
    assert_close(x, torch.zeros_like(x))


@pytest.mark.parametrize(
    ('custom_fn', 'bad_shape'),
    [
        (dF.dropout1d, (2, 3, 4, 5)),
        (dF.dropout2d, (2, 3)),
        (dF.dropout3d, (2, 3, 4)),
    ],
)
def test_dropout_nd_rejects_invalid_rank(
    custom_fn: DropoutFn, bad_shape: tuple[int, ...]
):
    with pytest.raises(AssertionError):
        custom_fn(torch.randn(bad_shape))
