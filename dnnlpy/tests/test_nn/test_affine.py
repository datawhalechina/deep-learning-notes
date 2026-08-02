import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


def test_bilinear_function_matches_torch_with_bias():
    x1 = torch.randn(4, 3, requires_grad=True)
    x2 = torch.randn(4, 2, requires_grad=True)
    w1 = torch.randn(5, 3, 2, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)

    y1 = _copy(x1)
    y2 = _copy(x2)
    w2 = _copy(w1)
    b2 = _copy(b1)

    actual = dF.bilinear(x1, x2, w1, b1)
    expected = F.bilinear(y1, y2, w2, b2)

    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(x1.grad, y1.grad)
    assert_close(x2.grad, y2.grad)
    assert_close(w1.grad, w2.grad)
    assert_close(b1.grad, b2.grad)


def test_bilinear_function_matches_torch_without_bias_for_batched_inputs():
    x1 = torch.randn(2, 4, 3)
    x2 = torch.randn(2, 4, 2)
    weight = torch.randn(5, 3, 2)

    actual = dF.bilinear(x1, x2, weight)
    expected = F.bilinear(x1, x2, weight)

    assert_close(actual, expected)


def test_bilinear_module_matches_torch():
    x1 = torch.randn(2, 4, 3)
    x2 = torch.randn(2, 4, 2)

    custom = dnn.Bilinear(3, 2, 5)
    reference = nn.Bilinear(3, 2, 5)
    reference.load_state_dict(custom.state_dict())

    assert custom.in1_features == reference.in1_features
    assert custom.in2_features == reference.in2_features
    assert custom.out_features == reference.out_features
    assert_close(custom(x1, x2), reference(x1, x2))


def test_fast_bilinear_module_matches_torch():
    x1 = torch.randn(2, 4, 3)
    x2 = torch.randn(2, 4, 2)

    custom = dnn.Bilinear(3, 2, 5, fast=True)
    reference = nn.Bilinear(3, 2, 5)
    reference.load_state_dict(custom.state_dict())

    assert custom.fast is True
    assert_close(custom(x1, x2), reference(x1, x2))


def test_bilinear_module_supports_no_bias():
    x1 = torch.randn(4, 3)
    x2 = torch.randn(4, 2)

    custom = dnn.Bilinear(3, 2, 5, bias=False)
    reference = nn.Bilinear(3, 2, 5, bias=False)
    reference.load_state_dict(custom.state_dict())

    assert custom.bias is None
    assert_close(custom(x1, x2), reference(x1, x2))


def test_linear_function_matches_torch_with_bias():
    x1 = torch.randn(4, 3, requires_grad=True)
    w1 = torch.randn(5, 3, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)

    x2 = _copy(x1)
    w2 = _copy(w1)
    b2 = _copy(b1)

    actual = dF.linear(x1, w1, b1)
    expected = F.linear(x2, w2, b2)

    assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)

    assert_close(x1.grad, x2.grad)
    assert_close(w1.grad, w2.grad)
    assert_close(b1.grad, b2.grad)


def test_linear_function_matches_torch_without_bias_for_batched_input():
    x = torch.randn(2, 4, 3)
    weight = torch.randn(5, 3)

    actual = dF.linear(x, weight)
    expected = F.linear(x, weight)

    assert_close(actual, expected)


def test_linear_module_matches_torch():
    x = torch.randn(2, 4, 3)

    custom = dnn.Linear(3, 5)
    reference = nn.Linear(3, 5)
    reference.load_state_dict(custom.state_dict())

    assert custom.in_features == reference.in_features
    assert custom.out_features == reference.out_features
    assert_close(custom(x), reference(x))


def test_fast_linear_module_matches_torch():
    x = torch.randn(2, 4, 3)

    custom = dnn.Linear(3, 5, fast=True)
    reference = nn.Linear(3, 5)
    reference.load_state_dict(custom.state_dict())

    assert custom.fast is True
    assert_close(custom(x), reference(x))


def test_linear_module_supports_no_bias():
    x = torch.randn(2, 3)

    custom = dnn.Linear(3, 5, bias=False)
    reference = nn.Linear(3, 5, bias=False)
    reference.load_state_dict(custom.state_dict())

    assert custom.bias is None
    assert_close(custom(x), reference(x))


def test_linear_module_initializes_bias_with_symmetric_bound():
    custom = dnn.Linear(16, 128)
    bound = 1 / math.sqrt(custom.in_features)

    assert custom.bias is not None
    assert custom.bias.min() >= -bound
    assert custom.bias.max() <= bound


def test_flatten_module_matches_torch():
    x = torch.randn(2, 3, 4, 5)

    custom = dnn.Flatten(start_dim=1, end_dim=2)
    reference = nn.Flatten(start_dim=1, end_dim=2)

    assert custom.start_dim == reference.start_dim
    assert custom.end_dim == reference.end_dim
    assert_close(custom(x), reference(x))


def test_flatten_module_preserves_fast_argument():
    x = torch.randn(2, 3, 4)

    custom = dnn.Flatten(fast=True)
    expected = torch.flatten(x, start_dim=1, end_dim=-1)

    assert custom.fast is True
    assert_close(custom(x), expected)


def test_unflatten_module_matches_torch():
    x = torch.randn(2, 12, 5)

    custom = dnn.Unflatten(dim=1, unflattened_size=(3, 4))
    reference = nn.Unflatten(dim=1, unflattened_size=(3, 4))

    assert custom.dim == reference.dim
    assert custom.unflattened_size == reference.unflattened_size
    assert_close(custom(x), reference(x))


def test_unflatten_module_preserves_fast_argument():
    x = torch.randn(2, 12, 5)

    custom = dnn.Unflatten(dim=1, unflattened_size=(3, 4), fast=True)
    expected = torch.unflatten(x, dim=1, sizes=(3, 4))

    assert custom.fast is True
    assert_close(custom(x), expected)


def test_identity_module_returns_same_tensor():
    x = torch.randn(2, 3)
    custom = dnn.Identity()

    assert custom(x) is x
