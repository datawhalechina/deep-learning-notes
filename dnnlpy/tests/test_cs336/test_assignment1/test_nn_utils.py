# pyright: reportPrivateImportUsage=false

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.testing import assert_close

from .adapters import run_cross_entropy, run_gradient_clipping, run_softmax

__all__ = [
    'test_cross_entropy',
    'test_gradient_clipping',
    'test_softmax',
]


def test_softmax():
    """Test the softmax implementation against PyTorch's softmax."""
    x = torch.tensor(
        [
            [0.4655, 0.8303, 0.9608, 0.9656, 0.6840],
            [0.2583, 0.2198, 0.9334, 0.2995, 0.1722],
            [0.1573, 0.6860, 0.1327, 0.7284, 0.6811],
        ]
    )

    actual = run_softmax(x, dim=-1)
    expected = F.softmax(x, dim=-1)
    assert_close(actual, expected)

    # Test that softmax handles numerical overflow issues
    actual = run_softmax(x + 100, dim=-1)
    expected = F.softmax(x + 100, dim=-1)
    assert_close(actual, expected)


def test_cross_entropy():
    """Test the cross-entropy implementation against PyTorch's cross-entropy."""
    inputs = torch.tensor(
        [
            [
                [0.1088, 0.1060, 0.6683, 0.5131, 0.0645],
                [0.4538, 0.6852, 0.2520, 0.3792, 0.2675],
                [0.4578, 0.3357, 0.6384, 0.0481, 0.5612],
                [0.9639, 0.8864, 0.1585, 0.3038, 0.0350],
            ],
            [
                [0.3356, 0.9013, 0.7052, 0.8294, 0.8334],
                [0.6333, 0.4434, 0.1428, 0.5739, 0.3810],
                [0.9476, 0.5917, 0.7037, 0.2987, 0.6208],
                [0.8541, 0.1803, 0.2054, 0.4775, 0.8199],
            ],
        ]
    )
    targets = torch.tensor([[1, 0, 2, 2], [4, 1, 4, 0]])

    inputs = inputs.view(-1, inputs.size(-1))
    targets = targets.view(-1)

    actual = run_cross_entropy(inputs, targets)
    expected = F.cross_entropy(inputs, targets)
    assert_close(actual, expected)

    # Test that cross-entropy handles numerical overflow issues
    inputs = 1000.0 * inputs
    actual = run_cross_entropy(inputs, targets)
    expected = F.cross_entropy(inputs, targets)
    assert_close(actual, expected)


def test_gradient_clipping():
    """Test the gradient clipping implementation against PyTorch's clip_grad_norm_."""
    tensors = [torch.randn(5, 5, requires_grad=True) for _ in range(5)]
    tensors.append(torch.randn(5, 5))
    max_norm = 1e-2

    actual = [
        tensor.detach().clone().requires_grad_(tensor.requires_grad)
        for tensor in tensors
    ]
    loss = torch.concat(actual)
    loss.backward(torch.ones_like(loss))

    run_gradient_clipping(actual, max_norm)
    actual_grads = [t.grad.clone() for t in actual if t.grad is not None]

    expected = [
        tensor.detach().clone().requires_grad_(tensor.requires_grad)
        for tensor in tensors
    ]
    loss = torch.concat(expected)
    loss.backward(torch.ones_like(loss))

    clip_grad_norm_(expected, max_norm)
    expected_grads = [t.grad.clone() for t in expected if t.grad is not None]

    assert len(actual_grads) == len(expected_grads)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert_close(actual_grad, expected_grad)
