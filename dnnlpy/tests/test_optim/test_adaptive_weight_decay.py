from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.optim as optim
from torch.testing import assert_close

import dnnlpy.optim as dopt

type Optimizer = Callable[..., optim.Optimizer]


@pytest.mark.parametrize(
    ('optimizer_cls', 'kwargs'),
    [
        (dopt.Adagrad, {'lr': 0.1, 'eps': 0.0}),
        (dopt.RMSprop, {'lr': 0.1, 'rho': 0.9, 'eps': 1e-8}),
        (dopt.Adadelta, {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6}),
        (dopt.Adam, {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8}),
    ],
)
def test_adaptive_optimizers_apply_weight_decay_as_gradient_term(
    optimizer_cls: Optimizer,
    kwargs: Any,
):
    actual_param = torch.tensor([1.0, -2.0], requires_grad=True)
    expected_param = actual_param.detach().clone().requires_grad_()
    weight_decay = 0.2

    actual_optimizer = optimizer_cls(
        [actual_param],
        weight_decay=weight_decay,
        **kwargs,
    )
    expected_optimizer = optimizer_cls([expected_param], **kwargs)

    for grad in [torch.tensor([0.5, -0.25]), torch.tensor([0.25, 0.5])]:
        actual_param.grad = grad.clone()
        expected_param.grad = grad + weight_decay * expected_param.detach()

        actual_optimizer.step()
        expected_optimizer.step()

    assert_close(actual_param, expected_param)


@pytest.mark.parametrize(
    ('optimizer_cls', 'kwargs'),
    [
        (dopt.Adagrad, {'lr': 0.1, 'eps': 0.0}),
        (dopt.RMSprop, {'lr': 0.1, 'rho': 0.9, 'eps': 1e-8}),
        (dopt.Adadelta, {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6}),
        (dopt.Adam, {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8}),
    ],
)
def test_adaptive_optimizers_do_not_mutate_gradients(
    optimizer_cls: Optimizer,
    kwargs: Any,
):
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])
    expected_grad = param.grad.clone()
    optimizer = optimizer_cls([param], weight_decay=0.2, **kwargs)

    optimizer.step()

    assert_close(param.grad, expected_grad)
