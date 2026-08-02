from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import dnnlpy.optim as dopt


def test_sgd_step_updates_parameters_and_skips_missing_gradients():
    trainable = torch.tensor([1.0, -2.0], requires_grad=True)
    trainable.grad = torch.tensor([0.25, -0.5])
    frozen = torch.tensor([3.0], requires_grad=True)

    optimizer = dopt.SGD([trainable, frozen], lr=0.1)
    optimizer.step()

    assert_close(trainable, torch.tensor([0.975, -1.95]))
    assert_close(frozen, torch.tensor([3.0]))


def test_sgd_with_momentum_accumulates_velocity():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])

    optimizer = dopt.SGD([param], lr=0.1, momentum=0.9)
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['velocity'], torch.tensor([0.5, -0.25]))
    assert_close(param, torch.tensor([0.95, -1.975]))

    param.grad = torch.tensor([0.1, 0.2])
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['velocity'], torch.tensor([0.55, -0.025]))
    assert_close(param, torch.tensor([0.895, -1.9725]))


def test_sgd_with_nesterov_momentum_uses_lookahead_update():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])

    optimizer = dopt.SGD([param], lr=0.1, momentum=0.9, nesterov=True)
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['velocity'], torch.tensor([0.5, -0.25]))
    assert_close(param, torch.tensor([0.905, -1.9525]))


@pytest.mark.parametrize(
    'optimizer',
    [
        lambda params: dopt.SGD(params, lr=0.1, weight_decay=0.2),
        lambda params: dopt.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.2),
        lambda params: dopt.SGD(
            params,
            lr=0.1,
            momentum=0.9,
            weight_decay=0.2,
            nesterov=True,
        ),
    ],
)
def test_sgd_step_does_not_mutate_gradients(optimizer: Callable[..., dopt.SGD]):
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])
    expected_grad = param.grad.clone()

    optimizer([param]).step()

    assert_close(param.grad, expected_grad)


def test_sgd_initializes_velocity_from_first_momentum_update():
    param = torch.tensor([1.0], requires_grad=True)
    param.grad = torch.tensor([0.5])

    optimizer = dopt.SGD([param], lr=0.1, momentum=0.9)
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['velocity'], torch.tensor([0.5]))
    assert_close(param, torch.tensor([0.95]))

    param.grad = torch.tensor([0.2])
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['velocity'], torch.tensor([0.65]))
    assert_close(param, torch.tensor([0.885]))
