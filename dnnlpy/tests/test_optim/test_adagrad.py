import torch
from torch.testing import assert_close

import dnnlpy.optim as dopt


def test_adagrad_skips_parameters_without_gradients():
    trainable = torch.tensor([1.0], requires_grad=True)
    trainable.grad = torch.tensor([0.5])
    skipped = torch.tensor([2.0], requires_grad=True)

    optimizer = dopt.Adagrad([trainable, skipped], lr=0.1, eps=0.0)
    optimizer.step()

    assert_close(trainable, torch.tensor([0.9]))
    assert_close(skipped, torch.tensor([2.0]))
    assert optimizer.state[skipped] == {}


def test_adagrad_accumulates_squared_gradients_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.Adagrad([param], lr=0.1, eps=0.0)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['sum_of_sq_grads'], torch.tensor([0.25, 0.0625]))
    assert_close(param, torch.tensor([0.9, -1.9]))

    param.grad = torch.tensor([0.5, 0.25])
    optimizer.step()

    expected_sum_sq_grads = torch.tensor([0.5, 0.125])

    state = optimizer.state[param]
    assert_close(state['sum_of_sq_grads'], expected_sum_sq_grads)
    assert_close(param, torch.tensor([0.8292893, -1.9707106]))


def test_adagrad_uses_lr_decay_and_initial_accumulator_value():
    param = torch.tensor([1.0], requires_grad=True)
    optimizer = dopt.Adagrad(
        [param],
        lr=0.2,
        lr_decay=0.5,
        initial_accumulator_value=0.25,
        eps=0.0,
    )

    assert optimizer.state[param] == {}

    param.grad = torch.tensor([0.5])
    optimizer.step()

    state = optimizer.state[param]
    assert state['step'] == 1
    assert_close(state['sum_of_sq_grads'], torch.tensor([0.5]))
    assert_close(param, torch.tensor([0.8585786]))

    param.grad = torch.tensor([0.5])
    optimizer.step()

    expected_clr = 0.2 / (1 + 0.5)
    expected_param = torch.tensor([0.8585786]) - expected_clr / (0.75**0.5) * 0.5

    state = optimizer.state[param]
    assert state['step'] == 2
    assert_close(state['sum_of_sq_grads'], torch.tensor([0.75]))
    assert_close(param, expected_param)
