import torch
from torch.testing import assert_close

import dnnlpy.optim as dopt


def test_rmsprop_skips_parameters_without_gradients():
    trainable = torch.tensor([1.0], requires_grad=True)
    trainable.grad = torch.tensor([0.5])
    skipped = torch.tensor([2.0], requires_grad=True)

    optimizer = dopt.RMSprop([trainable, skipped], lr=0.1, rho=0.9, eps=0.0)
    optimizer.step()

    assert_close(trainable, torch.tensor([0.6837722]))
    assert_close(skipped, torch.tensor([2.0]))
    assert optimizer.state[skipped] == {}


def test_rmsprop_accumulates_squared_gradients_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.RMSprop([param], lr=0.1, rho=0.9, eps=0.0)

    param.grad = torch.tensor([0.5, -0.25])
    optimizer.step()

    expected_square_avg = torch.tensor([0.025, 0.00625])

    state = optimizer.state[param]
    assert_close(state['square_avg'], expected_square_avg)
    assert_close(param, torch.tensor([0.6837722, -1.6837722]))


def test_rmsprop_accumulates_momentum_buffer():
    param = torch.tensor([1.0], requires_grad=True)
    optimizer = dopt.RMSprop([param], lr=0.1, rho=0.9, eps=0.0, momentum=0.5)

    param.grad = torch.tensor([0.5])
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['square_avg'], torch.tensor([0.025]))
    assert_close(state['momentum_buffer'], torch.tensor([3.1622777]))
    assert_close(param, torch.tensor([0.6837722]))

    param.grad = torch.tensor([0.25])
    optimizer.step()

    expected_ema = torch.tensor([0.02875])
    expected_buffer = 0.5 * torch.tensor([3.1622777]) + 0.25 / expected_ema.sqrt()

    state = optimizer.state[param]
    assert_close(state['square_avg'], expected_ema)
    assert_close(state['momentum_buffer'], expected_buffer)
    assert_close(param, torch.tensor([0.6837722]) - 0.1 * expected_buffer)
