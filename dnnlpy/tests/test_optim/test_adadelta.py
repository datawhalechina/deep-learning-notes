import torch
from torch.testing import assert_close

import dnnlpy.optim as dopt


def test_adadelta_skips_parameters_without_gradients():
    trainable = torch.tensor([1.0], requires_grad=True)
    trainable.grad = torch.tensor([0.5])
    skipped = torch.tensor([2.0], requires_grad=True)

    optimizer = dopt.Adadelta([trainable, skipped], lr=1.0, rho=0.9, eps=1e-6)
    optimizer.step()

    assert_close(trainable, torch.tensor([0.9968378]))
    assert_close(skipped, torch.tensor([2.0]))
    assert optimizer.state[skipped] == {}


def test_adadelta_accumulates_state_and_updates_parameters():
    param = torch.tensor([1.0, -2.0], requires_grad=True)
    param.grad = torch.tensor([0.5, -0.25])

    optimizer = dopt.Adadelta([param], lr=1.0, rho=0.9, eps=1e-6)
    optimizer.step()

    expected_square_avg = torch.tensor([0.025, 0.00625])
    expected_update = torch.tensor([-0.0031622, 0.0031620])
    expected_accumulate_update = 0.1 * expected_update.square()

    state = optimizer.state[param]
    assert_close(state['square_avg'], expected_square_avg)
    assert_close(state['acc_delta'], expected_accumulate_update)
    assert_close(param, torch.tensor([0.9968378, -1.9968380]))
