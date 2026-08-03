import pytest
import torch
from torch.testing import assert_close

import dnnlpy.optim as dopt
from dnnlpy.optim import muon


def test_muon_skips_parameters_without_gradients():
    trainable = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    trainable.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    skipped = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)

    optimizer = dopt.Muon(
        [trainable, skipped],
        lr=0.1,
        weight_decay=0.0,
        nesterov=False,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )
    optimizer.step()

    assert_close(trainable, torch.tensor([[0.94, -0.08], [0.0, 1.0]]))
    assert_close(skipped, torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    assert optimizer.state[skipped] == {}


def test_newton_schulz_5_preserves_tall_matrix_shape():
    update = torch.tensor([[3.0, 0.0], [4.0, 0.0], [0.0, 0.0]])
    actual = muon.newton_schulz_5(
        update,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    assert actual.shape == update.shape
    assert_close(actual, update / update.norm())


def test_muon_rejects_non_matrix_parameters():
    param = torch.tensor([1.0, 2.0], requires_grad=True)
    param.grad = torch.tensor([0.5, 0.25])
    optimizer = dopt.Muon([param])

    with pytest.raises(AssertionError, match='2D parameters'):
        optimizer.step()


def test_muon_accumulates_momentum_and_updates_parameters():
    param = torch.tensor([[1.0, -2.0], [0.5, 1.0]], requires_grad=True)
    optimizer = dopt.Muon(
        [param],
        lr=0.1,
        weight_decay=0.0,
        momentum=0.5,
        nesterov=False,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    param.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    optimizer.step()

    assert_close(optimizer.state[param]['momentum_buffer'], param.grad)
    assert_close(param, torch.tensor([[0.94, -2.08], [0.5, 1.0]]))

    param.grad = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    optimizer.step()

    expected_buffer = torch.tensor([[1.5, 2.0], [0.0, 5.0]])
    expected_update = expected_buffer / expected_buffer.norm()
    expected_param = torch.tensor([[0.94, -2.08], [0.5, 1.0]]) - 0.1 * expected_update

    state = optimizer.state[param]
    assert_close(state['momentum_buffer'], expected_buffer)
    assert_close(param, expected_param)


def test_muon_applies_decoupled_weight_decay():
    param = torch.tensor([[1.0, -2.0], [0.5, 1.0]], requires_grad=True)
    optimizer = dopt.Muon(
        [param],
        lr=0.1,
        weight_decay=0.2,
        momentum=0.0,
        nesterov=False,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    param.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    optimizer.step()

    assert_close(param, torch.tensor([[0.92, -2.04], [0.49, 0.98]]))


def test_muon_can_use_nesterov_direction():
    param = torch.tensor([[1.0, -2.0], [0.5, 1.0]], requires_grad=True)
    optimizer = dopt.Muon(
        [param],
        lr=0.1,
        weight_decay=0.0,
        momentum=0.5,
        nesterov=True,
        ns_steps=1,
        eps=0.0,
        ns_coefficients=(1.0, 0.0, 0.0),
    )

    param.grad = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    optimizer.step()

    state = optimizer.state[param]
    assert_close(state['momentum_buffer'], param.grad)
    assert_close(param, torch.tensor([[0.94, -2.08], [0.5, 1.0]]))
