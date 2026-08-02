from collections.abc import Callable

import pytest
import torch
import torch.optim as optim
from torch.testing import assert_close

import dnnlpy.optim as dopt

type Optimizer = Callable[..., optim.Optimizer]


def quadratic_loss(theta: torch.Tensor) -> torch.Tensor:
    return theta.square().sum() / 2


@pytest.mark.parametrize(
    ('optimizer_cls', 'group1', 'group2'),
    [
        (
            dopt.SGD,
            {'lr': 0.1, 'momentum': 0.0, 'weight_decay': 0.0, 'nesterov': False},
            {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.2, 'nesterov': True},
        ),
        (
            dopt.Adagrad,
            {
                'lr': 0.1,
                'lr_decay': 0.0,
                'weight_decay': 0.0,
                'initial_accumulator_value': 0.0,
                'eps': 1e-10,
            },
            {
                'lr': 0.01,
                'lr_decay': 0.1,
                'weight_decay': 0.2,
                'initial_accumulator_value': 0.5,
                'eps': 1e-8,
            },
        ),
        (
            dopt.Adadelta,
            {'lr': 1.0, 'rho': 0.9, 'eps': 1e-6, 'weight_decay': 0.0},
            {'lr': 0.5, 'rho': 0.8, 'eps': 1e-5, 'weight_decay': 0.2},
        ),
        (
            dopt.RMSprop,
            {'lr': 0.1, 'rho': 0.9, 'eps': 1e-8, 'weight_decay': 0.0, 'momentum': 0.0},
            {'lr': 0.01, 'rho': 0.8, 'eps': 1e-6, 'weight_decay': 0.2, 'momentum': 0.9},
        ),
        (
            dopt.Adam,
            {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0},
            {'lr': 0.01, 'betas': (0.8, 0.9), 'eps': 1e-6, 'weight_decay': 0.2},
        ),
        (
            dopt.AdamW,
            {'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0},
            {'lr': 0.01, 'betas': (0.8, 0.9), 'eps': 1e-6, 'weight_decay': 0.2},
        ),
        (
            dopt.Muon,
            {
                'lr': 0.1,
                'weight_decay': 0.0,
                'momentum': 0.0,
                'nesterov': False,
                'ns_steps': 2,
                'eps': 1e-7,
            },
            {
                'lr': 0.01,
                'weight_decay': 0.2,
                'momentum': 0.9,
                'nesterov': True,
                'ns_steps': 3,
                'eps': 1e-6,
            },
        ),
    ],
)
def test_optimizer_param_groups_match_independent_optimizers(
    optimizer_cls: Optimizer,
    group1: dict[str, object],
    group2: dict[str, object],
):
    params = [
        torch.tensor([[1.0, -2.0], [0.5, -0.25]], requires_grad=True),
        torch.tensor([[0.25, -0.75], [1.5, -1.0]], requires_grad=True),
    ]
    expected = [p.detach().clone().requires_grad_() for p in params]
    grads = [
        torch.tensor([[0.5, -0.25], [0.1, 0.2]]),
        torch.tensor([[0.25, 0.5], [-0.3, 0.4]]),
    ]

    grouped_optimizer = optimizer_cls(
        [
            {'params': [params[0]], **group1},
            {'params': [params[1]], **group2},
        ]
    )
    expected_optimizers = [
        optimizer_cls([expected[0]], **group1),
        optimizer_cls([expected[1]], **group2),
    ]

    assert grouped_optimizer.param_groups[0]['params'] == [params[0]]
    assert grouped_optimizer.param_groups[1]['params'] == [params[1]]

    for _ in range(2):
        for param, expected_param, grad, expected_optimizer in zip(
            params,
            expected,
            grads,
            expected_optimizers,
            strict=True,
        ):
            param.grad = grad.clone()
            expected_param.grad = grad.clone()
            expected_optimizer.step()

        grouped_optimizer.step()

    for param, expected_param in zip(params, expected, strict=True):
        assert_close(param, expected_param)


def test_run_optimizer_records_parameter_history():
    params = torch.tensor([1.0, -2.0], requires_grad=True)
    optimizer = dopt.SGD([params], lr=0.1)

    history = dopt.run_optimizer(optimizer, quadratic_loss, steps=2)

    assert history.shape == (3, 2)
    assert_close(history[0], torch.tensor([1.0, -2.0]))
    assert_close(history[1], torch.tensor([0.9, -1.8]))
    assert_close(history[2], torch.tensor([0.81, -1.62]))
    assert_close(params, torch.tensor([0.81, -1.62]))
