import inspect

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr
from torch.testing import assert_close

import dnnlpy.optim as dopt
from dnnlpy.optim.lr_schedule import ConstantLR, CosineAnnealingLR, LinearLR


def _make_optimizer() -> optim.SGD:
    parameters = [nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    return optim.SGD(
        [
            {'params': [parameters[0]], 'lr': 0.5},
            {'params': [parameters[1]], 'lr': 0.1},
        ]
    )


@pytest.mark.parametrize(
    ('actual_class', 'expected_class'),
    [
        (LinearLR, lr.LinearLR),
        (ConstantLR, lr.ConstantLR),
        (CosineAnnealingLR, lr.CosineAnnealingLR),
    ],
)
def test_lr_scheduler_api_and_exports(actual_class, expected_class):
    assert issubclass(actual_class, lr.LRScheduler)
    assert getattr(dopt, actual_class.__name__) is actual_class

    actual = inspect.signature(actual_class)
    expected = inspect.signature(expected_class)
    assert list(actual.parameters) == list(expected.parameters)

    for name in actual.parameters:
        assert actual.parameters[name].default == expected.parameters[name].default


@pytest.mark.parametrize(
    ('actual_class', 'expected_class', 'options', 'num_steps'),
    [
        (
            LinearLR,
            lr.LinearLR,
            {'start_factor': 0.2, 'end_factor': 0.8, 'total_iters': 4},
            8,
        ),
        (ConstantLR, lr.ConstantLR, {'factor': 0.25, 'total_iters': 4}, 8),
    ],
)
def test_lr_scheduler_matches_pytorch(
    actual_class,
    expected_class,
    options,
    num_steps,
):
    actual_optimizer = _make_optimizer()
    expected_optimizer = _make_optimizer()
    actual_scheduler = actual_class(actual_optimizer, **options)
    expected_scheduler = expected_class(expected_optimizer, **options)

    assert_close(
        torch.tensor(actual_scheduler.get_last_lr()),
        torch.tensor(expected_scheduler.get_last_lr()),
    )

    for _ in range(num_steps):
        actual_optimizer.step()
        expected_optimizer.step()
        actual_scheduler.step()
        expected_scheduler.step()

        assert_close(
            torch.tensor(actual_scheduler.get_last_lr()),
            torch.tensor(expected_scheduler.get_last_lr()),
        )
        assert actual_scheduler.state_dict() == expected_scheduler.state_dict()


@pytest.mark.parametrize(
    ('actual_class', 'expected_class', 'options', 'epochs'),
    [
        (
            LinearLR,
            lr.LinearLR,
            {'start_factor': 0.2, 'end_factor': 0.8, 'total_iters': 4},
            [0, 1, 4, 6],
        ),
        (
            ConstantLR,
            lr.ConstantLR,
            {'factor': 0.25, 'total_iters': 4},
            [0, 1, 4, 6],
        ),
        (
            CosineAnnealingLR,
            lr.CosineAnnealingLR,
            {'T_max': 4, 'eta_min': 0.01},
            [0, 1, 4, 6],
        ),
    ],
)
def test_closed_form_lr_matches_pytorch(
    actual_class,
    expected_class,
    options,
    epochs,
):
    actual_scheduler = actual_class(_make_optimizer(), **options)
    expected_scheduler = expected_class(_make_optimizer(), **options)

    for epoch in epochs:
        actual_scheduler.last_epoch = epoch
        expected_scheduler.last_epoch = epoch
        assert_close(
            torch.tensor(actual_scheduler._get_closed_form_lr()),
            torch.tensor(expected_scheduler._get_closed_form_lr()),
        )


def test_cosine_annealing_lr_matches_pytorch_across_cycles():
    actual_optimizer = _make_optimizer()
    expected_optimizer = _make_optimizer()
    actual_scheduler = CosineAnnealingLR(actual_optimizer, T_max=4, eta_min=0.01)
    expected_scheduler = lr.CosineAnnealingLR(expected_optimizer, T_max=4, eta_min=0.01)

    actual = torch.tensor(actual_scheduler.get_last_lr())
    expected = torch.tensor(expected_scheduler.get_last_lr())

    assert_close(actual, expected)

    for _ in range(12):
        actual_optimizer.step()
        expected_optimizer.step()
        actual_scheduler.step()
        expected_scheduler.step()

        actual = torch.tensor(actual_scheduler.get_last_lr())
        expected = torch.tensor(expected_scheduler.get_last_lr())
        assert_close(actual, expected)

        assert actual_scheduler.state_dict() == expected_scheduler.state_dict()


def test_cosine_annealing_lr_matches_pytorch_when_resuming():
    actual_optimizer = _make_optimizer()
    expected_optimizer = _make_optimizer()

    for actual_group, expected_group in zip(
        actual_optimizer.param_groups,
        expected_optimizer.param_groups,
        strict=True,
    ):
        actual_group['initial_lr'] = actual_group['lr']
        expected_group['initial_lr'] = expected_group['lr']

    options = {'T_max': 8, 'eta_min': 0.02, 'last_epoch': 3}
    actual_scheduler = CosineAnnealingLR(actual_optimizer, **options)
    expected_scheduler = lr.CosineAnnealingLR(expected_optimizer, **options)

    actual = torch.tensor(actual_scheduler.get_last_lr())
    expected = torch.tensor(expected_scheduler.get_last_lr())
    assert_close(actual, expected)
