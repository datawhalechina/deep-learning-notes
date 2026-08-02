import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

import dnnlpy.nn.utils as dutils


def _parameters() -> list[nn.Parameter]:
    params = [
        nn.Parameter(torch.zeros(2, dtype=torch.float64)),
        nn.Parameter(torch.zeros(3, dtype=torch.float64)),
        nn.Parameter(torch.zeros(1, dtype=torch.float64)),
    ]
    params[0].grad = torch.tensor([3.0, -4.0], dtype=torch.float64)
    params[1].grad = torch.tensor([1.0, -2.0, 6.0], dtype=torch.float64)
    return params


def _clone_parameters(params: list[nn.Parameter]) -> list[nn.Parameter]:
    cloned = [nn.Parameter(param.detach().clone()) for param in params]
    for param, clone in zip(params, cloned, strict=True):
        if param.grad is not None:
            clone.grad = param.grad.clone()
    return cloned


@pytest.mark.parametrize('norm_type', [1.0, 2.0, float('inf')])
@pytest.mark.parametrize('foreach', [None, False, True])
def test_clip_grad_norm_matches_pytorch(norm_type: float, foreach: bool | None):
    actual_params = _parameters()
    expected_params = _clone_parameters(actual_params)

    actual_norm = dutils.clip_grad_norm_(
        (param for param in actual_params),
        max_norm=2.5,
        norm_type=norm_type,
        foreach=foreach,
    )
    expected_norm = torch.nn.utils.clip_grad_norm_(
        expected_params,
        max_norm=2.5,
        norm_type=norm_type,
        foreach=foreach,
    )

    assert_close(actual_norm, expected_norm)
    for actual, expected in zip(actual_params, expected_params, strict=True):
        assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize('foreach', [None, False, True])
def test_clip_grad_value_matches_pytorch(foreach: bool | None):
    actual_params = _parameters()
    expected_params = _clone_parameters(actual_params)

    dutils.clip_grad_value_(actual_params, clip_value=1.25, foreach=foreach)
    nn.utils.clip_grad_value_(expected_params, clip_value=1.25, foreach=foreach)

    for actual, expected in zip(actual_params, expected_params, strict=True):
        assert_close(actual.grad, expected.grad)


def test_clip_grad_norm_errors_on_nonfinite_norm_when_requested():
    param = torch.nn.Parameter(torch.tensor(0.0))
    param.grad = torch.tensor(float('inf'))

    with pytest.raises(RuntimeError, match='non-finite'):
        dutils.clip_grad_norm_(param, 1.0, error_if_nonfinite=True)


def test_empty_generator_returns_zero_norm():
    total_norm = dutils.clip_grad_norm_((param for param in []), 1.0)
    assert_close(total_norm, torch.tensor(0.0))
