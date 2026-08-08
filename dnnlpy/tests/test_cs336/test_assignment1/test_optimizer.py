import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.optim as dopt

from .adapters import get_adamw_cls, run_get_lr_cosine_schedule
from .conftest import SnapShot

__all__ = [
    'test_adamw',
    'test_get_lr_cosine_schedule',
]


def optimize(opt_class: type[optim.Optimizer]) -> Tensor:
    """Run a simple optimization problem using the provided optimizer class."""
    torch.manual_seed(42)

    model = nn.Linear(3, 2, bias=False)
    x = torch.rand(model.in_features)
    y = torch.tensor([x[0] + x[1], -x[2]])

    assert opt_class is dopt.AdamW or opt_class is optim.AdamW

    optimizer = opt_class(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Use 1000 optimization steps for testing
    for _ in range(1000):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')
        loss.backward()
        optimizer.step()

    return model.weight.detach()


def test_adamw(snapshot: SnapShot):
    """Our reference implementation yields slightly different results than the
    PyTorch AdamW, since there are a couple different ways that you can apply
    weight decay that are equivalent in principle, but differ in practice due to
    floating point behavior. So, we test that the provided implementation matches
    _either_ our reference implementation's expected results or those from the
    PyTorch AdamW.
    """
    actual = optimize(get_adamw_cls())
    expected = optimize(optim.AdamW)

    try:
        assert_close(actual, expected)
    except AssertionError:
        try:
            snapshot.assert_close(actual, atol=1e-4)
        except AssertionError:
            raise AssertionError(
                'The provided implementation of AdamW does not match either '
                "the reference implementation or PyTorch's AdamW."
            )


def test_get_lr_cosine_schedule():
    """Test the learning rate schedule for the cosine annealing with warmup."""
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]

    actual_lrs = [
        run_get_lr_cosine_schedule(
            iteration=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]

    actual_lrs = torch.tensor(actual_lrs)
    expected_lrs = torch.tensor(expected_lrs)

    assert_close(actual_lrs, expected_lrs)
