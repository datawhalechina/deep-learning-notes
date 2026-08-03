import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.testing import assert_close

import dnnlpy.nn as dnn
import dnnlpy.nn.functional as dF
import dnnlpy.nn.functional.loss as dF_loss


def _copy(x: Tensor, mode: bool = True) -> Tensor:
    """Returns a copy of the input tensor with `requires_grad` set to True."""
    return x.detach().clone().requires_grad_(mode)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_bce_loss_function_matches_torch(reduction: str):
    x = torch.rand(2, 3)
    target = torch.rand(2, 3)

    actual = dF.bce_loss(x, target, reduction=reduction)
    expected = F.binary_cross_entropy(x, target, reduction=reduction)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_bce_loss_function_matches_torch_with_weight(reduction: str):
    x = torch.rand(2, 3)
    target = torch.rand(2, 3)
    weight = torch.rand(2, 3)

    actual = dF.bce_loss(x, target, weight=weight, reduction=reduction)
    expected = F.binary_cross_entropy(
        x,
        target,
        weight=weight,
        reduction=reduction,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_bce_with_logits_loss_function_matches_torch(reduction: str):
    x = torch.randn(2, 3)
    target = torch.rand(2, 3)

    actual = dF.bce_with_logits_loss(x, target, reduction=reduction)
    expected = F.binary_cross_entropy_with_logits(
        x,
        target,
        reduction=reduction,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_bce_with_logits_loss_function_matches_torch_with_weights(
    reduction: str,
):
    x = torch.randn(2, 3)
    target = torch.rand(2, 3)
    weight = torch.rand(2, 3)
    pos_weight = torch.rand(3) + 0.5

    actual = dF.bce_with_logits_loss(
        x,
        target,
        weight=weight,
        reduction=reduction,
        pos_weight=pos_weight,
    )
    expected = F.binary_cross_entropy_with_logits(
        x,
        target,
        weight=weight,
        reduction=reduction,
        pos_weight=pos_weight,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_bce_with_logits_loss_function_is_stable_for_extreme_logits(
    reduction: str,
):
    x = torch.tensor([[-1000.0, -100.0, 0.0, 100.0, 1000.0]])
    target = torch.tensor([[0.0, 1.0, 0.25, 0.75, 1.0]])
    pos_weight = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])

    actual = dF.bce_with_logits_loss(
        x,
        target,
        reduction=reduction,
        pos_weight=pos_weight,
    )
    expected = F.binary_cross_entropy_with_logits(
        x,
        target,
        reduction=reduction,
        pos_weight=pos_weight,
    )

    assert torch.isfinite(actual).all()
    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['batchmean', 'mean', 'sum', 'none'])
def test_kl_div_loss_function_matches_torch(reduction: str):
    x = torch.randn(2, 3).log_softmax(dim=1)
    target = torch.rand(2, 3).softmax(dim=1)

    actual = dF.kl_div_loss(x, target, reduction=reduction)
    if reduction == 'mean':
        with pytest.warns(UserWarning, match='batchmean'):
            expected = F.kl_div(x, target, reduction=reduction)
    else:
        expected = F.kl_div(x, target, reduction=reduction)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['batchmean', 'mean', 'sum', 'none'])
def test_kl_div_loss_function_matches_torch_with_log_target(reduction: str):
    x = torch.randn(2, 3).log_softmax(dim=1)
    target = torch.randn(2, 3).log_softmax(dim=1)

    actual = dF.kl_div_loss(x, target, reduction=reduction, log_target=True)
    if reduction == 'mean':
        with pytest.warns(UserWarning, match='batchmean'):
            expected = F.kl_div(x, target, reduction=reduction, log_target=True)
    else:
        expected = F.kl_div(x, target, reduction=reduction, log_target=True)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_mse_loss_function_matches_torch(reduction: str):
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    actual = dF.mse_loss(x, target, reduction=reduction)
    expected = F.mse_loss(x, target, reduction=reduction)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_mse_loss_function_matches_torch_with_weight(reduction: str):
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)
    weight = torch.rand(2, 3)

    actual = dF.mse_loss(x, target, reduction=reduction, weight=weight)
    expected = F.mse_loss(x, target, reduction=reduction, weight=weight)  # type: ignore[call-arg]

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_l1_loss_function_matches_torch(reduction: str):
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    actual = dF.l1_loss(x, target, reduction=reduction)
    expected = F.l1_loss(x, target, reduction=reduction)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_l1_loss_function_matches_torch_with_weight(reduction: str):
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)
    weight = torch.rand(2, 3)

    actual = dF.l1_loss(x, target, reduction=reduction, weight=weight)
    expected = F.l1_loss(x, target, reduction=reduction, weight=weight)  # type: ignore[call-arg]

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_smooth_l1_loss_function_matches_torch(reduction: str):
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    actual = dF.smooth_l1_loss(x, target, reduction=reduction, beta=0.5)
    expected = F.smooth_l1_loss(x, target, reduction=reduction, beta=0.5)

    assert_close(actual, expected)


def test_smooth_l1_loss_function_matches_torch_with_zero_beta():
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    actual = dF.smooth_l1_loss(x, target, beta=0.0)
    expected = F.smooth_l1_loss(x, target, beta=0.0)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_huber_loss_function_matches_torch(reduction: str):
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    actual = dF.huber_loss(x, target, reduction=reduction, delta=0.5)
    expected = F.huber_loss(x, target, reduction=reduction, delta=0.5)

    assert_close(actual, expected)


def test_huber_loss_function_backward_matches_torch():
    x = torch.randn(2, 3, requires_grad=True)
    target = torch.randn(2, 3)
    expected_x = x.detach().clone().requires_grad_()

    actual = dF.huber_loss(x, target, delta=0.5)
    expected = F.huber_loss(expected_x, target, delta=0.5)

    actual.backward()
    expected.backward()

    assert_close(actual, expected)
    assert_close(x.grad, expected_x.grad)


def test_loss_function_gradients_match_torch():
    probability = torch.rand(2, 3, dtype=torch.float64).clamp(0.1, 0.9)
    logits = torch.randn(2, 3, dtype=torch.float64)
    regression_target = torch.randn(2, 3, dtype=torch.float64)
    probability_target = torch.rand(2, 3, dtype=torch.float64)
    class_target = torch.tensor([0, 2])
    distribution_target = torch.rand(2, 3, dtype=torch.float64).softmax(dim=1)

    cases = [
        (dF.bce_loss, F.binary_cross_entropy, probability, probability_target, {}),
        (
            dF.bce_with_logits_loss,
            F.binary_cross_entropy_with_logits,
            logits,
            probability_target,
            {},
        ),
        (dF.kl_div_loss, F.kl_div, logits.log_softmax(dim=1), distribution_target, {}),
        (dF.mse_loss, F.mse_loss, logits, regression_target, {}),
        (dF.l1_loss, F.l1_loss, logits, regression_target, {}),
        (dF.smooth_l1_loss, F.smooth_l1_loss, logits, regression_target, {'beta': 0.5}),
        (dF.huber_loss, F.huber_loss, logits, regression_target, {'delta': 0.5}),
        (dF.nll_loss, F.nll_loss, logits.log_softmax(dim=1), class_target, {}),
        (dF.cross_entropy_loss, F.cross_entropy, logits, class_target, {}),
    ]

    for custom_fn, reference_fn, base, target, kwargs in cases:
        x1 = _copy(base)
        x2 = _copy(base)

        actual = custom_fn(x1, target, reduction='sum', **kwargs)
        expected = reference_fn(x2, target, reduction='sum', **kwargs)

        assert_close(actual, expected)

        actual.backward()
        expected.backward()

        assert_close(x1.grad, x2.grad)


@pytest.mark.parametrize(
    'loss_fn,args',
    [
        (dF.mse_loss, (torch.randn(2, 3), torch.randn(2, 3))),
        (dF.l1_loss, (torch.randn(2, 3), torch.randn(2, 3))),
        (dF.smooth_l1_loss, (torch.randn(2, 3), torch.randn(2, 3))),
        (dF.huber_loss, (torch.randn(2, 3), torch.randn(2, 3))),
        (dF.bce_loss, (torch.rand(2, 3), torch.rand(2, 3))),
        (dF.bce_with_logits_loss, (torch.randn(2, 3), torch.rand(2, 3))),
    ],
)
def test_pointwise_loss_functions_reject_invalid_reduction(loss_fn, args):
    with pytest.raises(AssertionError, match='reduction'):
        loss_fn(*args, reduction='batchmean')


def test_kl_div_loss_rejects_invalid_reduction():
    x = torch.randn(2, 3)
    target = torch.rand(2, 3)

    with pytest.raises(AssertionError, match='reduction'):
        dF.kl_div_loss(x, target, reduction='invalid')


def test_smooth_l1_loss_rejects_negative_beta():
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    with pytest.raises(AssertionError, match='beta'):
        dF.smooth_l1_loss(x, target, beta=-0.5)


def test_huber_loss_rejects_non_positive_delta():
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)

    with pytest.raises(AssertionError, match='delta'):
        dF.huber_loss(x, target, delta=0.0)


@pytest.mark.parametrize(
    ('custom', 'reference', 'x', 'target'),
    [
        (
            dnn.MSELoss(reduction='sum'),
            nn.MSELoss(reduction='sum'),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.L1Loss(reduction='sum'),
            nn.L1Loss(reduction='sum'),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.SmoothL1Loss(reduction='sum', beta=0.5),
            nn.SmoothL1Loss(reduction='sum', beta=0.5),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.HuberLoss(reduction='sum', delta=0.5),
            nn.HuberLoss(reduction='sum', delta=0.5),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.BCELoss(weight=torch.rand(2, 3), reduction='sum'),
            nn.BCELoss(weight=torch.rand(2, 3), reduction='sum'),
            torch.rand(2, 3),
            torch.rand(2, 3),
        ),
        (
            dnn.BCEWithLogitsLoss(
                weight=torch.rand(2, 3),
                pos_weight=torch.rand(3) + 0.5,
                reduction='sum',
            ),
            nn.BCEWithLogitsLoss(
                weight=torch.rand(2, 3),
                pos_weight=torch.rand(3) + 0.5,
                reduction='sum',
            ),
            torch.randn(2, 3),
            torch.rand(2, 3),
        ),
        (
            dnn.KLDivLoss(reduction='batchmean', log_target=True),
            nn.KLDivLoss(reduction='batchmean', log_target=True),
            torch.randn(2, 3).log_softmax(dim=1),
            torch.randn(2, 3).log_softmax(dim=1),
        ),
    ],
)
def test_loss_modules_match_reference_clss(
    custom: nn.Module,
    reference: nn.Module,
    x: Tensor,
    target: Tensor,
):
    if isinstance(custom, (dnn.BCELoss, dnn.BCEWithLogitsLoss)):
        assert custom.weight is not None
        reference.weight = custom.weight
    if isinstance(custom, dnn.BCEWithLogitsLoss):
        assert custom.pos_weight is not None
        reference.pos_weight = custom.pos_weight

    assert_close(custom(x, target), reference(x, target))


@pytest.mark.parametrize(
    ('custom', 'reference', 'x', 'target'),
    [
        (
            dnn.MSELoss(reduction='sum', fast=True),
            nn.MSELoss(reduction='sum'),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.L1Loss(reduction='sum', fast=True),
            nn.L1Loss(reduction='sum'),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.SmoothL1Loss(reduction='sum', beta=0.5, fast=True),
            nn.SmoothL1Loss(reduction='sum', beta=0.5),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.HuberLoss(reduction='sum', delta=0.5, fast=True),
            nn.HuberLoss(reduction='sum', delta=0.5),
            torch.randn(2, 3),
            torch.randn(2, 3),
        ),
        (
            dnn.BCELoss(weight=torch.rand(2, 3), reduction='sum', fast=True),
            nn.BCELoss(weight=torch.rand(2, 3), reduction='sum'),
            torch.rand(2, 3),
            torch.rand(2, 3),
        ),
        (
            dnn.BCEWithLogitsLoss(
                weight=torch.rand(2, 3),
                pos_weight=torch.rand(3) + 0.5,
                reduction='sum',
                fast=True,
            ),
            nn.BCEWithLogitsLoss(
                weight=torch.rand(2, 3),
                pos_weight=torch.rand(3) + 0.5,
                reduction='sum',
            ),
            torch.randn(2, 3),
            torch.rand(2, 3),
        ),
        (
            dnn.KLDivLoss(reduction='batchmean', log_target=True, fast=True),
            nn.KLDivLoss(reduction='batchmean', log_target=True),
            torch.randn(2, 3).log_softmax(dim=1),
            torch.randn(2, 3).log_softmax(dim=1),
        ),
    ],
)
def test_fast_loss_modules_match_torch(
    custom: nn.Module,
    reference: nn.Module,
    x: Tensor,
    target: Tensor,
):
    if isinstance(custom, (dnn.BCELoss, dnn.BCEWithLogitsLoss)):
        assert custom.weight is not None
        reference.weight = custom.weight
    if isinstance(custom, dnn.BCEWithLogitsLoss):
        assert custom.pos_weight is not None
        reference.pos_weight = custom.pos_weight

    assert custom.fast is True
    assert_close(custom(x, target), reference(x, target))


def test_weighted_mse_loss_module_matches_functional():
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)
    weight = torch.rand(2, 3)

    custom = dnn.MSELoss(weight=weight, reduction='sum')
    actual = dF.mse_loss(x, target, weight=weight, reduction='sum')

    assert_close(custom(x, target), actual)


def test_weighted_l1_loss_module_matches_functional():
    x = torch.randn(2, 3)
    target = torch.randn(2, 3)
    weight = torch.rand(2, 3)

    custom = dnn.L1Loss(weight=weight, reduction='sum')
    actual = dF.l1_loss(x, target, weight=weight, reduction='sum')

    assert_close(custom(x, target), actual)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_nll_loss_function_matches_torch(reduction: str):
    x = torch.randn(5, 4).log_softmax(dim=1)
    target = torch.tensor([0, -1, 1, 2, -1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.nll_loss(
        x,
        target,
        weight=weight,
        ignore_index=-1,
        reduction=reduction,
    )
    expected = F.nll_loss(
        x,
        target,
        weight=weight,
        ignore_index=-1,
        reduction=reduction,
    )

    assert_close(actual, expected)


def test_nll_loss_function_matches_torch_for_spatial_targets():
    x = torch.randn(2, 4, 3, 2).log_softmax(dim=1)
    target = torch.tensor([[[0, 1], [2, -1], [3, 0]], [[1, 2], [-1, 3], [0, 1]]])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.nll_loss(
        x,
        target,
        weight=weight,
        ignore_index=-1,
        reduction='mean',
    )
    expected = F.nll_loss(
        x,
        target,
        weight=weight,
        ignore_index=-1,
        reduction='mean',
    )

    assert_close(actual, expected)


def test_nll_loss_module_matches_torch():
    x = torch.randn(5, 4).log_softmax(dim=1)
    target = torch.tensor([0, -1, 1, 2, -1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    custom = dnn.NLLLoss(weight=weight, ignore_index=-1, reduction='sum')
    reference = nn.NLLLoss(weight=weight, ignore_index=-1, reduction='sum')

    assert_close(custom(x, target), reference(x, target))


def test_fast_nll_loss_module_matches_torch():
    x = torch.randn(5, 4).log_softmax(dim=1)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    custom = dnn.NLLLoss(weight=weight, reduction='sum', fast=True)
    reference = nn.NLLLoss(weight=weight, reduction='sum')

    assert custom.fast is True
    assert_close(custom(x, target), reference(x, target))


def test_nll_loss_rejects_invalid_reduction():
    x = torch.randn(5, 4).log_softmax(dim=1)
    target = torch.tensor([0, 3, 1, 2, 1])

    with pytest.raises(AssertionError, match='reduction'):
        dF.nll_loss(x, target, reduction='batchmean')


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_cross_entropy_loss_function_matches_torch(reduction: str):
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.cross_entropy_loss(x, target, weight=weight, reduction=reduction)
    expected = F.cross_entropy(x, target, weight=weight, reduction=reduction)

    assert_close(actual, expected)


def test_cross_entropy_loss_function_matches_torch_without_weight():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])

    actual = dF.cross_entropy_loss(x, target)
    expected = F.cross_entropy(x, target)

    assert_close(actual, expected)


def test_cross_entropy_loss_uses_nll_loss_for_index_targets(monkeypatch):
    called = False

    def fake_nll_loss(input, target, weight=None, ignore_index=-100, reduction='mean'):
        nonlocal called
        called = True
        return torch.tensor(3.0)

    monkeypatch.setattr(dF_loss, 'nll_loss', fake_nll_loss)

    actual = dF_loss.cross_entropy_loss(
        torch.randn(5, 4),
        torch.tensor([0, 3, 1, 2, 1]),
    )

    assert called is True
    assert_close(actual, torch.tensor(3.0))


def test_cross_entropy_loss_does_not_use_nll_loss_for_probability_targets(monkeypatch):
    def fail_nll_loss(*args, **kwargs):
        raise AssertionError('nll_loss should not be used for probability targets')

    monkeypatch.setattr(dF_loss, 'nll_loss', fail_nll_loss)

    x = torch.randn(3, 4)
    target = torch.rand(3, 4)
    target = target / target.sum(dim=1, keepdim=True)

    actual = dF_loss.cross_entropy_loss(x, target)
    expected = F.cross_entropy(x, target)

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_cross_entropy_loss_function_matches_torch_with_ignore_index(reduction: str):
    x = torch.randn(5, 4)
    target = torch.tensor([0, -1, 1, 2, -1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.cross_entropy_loss(
        x,
        target,
        weight=weight,
        ignore_index=-1,
        reduction=reduction,
        label_smoothing=0.2,
    )
    expected = F.cross_entropy(
        x,
        target,
        weight=weight,
        ignore_index=-1,
        reduction=reduction,
        label_smoothing=0.2,
    )

    assert_close(actual, expected)


@pytest.mark.parametrize('reduction', ['mean', 'sum', 'none'])
def test_cross_entropy_loss_function_matches_torch_with_label_smoothing(
    reduction: str,
):
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.cross_entropy_loss(
        x,
        target,
        weight=weight,
        reduction=reduction,
        label_smoothing=0.2,
    )
    expected = F.cross_entropy(
        x,
        target,
        weight=weight,
        reduction=reduction,
        label_smoothing=0.2,
    )

    assert_close(actual, expected)


def test_cross_entropy_loss_function_matches_torch_for_spatial_targets():
    x = torch.randn(2, 4, 3, 2)
    target = torch.tensor([[[0, 1], [2, -1], [3, 0]], [[1, 2], [-1, 3], [0, 1]]])

    actual = dF.cross_entropy_loss(
        x,
        target,
        ignore_index=-1,
        reduction='none',
        label_smoothing=0.1,
    )
    expected = F.cross_entropy(
        x,
        target,
        ignore_index=-1,
        reduction='none',
        label_smoothing=0.1,
    )

    assert_close(actual, expected)


def test_cross_entropy_loss_function_matches_torch_for_probability_targets():
    x = torch.randn(3, 4)
    target = torch.rand(3, 4)
    target = target / target.sum(dim=1, keepdim=True)
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    actual = dF.cross_entropy_loss(
        x,
        target,
        weight=weight,
        reduction='mean',
        label_smoothing=0.1,
    )
    expected = F.cross_entropy(
        x,
        target,
        weight=weight,
        reduction='mean',
        label_smoothing=0.1,
    )

    assert_close(actual, expected)


def test_cross_entropy_loss_module_matches_torch():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    custom = dnn.CrossEntropyLoss(
        weight=weight,
        ignore_index=-1,
        reduction='sum',
        label_smoothing=0.2,
    )
    reference = nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=-1,
        reduction='sum',
        label_smoothing=0.2,
    )

    assert_close(custom(x, target), reference(x, target))


def test_fast_cross_entropy_loss_module_matches_torch():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])

    custom = dnn.CrossEntropyLoss(weight=weight, reduction='sum', fast=True)
    reference = nn.CrossEntropyLoss(weight=weight, reduction='sum')

    assert custom.fast is True
    assert_close(custom(x, target), reference(x, target))


def test_cross_entropy_loss_rejects_invalid_reduction():
    x = torch.randn(5, 4)
    target = torch.tensor([0, 3, 1, 2, 1])

    with pytest.raises(AssertionError, match='reduction'):
        dF.cross_entropy_loss(x, target, reduction='batchmean')
