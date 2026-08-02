# pyright: reportOptionalMemberAccess=false

import torch
from torch import Tensor

from .activation import log_softmax, softplus

__all__ = [
    'bce_loss',
    'bce_with_logits_loss',
    'cross_entropy_loss',
    'huber_loss',
    'kl_div_loss',
    'l1_loss',
    'mse_loss',
    'nll_loss',
    'smooth_l1_loss',
]


def _validate_inputs(
    x: Tensor,
    target: Tensor,
    reduction: str,
    weight: Tensor | None = None,
    include_batchmean: bool = False,
    check_size: bool = True,
) -> None:
    """Validate the inputs for loss functions."""
    if include_batchmean:
        if reduction not in {'batchmean', 'mean', 'sum', 'none'}:
            raise AssertionError(
                '`reduction` must be `batchmean`, `mean`, `sum`, or `none`.'
            )
    else:
        if reduction not in {'mean', 'sum', 'none'}:
            raise AssertionError('`reduction` must be `mean`, `sum`, or `none`.')
    if check_size and x.size() != target.size():
        raise AssertionError(
            f'Target size ({target.size()}) must be the same as input size ({x.size()}).'
        )
    if weight is not None and weight.size() != x.size():
        raise AssertionError(
            f'Weight size ({weight.size()}) must be the same as input size ({x.size()}).'
        )


def _reduce_loss(
    loss: Tensor,
    reduction: str,
    total_weight: Tensor | None = None,
) -> Tensor:
    """Reduce the loss tensor according to the specified reduction method."""
    if reduction == 'batchmean':
        return loss.sum() / loss.size(0)

    if reduction == 'mean':
        if total_weight is not None:
            return loss.sum() / total_weight
        return loss.mean()

    if reduction == 'sum':
        return loss.sum()

    return loss


def bce_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    weight: Tensor | None = None,
) -> Tensor:
    """Compute binary cross entropy loss.

    Args:
        x (Tensor): Input tensor containing probabilities (values between 0 and 1).
        target (Tensor): Target tensor containing binary labels (0 or 1).
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        weight (Tensor, optional): A manual rescaling weight given to each element. If
            provided, it must be broadcastable to the shape of `x`.

    Returns:
        Tensor: The computed binary cross entropy loss. If reduction is 'none', the shape
            will be the same as the input. If reduction is 'mean' or 'sum', the shape will
            be scalar.

    .. note::
        The logarithmic terms are clamped to a minimum of -100 to prevent log(0) from
        producing negative infinity. This avoids subsequent operations such as 0 * inf,
        which would otherwise result in NaN.
    """
    _validate_inputs(x, target, reduction)

    if torch.any((x < 0) | (x > 1)):
        raise AssertionError('All elements of x should be between 0 and 1.')
    if torch.any((target < 0) | (target > 1)):
        raise AssertionError('All elements of target should be between 0 and 1.')

    loss = -(target * x.log().clamp(min=-100))
    loss = loss - (1 - target) * (1 - x).log().clamp(min=-100)
    if weight is not None:
        loss = loss * weight

    return _reduce_loss(loss, reduction)


def bce_with_logits_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    weight: Tensor | None = None,
    pos_weight: Tensor | None = None,
) -> Tensor:
    """Compute binary cross entropy loss from logits.

    Args:
        x (Tensor): Input tensor containing logits.
        target (Tensor): Target tensor containing binary labels (0 or 1).
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        weight (Tensor, optional): A manual rescaling weight given to each element. If
            provided, it must be broadcastable to the shape of `x`.
        pos_weight (Tensor, optional): a weight of positive examples to be broadcasted
            with target. Must be a tensor with equal size along the class dimension to
            the number of classes. Pay close attention to PyTorch's broadcasting semantics
            in order to achieve the desired operations.

    Returns:
        Tensor: The computed binary cross entropy loss. If reduction is 'none', the shape
            will be the same as the input. If reduction is 'mean' or 'sum', the shape will
            be scalar.
    """
    _validate_inputs(x, target, reduction)

    if pos_weight is None:
        loss = softplus(x) - x * target
    else:
        log_weight = (pos_weight - 1) * target + 1
        loss = (1 - target) * x + log_weight * softplus(-x)

    if weight is not None:
        loss = loss * weight

    return _reduce_loss(loss, reduction)


def _unreduced_nll_loss(
    x: Tensor,
    target: Tensor,
    weight: Tensor | None,
    ignore_index: int,
) -> tuple[Tensor, Tensor, int]:
    """Compute unreduced negative log likelihood loss for log-probabilities."""
    if x.ndim == 1:
        class_count = x.shape[0]
        flat_input = x.unsqueeze(0)
        flat_target = target.reshape(-1)
        output_shape = target.shape
    else:
        class_count = x.shape[1]
        flat_input = x.movedim(1, -1).reshape(-1, class_count)
        flat_target = target.reshape(-1)
        output_shape = target.shape

    valid_target = flat_target != ignore_index
    safe_target = torch.where(valid_target, flat_target, torch.zeros_like(flat_target))
    loss = -flat_input.gather(1, safe_target.unsqueeze(1)).squeeze(1)

    if weight is not None:
        sample_weight = weight[safe_target]
        loss = loss * sample_weight
        total_weight = sample_weight[valid_target].sum()
    else:
        total_weight = valid_target.sum()

    loss = torch.where(valid_target, loss, torch.zeros_like(loss))
    return loss.reshape(output_shape), total_weight, class_count


def nll_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    weight: Tensor | None = None,
    ignore_index: int = -100,
) -> Tensor:
    """Compute negative log likelihood loss for log-probabilities.

    Args:
        x (Tensor): Input tensor containing log-probabilities.
        target (Tensor): Target tensor containing class indices.
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        weight (Tensor, optional): A manual rescaling weight given to each class. If
            provided, it must be a 1D tensor with length equal to the number of classes.
        ignore_index (int, default: -100): Specifies a target value that is ignored and
            does not contribute to the input gradient.

    Returns:
        Tensor: The computed negative log likelihood loss. If reduction is 'none', the shape
            will be the same as the input. If reduction is 'mean' or 'sum', the shape will be
            scalar.
    """
    _validate_inputs(x, target, reduction, check_size=False)

    loss, total_weight, _ = _unreduced_nll_loss(
        x,
        target,
        weight=weight,
        ignore_index=ignore_index,
    )

    return _reduce_loss(loss, reduction, total_weight=total_weight)


def cross_entropy_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    weight: Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute cross entropy loss between logits and class indices.

    Args:
        x (Tensor): Input tensor containing logits.
        target (Tensor): Target tensor containing class indices.
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        weight (Tensor, optional): A manual rescaling weight given to each class. If
            provided, it must be a 1D tensor with length equal to the number of classes.
        ignore_index (int, default: -100): Specifies a target value that is ignored and
            does not contribute to the input gradient.
        label_smoothing (float, default: 0.0): Specifies the amount of label smoothing
            to apply. Must be between 0 and 1.

    Returns:
        Tensor: The computed cross entropy loss. If reduction is 'none', the shape will be
            the same as the input. If reduction is 'mean' or 'sum', the shape will be scalar.
    """
    _validate_inputs(x, target, reduction, check_size=target.is_floating_point())

    if x.ndim == 1:
        log_probs = log_softmax(x, dim=0)
        class_count = x.size(0)
    else:
        log_probs = log_softmax(x, dim=1)
        class_count = x.size(1)

    if target.is_floating_point():
        smoothed_target = target
        if label_smoothing > 0.0:
            smoothed_target = (
                target * (1.0 - label_smoothing) + label_smoothing / class_count
            )

        loss = -smoothed_target * log_probs
        if weight is not None:
            if x.ndim == 1:
                loss = loss * weight
            else:
                weight_shape = [1] * x.ndim
                weight_shape[1] = -1
                loss = loss * weight.reshape(weight_shape)
        loss = loss.sum(dim=0 if x.ndim == 1 else 1)

        return _reduce_loss(loss, reduction)

    if label_smoothing == 0.0:
        return nll_loss(
            log_probs,
            target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    nll, total_weight, _ = _unreduced_nll_loss(
        log_probs,
        target,
        weight,
        ignore_index,
    )

    if x.ndim == 1:
        flat_log_probs = log_probs.unsqueeze(0)
        flat_target = target.reshape(-1)
    else:
        flat_log_probs = log_probs.movedim(1, -1).reshape(-1, class_count)
        flat_target = target.reshape(-1)

    valid_target = flat_target != ignore_index

    if weight is not None:
        smooth_loss = -(flat_log_probs * weight).sum(dim=1)
    else:
        smooth_loss = -flat_log_probs.sum(dim=1)

    smooth_loss = torch.where(valid_target, smooth_loss, torch.zeros_like(smooth_loss))
    smooth_loss = smooth_loss.reshape(target.shape)

    loss = (1.0 - label_smoothing) * nll
    loss = loss + label_smoothing * smooth_loss / class_count

    return _reduce_loss(loss, reduction, total_weight=total_weight)


def mse_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    weight: Tensor | None = None,
) -> Tensor:
    """Compute mean squared error loss.

    Args:
        x (Tensor): Input tensor containing probabilities (values between 0 and 1).
        target (Tensor): Target tensor containing binary labels (0 or 1).
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        weight (Tensor, optional): A manual rescaling weight given to each element. If
            provided, it must be broadcastable to the shape of `x`.

    Returns:
        Tensor: The computed mean squared error loss. If reduction is 'none', the shape
            will be the same as the input. If reduction is 'mean' or 'sum', the shape will
            be scalar.
    """
    _validate_inputs(x, target, reduction)
    loss = (x - target).square()

    if weight is not None:
        loss, weight = torch.broadcast_tensors(loss, weight)
        loss = loss * weight
        return _reduce_loss(loss, reduction, total_weight=weight.sum())

    return _reduce_loss(loss, reduction)


def l1_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    weight: Tensor | None = None,
) -> Tensor:
    """Compute mean absolute error loss.

    Args:
        x (Tensor): Input tensor containing probabilities (values between 0 and 1).
        target (Tensor): Target tensor containing binary labels (0 or 1).
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        weight (Tensor, optional): A manual rescaling weight given to each element. If
            provided, it must be broadcastable to the shape of `x`.

    Returns:
        Tensor: The computed mean absolute error loss. If reduction is 'none', the shape
            will be the same as the input. If reduction is 'mean' or 'sum', the shape will
            be scalar.
    """
    _validate_inputs(x, target, reduction)
    loss = (x - target).abs()

    if weight is not None:
        loss, weight = torch.broadcast_tensors(loss, weight)
        loss = loss * weight
        return _reduce_loss(loss, reduction, total_weight=weight.sum())

    return _reduce_loss(loss, reduction)


def smooth_l1_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    beta: float = 1.0,
) -> Tensor:
    """Compute smooth L1 loss.

    Args:
        x (Tensor): Input tensor.
        target (Tensor): Target tensor.
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        beta (float, default: 1.0): The threshold at which to change between L1 and L2 loss.
            The value must be non-negative. When beta is 0, the loss will be equivalent to L1
            loss. When beta is infinity, the loss will be equivalent to 0.

    Returns:
        Tensor: The computed smooth L1 loss. If reduction is 'none', the shape will be the same
        as the input. If reduction is 'mean' or 'sum', the shape will be scalar.
    """
    if beta < 0:
        raise AssertionError(
            '`F.smooth_l1_loss` does not support negative values for beta.'
        )

    _validate_inputs(x, target, reduction)

    diff = (x - target).abs()
    if beta == 0:
        loss = diff
    else:
        loss = torch.where(diff < beta, 0.5 * diff.square() / beta, diff - 0.5 * beta)

    return _reduce_loss(loss, reduction)


def huber_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    delta: float = 1.0,
) -> Tensor:
    """Compute Huber loss.

    Args:
        x (Tensor): Input tensor.
        target (Tensor): Target tensor.
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
        delta (float, default: 1.0): Threshold where the loss changes from quadratic to
            linear. Must be positive.

    Returns:
        Tensor: The computed Huber loss. If reduction is 'none', the shape will be the same
            as the input. If reduction is 'mean' or 'sum', the shape will be scalar.
    """
    if delta <= 0:
        raise AssertionError(
            '`F.huber_loss` does not support non-positive values for `delta`.'
        )

    _validate_inputs(x, target, reduction)

    diff = (x - target).abs()
    loss = torch.where(
        diff < delta,
        0.5 * diff.square(),
        delta * (diff - 0.5 * delta),
    )

    return _reduce_loss(loss, reduction)


def kl_div_loss(
    x: Tensor,
    target: Tensor,
    reduction: str = 'mean',
    log_target: bool = False,
) -> Tensor:
    """Compute Kullback-Leibler divergence loss.

    Args:
        x (Tensor): x tensor containing log-probabilities.
        target (Tensor): Target tensor containing probabilities or log-probabilities.
        reduction (str, default: 'mean'): Specifies the reduction to apply to the output.
            - 'none': No reduction will be applied.
            - 'mean': The sum of the output will be divided by the number of elements in
                the output.
            - 'sum': The output will be summed.
            - 'batchmean': The sum of the output will be divided by the batch size (the
                first dimension of the x tensor).
        log_target (bool, default: False): If True, the target is expected to contain
            log-probabilities. If False, the target is expected to contain probabilities.

    Returns:
        Tensor: The computed Kullback-Leibler divergence loss. If reduction is 'none', the
            shape will be the same as the input. If reduction is 'mean' or 'sum', the shape
            will be scalar.
    """
    _validate_inputs(x, target, reduction, include_batchmean=True)

    if log_target:
        loss = target.exp() * (target - x)
    else:
        loss = target.xlogy(target) - target * x

    return _reduce_loss(loss, reduction)
