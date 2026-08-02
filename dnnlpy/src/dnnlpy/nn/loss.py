from typing import Literal

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

type Reduction = Literal['mean', 'sum', 'none']
type KLReduction = Literal['batchmean', 'mean', 'sum', 'none']

__all__ = [
    'BCELoss',
    'BCEWithLogitsLoss',
    'CrossEntropyLoss',
    'HuberLoss',
    'KLDivLoss',
    'L1Loss',
    'MSELoss',
    'NLLLoss',
    'SmoothL1Loss',
]


class BCELoss(nn.Module):
    """This criterion computes binary cross entropy loss."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Tensor | None = None,
        *,
        fast: bool = False,
    ):
        """Initializes the BCELoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            weight (Tensor | None, default: None): A manual rescaling weight given to
                each element. If provided, it must be broadcastable to the input shape.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.binary_cross_entropy(
                x,
                target,
                weight=self.weight,
                reduction=self.reduction,
            )
        return dF.bce_loss(
            x,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return f'reduction={self.reduction!r}, weight={self.weight is not None}'


class BCEWithLogitsLoss(nn.Module):
    """This criterion computes binary cross entropy loss from logits."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Tensor | None = None,
        pos_weight: Tensor | None = None,
        *,
        fast: bool = False,
    ):
        """Initializes the BCEWithLogitsLoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            weight (Tensor | None, default: None): A manual rescaling weight given to
                each element. If provided, it must be broadcastable to the input shape.
            pos_weight (Tensor | None, default: None): A weight of positive examples to
                be broadcast with the target.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.binary_cross_entropy_with_logits(
                x,
                target,
                weight=self.weight,
                reduction=self.reduction,
                pos_weight=self.pos_weight,
            )
        return dF.bce_with_logits_loss(
            x,
            target,
            weight=self.weight,
            reduction=self.reduction,
            pos_weight=self.pos_weight,
        )

    def extra_repr(self) -> str:
        return (
            f'reduction={self.reduction!r}, '
            f'weight={self.weight is not None}, '
            f'pos_weight={self.pos_weight is not None}'
        )


class NLLLoss(nn.Module):
    """This criterion computes negative log likelihood loss."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Tensor | None = None,
        ignore_index: int = -100,
        *,
        fast: bool = False,
    ):
        """Initializes the NLLLoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            weight (Tensor | None, default: None): A manual rescaling weight given to
                each class. If provided, it must be a 1D tensor with one entry per class.
            ignore_index (int, default: -100): Specifies a target value that is ignored
                and does not contribute to the input gradient.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`:func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.nll_loss(
                x,
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )
        return dF.nll_loss(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        return (
            f'reduction={self.reduction!r}, '
            f'weight={self.weight is not None}, '
            f'ignore_index={self.ignore_index}'
        )


class CrossEntropyLoss(nn.Module):
    """This criterion computes the cross entropy loss between input logits and target."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Tensor | None = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        *,
        fast: bool = False,
    ):
        """Initializes the CrossEntropyLoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            weight (Tensor | None, default: None): A manual rescaling weight given to
                each class. If provided, it must be a 1D tensor with one entry per class.
            ignore_index (int, default: -100): Specifies a target value that is ignored
                and does not contribute to the input gradient.
            label_smoothing (float, default: 0.0): Amount of label smoothing to apply.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.cross_entropy(
                x,
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )
        return dF.cross_entropy_loss(
            x,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )

    def extra_repr(self) -> str:
        return (
            f'reduction={self.reduction!r}, '
            f'weight={self.weight is not None}, '
            f'ignore_index={self.ignore_index}, '
            f'label_smoothing={self.label_smoothing}'
        )


class MSELoss(nn.Module):
    """This criterion computes mean squared error loss."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Tensor | None = None,
        *,
        fast: bool = False,
    ):
        """Initializes the MSELoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            weight (Tensor | None, default: None): A manual rescaling weight given to
                each element. If provided, it must be broadcastable to the input shape.
            fast (bool, default: False): If True and weight is None, uses
                :func:`torch.nn.functional`.
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast and self.weight is None:
            return F.mse_loss(x, target, reduction=self.reduction)
        return dF.mse_loss(
            x,
            target,
            reduction=self.reduction,
            weight=self.weight,
        )

    def extra_repr(self) -> str:
        return f'reduction={self.reduction!r}, weight={self.weight is not None}'


class L1Loss(nn.Module):
    """This criterion computes mean absolute error loss."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        weight: Tensor | None = None,
        *,
        fast: bool = False,
    ):
        """Initializes the L1Loss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            weight (Tensor | None, default: None): A manual rescaling weight given to
                each element. If provided, it must be broadcastable to the input shape.
            fast (bool, default: False): If True and weight is None, uses
                :func:`torch.nn.functional`.
        """
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast and self.weight is None:
            return F.l1_loss(x, target, reduction=self.reduction)
        return dF.l1_loss(
            x,
            target,
            reduction=self.reduction,
            weight=self.weight,
        )

    def extra_repr(self) -> str:
        return f'reduction={self.reduction!r}, weight={self.weight is not None}'


class SmoothL1Loss(nn.Module):
    """This criterion computes smooth L1 loss."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        beta: float = 1.0,
        *,
        fast: bool = False,
    ):
        """Initializes the SmoothL1Loss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            beta (float, default: 1.0): Threshold where the loss changes from L2 to L1.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.beta = beta
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.smooth_l1_loss(
                x,
                target,
                reduction=self.reduction,
                beta=self.beta,
            )
        return dF.smooth_l1_loss(
            x,
            target,
            reduction=self.reduction,
            beta=self.beta,
        )

    def extra_repr(self) -> str:
        return f'reduction={self.reduction!r}, beta={self.beta}'


class HuberLoss(nn.Module):
    """This criterion computes Huber loss."""

    def __init__(
        self,
        reduction: Reduction = 'mean',
        delta: float = 1.0,
        *,
        fast: bool = False,
    ):
        """Initializes the HuberLoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', or 'sum'.
            delta (float, default: 1.0): Threshold where the loss changes from quadratic
                to linear. Must be positive.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.delta = delta
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.huber_loss(
                x,
                target,
                reduction=self.reduction,
                delta=self.delta,
            )
        return dF.huber_loss(
            x,
            target,
            reduction=self.reduction,
            delta=self.delta,
        )

    def extra_repr(self) -> str:
        return f'reduction={self.reduction!r}, delta={self.delta}'


class KLDivLoss(nn.Module):
    """This criterion computes Kullback-Leibler divergence loss."""

    def __init__(
        self,
        reduction: KLReduction = 'mean',
        log_target: bool = False,
        *,
        fast: bool = False,
    ):
        """Initializes the KLDivLoss module.

        Args:
            reduction (str, default: 'mean'): Specifies the reduction to apply to the
                output: 'none', 'mean', 'sum', or 'batchmean'.
            log_target (bool, default: False): If True, target is expected to contain
                log-probabilities.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.reduction = reduction
        self.log_target = log_target
        self.fast = fast

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        if self.fast:
            return F.kl_div(
                x,
                target,
                reduction=self.reduction,
                log_target=self.log_target,
            )
        return dF.kl_div_loss(
            x,
            target,
            reduction=self.reduction,
            log_target=self.log_target,
        )

    def extra_repr(self) -> str:
        return f'reduction={self.reduction!r}, log_target={self.log_target}'
