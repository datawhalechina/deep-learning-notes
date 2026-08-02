import math

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

__all__ = [
    'ConstantLR',
    'CosineAnnealingLR',
    'LinearLR',
]


class LinearLR(LRScheduler):
    """Linearly change the learning rate between two multiplicative factors."""

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        """Initialize a linear learning-rate schedule.

        Args:
            optimizer (Optimizer): Optimizer whose learning rate is scheduled.
            start_factor (float, default: 1/3): Initial learning-rate multiplier.
            end_factor (float, default: 1.0): Final learning-rate multiplier.
            total_iters (int, default: 5): Number of iterations used for interpolation.
            last_epoch (int, default: -1): Index of the last completed epoch.
        """
        if not (0 < start_factor <= 1):
            raise AssertionError(
                '`start_factor` must be greater than 0 and less than or equal to 1.'
            )
        if not (0 <= end_factor <= 1):
            raise AssertionError('`end_factor` must be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Compute learning rates from the recursive linear schedule."""
        param_groups = self.optimizer.param_groups

        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in param_groups]

        if self._is_initial or self.last_epoch > self.total_iters:
            return [group['lr'] for group in param_groups]

        step_size = (self.end_factor - self.start_factor) / (
            self.total_iters * self.start_factor
            + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
        )
        return [group['lr'] * (1.0 + step_size) for group in param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        """Compute learning rates directly from the current epoch."""
        factor = self.start_factor + (
            (self.end_factor - self.start_factor)
            * min(self.total_iters, self.last_epoch)
            / self.total_iters
        )
        return [base_lr * factor for base_lr in self.base_lrs]


class ConstantLR(LRScheduler):
    """Multiply the learning rate by a constant factor for a fixed duration."""

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3,
        total_iters: int = 5,
        last_epoch: int = -1,
    ) -> None:
        """Initialize a constant learning-rate schedule.

        Args:
            optimizer (Optimizer): Optimizer whose learning rate is scheduled.
            factor (float, default: 1/3): Learning-rate multiplier.
            total_iters (int, default: 5): Number of iterations using the multiplier.
            last_epoch (int, default: -1): Index of the last completed epoch.
        """
        if not (0 <= factor <= 1):
            raise AssertionError('`factor` must be between 0 and 1.')

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Compute learning rates from the recursive constant schedule."""
        param_groups = self.optimizer.param_groups

        if self.last_epoch == 0:
            return [group['lr'] * self.factor for group in param_groups]

        if self.last_epoch != self.total_iters:
            return [group['lr'] for group in param_groups]

        return [group['lr'] / self.factor for group in param_groups]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        """Compute learning rates directly from the current epoch."""
        factor = self.factor if self.last_epoch < self.total_iters else 1.0
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    """Anneal each parameter group's learning rate with a cosine schedule."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        """Set the learning rate of each parameter group using a cosine annealing schedule.

        Args:
            optimizer (Optimizer): Optimizer whose learning rate is scheduled.
            T_max (int): Maximum number of iterations in the cosine cycle.
            eta_min (float, default: 0.0): Minimum learning rate.
            last_epoch (int, default: -1): Index of the last completed epoch.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Tensor]:
        """Compute learning rates from the recursive cosine schedule."""
        param_groups = self.optimizer.param_groups

        if getattr(self, '_is_initial', self.last_epoch == 0):
            return [group['lr'] for group in param_groups]

        if self._step_count == 1 and self.last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
                / 2
                for base_lr in self.base_lrs
            ]

        # The usual recurrence divides by zero at each cosine minimum.
        if (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr']
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, param_groups, strict=True)
            ]

        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group['lr'] - self.eta_min)
            + self.eta_min
            for group in param_groups
        ]

    def _get_closed_form_lr(self) -> list[float | Tensor]:
        """Compute learning rates directly from the current epoch."""
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
