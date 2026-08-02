from collections.abc import Iterable
from typing import cast

import torch
import torch.optim as optim
from torch import Tensor

__all__ = ['Adagrad']


class Adagrad(optim.Optimizer):
    """Adaptive gradient optimizer with per-parameter learning rates."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
    ):
        """Create an Adagrad optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-2): Base learning rate.
            lr_decay (float, default: 0.0): Learning-rate decay multiplier.
            weight_decay (float, default: 0.0): Coefficient applied to the
                parameters before adding them to the gradient.
            initial_accumulator_value (float, default: 0.0): Initial value for
                each squared-gradient accumulator.
            eps (float, default: 1e-10): Small value added to the denominator
                for numerical stability.
        """
        defaults = {
            'lr': lr,
            'lr_decay': lr_decay,
            'weight_decay': weight_decay,
            'initial_accumulator_value': initial_accumulator_value,
            'eps': eps,
        }
        super().__init__(params, defaults=defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        """Accumulate squared gradients and apply an Adagrad update."""
        for group in self.param_groups:
            lr = group['lr']
            lr_decay = group['lr_decay']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                state: dict[str, Tensor] = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['sum_of_sq_grads'] = torch.full_like(
                        p, group['initial_accumulator_value']
                    )

                state['step'] += 1
                step = state['step']

                grad = p.grad
                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                s = state['sum_of_sq_grads']
                s.add_(grad.square())
                clr = lr / (1 + (step - 1) * lr_decay)
                p.sub_(clr / (s.sqrt() + eps) * grad)
