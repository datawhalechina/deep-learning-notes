from collections.abc import Iterable
from typing import cast

import torch
import torch.optim as optim
from torch import Tensor

__all__ = ['Adadelta']


class Adadelta(optim.Optimizer):
    """Adadelta optimizer with running averages of gradients and updates."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        """Create an Adadelta optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1.0): Learning rate used to scale each update.
            rho (float, default: 0.9): Decay factor for the moving averages.
            eps (float, default: 1e-6): Small value added to root-mean-square
                terms for numerical stability.
            weight_decay (float, default: 0.0): Coefficient applied to the
                parameters before adding them to the gradient.
        """
        defaults = {
            'lr': lr,
            'rho': rho,
            'eps': eps,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults=defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        """Update parameters using the current Adadelta state."""
        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                state: dict[str, Tensor] = self.state[p]
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    state['acc_delta'] = torch.zeros_like(p)

                grad = p.grad
                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                v = state['square_avg']
                u = state['acc_delta']

                v.mul_(rho).add_(grad.square(), alpha=1 - rho)
                delta_x = (u + eps).sqrt() / (v + eps).sqrt() * grad

                u.mul_(rho).add_(delta_x.square(), alpha=1 - rho)
                p.sub_(delta_x, alpha=lr)
