from collections.abc import Iterable
from typing import cast

import torch
import torch.optim as optim
from torch import Tensor

__all__ = ['RMSprop']


class RMSprop(optim.Optimizer):
    """RMSprop optimizer with an exponential average of squared gradients."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        rho: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Create an RMSprop optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-2): Base learning rate.
            rho (float, default: 0.99): Decay factor for the squared-gradient
                moving average.
            eps (float, default: 1e-8): Small value added to the denominator
                for numerical stability.
            weight_decay (float, default: 0.0): Coefficient applied to the
                parameters before adding them to the gradient.
            momentum (float, default: 0.0): Momentum coefficient.
        """
        defaults = {
            'lr': lr,
            'rho': rho,
            'eps': eps,
            'weight_decay': weight_decay,
            'momentum': momentum,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        """Update parameters using the current RMSprop state."""
        for group in self.param_groups:
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                state: dict[str, Tensor] = self.state[p]
                if len(state) == 0:
                    state['square_avg'] = torch.zeros_like(p)
                    state['momentum_buffer'] = torch.zeros_like(p)

                grad = p.grad
                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                v = state['square_avg']
                buffer = state['momentum_buffer']
                v.mul_(rho).add_(grad.square(), alpha=1 - rho)

                if momentum > 0:
                    buffer.mul_(momentum).addcdiv_(grad, v.sqrt() + eps)
                    p.sub_(buffer, alpha=lr)
                else:
                    p.addcdiv_(grad, v.sqrt() + eps, value=-lr)
