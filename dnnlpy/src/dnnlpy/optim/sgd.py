from collections.abc import Iterable
from typing import cast

import torch
import torch.optim as optim
from torch import Tensor

__all__ = ['SGD']


class SGD(optim.Optimizer):
    """Stochastic gradient descent with optional momentum variants."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        """Create a stochastic gradient descent optimizer with optional momentum and
        Nesterov momentum.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Learning rate used to scale each velocity update.
            momentum (float, default: 0.0): Coefficient applied to the previous velocity
                before adding the current gradient.
            weight_decay (float, default: 0.0): Coefficient applied to the parameters
                before adding them to the gradient. Must be non-negative.
            nesterov (bool, default: False): Whether to use Nesterov momentum.
        """
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        """Update velocities from gradients and apply them to parameters."""
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay > 0:
                    grad = grad.add(p, alpha=weight_decay)

                if momentum > 0:
                    state: dict[str, Tensor] = self.state[p]
                    v = state.get('velocity')

                    if v is None:
                        v = grad.clone()
                        self.state[p]['velocity'] = v
                    else:
                        v.mul_(momentum).add_(grad)

                    if nesterov:
                        update = grad.add(v, alpha=momentum)
                    else:
                        update = v
                else:
                    update = grad

                p.sub_(update, alpha=lr)
