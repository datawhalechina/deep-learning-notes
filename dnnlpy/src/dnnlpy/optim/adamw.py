from collections.abc import Iterable
from typing import cast

import torch
import torch.optim as optim
from torch import Tensor

__all__ = ['AdamW']


class AdamW(optim.Optimizer):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        """Create an AdamW optimizer.

        Args:
            params (Iterable[Tensor]): Parameters to update.
            lr (float, default: 1e-3): Base learning rate.
            betas (tuple[float, float], default: (0.9, 0.999)): Coefficients
                used to compute running averages of the gradient and its square.
            eps (float, default: 1e-8): Small value added to the denominator
                for numerical stability.
            weight_decay (float, default: 1e-2): Decoupled weight decay
                coefficient applied directly to parameters before the Adam
                update.
        """
        defaults = {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        """Update parameters using Adam moments and decoupled weight decay."""
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                state: dict[str, Tensor] = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # Decoupled weight decay: directly shrink parameters.
                grad = p.grad
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                m = state['exp_avg']
                v = state['exp_avg_sq']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - torch.pow(beta1, step)
                bias_correction2 = 1 - torch.pow(beta2, step)

                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Adam update: use the original gradient statistics.
                p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
