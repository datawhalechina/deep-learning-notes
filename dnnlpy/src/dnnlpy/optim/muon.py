from collections.abc import Iterable
from typing import cast

import torch
import torch.optim as optim
from torch import Tensor

__all__ = ['Muon']

NS_COEFFS_DEFAULT = (3.4445, -4.7750, 2.0315)


def newton_schulz_5(
    X: Tensor,
    ns_steps: int = 5,
    ns_coefficients: tuple[float, float, float] = NS_COEFFS_DEFAULT,
    eps: float = 1e-7,
) -> Tensor:
    """Approximate the orthogonalized Muon update with Newton-Schulz steps.

    Args:
        X (Tensor): Two-dimensional update matrix to orthogonalize.
        ns_steps (int, default: 5): Number of Newton-Schulz iterations.
        ns_coefficients (tuple[float, float, float], default: (3.4445, -4.7750, 2.0315)):
            Coefficients for the quintic Newton-Schulz iteration.
        eps (float, default: 1e-7): Small value used when normalizing `X`.

    Returns:
        Orthogonalized update matrix with the same shape as `X`.
    """
    if X.ndim != 2:
        raise AssertionError('Muon only supports 2D parameters.')

    a, b, c = ns_coefficients
    X = X / (X.norm() + eps)
    should_transpose = X.size(0) > X.size(1)

    if should_transpose:
        X = X.T

    for _ in range(ns_steps):
        A = torch.mm(X, X.T)
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)

    if should_transpose:
        X = X.T

    return X


class Muon(optim.Optimizer):
    """Muon optimizer for two-dimensional parameters."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = NS_COEFFS_DEFAULT,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        """Create a Muon optimizer.

        Args:
            params (Iterable[Tensor]): Two-dimensional parameters to update.
            lr (float, default: 1e-3): Base learning rate.
            weight_decay (float, default: 0.1): Coefficient applied to the
                parameters before adding them to the gradient.
            momentum (float, default: 0.95): Momentum coefficient applied to
                the update buffer.
            nesterov (bool, default: True): Whether to use Nesterov momentum.
            ns_coefficients (tuple[float, float, float], default: (3.4445,
                -4.7750, 2.0315)): Coefficients for the Newton-Schulz
                orthogonalization iteration.
            ns_steps (int, default: 5): Number of Newton-Schulz iterations.
            eps (float, default: 1e-7): Small value used when normalizing the
                update matrix.
        """
        defaults = {
            'lr': lr,
            'weight_decay': weight_decay,
            'momentum': momentum,
            'nesterov': nesterov,
            'ns_coefficients': ns_coefficients,
            'ns_steps': ns_steps,
            'eps': eps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        """Update parameters using momentum and a Muon orthogonalized step."""
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_coefficients = group['ns_coefficients']
            ns_steps = group['ns_steps']
            eps = group['eps']

            for p in group['params']:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                state: dict[str, Tensor] = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                # Decoupled weight decay: directly shrink parameters.
                grad = p.grad
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                buffer = state['momentum_buffer']
                buffer.mul_(momentum).add_(grad)

                if nesterov:
                    direction = grad + momentum * buffer
                else:
                    direction = buffer

                update = newton_schulz_5(
                    direction,
                    ns_steps=ns_steps,
                    ns_coefficients=ns_coefficients,
                    eps=eps,
                )

                p.sub_(update, alpha=lr)
