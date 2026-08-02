import torch
from torch import Tensor

__all__ = [
    'add_noise',
    'denoise',
]


def add_noise(x0: Tensor, betas: Tensor, timestep: int) -> Tensor:
    """Sample `x_t` directly from `x_0` at a given diffusion timestep.

    Args:
        x0 (Tensor): The original data tensor.
        betas (Tensor): The beta schedule tensor.
        timestep (int): The diffusion timestep.

    Returns:
        Tensor: The noisy data tensor at the given timestep.
    """
    t = timestep
    noise = torch.randn_like(x0)
    alphas = 1.0 - betas
    alpha_bars = alphas.cumprod(dim=0)
    noisy = alpha_bars[t].sqrt() * x0 + (1 - alpha_bars[t]).sqrt() * noise
    return noisy


def denoise(x0: Tensor, xt: Tensor, timestep: int, betas: Tensor) -> Tensor:
    """Compute one DDPM reverse step and sample noise when `timestep > 0`.

    Args:
        x0 (Tensor): The original data tensor.
        xt (Tensor): The noisy data tensor at the given timestep.
        timestep (int): The diffusion timestep.
        betas (Tensor): The beta schedule tensor.

    Returns:
        Tensor: The denoised data tensor at the previous timestep.
    """
    alphas = 1.0 - betas
    alpha_t = alphas[timestep]
    alpha_bars = alphas.cumprod(dim=0)
    alpha_bar_t = alpha_bars[timestep]
    alpha_bar_prev_t = (
        alpha_bars[timestep - 1]
        if timestep > 0
        else torch.tensor(1.0, device=x0.device)
    )
    beta_t = betas[timestep]

    param1 = alpha_bar_prev_t.sqrt() * beta_t / (1 - alpha_bar_t)
    param2 = alpha_t.sqrt() * (1 - alpha_bar_prev_t) / (1 - alpha_bar_t)
    mean = param1 * x0 + param2 * xt
    variance = (1 - alpha_bar_prev_t) / (1 - alpha_bar_t) * beta_t

    if timestep > 0:
        noise = torch.randn_like(x0)
        return mean + variance.sqrt() * noise

    return mean
