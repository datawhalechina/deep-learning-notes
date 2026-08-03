import torch
import torch.nn.functional as F
from torch import Tensor
from torch.types import Device

import dnnlpy.nn.functional as dF

__all__ = [
    'get_batch',
    'greedy_sampling',
    'sample_next_token',
    'top_k_sampling',
    'top_p_sampling',
]


def get_batch(
    token_ids: Tensor,
    block_size: int,
    batch_size: int,
    device: Device = 'cpu',
) -> tuple[Tensor, Tensor]:
    """Cut out a batch of inputs and targets from the token stream.

    Args:
        token_ids (Tensor): The token ids of the text data.
        block_size (int): The length of each input sequence.
        batch_size (int): The number of sequences in the batch.
        device (Device, default: 'cpu'): The device to place the tensors on.

    Returns:
        tuple[Tensor, Tensor]: A tuple containing the input tensor `x` and the target tensor `y`.
    """
    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(max_start + 1, (batch_size,))

    x = torch.stack([token_ids[i : i + block_size] for i in starts])
    y = torch.stack([token_ids[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def greedy_sampling(logits: Tensor, temperature: float) -> Tensor:
    """Sample the next token greedily from the logits.

    Args:
        logits (Tensor): The logits from the model of shape (B, V).
        temperature (float): The temperature for sampling. Must be positive.

    Returns:
        Tensor: The sampled token ids of shape (B, 1).
    """
    if temperature <= 0:
        raise AssertionError('`temperature` must be positive.')
    return dF.softmax(logits / temperature, dim=-1)


def top_k_sampling(logits: Tensor, top_k: int) -> Tensor:
    """Sample the next token from the logits using top-k sampling.

    Args:
        logits (Tensor): The logits from the model of shape (B, V).
        top_k (int): The number of top tokens to consider for sampling. Must be positive.

    Returns:
        Tensor: The logits with tokens outside the top-k set to -inf, of shape (B, V).
    """
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))
    values = logits.topk(top_k, dim=-1).values
    threshold = values[..., -1]

    logits = logits.masked_fill(logits < threshold, -torch.inf)
    return logits


def top_p_sampling(logits: Tensor, top_p: float) -> Tensor:
    """Sample the next token from the logits using top-p sampling.

    Args:
        logits (Tensor): The logits from the model of shape (B, V).
        top_p (float): The cumulative probability threshold for sampling. Must be in (0, 1].

    Returns:
        Tensor: The logits with tokens outside the top-p set to -inf, of shape (B, V).
    """
    if not 0 < top_p <= 1:
        raise AssertionError('`top_p` must be in (0, 1].')

    if top_p == 1.0:
        return logits

    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    sorted_probs = dF.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # Remove tokens whose cumulative probability is above top_p.
    mask = cumulative_probs > top_p
    # Keep the first token above the threshold as well, so the kept set reaches top_p.
    mask = F.pad(mask[..., :-1], (1, 0), value=False)

    remove_mask = torch.zeros_like(logits, dtype=torch.bool)
    remove_mask.scatter_(dim=-1, index=sorted_indices, src=mask)

    logits = logits.masked_fill(remove_mask, -torch.inf)
    return logits


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
) -> Tensor:
    """Sample next token ids from logits with temperature, top-k, and top-p.

    Args:
        logits (Tensor): The logits from the model of shape (B, V).
        temperature (float, default: 1.0): The temperature for sampling. Must be positive.
        top_k (int | None, default: None): The number of top tokens to consider for sampling.
            If None, no top-k filtering is applied.
        top_p (float | None, default: None): The cumulative probability threshold for sampling.
            If None, no top-p filtering is applied.
        greedy (bool, default: False): If True, use greedy sampling instead of probabilistic
            sampling.

    Returns:
        Tensor: The sampled token ids of shape (B, 1).
    """
    if logits.ndim != 2:
        raise AssertionError('`logits` must have shape (B, V).')
    if temperature <= 0:
        raise AssertionError('`temperature` must be positive.')

    if greedy:
        next_token = logits.argmax(dim=-1, keepdim=True)
        return next_token

    logits = logits / temperature

    if top_k is not None:
        logits = top_k_sampling(logits, top_k=top_k)

    if top_p is not None:
        logits = top_p_sampling(logits, top_p=top_p)

    probs = dF.softmax(logits, dim=-1)
    next_token = probs.multinomial(num_samples=1)
    return next_token
