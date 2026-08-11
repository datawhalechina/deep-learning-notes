import torch
from torch import Tensor
from torch.types import Device

__all__ = ['get_batch']


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
        tuple[Tensor, Tensor]: A tuple containing the input tensor `x` and target
            tensor `y`.
    """
    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(max_start + 1, (batch_size,))

    x = torch.stack([token_ids[i : i + block_size] for i in starts])
    y = torch.stack([token_ids[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
