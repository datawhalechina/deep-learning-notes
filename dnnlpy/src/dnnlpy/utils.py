import torch.nn as nn

__all__ = ['count_params']


def count_params(module: nn.Module) -> int:
    """Count the number of parameters in a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to count parameters for.

    Returns:
        num_params (int): The total number of parameters in the module.
    """
    return sum(param.numel() for param in module.parameters())
