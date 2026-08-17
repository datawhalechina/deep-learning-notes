import torch.nn as nn

GIB = 1024**3
MIB = 1024**2

__all__ = [
    'bytes_to_gib',
    'bytes_to_mib',
    'count_params',
]


def count_params(module: nn.Module) -> int:
    """Count the number of parameters in a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to count parameters for.

    Returns:
        num_params (int): The total number of parameters in the module.
    """
    return sum(param.numel() for param in module.parameters())


def bytes_to_gib(num_bytes: float) -> float:
    """Convert bytes to gibibytes (GiB).

    Args:
        num_bytes (float): The number of bytes to convert.

    Returns:
        GiB (float): The equivalent number of gibibytes (GiB).
    """
    return num_bytes / GIB


def bytes_to_mib(num_bytes: float) -> float:
    """Convert bytes to mebibytes (MiB).

    Args:
        num_bytes (float): The number of bytes to convert.

    Returns:
        MiB (float): The equivalent number of mebibytes (MiB).
    """
    return num_bytes / MIB
