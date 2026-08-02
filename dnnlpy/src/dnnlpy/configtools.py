import os
import sys

import torch
import torch.accelerator as accl

__all__ = [
    'get_data_root',
    'get_default_device',
    'get_num_workers',
    'has_gil',
    'set_seed',
]


def has_gil() -> bool:
    """Check if the current Python interpreter has a Global Interpreter Lock (GIL)."""
    if sys.version_info >= (3, 13):
        return sys._is_gil_enabled()
    return True


def set_seed(
    seed: int | None = None,
    *,
    deterministic: bool = False,
    benchmark: bool = False,
    warn_only: bool = True,
) -> torch.Generator:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed (int, default: None): Seed value to apply to all supported random
            number generators.
        deterministic (bool, default: False): Whether to request deterministic
            PyTorch algorithms.
        benchmark (bool, default: False): Whether to enable cuDNN benchmark mode.
        warn_only (bool, default: True): Whether nondeterministic PyTorch
            operations should warn instead of raising an error.

    Returns:
        Generator: The PyTorch generator returned by `torch.manual_seed`.
    """
    if seed is not None:
        torch_rng = torch.manual_seed(seed)
    else:
        torch_rng = torch.default_generator

    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    return torch_rng


def get_default_device() -> torch.device:
    """Return the current accelerator device, or CPU when none is available."""
    device = accl.current_accelerator(check_available=True)
    if device is not None:
        return device
    return torch.device('cpu')


def get_data_root() -> str:
    """Return the dataset root directory, creating it when necessary.

    The `DNNL_DATA_ROOT` environment variable overrides the default
    `~/datasets` location.
    """
    root = os.getenv('DNNL_DATA_ROOT', os.path.expanduser('~/datasets'))
    if not os.path.exists(root):
        os.mkdir(root)
    return root


def get_num_workers(num_workers: int | None = None) -> int:
    """Get the number of worker threads to use for parallel processing.

    Args:
        num_workers (int | None, optional): The number of workers to use. This function
            can automatically determine the number of workers based on whether the Python
            interpreter has a Global Interpreter Lock (GIL) and the number of CPU cores
            available. If None, the default will be all available CPU cores if GIL is not
            present, or 1/2 of the available CPU cores if GIL is present.

    Returns:
        num_workers (int): The number of worker threads to use for parallel processing.
    """
    if num_workers is None:
        if sys.version_info >= (3, 13):
            num_workers = os.process_cpu_count() or 1
        else:
            num_workers = os.cpu_count() or 1

        if has_gil() and num_workers > 1:
            num_workers = max(1, num_workers // 2)

    return num_workers
