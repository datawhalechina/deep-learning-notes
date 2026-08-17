import torch

from .configtools import get_default_device

DEVICE = get_default_device()

__all__ = [
    'max_memory_allocated',
    'memory_allocated',
    'memory_reserved',
    'reset_max_memory_allocated',
    'reset_peak_memory_stats',
]


def memory_allocated() -> int:
    """Returns the current GPU memory usage in bytes."""
    match DEVICE.type:
        case 'cuda':
            return torch.cuda.memory_allocated()
        case 'xpu':
            return torch.xpu.memory_allocated()
        case 'mtia':
            return torch.mtia.memory_allocated()
        case 'cpu':
            return 0  # No memory tracking for CPU
        case _:
            raise NotImplementedError(f'Unsupported device type: {DEVICE.type}.')


def memory_reserved() -> int:
    """Returns the current GPU memory reserved in bytes."""
    match DEVICE.type:
        case 'cuda':
            return torch.cuda.memory_reserved()
        case 'xpu':
            return torch.xpu.memory_reserved()
        case 'cpu':
            return 0  # No memory tracking for CPU
        case _:
            raise NotImplementedError(f'Unsupported device type: {DEVICE.type}.')


def max_memory_allocated() -> int:
    """Returns the maximum GPU memory usage in bytes."""
    match DEVICE.type:
        case 'cuda':
            return torch.cuda.max_memory_allocated()
        case 'xpu':
            return torch.xpu.max_memory_allocated()
        case 'mtia':
            return torch.mtia.max_memory_allocated()
        case 'cpu':
            return 0  # No memory tracking for CPU
        case _:
            raise NotImplementedError(f'Unsupported device type: {DEVICE.type}.')


def reset_max_memory_allocated() -> None:
    """Resets the maximum GPU memory usage statistics."""
    match DEVICE.type:
        case 'cuda':
            torch.cuda.reset_peak_memory_stats()
        case 'xpu':
            torch.xpu.reset_peak_memory_stats()
        case 'mtia':
            torch.mtia.reset_peak_memory_stats()
        case 'cpu':
            pass  # No action needed for CPU
        case _:
            raise NotImplementedError(f'Unsupported device type: {DEVICE.type}.')


def reset_peak_memory_stats() -> None:
    """Resets the peak memory usage statistics."""
    match DEVICE.type:
        case 'cuda':
            torch.cuda.reset_peak_memory_stats()
        case 'xpu':
            torch.xpu.reset_peak_memory_stats()
        case 'mtia':
            torch.mtia.reset_peak_memory_stats()
        case 'cpu':
            pass  # No action needed for CPU
        case _:
            raise NotImplementedError(f'Unsupported device type: {DEVICE.type}.')
