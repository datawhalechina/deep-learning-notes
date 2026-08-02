import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    'batch_norm',
    'group_norm',
    'instance_norm',
    'layer_norm',
    'local_response_norm',
    'rms_norm',
]


def batch_norm(
    x: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_batch_stats: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Apply batch normalization to an input tensor.

    Args:
        x (Tensor): Input tensor with shape `(N, C, ...)`.
        running_mean (Tensor | None): Running mean with shape `(C,)`. May be `None`
            when `use_batch_stats=True`.
        running_var (Tensor | None): Running variance with shape `(C,)`. May be `None`
            when `use_batch_stats=True`.
        weight (Tensor | None, default: None): Optional scale parameter with shape `(C,)`.
        bias (Tensor | None, default: None): Optional shift parameter with shape `(C,)`.
        use_batch_stats (bool, default: False): If `True`, use batch statistics and update
            running statistics when they are provided.
        momentum (float, default: 0.1): Momentum used to update running statistics.
        eps (float, default: 1e-5): Value added to the variance for numerical stability.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    if (running_mean is None) != (running_var is None):
        raise AssertionError(
            '`running_mean` and `running_var` must either both be tensors or both be None.'
        )

    # (N, C, H, W) -> reduce_dims = (0, 2, 3)
    reduce_dims = (0, *range(2, x.ndim))
    # (C,) -> broadcast_shape = (1, C, 1, 1)
    broadcast_shape = (1, x.size(1)) + (1,) * (x.ndim - 2)

    # Hit this branch when:
    # 1) In training mode, regardless of whether running stats are provided, or
    # 2) In evaluation mode when running stats are not provided.
    if use_batch_stats:
        sample_count = x.numel() // x.size(1)
        if sample_count <= 1:
            raise RuntimeError(
                'Expected more than 1 value per channel when training, '
                f'but got input shape {tuple(x.shape)}.'
            )

        batch_mean = x.mean(dim=reduce_dims)
        batch_var = x.var(dim=reduce_dims, correction=0)

        # Only update running stats when in training mode and running stats are provided
        if running_mean is not None and running_var is not None:
            unbiased_var = batch_var * sample_count / (sample_count - 1)

            with torch.no_grad():
                running_mean.lerp_(batch_mean, momentum)
                running_var.lerp_(unbiased_var, momentum)

    # Hit this branch when in evaluation mode and running stats are provided.
    else:
        assert running_mean is not None and running_var is not None
        batch_mean = running_mean
        batch_var = running_var

    batch_mean = batch_mean.reshape(broadcast_shape)
    batch_var = batch_var.reshape(broadcast_shape)

    y = (x - batch_mean) * (batch_var + eps).rsqrt()
    if weight is not None:
        y = y * weight.reshape(broadcast_shape)
    if bias is not None:
        y = y + bias.reshape(broadcast_shape)

    return y


def group_norm(
    x: Tensor,
    num_groups: int,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Apply group normalization to an input tensor.

    Args:
        x (Tensor): Input tensor with shape `(N, C, ...)`.
        num_groups (int): Number of groups used to divide the channels. `C`must be
            divisible by `num_groups`.
        weight (Tensor | None, default: None): Optional scale parameter with shape `(C,)`.
        bias (Tensor | None, default: None): Optional shift parameter with shape `(C,)`.
        eps (float, default: 1e-5): Value added to the variance for numerical stability.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    if x.ndim < 2:
        raise AssertionError(
            f'Expected input tensor to have at least 2 dimensions, but got {x.ndim}.'
        )

    if num_groups <= 0:
        raise AssertionError(
            f'Expected `num_groups` to be greater than 0, but got {num_groups}.'
        )

    num_channels = x.size(1)
    channels_per_group = num_channels // num_groups
    if num_channels % num_groups != 0:
        raise AssertionError(
            f'Expected the number of channels ({num_channels}) to be divisible '
            f'by `num_groups` ({num_groups}).'
        )

    # (N, C, H, W) -> (N, G, C // G, H, W)
    grouped_shape = (x.size(0), num_groups, channels_per_group, *x.shape[2:])
    grouped_x = x.reshape(grouped_shape)

    # Reduce over the channels in each group and all spatial dimensions.
    # (N, G, C // G, H, W) -> reduce_dims = (2, 3, 4)
    reduce_dims = tuple(range(2, grouped_x.ndim))

    group_mean = grouped_x.mean(dim=reduce_dims, keepdim=True)
    group_var = grouped_x.var(dim=reduce_dims, correction=0, keepdim=True)

    grouped_y = (grouped_x - group_mean) * (group_var + eps).rsqrt()
    y = grouped_y.reshape_as(x)

    # (C,) -> (1, C, 1, 1)
    broadcast_shape = (1, num_channels) + (1,) * (x.ndim - 2)

    if weight is not None:
        y = y * weight.reshape(broadcast_shape)

    if bias is not None:
        y = y + bias.reshape(broadcast_shape)

    return y


def instance_norm(
    x: Tensor,
    running_mean: Tensor | None,
    running_var: Tensor | None,
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    use_instance_stats: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Apply instance normalization to an input tensor.

    Args:
        x (Tensor): Input tensor with shape `(N, C, ...)`.
        running_mean (Tensor | None): Running mean with shape `(C,)`. May be `None`
            when `use_instance_stats=True`.
        running_var (Tensor | None): Running variance with shape `(C,)`. May be `None`
            when `use_instance_stats=True`.
        weight (Tensor | None, default: None): Optional scale parameter with shape `(C,)`.
        bias (Tensor | None, default: None): Optional shift parameter with shape `(C,)`.
        use_instance_stats (bool, default: False): If `True`, use statistics computed
            independently for each instance and update running statistics when they are
            provided.
        momentum (float, default: 0.1): Momentum used to update running statistics.
        eps (float, default: 1e-5): Value added to the variance for numerical stability.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    if (running_mean is None) != (running_var is None):
        raise AssertionError(
            '`running_mean` and `running_var` must either both be tensors or both be None.'
        )

    # (N, C, H, W) -> reduce_dims = (2, 3)
    reduce_dims = tuple(range(2, x.ndim))

    # Per-instance statistics have shape (N, C).
    input_stats_shape = (x.size(0), x.size(1)) + (1,) * (x.ndim - 2)

    # Running statistics and affine parameters have shape (C,).
    broadcast_shape = (1, x.size(1)) + (1,) * (x.ndim - 2)

    # Hit this branch when:
    # 1) In training mode, regardless of whether running stats are provided, or
    # 2) In evaluation mode when running stats are not provided.
    if use_instance_stats:
        sample_count = x[0, 0].numel()
        if sample_count <= 1:
            raise RuntimeError(
                'Expected more than 1 spatial value when using input statistics, '
                f'but got input shape {tuple(x.shape)}.'
            )

        # Each sample and channel has its own mean and variance.
        instance_mean = x.mean(dim=reduce_dims)
        instance_var = x.var(dim=reduce_dims, correction=0)

        # Running statistics are shared across the batch, so average the
        # per-instance statistics over the batch dimension.
        if running_mean is not None and running_var is not None:
            mean_for_running = instance_mean.mean(dim=0)

            unbiased_var = instance_var * sample_count / (sample_count - 1)
            var_for_running = unbiased_var.mean(dim=0)

            with torch.no_grad():
                running_mean.lerp_(mean_for_running, momentum)
                running_var.lerp_(var_for_running, momentum)

        instance_mean = instance_mean.reshape(input_stats_shape)
        instance_var = instance_var.reshape(input_stats_shape)

    # Hit this branch when in evaluation mode and running stats are provided.
    else:
        assert running_mean is not None and running_var is not None
        instance_mean = running_mean.reshape(broadcast_shape)
        instance_var = running_var.reshape(broadcast_shape)

    y = (x - instance_mean) * (instance_var + eps).rsqrt()

    if weight is not None:
        y = y * weight.reshape(broadcast_shape)

    if bias is not None:
        y = y + bias.reshape(broadcast_shape)

    return y


def layer_norm(
    x: Tensor,
    normalized_shape: int | tuple[int, ...],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Applies layer normalization to an input tensor.

    Args:
        x (Tensor): Input tensor with shape `(N, *)`, where `*` means any number of
            additional dimensions.
        normalized_shape (tuple[int, ...]): Input shape from an expected input of size
            `(*)`. If a single integer is used, it is treated as a singleton tuple.
        weight (Tensor | None, default: None): Optional learnable scale parameter of shape
            `normalized_shape`.
        bias (Tensor | None, default: None): Optional learnable shift parameter of shape
            `normalized_shape`.
        eps (float, default: 1e-5): A value added to the denominator for numerical stability.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    if x.shape[-len(normalized_shape) :] != normalized_shape:
        raise AssertionError(
            f'Expected the trailing input dimensions to match '
            f'`normalized_shape={normalized_shape}`, '
            f'but got input shape {tuple(x.shape)}.'
        )

    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    layer_mean = x.mean(dim=dims, keepdim=True)
    layer_var = x.var(dim=dims, correction=0, keepdim=True)

    y = (x - layer_mean) * (layer_var + eps).rsqrt()

    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias

    return y


def local_response_norm(
    x: Tensor,
    size: int,
    alpha: float = 1e-4,
    beta: float = 0.75,
    k: float = 1.0,
) -> Tensor:
    """Apply local response normalization to an input tensor.

    Args:
        x (Tensor): Input tensor wOth shape `(N, C, ...)`.
        size (int): Number of neighboring channels used for normalization.
        alpha (float, default: 1e-4): Scaling factor applied to the local squared response.
        beta (float, default: 0.75): Exponent applied to the normalization term.
        k (float, default: 1.0): Additive constant in the normalization term.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    if x.ndim < 3:
        raise AssertionError(
            'Expected an input with at least 3 dimensions, '
            f'but got input shape {tuple(x.shape)}.'
        )

    # Move the channel dimension to the end:
    # (N, C, ...) -> (N, ..., C)
    squared = x.square().movedim(1, -1)

    # avg_pool1d expects an input with shape (B, C, L).
    # Flatten every dimension except the channel dimension.
    flat_squared = squared.reshape(-1, 1, x.size(1))

    # PyTorch pads asymmetrically when size is even.
    left_padding = size // 2
    right_padding = (size - 1) // 2

    padded_squared = F.pad(flat_squared, (left_padding, right_padding))
    local_mean_square = F.avg_pool1d(padded_squared, kernel_size=size, stride=1)
    local_mean_square = local_mean_square.reshape_as(squared).movedim(-1, 1)

    scale = k + alpha * local_mean_square
    return x * scale.pow(-beta)


def rms_norm(
    x: Tensor,
    normalized_shape: int | tuple[int, ...],
    weight: Tensor | None = None,
    eps: float | None = None,
) -> Tensor:
    """Apply root mean square normalization to an input tensor.

    Args:
        x (Tensor): Input tensor whose trailing dimensions match `normalized_shape`.
        normalized_shape (int | tuple[int, ...]): Shape of the trailing dimensions
            to normalize.
        weight (Tensor | None, default: None): Optional scale parameter with shape
            `normalized_shape`.
        eps (float | None, default: None): Value added to the mean square for numerical
            stability. If `None`, use the machine epsilon of `x.dtype`.

    Returns:
        Tensor: Normalized tensor with the same shape as `x`.
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    if x.shape[-len(normalized_shape) :] != normalized_shape:
        raise AssertionError(
            f'Expected the trailing input dimensions to match '
            f'`normalized_shape={normalized_shape}`, '
            f'but got input shape {tuple(x.shape)}.'
        )

    if eps is None:
        eps = torch.finfo(x.dtype).eps

    # Normalize over the trailing dimensions specified by normalized_shape.
    reduce_dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))

    dtype = x.dtype
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.to(torch.float32)

    mean_square = x.square().mean(dim=reduce_dims, keepdim=True)
    y = x * (mean_square + eps).rsqrt()
    y = y.to(dtype)

    if weight is not None:
        broadcast_shape = (1,) * (x.ndim - len(normalized_shape)) + normalized_shape
        y = y * weight.reshape(broadcast_shape)

    return y
