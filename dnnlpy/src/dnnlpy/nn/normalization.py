# pyright: reportOptionalMemberAccess=false

from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

__all__ = [
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'GroupNorm',
    'InstanceNorm1d',
    'InstanceNorm2d',
    'InstanceNorm3d',
    'LayerNorm',
    'LocalResponseNorm',
    'RMSNorm',
]


class _BatchNorm(ABC, nn.Module):
    """Base class for batch normalization modules."""

    weight: Tensor | None
    bias: Tensor | None
    running_mean: Tensor | None
    running_var: Tensor | None
    num_batches_tracked: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a batch normalization module.

        Args:
            num_features (int): Number of feature channels `C` in the input.
            eps (float, default: 1e-5): Value added to the variance for numerical stability.
            momentum (float | None, default: 0.1): Momentum used to update running statistics.
                If `None`, use a cumulative moving average.
            affine (bool, default: True): If `True`, learn per-channel scale and shift
                parameters.
            track_running_stats (bool, default: True): If `True`, track running mean and
                variance for evaluation. If `False`, running statistics are `None` and
                batch statistics are used in both training and evaluation.
            bias (bool, default: True): If `True` and `affine=True`, learn the additive
                shift parameter.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.fast = fast

        if affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(num_features))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.empty(num_features))
            self.register_buffer('running_var', torch.empty(num_features))
            self.register_buffer(
                'num_batches_tracked', torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @abstractmethod
    def _check_input_dim(self, x: Tensor) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()

        if self.training:
            use_batch_stats = True
        else:
            use_batch_stats = (self.running_mean is None) and (self.running_var is None)

        if self.fast:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                weight=self.weight,
                bias=self.bias,
                training=use_batch_stats,
                momentum=exponential_average_factor,
                eps=self.eps,
            )

        return dF.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            use_batch_stats=use_batch_stats,
            momentum=exponential_average_factor,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, '
            f'eps={self.eps}, '
            f'momentum={self.momentum}, '
            f'affine={self.affine}, '
            f'bias={self.bias is not None}, '
            f'track_running_stats={self.track_running_stats}'
        )


class BatchNorm1d(_BatchNorm):
    """Apply batch normalization over 2D or 3D inputs."""

    @override
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim not in (2, 3):
            raise AssertionError(
                f'Expected 2D or 3D input, but got shape {tuple(x.shape)}.'
            )
        if x.size(1) != self.num_features:
            raise AssertionError(
                f'Expected {self.num_features} channels, but got {x.size(1)} channels.'
            )


class BatchNorm2d(_BatchNorm):
    """Apply batch normalization over 4D inputs."""

    @override
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise AssertionError(f'Expected 4D input, but got shape {tuple(x.shape)}.')
        if x.size(1) != self.num_features:
            raise AssertionError(
                f'Expected {self.num_features} channels, but got {x.size(1)} channels.'
            )


class BatchNorm3d(_BatchNorm):
    """Apply batch normalization over 5D inputs."""

    @override
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 5:
            raise AssertionError(f'Expected 5D input, but got shape {tuple(x.shape)}.')
        if x.size(1) != self.num_features:
            raise AssertionError(
                f'Expected {self.num_features} channels, but got {x.size(1)} channels.'
            )


class GroupNorm(nn.Module):
    """Apply group normalization over channel groups."""

    weight: Tensor | None
    bias: Tensor | None

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a group normalization module.

        Args:
            num_groups (int): Number of channel groups used for normalization.
            num_channels (int): Number of input channels `C`. Must be divisible by
                `num_groups`.
            eps (float, default: 1e-5): Value added to the variance for numerical stability.
            affine (bool, default: True): If `True`, learn per-channel scale and shift
                parameters.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        if num_channels % num_groups != 0:
            raise AssertionError('`num_channels` must be divisible by `num_groups`.')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.fast = fast

        if affine:
            self.weight = nn.Parameter(torch.empty(num_channels))
            self.bias = nn.Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) != self.num_channels:
            raise AssertionError(
                f'Expected {self.num_channels} channels, but got {x.size(1)} channels.'
            )

        if self.fast:
            return F.group_norm(
                x,
                self.num_groups,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )

        return dF.group_norm(
            x,
            self.num_groups,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f'{self.num_groups}, '
            f'{self.num_channels}, '
            f'eps={self.eps}, '
            f'affine={self.affine}'
        )


class _InstanceNorm(ABC, nn.Module):
    """Base class for instance normalization modules."""

    weight: Tensor | None
    bias: Tensor | None
    running_mean: Tensor | None
    running_var: Tensor | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize an instance normalization module.

        Args:
            num_features (int): Number of feature channels `C` in the input.
            eps (float, default: 1e-5): Value added to the variance for numerical stability.
            momentum (float, default: 0.1): Momentum used to update running statistics.
            affine (bool, default: False): If `True`, learn per-channel scale and shift
                parameters.
            track_running_stats (bool, default: False): If `True`, track running mean and
                variance for evaluation. If `False`, input statistics are used in both
                training and evaluation.
            bias (bool, default: True): If `True` and `affine=True`, learn the additive
                shift parameter.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.fast = fast

        if affine:
            self.weight = nn.Parameter(torch.empty(num_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(num_features))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if track_running_stats:
            self.register_buffer('running_mean', torch.empty(num_features))
            self.register_buffer('running_var', torch.empty(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @abstractmethod
    def _check_input_dim(self, x: Tensor) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        self._check_input_dim(x)
        use_instance_stats = self.training or not self.track_running_stats

        if self.fast:
            return F.instance_norm(
                x,
                self.running_mean,
                self.running_var,
                weight=self.weight,
                bias=self.bias,
                use_input_stats=use_instance_stats,
                momentum=self.momentum,
                eps=self.eps,
            )
        return dF.instance_norm(
            x,
            self.running_mean,
            self.running_var,
            weight=self.weight,
            bias=self.bias,
            use_instance_stats=use_instance_stats,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f'{self.num_features}, '
            f'eps={self.eps}, '
            f'momentum={self.momentum}, '
            f'affine={self.affine}, '
            f'bias={self.bias is not None}, '
            f'track_running_stats={self.track_running_stats}'
        )


class InstanceNorm1d(_InstanceNorm):
    """Apply instance normalization over 3D inputs."""

    @override
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise AssertionError(f'Expected 3D input, but got shape {tuple(x.shape)}.')
        if x.size(1) != self.num_features:
            raise AssertionError(
                f'Expected {self.num_features} channels, but got {x.size(1)} channels.'
            )


class InstanceNorm2d(_InstanceNorm):
    """Apply instance normalization over 4D inputs."""

    @override
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise AssertionError(f'Expected 4D input, but got shape {tuple(x.shape)}.')
        if x.size(1) != self.num_features:
            raise AssertionError(
                f'Expected {self.num_features} channels, but got {x.size(1)} channels.'
            )


class InstanceNorm3d(_InstanceNorm):
    """Apply instance normalization over 5D inputs."""

    @override
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 5:
            raise AssertionError(f'Expected 5D input, but got shape {tuple(x.shape)}.')
        if x.size(1) != self.num_features:
            raise AssertionError(
                f'Expected {self.num_features} channels, but got {x.size(1)} channels.'
            )


class LayerNorm(nn.Module):
    """Apply layer normalization over the trailing input dimensions."""

    weight: Tensor | None
    bias: Tensor | None

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a layer normalization module.

        Args:
            normalized_shape (int | tuple[int, ...]): Shape of the trailing dimensions
                to normalize. A single integer is treated as a singleton tuple.
            eps (float, default: 1e-5): Value added to the variance for numerical stability.
            elementwise_affine (bool, default: True): If `True`, learn scale and shift
                parameters with shape `normalized_shape`.
            bias (bool, default: True): If `True` and `elementwise_affine=True`, learn
                the additive shift parameter.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.fast = fast

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.normalized_shape))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
            )
        return dF.layer_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f'normalized_shape={self.normalized_shape}, '
            f'eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}, '
            f'bias={self.bias is not None}'
        )


class LocalResponseNorm(nn.Module):
    """Apply local response normalization across neighboring channels."""

    def __init__(
        self,
        size: int,
        alpha: float = 1e-4,
        beta: float = 0.75,
        k: float = 1.0,
        *,
        fast: bool = False,
    ):
        """Initialize a local response normalization module.

        Args:
            size (int): Number of neighboring channels used for normalization.
            alpha (float, default: 1e-4): Scaling factor applied to the local squared
                response.
            beta (float, default: 0.75): Exponent applied to the normalization term.
            k (float, default: 1.0): Additive constant in the normalization term.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.fast = fast

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.local_response_norm(
                x,
                self.size,
                alpha=self.alpha,
                beta=self.beta,
                k=self.k,
            )
        return dF.local_response_norm(
            x,
            self.size,
            alpha=self.alpha,
            beta=self.beta,
            k=self.k,
        )

    def extra_repr(self) -> str:
        return f'{self.size}, alpha={self.alpha}, beta={self.beta}, k={self.k}'


class RMSNorm(nn.Module):
    """Apply root mean square normalization over the trailing input dimensions."""

    weight: Tensor | None

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float | None = None,
        elementwise_affine: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a root mean square normalization module.

        Args:
            normalized_shape (int | tuple[int, ...]): Shape of the trailing dimensions
                to normalize. A single integer is treated as a singleton tuple.
            eps (float | None, default: None): Value added to the mean square for
                numerical stability. If `None`, use the machine epsilon of `x.dtype`.
            elementwise_affine (bool, default: True): If `True`, learn a scale
                parameter with shape `normalized_shape`.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.fast = fast

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.rms_norm(
                x,
                self.normalized_shape,
                weight=self.weight,
                eps=self.eps,
            )
        return dF.rms_norm(
            x,
            self.normalized_shape,
            weight=self.weight,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f'normalized_shape={self.normalized_shape}, '
            f'eps={self.eps}, '
            f'elementwise_affine={self.elementwise_affine}'
        )
