import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF
from .common_types import Size1D, Size2D, Size3D, SizeND

__all__ = [
    'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d',
    'AdaptiveAvgPool3d',
    'AdaptiveMaxPool1d',
    'AdaptiveMaxPool2d',
    'AdaptiveMaxPool3d',
    'AvgPool1d',
    'AvgPool2d',
    'AvgPool3d',
    'MaxPool1d',
    'MaxPool2d',
    'MaxPool3d',
]


class _MaxPoolNd(nn.Module):
    """Base class for max pooling modules."""

    def __init__(
        self,
        kernel_size: SizeND,
        stride: SizeND | None = None,
        padding: SizeND = 0,
        dilation: SizeND = 1,
        ceil_mode: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize the max pooling module.

        Args:
            kernel_size (SizeND): Size of the pooling kernel.
            stride (SizeND, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (SizeND, default: 0): Padding added to all four sides of the input.
            dilation (SizeND, default: 1): Dilation of the pooling kernel.
            ceil_mode (bool, default: False): When True, will use ceil instead of floor to
                compute the output shape.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.fast = fast

    def extra_repr(self) -> str:
        return (
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}, '
            f'dilation={self.dilation}, '
            f'ceil_mode={self.ceil_mode}'
        )


class MaxPool1d(_MaxPoolNd):
    """Apply a 1D max pooling operation."""

    def __init__(
        self,
        kernel_size: Size1D,
        stride: Size1D | None = None,
        padding: Size1D = 0,
        dilation: Size1D = 1,
        ceil_mode: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize a 1D max pooling module.

        Args:
            kernel_size (Size1D): Size of the pooling kernel.
            stride (Size1D, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (Size1D, default: 0): Implicit negative infinity padding added to both
                sides of the input.
            dilation (Size1D, default: 1): Spacing between kernel elements.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor
                to compute the output shape.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.max_pool1d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )
        else:
            return dF.max_pool1d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )


class MaxPool2d(_MaxPoolNd):
    """Apply a 2D max pooling operation."""

    def __init__(
        self,
        kernel_size: Size2D,
        stride: Size2D | None = None,
        padding: Size2D = 0,
        dilation: Size2D = 1,
        ceil_mode: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize a 2D max pooling module.

        Args:
            kernel_size (Size2D): Size of the pooling kernel.
            stride (Size2D, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (Size2D, default: 0): Implicit negative infinity padding added to both
                sides of the input.
            dilation (Size2D, default: 1): Spacing between kernel elements.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor to
                compute the output shape.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )
        else:
            return dF.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )


class MaxPool3d(_MaxPoolNd):
    """Apply a 3D max pooling operation."""

    def __init__(
        self,
        kernel_size: Size3D,
        stride: Size3D | None = None,
        padding: Size3D = 0,
        dilation: Size3D = 1,
        ceil_mode: bool = False,
        *,
        fast: bool = False,
    ):
        """Initialize a 3D max pooling module.

        Args:
            kernel_size (Size3D): Size of the pooling kernel.
            stride (Size3D, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (Size3D, default: 0): Implicit negative infinity padding added to both
                sides of the input.
            dilation (Size3D, default: 1): Spacing between kernel elements.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor to
                compute the output shape.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )
        else:
            return dF.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                ceil_mode=self.ceil_mode,
            )


class _AvgPoolNd(nn.Module):
    """Base class for average pooling modules."""

    def __init__(
        self,
        kernel_size: SizeND,
        stride: SizeND | None = None,
        padding: SizeND = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
        *,
        fast: bool = False,
    ):
        """Initialize the average pooling module.

        Args:
            kernel_size (SizeND): Size of the pooling kernel.
            stride (SizeND, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (SizeND, default: 0): Implicit zero padding added to both sides of
                the input.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor
                to compute the output shape.
            count_include_pad (bool, default: True): If set to True, include zero padding
                in the averaging calculation.
            divisor_override (int, optional): If specified, use this value as the divisor
                instead of the pooling region size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.fast = fast

    def extra_repr(self) -> str:
        return (
            f'kernel_size={self.kernel_size}, '
            f'stride={self.stride}, '
            f'padding={self.padding}'
        )


class AvgPool1d(_AvgPoolNd):
    """Apply a 1D average pooling operation."""

    def __init__(
        self,
        kernel_size: Size1D,
        stride: Size1D | None = None,
        padding: Size1D = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        *,
        fast: bool = False,
    ):
        """Initialize a 1D average pooling module.

        Args:
            kernel_size (Size1D): Size of the pooling kernel.
            stride (Size1D, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (Size1D, default: 0): Implicit zero padding added to both sides of
                the input.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor
                to compute the output shape.
            count_include_pad (bool, default: True): If set to True, include zero padding
                in the averaging calculation.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
            )
        else:
            return dF.avg_pool1d(
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
            )


class AvgPool2d(_AvgPoolNd):
    """Apply a 2D average pooling operation."""

    def __init__(
        self,
        kernel_size: Size2D,
        stride: Size2D | None = None,
        padding: Size2D = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
        *,
        fast: bool = False,
    ):
        """Initialize a 2D average pooling module.

        Args:
            kernel_size (Size2D): Size of the pooling kernel.
            stride (Size2D, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (Size2D, default: 0): Implicit zero padding added to both sides of
                the input.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor
                to compute the output shape.
            count_include_pad (bool, default: True): If set to True, include zero padding
                in the averaging calculation.
            divisor_override (int, optional): If specified, use this value as the divisor
                instead of the pooling region size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            )
        else:
            return dF.avg_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            )


class AvgPool3d(_AvgPoolNd):
    """Apply a 3D average pooling operation."""

    def __init__(
        self,
        kernel_size: Size3D,
        stride: Size3D | None = None,
        padding: Size3D = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
        *,
        fast: bool = False,
    ):
        """Initialize a 3D average pooling module.

        Args:
            kernel_size (Size3D): Size of the pooling kernel.
            stride (Size3D, optional): Stride of the pooling operation. If `None`, use
                `kernel_size`.
            padding (Size3D, default: 0): Implicit zero padding added to both sides of
                the input.
            ceil_mode (bool, default: False): If set to True, use ceil instead of floor
                to compute the output shape.
            count_include_pad (bool, default: True): If set to True, include zero padding
                in the averaging calculation.
            divisor_override (int, optional): If specified, use this value as the divisor
                instead of the pooling region size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
            fast=fast,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.avg_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            )
        else:
            return dF.avg_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                ceil_mode=self.ceil_mode,
                count_include_pad=self.count_include_pad,
                divisor_override=self.divisor_override,
            )


class _AdaptivePoolNd(nn.Module):
    """Base class for adaptive pooling modules."""

    def __init__(self, output_size: SizeND, *, fast: bool = False):
        """Initialize the adaptive pooling module.

        Args:
            output_size (SizeND): Target spatial output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.output_size = output_size
        self.fast = fast

    def extra_repr(self) -> str:
        return f'output_size={self.output_size}'


class AdaptiveAvgPool1d(_AdaptivePoolNd):
    """Apply a 1D adaptive average pooling operation."""

    def __init__(self, output_size: Size1D, *, fast: bool = False):
        """Initialize a 1D adaptive average pooling module.

        Args:
            output_size (Size1D): Target output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(output_size, fast=fast)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.adaptive_avg_pool1d(x, self.output_size)
        else:
            return dF.adaptive_avg_pool1d(x, self.output_size)


class AdaptiveAvgPool2d(_AdaptivePoolNd):
    """Apply a 2D adaptive average pooling operation."""

    def __init__(self, output_size: Size2D, *, fast: bool = False):
        """Initialize a 2D adaptive average pooling module.

        Args:
            output_size (Size2D): Target output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(output_size, fast=fast)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.adaptive_avg_pool2d(x, self.output_size)  # type: ignore[arg-type]
        else:
            return dF.adaptive_avg_pool2d(x, self.output_size)


class AdaptiveAvgPool3d(_AdaptivePoolNd):
    """Apply a 3D adaptive average pooling operation."""

    def __init__(self, output_size: Size3D, *, fast: bool = False):
        """Initialize a 3D adaptive average pooling module.

        Args:
            output_size (Size3D): Target output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(output_size, fast=fast)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.adaptive_avg_pool3d(x, self.output_size)  # type: ignore[arg-type]
        else:
            return dF.adaptive_avg_pool3d(x, self.output_size)


class AdaptiveMaxPool1d(_AdaptivePoolNd):
    """Apply a 1D adaptive max pooling operation."""

    def __init__(self, output_size: Size1D, *, fast: bool = False):
        """Initialize a 1D adaptive max pooling module.

        Args:
            output_size (Size1D): Target output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(output_size, fast=fast)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.adaptive_max_pool1d(x, self.output_size)
        else:
            return dF.adaptive_max_pool1d(x, self.output_size)


class AdaptiveMaxPool2d(_AdaptivePoolNd):
    """Apply a 2D adaptive max pooling operation."""

    def __init__(self, output_size: Size2D, *, fast: bool = False):
        """Initialize a 2D adaptive max pooling module.

        Args:
            output_size (Size2D): Target output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(output_size, fast=fast)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.adaptive_max_pool2d(x, self.output_size)
        else:
            return dF.adaptive_max_pool2d(x, self.output_size)


class AdaptiveMaxPool3d(_AdaptivePoolNd):
    """Apply a 3D adaptive max pooling operation."""

    def __init__(self, output_size: Size3D, *, fast: bool = False):
        """Initialize a 3D adaptive max pooling module.

        Args:
            output_size (Size3D): Target output size.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__(output_size, fast=fast)

    def forward(self, x: Tensor) -> Tensor:
        if self.fast:
            return F.adaptive_max_pool3d(x, self.output_size)
        else:
            return dF.adaptive_max_pool3d(x, self.output_size)
