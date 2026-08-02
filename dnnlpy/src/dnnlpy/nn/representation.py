from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import functional as dF

__all__ = ['Embedding']


class Embedding(nn.Module):
    """Store embeddings for a fixed-size dictionary and retrieve them by index."""

    weight: Tensor

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        *,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        fast: bool = False,
    ) -> None:
        """Initialize an embedding table.

        Args:
            num_embeddings (int): Number of entries in the embedding table.
            embedding_dim (int): Size of each embedding vector.
            padding_idx (int | None, default: None): Entry excluded from gradient
                updates. Its initial value is zero when `_weight` is not supplied.
            max_norm (float | None, default: None): Maximum norm allowed for rows
                selected during the forward pass.
            norm_type (float, default: 2.0): The p value used to calculate row norms.
            scale_grad_by_freq (bool, default: False): Scale each row's gradient by
                the inverse frequency of its index in the input.
            _weight (Tensor | None, default: None): Optional preinitialized weight.
            _freeze (bool, default: False): Whether the supplied or initialized weight
                should be frozen.
            fast (bool, default: False): If set to True, will use the fast implementation
                from :func:`torch.nn.functional`. Default: False.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fast = fast

        if padding_idx is not None:
            if padding_idx > 0:
                if padding_idx >= num_embeddings:
                    raise AssertionError(
                        '`padding_idx` must be within `num_embeddings`.'
                    )
            elif padding_idx < 0:
                if padding_idx < -num_embeddings:
                    raise AssertionError(
                        '`padding_idx` must be within `num_embeddings`.'
                    )
                padding_idx += num_embeddings

        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if _weight is None:
            self.weight = nn.Parameter(
                torch.empty((num_embeddings, embedding_dim)),
                requires_grad=not _freeze,
            )
            self.reset_parameters()
        else:
            if _weight.size() != (num_embeddings, embedding_dim):
                raise AssertionError(
                    'Shape of weight does not match `num_embeddings` and `embedding_dim`.'
                )
            self.weight = nn.Parameter(_weight, requires_grad=not _freeze)

    def reset_parameters(self) -> None:
        """Initialize weights from a standard normal distribution."""
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)

    def forward(self, x: Tensor) -> Tensor:
        """Retrieve embedding vectors for `x`."""
        if self.fast:
            return F.embedding(
                x,
                self.weight,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
            )
        return dF.embedding(
            x,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
        )

    def extra_repr(self) -> str:
        """Return the module configuration used by `repr`."""
        parts = [str(self.num_embeddings), str(self.embedding_dim)]
        if self.padding_idx is not None:
            parts.append(f'padding_idx={self.padding_idx}')
        if self.max_norm is not None:
            parts.append(f'max_norm={self.max_norm}')
        if self.norm_type != 2:
            parts.append(f'norm_type={self.norm_type}')
        if self.scale_grad_by_freq:
            parts.append(f'scale_grad_by_freq={self.scale_grad_by_freq}')
        return ', '.join(parts)

    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor,
        freeze: bool = True,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
    ) -> Self:
        """Create an embedding module from a two-dimensional weight tensor.

        Args:
            embeddings (Tensor): A 2D tensor containing the embedding weights.
            freeze (bool, default: True): If True, the embedding weights will not be updated
                during training. If False, the weights will be trainable.
            padding_idx (int | None, default: None): If specified, the embedding at this
                index will not be updated during training and will be initialized to zero.
            max_norm (float | None, default: None): If specified, renormalize selected
                rows whose norm exceeds this value. This modifies `embeddings` in-place.
            norm_type (float, default: 2.0): The p value used to calculate row norms.
            scale_grad_by_freq (bool, default: False): Scale each row's gradient by the
                inverse frequency of its index in the input.

        Returns:
            Embedding: An instance of the Embedding class initialized with the provided weights.
        """
        if embeddings.ndim != 2:
            raise AssertionError('The embedding weight tensor must be 2-dimensional.')

        return cls(
            embeddings.size(0),
            embeddings.size(1),
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            _weight=embeddings,
            _freeze=freeze,
        )
