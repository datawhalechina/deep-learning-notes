from typing import Any, cast

import torch
from torch import Tensor
from torch.autograd import Function

__all__ = [
    'embedding',
]


def _normalize_padding_idx(padding_idx: int | None, num_embeddings: int) -> int | None:
    """Normalize the padding index to be a non-negative integer."""
    if padding_idx is None:
        return None
    if padding_idx > 0:
        if padding_idx >= num_embeddings:
            raise AssertionError('`padding_idx` must be within `num_embeddings`.')
    elif padding_idx < 0:
        if padding_idx < -num_embeddings:
            raise AssertionError('`padding_idx` must be within `num_embeddings`.')
        padding_idx += num_embeddings
    return padding_idx


@torch.no_grad()
def _renorm_embedding_weight_(
    x: Tensor,
    weight: Tensor,
    max_norm: float,
    norm_type: float,
) -> None:
    """Renormalize the rows of the embedding weight tensor in-place."""
    # Only renormalize the rows that are actually used in the input tensor.
    indices = x.reshape(-1).unique()
    if indices.numel() == 0:
        return

    rows = weight.index_select(0, indices)
    norms = rows.norm(norm_type, dim=1, keepdim=True)
    scale = max_norm / (norms + 1e-7)
    rows = torch.where(norms > max_norm, rows * scale, rows)
    weight.index_copy_(0, indices, rows)


class _Embedding(Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        weight: Tensor,
        padding_idx: int | None,
        scale_grad_by_freq: bool,
    ) -> Tensor:
        """Look up embeddings for indices in a fixed embedding table."""
        ctx.save_for_backward(x)
        ctx.num_embeddings = weight.size(0)
        ctx.embedding_dim = weight.size(1)
        ctx.padding_idx = padding_idx
        ctx.scale_grad_by_freq = scale_grad_by_freq

        flat_input = x.reshape(-1)
        output = weight.index_select(0, flat_input)
        return output.reshape(*x.size(), ctx.embedding_dim)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Any:
        """Compute the gradient of the embedding lookup with respect to the weight tensor."""
        (x,) = cast(tuple[Tensor], ctx.saved_tensors)
        grad_output = grad_outputs[0]

        all_indices = x.reshape(-1)
        indices = all_indices
        grad = grad_output.reshape(-1, ctx.embedding_dim)

        if ctx.padding_idx is not None:
            mask = indices != ctx.padding_idx
            indices = indices[mask]
            grad = grad[mask]

        grad_weight = grad_output.new_zeros(ctx.num_embeddings, ctx.embedding_dim)

        if indices.numel() > 0:
            if ctx.scale_grad_by_freq:
                counts = torch.bincount(
                    all_indices.to(dtype=torch.long),
                    minlength=ctx.num_embeddings,
                )
                frequency = counts.index_select(0, indices.to(dtype=torch.long))
                grad = grad / frequency.to(dtype=grad.dtype).unsqueeze(1)
            grad_weight.index_add_(0, indices, grad)

        return None, grad_weight, None, None


def embedding(
    x: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
) -> Tensor:
    """Look up embeddings for indices in a fixed embedding table.

    Args:
        x (Tensor): Integer indices with arbitrary shape.
        weight (Tensor): Embedding table with shape
            ``(num_embeddings, embedding_dim)``.
        padding_idx (int | None, default: None): Row that does not contribute to
            the gradient. A negative index is resolved from the end of `weight`.
        max_norm (float | None, default: None): If specified, renormalize selected
            rows whose norm exceeds this value. This modifies `weight` in-place.
        norm_type (float, default: 2.0): The p value used to calculate row norms.
        scale_grad_by_freq (bool, default: False): Scale each row's gradient by the
            inverse frequency of its index in `input`.

    Returns:
        Tensor: Selected embeddings with shape `(*input.shape, embedding_dim)`.
    """
    if weight.ndim != 2:
        raise AssertionError('The embedding weight tensor must be 2-dimensional.')

    padding_idx = _normalize_padding_idx(padding_idx, weight.size(0))
    if max_norm is not None:
        _renorm_embedding_weight_(x, weight, max_norm, norm_type)

    return _Embedding.apply(x, weight, padding_idx, scale_grad_by_freq)
