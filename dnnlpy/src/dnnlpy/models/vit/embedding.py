import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import dnnlpy.nn as dnn

__all__ = [
    'ViTConvPatchEmbedding',
    'ViTEmbedding',
    'ViTLinearPatchEmbedding',
    'ViTPositionalEmbedding',
]


class ViTLinearPatchEmbedding(nn.Module):
    """Embed image patches by unfolding them and applying a linear projection."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """Initialize an unfold-based patch embedding layer.

        Args:
            image_size (int, default: 224): Height and width of the square input image.
            patch_size (int, default: 16): Height and width of each square image patch.
            in_channels (int, default: 3): Number of input image channels.
            embed_dim (int, default: 768): Output embedding dimension for each patch.
        """
        super().__init__()
        if image_size % patch_size != 0:
            raise AssertionError('`image_size` must be divisible by `patch_size`.')

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.unfold = dnn.Unfold(patch_size, stride=patch_size)
        self.proj = dnn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Convert images of shape `(batch, channels, height, width)` to patch tokens."""
        x = self.unfold(x)  # (B, C*P*P, N)
        x = x.transpose(1, 2)  # (B, N, C*P*P)
        x = self.proj(x)
        return x


class ViTConvPatchEmbedding(nn.Module):
    """Embed image patches with a strided convolution."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        """Initialize a convolution-based patch embedding layer.

        Args:
            image_size (int, default: 224): Height and width of the square input image.
            patch_size (int, default: 16): Height and width of each square image patch.
            in_channels (int, default: 3): Number of input image channels.
            embed_dim (int, default: 768): Output embedding dimension for each patch.
        """
        super().__init__()
        if image_size % patch_size != 0:
            raise AssertionError('`image_size` must be divisible by `patch_size`.')

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.proj = dnn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Convert images of shape `(batch, channels, height, width)` to patch tokens."""
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class ViTAddClassToken(nn.Module):
    """Prepend a learnable class token to patch embeddings."""

    def __init__(self, embed_dim: int):
        """Initialize the class token parameter.

        Args:
            embed_dim (int): Embedding dimension of each token.
        """
        super().__init__()
        cls_token = torch.zeros(1, 1, embed_dim)
        self.cls_token = nn.Parameter(cls_token)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Prepend the class token to each batch item."""
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.concat([cls_token, x], dim=1)
        return x


class ViTPositionalEmbedding(nn.Module):
    """Learnable ViT positional embeddings with optional class-token support."""

    def __init__(
        self,
        embed_dim: int,
        num_patches: int,
        use_cls_token: bool = True,
    ):
        """Initialize learnable positional embeddings.

        Args:
            embed_dim (int): Embedding dimension of each token.
            num_patches (int): Number of image patch tokens.
            use_cls_token (bool, default: True): Whether the sequence includes a leading
                class token.
        """
        super().__init__()
        self.use_cls_token = use_cls_token

        num_tokens = num_patches + int(use_cls_token)
        pos_embed = torch.zeros(1, num_tokens, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional embeddings to an input token sequence."""
        if x.size(1) != self.pos_embed.size(1):
            raise AssertionError(
                f'Expected sequence length {self.pos_embed.size(1)}, but got {x.size(1)}.'
            )
        return x + self.pos_embed

    def interpolate(
        self,
        old_grid_size: tuple[int, int],
        new_grid_size: tuple[int, int],
    ) -> Tensor:
        """Resize patch positional embeddings to a new image grid.

        Args:
            old_grid_size (tuple[int, int]): Original patch grid as `(height, width)`.
            new_grid_size (tuple[int, int]): Target patch grid as `(height, width)`.

        Returns:
            Tensor: Positional embeddings resized to the target grid. The class token
            embedding is preserved when `use_cls_token` is `True`.
        """
        if self.use_cls_token:
            cls_pos_embed = self.pos_embed[:, :1]
            patch_pos_embed = self.pos_embed[:, 1:]
        else:
            cls_pos_embed = None
            patch_pos_embed = self.pos_embed

        old_h, old_w = old_grid_size
        new_h, new_w = new_grid_size
        embed_dim = self.pos_embed.size(-1)

        if patch_pos_embed.size(1) != old_h * old_w:
            raise AssertionError(
                f'Expected old grid with {patch_pos_embed.size(1)} patches, '
                f'but got {old_h * old_w}.'
            )

        patch_pos_embed = patch_pos_embed.reshape(1, old_h, old_w, embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        patch_pos_embed = patch_pos_embed.reshape(1, new_h * new_w, embed_dim)

        if cls_pos_embed is not None:
            return torch.concat([cls_pos_embed, patch_pos_embed], dim=1)

        return patch_pos_embed


class ViTEmbedding(nn.Module):
    """ViT input embedding that combines patch, class, and positional embeddings."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0,
    ):
        """Initialize the complete ViT embedding stem.

        Args:
            image_size (int, default: 224): Height and width of the square input image.
            patch_size (int, default: 16): Height and width of each square patch.
            in_channels (int, default: 3): Number of input image channels.
            embed_dim (int, default: 768): Output embedding dimension for each token.
            dropout (float, default: 0.0): Dropout probability applied after embeddings.
        """
        super().__init__()
        self.patch_embed = ViTConvPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.add_cls_token = ViTAddClassToken(embed_dim)
        self.pos_embed = ViTPositionalEmbedding(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim,
            use_cls_token=True,
        )
        self.dropout = dnn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Convert images into class-token-prefixed ViT embeddings."""
        x = self.patch_embed(x)
        x = self.add_cls_token(x)
        x = self.pos_embed(x)
        x = self.dropout(x)
        return x

    def interpolate_pos_embedding(
        self,
        old_grid_size: tuple[int, int],
        new_grid_size: tuple[int, int],
    ) -> Tensor:
        """Resize patch positional embeddings to a new image grid.

        Args:
            old_grid_size (tuple[int, int]): Original patch grid as `(height, width)`.
            new_grid_size (tuple[int, int]): Target patch grid as `(height, width)`.

        Returns:
            Tensor: Positional embeddings resized to the target grid. The class token
            embedding is preserved when `use_cls_token` is `True`.
        """
        return self.pos_embed.interpolate(old_grid_size, new_grid_size)
