import torch.nn as nn
from torch import Tensor

import dnnlpy.nn as dnn

from .embedding import ViTEmbedding

__all__ = [
    'ViTClassificationHead',
    'ViTEncoder',
    'ViTEncoderLayer',
    'ViTForImageClassification',
    'ViTMLP',
    'ViTModel',
]


class ViTMLP(nn.Module):
    """Feed-forward MLP block used inside a Vision Transformer encoder layer."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
    ):
        """Initialize the two-layer MLP block.

        Args:
            embed_dim (int): Input and output token embedding dimension.
            hidden_dim (int, default: 3072): Hidden dimension of the feed-forward layer.
                Defaults to `4 * embed_dim`.
            dropout (float, default: 0.0): Dropout probability after each linear projection.
        """
        super().__init__()
        self.net = nn.Sequential(
            dnn.Linear(embed_dim, hidden_dim),
            dnn.GELU(),
            dnn.Dropout(dropout),
            dnn.Linear(hidden_dim, embed_dim),
            dnn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MLP to each token in the input sequence."""
        return self.net(x)


class ViTEncoderLayer(nn.Module):
    """Pre-LayerNorm Vision Transformer encoder layer."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize a ViT encoder layer.

        Args:
            embed_dim (int): Token embedding dimension.
            num_heads (int, default: 12): Number of attention heads.
            hidden_dim (int, default: 3072): Hidden dimension of the feed-forward layer.
                Defaults to `4 * embed_dim`.
            dropout (float, default: 0.0): Dropout probability for residual paths and
                the feed-forward block.
            attn_dropout (float, default: 0.0): Dropout probability inside multi-head
                self-attention.
        """
        super().__init__()
        self.norm1 = dnn.LayerNorm(embed_dim)
        self.attn = dnn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
        )
        self.dropout1 = dnn.Dropout(dropout)
        self.norm2 = dnn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return encoded token representations for a batch of ViT tokens."""
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=False,
        )
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Stack of Vision Transformer encoder layers with a final layer norm."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize a ViT encoder stack.

        Args:
            embed_dim (int): Token embedding dimension.
            num_heads (int, default: 12): Number of attention heads per layer.
            num_layers (int, default: 12): Number of encoder layers.
            hidden_dim (int, default: 3072): Hidden dimension of each feed-forward layer.
                Defaults to `4 * embed_dim`.
            dropout (float, default: 0.0): Dropout probability for residual paths and
                feed-forward blocks.
            attn_dropout (float, default: 0.0): Dropout probability inside multi-head
                self-attention.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ViTEncoderLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = dnn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a sequence of ViT tokens."""
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class ViTModel(nn.Module):
    """Vision Transformer backbone without a task-specific prediction head."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize a ViT backbone.

        Args:
            image_size (int, default: 224): Height and width of the square input image.
            patch_size (int, default: 16): Height and width of each square patch.
            in_channels (int, default: 3): Number of input image channels.
            embed_dim (int, default: 768): Token embedding dimension.
            num_heads (int, default: 12): Number of attention heads per layer.
            num_layers (int, default: 12): Number of encoder layers.
            hidden_dim (int, default: 3072): Hidden dimension of each feed-forward layer.
                Defaults to `4 * embed_dim`.
            dropout (float, default: 0.0): Dropout probability for residual paths and
                feed-forward blocks.
            attn_dropout (float, default: 0.0): Dropout probability inside multi-head
                self-attention.
        """
        super().__init__()
        self.embedding = ViTEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.encoder = ViTEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return class and patch token representations for a batch of images."""
        x = self.embedding(x)
        x = self.encoder(x)
        return x


class ViTClassificationHead(nn.Module):
    """Classification head that predicts logits from the class token."""

    def __init__(self, embed_dim: int, num_classes: int):
        """Initialize the linear classification head.

        Args:
            embed_dim (int): Class-token embedding dimension.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.head = dnn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits from the first token of each sequence."""
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return logits


class ViTForImageClassification(nn.Module):
    """Vision Transformer with a class-token image classification head."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        hidden_dim: int = 3072,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        """Initialize a Vision Transformer classifier.

        Args:
            image_size (int, default: 224): Height and width of the square input image.
            patch_size (int, default: 16): Height and width of each square patch.
            in_channels (int, default: 3): Number of input image channels.
            num_classes (int, default: 1000): Number of output classes.
            embed_dim (int, default: 768): Token embedding dimension.
            num_heads (int, default: 12): Number of attention heads per layer.
            num_layers (int, default: 12): Number of encoder layers.
            hidden_dim (int, default: 3072): Hidden dimension of each feed-forward layer.
                Defaults to `4 * embed_dim`.
            dropout (float, default: 0.0): Dropout probability in embeddings, residual paths,
                and feed-forward blocks.
            attn_dropout (float, default: 0.0): Dropout probability inside multi-head
                self-attention.
        """
        super().__init__()
        self.backbone = ViTModel(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.head = ViTClassificationHead(embed_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Return image-class logits for a batch of images."""
        x = self.backbone(x)
        logits = self.head(x)
        return logits
