import math

import torch.nn as nn
from torch import Tensor

import dnnlpy.nn as dnn

__all__ = ['Seq2SeqTransformer']


class Seq2SeqTransformer(nn.Module):
    """A token-level sequence-to-sequence Transformer with positional encodings."""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """Initialize embeddings, Transformer blocks, and output projection.

        Args:
            src_vocab_size (int): Number of source vocabulary entries.
            tgt_vocab_size (int): Number of target vocabulary entries.
            d_model (int, default: 512): Token embedding and Transformer hidden size.
            num_heads (int, default: 8): Number of attention heads.
            num_encoder_layers (int, default: 6): Number of encoder layers.
            num_decoder_layers (int, default: 6): Number of decoder layers.
            dim_feedforward (int, default: 2048): Feed-forward hidden dimension.
            dropout (float, default: 0.1): Dropout probability inside the Transformer.
            max_len (int, default: 5000): Maximum supported sequence length.
        """
        super().__init__()
        self.d_model = d_model
        self.src_embedding = dnn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = dnn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = dnn.SinusoidalPositionalEncoding(d_model, max_len)

        self.transformer = dnn.Transformer(
            d_model=d_model,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=True,  # Pre-LN Transformer
        )
        self.output_proj = dnn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return target-token logits for a batch of source and target token ids.

        Args:
            src (Tensor): Source token ids of shape (batch_size, src_seq_len).
            tgt (Tensor): Target token ids of shape (batch_size, tgt_seq_len).
            src_mask (Tensor | None, optional): Source attention mask of shape
                (src_seq_len, src_seq_len). Defaults to None.
            tgt_mask (Tensor | None, optional): Target attention mask of shape
                (tgt_seq_len, tgt_seq_len). Defaults to None.
            src_key_padding_mask (Tensor | None, optional): Source key padding mask
                of shape (batch_size, src_seq_len). Defaults to None.
            tgt_key_padding_mask (Tensor | None, optional): Target key padding mask
                of shape (batch_size, tgt_seq_len). Defaults to None.
            memory_key_padding_mask (Tensor | None, optional): Memory key padding mask
                of shape (batch_size, src_seq_len). Defaults to None.

        Returns:
            Tensor: Target token logits of shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        # We scale the embeddings by sqrt(d_model) to maintain the variance
        # of the input to the Transformer.
        scale = math.sqrt(self.d_model)
        src_emb = self.src_embedding(src) * scale
        tgt_emb = self.tgt_embedding(tgt) * scale

        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)

        hidden_states = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_proj(hidden_states)
        return logits
