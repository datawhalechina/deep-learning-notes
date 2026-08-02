import torch
from torch.testing import assert_close

import dnnlpy.models.seq2seq as seq


def _make_model() -> seq.Seq2SeqTransformer:
    return seq.Seq2SeqTransformer(
        src_vocab_size=11,
        tgt_vocab_size=13,
        d_model=8,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=16,
        dropout=0.0,
        max_len=6,
    )


def test_seq2seq_transformer_returns_vocab_logits_and_supports_backward():
    model = _make_model()
    src = torch.randint(0, 11, (2, 4))
    tgt = torch.randint(0, 13, (2, 5))

    logits = model(src, tgt)
    logits.mean().backward()

    assert logits.shape == (2, 5, 13)
    assert model.src_embedding.weight.grad is not None
    assert model.tgt_embedding.weight.grad is not None
    assert model.output_proj.weight.grad is not None


def test_seq2seq_causal_mask_hides_future_target_tokens():
    model = _make_model()
    src = torch.tensor([[1, 2, 3, 4]])
    tgt = torch.tensor([[1, 2, 3, 4, 5]])
    changed_future = tgt.clone()
    changed_future[:, -1] = 6
    tgt_mask = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    actual = model(src, tgt, tgt_mask=tgt_mask)
    changed = model(src, changed_future, tgt_mask=tgt_mask)

    assert_close(actual[:, :-1], changed[:, :-1])


def test_seq2seq_accepts_padding_masks_for_source_and_target():
    model = _make_model()
    src = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    tgt = torch.tensor([[1, 2, 3], [4, 5, 0]])
    src_padding = src == 0
    tgt_padding = tgt == 0

    output = model(
        src,
        tgt,
        src_key_padding_mask=src_padding,
        tgt_key_padding_mask=tgt_padding,
        memory_key_padding_mask=src_padding,
    )

    assert output.shape == (2, 3, 13)
    assert torch.isfinite(output).all()
