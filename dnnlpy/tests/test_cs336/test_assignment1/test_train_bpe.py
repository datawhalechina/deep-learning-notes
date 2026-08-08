import json
import time
from typing import cast

from .adapters import run_train_bpe
from .conftest import FIXTURES_PATH, SNAPSHOT_PATH

__all__ = [
    'test_train_bpe',
    'test_train_bpe_special_tokens',
    'test_train_bpe_speed',
]


def test_train_bpe_speed():
    """Ensure that BPE training is relatively efficient by measuring training time on
    this small dataset and throwing an error if it takes more than 1.5 seconds. This is
    a pretty generous upper-bound, it takes 0.38 seconds with the  reference implementation
    on my laptop. In contrast, the toy implementation takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / 'corpus.en.txt'

    start_time = time.perf_counter()
    run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=['<|endoftext|>'],
        num_workers=1,
    )
    end_time = time.perf_counter()

    assert end_time - start_time < 1.5


def test_train_bpe():
    """Ensure that the BPE training produces the expected vocab and merges on a small
    dataset. This test uses a small dataset and compares the learned vocab and merges
    to a reference implementation. The reference vocab and merges were generated using
    the reference implementation of BPE training from the Hugging Face tokenizers library.
    The reference vocab and merges are stored in the `tests/fixtures` directory.
    """
    input_path = FIXTURES_PATH / 'corpus.en.txt'
    tokenizer = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=['<|endoftext|>'],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / 'train-bpe-reference-vocab.json'
    reference_merges_path = FIXTURES_PATH / 'train-bpe-reference-merges.txt'

    # Compare the learned merges to the expected output merges
    with open(reference_merges_path) as fp:
        reference_merges = [tuple(line.split()) for line in fp]
        reference_merges = cast(list[tuple[str, str]], reference_merges)

    assert tokenizer.model.merges is not None
    assert tokenizer.model.merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as fp:
        reference_vocab = json.load(fp)
        reference_vocab = cast(dict[str, int], reference_vocab)

    assert tokenizer.model.vocab == reference_vocab


def test_train_bpe_special_tokens():
    """Ensure that the special tokens are added to the vocabulary and not merged with
    other tokens. This is important because the special tokens are used to indicate the
    end of a text sequence, and should not be split into multiple tokens.
    """
    input_path = FIXTURES_PATH / 'tinystories_sample_5M.txt'
    tokenizer = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=['<|endoftext|>'],
    )

    # Check that the special token is not in the vocab
    for word in tokenizer.vocab.values():
        assert word != '<|' and word != '|>'

    with open(SNAPSHOT_PATH / 'test_train_bpe_vocab.json') as fp:
        vocab = json.load(fp)
        vocab = cast(dict[str, int], vocab)

    with open(SNAPSHOT_PATH / 'test_train_bpe_merges.txt') as fp:
        merges = [tuple(line.split()) for line in fp]
        merges = cast(list[tuple[str, str]], merges)

    assert vocab == tokenizer.model.vocab
    assert merges == tokenizer.model.merges
