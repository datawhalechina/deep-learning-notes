import time
from collections.abc import Iterator

import datasets as ds
import pytest

import dnnlpy
import dnnlpy.tokenizers as tk

NUM_STORIES = 2119719
VOCAB_SIZE = 10000
INITIAL_ALPHABET = tk.ByteLevelPreTokenizer.alphabet()

if dnnlpy.has_gil():
    MAX_TRAINING_SECONDS = 45
else:
    MAX_TRAINING_SECONDS = 20

SAMPLE_TEXT = 'The little robot learned to read.'
SAMPLE_TEXTS = [
    'The little robot learned to read.',
    'Once upon a time, a curious fox found a golden key.',
    'Mia and Ben built a tiny boat together.',
]


def batch_iterator(dataset: ds.Dataset, batch_size: int = 1024) -> Iterator[list[str]]:
    """Yield batches of story strings to train_from_iterator."""
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield dataset[start:end]['text']


@pytest.fixture(scope='module')
def trained_tokenizer() -> tk.Tokenizer:
    """Train on cached TinyStories data and return the tokenizer and elapsed time."""
    try:
        dataset = ds.load_dataset('roneneldan/TinyStories', split='train')
    except (ConnectionError, FileNotFoundError) as err:
        pytest.skip(f'TinyStories is not available in the local cache: {err}.')

    assert len(dataset) == NUM_STORIES

    tokenizer = tk.Tokenizer(
        tk.BPE(),
        pre_tokenizer=tk.ByteLevelPreTokenizer(add_prefix_space=False),
        decoder=tk.ByteLevelDecoder(),
    )
    iterator = batch_iterator(dataset)

    start = time.perf_counter()
    tokenizer.train_from_iterator(
        iterator,
        vocab_size=VOCAB_SIZE,
        initial_alphabet=INITIAL_ALPHABET,
    )
    elapsed = time.perf_counter() - start

    setattr(tokenizer, 'training_time', elapsed)  # ruff: ignore[B010]
    return tokenizer


@pytest.mark.slow
def test_tinystories_training(trained_tokenizer: tk.Tokenizer) -> None:
    """Train a byte-level BPE tokenizer on TinyStories in under 20 seconds."""
    training_time = getattr(trained_tokenizer, 'training_time', None)

    assert training_time, 'Training time not recorded.'
    assert trained_tokenizer.get_vocab_size() == VOCAB_SIZE
    assert training_time < MAX_TRAINING_SECONDS, (
        f'Training took {training_time:.2f} seconds; '
        f'expected less than {MAX_TRAINING_SECONDS} seconds.'
    )


@pytest.mark.slow
def test_tinystories_encode_decode(trained_tokenizer: tk.Tokenizer) -> None:
    """Encode and decode one string with the trained tokenizer."""
    encoding = trained_tokenizer.encode(SAMPLE_TEXT)

    assert encoding.ids
    assert len(encoding.ids) == len(encoding.tokens) == len(encoding.offsets)
    assert trained_tokenizer.decode(encoding.ids) == SAMPLE_TEXT

    for token, token_id in zip(encoding.tokens, encoding.ids, strict=True):
        assert trained_tokenizer.token_to_id(token) == token_id
        assert trained_tokenizer.id_to_token(token_id) == token


@pytest.mark.slow
def test_tinystories_encode_decode_batch(trained_tokenizer: tk.Tokenizer) -> None:
    """Encode and decode a batch while preserving its contents and order."""
    encodings = trained_tokenizer.encode_batch(SAMPLE_TEXTS, batch_size=1)
    decoded = trained_tokenizer.decode_batch(
        [encoding.ids for encoding in encodings],
        batch_size=1,
    )

    assert len(encodings) == len(SAMPLE_TEXTS)
    assert all(encoding.ids for encoding in encodings)
    assert decoded == SAMPLE_TEXTS
