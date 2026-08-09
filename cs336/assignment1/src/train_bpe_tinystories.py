# TODO: Replace cProfile with profiling in Python 3.15. See <PEP 799> for details.
# TODO: Add lazy import in Python 3.15. See <PEP 810> for details.
import argparse
import cProfile
import pstats
import time
from collections.abc import Iterator

import datasets as ds
import dnnlpy.tokenizers as dltk
import psutil

__all__ = [
    'profile_train_bpe_tinystories',
    'train_bpe_tinystories',
]

NUM_STORIES = 2119719
VOCAB_SIZE = 10000


def _batch_iterator(dataset: ds.Dataset, batch_size: int = 1024) -> Iterator[list[str]]:
    """Yield batches of story strings to train_from_iterator."""
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        yield dataset[start:end]['text']


def train_bpe_tinystories() -> dltk.Tokenizer:
    """Train a BPE tokenizer on the TinyStories dataset."""
    dataset = ds.load_dataset('roneneldan/TinyStories', split='train', num_proc=2)
    assert len(dataset) == NUM_STORIES

    tokenizer = dltk.Tokenizer(
        dltk.BPE(unk_token='<unk>'),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(add_prefix_space=False),
        decoder=dltk.ByteLevelDecoder(),
    )
    text_iterator = _batch_iterator(dataset)

    print('Training tokenizer on TinyStories...')
    start = time.perf_counter()
    tokenizer.train_from_iterator(
        text_iterator,
        vocab_size=VOCAB_SIZE,
        initial_alphabet=dltk.ByteLevelPreTokenizer.alphabet(),
    )
    end = time.perf_counter()
    print(f'Training completed in {end - start:.4f} seconds.')
    print('[NOTE] This should be less than 30 minutes for the TinyStories dataset.')
    print(f'Tokenizer vocabulary size: {tokenizer.get_vocab_size()}.')
    print('LRU cache info:', dltk.utils.bytes_to_unicode.cache_info())

    process = psutil.Process()
    peak_mem = process.memory_info().peak_wset
    print(f'Peak memory usage: {peak_mem / 1024**3:.4f} GB.')
    print('[NOTE] This should be less than 30 GB for the TinyStories dataset.')

    file_name = 'bpe_tinystories.json'
    tokenizer.save(file_name)
    print(f'Tokenizer saved to {file_name}.')

    return tokenizer


def profile_train_bpe_tinystories():
    """Profile the training of the BPE tokenizer on the TinyStories dataset."""
    with cProfile.Profile() as profiler:
        train_bpe_tinystories()

    print('\nProfiling results:')
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumtime')
    stats.print_stats(10)  # Print top 10 functions by cumulative time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a BPE tokenizer on the TinyStories dataset.'
    )
    parser.add_argument(
        '--profile', action='store_true', help='Profile the training process.'
    )
    args = parser.parse_args()

    if args.profile:
        print('Profiling enabled. Training may take longer.')
        profile_train_bpe_tinystories()
    else:
        train_bpe_tinystories()
