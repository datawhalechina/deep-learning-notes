from __future__ import annotations

import heapq
import itertools as it
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, override

import regex as re

from .base import Tokenizer, Trainer

if TYPE_CHECKING:
    from .model import BPE

type Pair = tuple[str, str]
type WordSymbols = tuple[str, ...]
type PairIndices = dict[Pair, set[int]]
type PairFreqBuckets = dict[int, set[Pair]]

__all__ = ['BPETrainer']


class BPETrainer(Trainer):
    """Internal trainer containing the complete BPE training pipeline."""

    def __init__(
        self,
        model: BPE,
        tokenizer: Tokenizer,
        vocab_size: int,
        min_frequency: int = 2,
        initial_alphabet: list[str] | None = None,
    ):
        """Initialize the BPETrainer with the given parameters.

        Args:
            model (BPE): The BPE model to train.
            tokenizer (Tokenizer): The tokenizer instance to use for training.
            vocab_size (int): The maximum size of the vocabulary to build.
            min_frequency (int, default: 2): The minimum frequency for a token to be
                included in the vocabulary.
            initial_alphabet (list[str] | None, optional): Characters to include in the
                initial alphabet, even when they are not present in the training corpus.
                If an entry contains multiple characters, only the first one is used.

        Raises:
            AssertionError: If `vocab_size` is less than 1 or `min_frequency` is negative.
        """
        if vocab_size < 1:
            raise AssertionError('`vocab_size` must be at least 1.')
        if min_frequency < 0:
            raise AssertionError('`min_frequency` must be non-negative.')

        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.num_workers = tokenizer.num_workers
        self.special_tokens = list(dict.fromkeys(tokenizer.special_tokens))
        self.initial_alphabet = self._prepare_initial_alphabet(initial_alphabet)

    def _prepare_initial_alphabet(
        self,
        initial_alphabet: list[str] | None = None,
    ) -> list[str]:
        """Prepare unique one-character alphabet tokens for initial vocabulary.

        Args:
            initial_alphabet (list[str] | None): A list of characters to include in the
                initial alphabet, even if they are not present in the training texts.

        Returns:
            alphabet (list[str]): A list of unique one-character tokens to include in the
                initial vocabulary. If an entry has multiple characters, only the first
                character is used.
        """
        if initial_alphabet is None:
            return []

        alphabet = []
        seen_alphabet = set()
        for token in initial_alphabet:
            if not token:
                continue

            char = token[0]
            if char not in seen_alphabet:
                alphabet.append(char)
                seen_alphabet.add(char)

        return alphabet

    @override
    def train(self, texts: Iterable[str | Iterable[str]]) -> None:
        """Train the BPE model from an iterator of input texts.

        The whole training process is as follows:

        1. Flatten the input texts into a single iterator of strings.
        2. Collect word counts from the input texts, normalizing and pre-tokenizing them.
        3. Initialize the vocabulary with special tokens and the byte-level alphabet.
        4. Count adjacent symbol pairs, then incrementally update the counts while merging
        the most frequent pair until the desired vocabulary size is reached or no more
        eligible pairs are found.
        5. Save the trained vocabulary and merges to the model and tokenizer.

        Args:
            inputs (Iterable[str | Iterable[str]]): An iterable of input texts or
                pre-tokenized sequences.

        Raises:
            ValueError: If `vocab_size` is less than 1 or `min_frequency` is negative.
        """
        texts = self._iter_texts(texts)

        # word_counts: word -> frequency.
        word_counts = self._count_pre_tokens(texts)

        # sorted_word_counts: list of (word, freq) tuples sorted by word.
        sorted_word_counts = sorted(word_counts.items())

        # word_symbols: list of tuples of symbols for each word.
        # e.g. word_symbols = [('l', 'o', 'w'), ('l', 'o', 'w', 'e', 'r')]
        word_symbols = [tuple(word) for word, _ in sorted_word_counts]

        # word_freqs: list of frequencies corresponding to each word in `word_symbols`.
        # e.g. word_freqs = [5, 2] means 'low' appears 5 times and 'lower' appears 2 times.
        word_freqs = [freq for _, freq in sorted_word_counts]

        # vocab_tokens: list of vocabulary tokens, starting with special tokens and followed
        # by unique characters from the input texts.
        vocab_tokens, known_tokens = self._init_vocab(word_counts)

        # pair_counts: word pair -> frequency.
        # pair_word_indices: mapping from each pair to the indices of words that contain that pair.
        # This allows us to efficiently update pair counts when a pair is merged in a word.
        pair_counts, pair_indices = self._init_pair_counts(word_symbols, word_freqs)

        # pair_freq_buckets: frequency -> set of pairs with that frequency.
        # Basically, it is the reverse mapping of `pair_counts`.
        # freq_heap: max-heap of frequencies, allowing efficient retrieval of the highest frequency.
        pair_freq_buckets, freq_heap = self._init_pair_freq_index(pair_counts)

        merges = []

        while len(vocab_tokens) < self.vocab_size:
            best_pair = self._select_best_pair(
                pair_counts,
                pair_freq_buckets,
                freq_heap,
            )

            # If no eligible pair is found, we have reached the end of the training process.
            if best_pair is None:
                break

            new_token = ''.join(best_pair)

            if new_token not in known_tokens:
                vocab_tokens.append(new_token)
                known_tokens.add(new_token)

            merges.append(best_pair)

            # affected_word_indices: indices of words that contain the best pair to be merged.
            affected_word_indices = tuple(pair_indices.pop(best_pair))
            for word_index in affected_word_indices:
                self._merge_pair_and_update_pair_counts(
                    word_index,
                    best_pair,
                    word_symbols,
                    word_freqs,
                    pair_counts,
                    pair_indices,
                    pair_freq_buckets,
                    freq_heap,
                )

        self._save_result(vocab_tokens, merges)

    def _iter_texts(self, inputs: Iterable[str | Iterable[str]]) -> Iterator[str]:
        """Flatten inputs and yield pieces outside registered special tokens."""
        special_tokens = sorted(
            (token for token in self.special_tokens if token),
            key=len,
            reverse=True,
        )
        pattern = (
            re.compile('|'.join(re.escape(token) for token in special_tokens))
            if special_tokens
            else None
        )

        for item in inputs:
            texts = (item,) if isinstance(item, str) else item
            for text in texts:
                if pattern is None:
                    yield text
                    continue

                start = 0
                for match in pattern.finditer(text):
                    if start < match.start():
                        yield text[start : match.start()]
                    start = match.end()

                if start < len(text):
                    yield text[start:]

    def _count_pre_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count how often each pre-token appears in the input texts. The texts are
        pre-tokenized in batches, and the resulting frequency counts are combined
        into a single Counter.

        The registered pre-tokenizer may count tokens directly so training does not
        need to materialize offsets or a complete encoded-token list for every text.

        Args:
            texts (Iterable[str]): An iterable of input texts to process.

        Returns:
            counter (Counter[str]): A counter mapping each pre-tokenized word to its
                frequency in the input texts.

        Examples:
            >>> texts = ['Hello world!', 'Hello again.']
            >>> _count_pre_tokens(texts)
            Counter({'Hello': 2, 'world!': 1, 'again.': 1})
        """
        pre_tokens = map(self.tokenizer._normalize, texts)
        return self.tokenizer._count_pre_tokens(pre_tokens)

    def _init_vocab(self, word_counts: Counter[str]) -> tuple[list[str], set[str]]:
        """Initialize the vocabulary with special tokens and the byte-level alphabet.

        Args:
            word_counts (Counter[str]): A counter mapping each pre-tokenized word to
                its frequency in the input texts.

        Returns:
            vocab_tokens (list[str]): A list of vocabulary tokens, starting with special
                tokens and followed by unique characters from the input texts.
            known_tokens (set[str]): A set of known tokens for O(1) membership checks.
        """
        alphabet = {char for word in word_counts for char in word}
        alphabet = alphabet.union(self.initial_alphabet)
        alphabet = sorted(alphabet)

        vocab_tokens = list(self.special_tokens)
        known_tokens = set(vocab_tokens)

        for token in alphabet:
            if token not in known_tokens:  # O(1) lookup in a set
                vocab_tokens.append(token)
                known_tokens.add(token)

        return vocab_tokens, known_tokens

    def _init_pair_counts(
        self,
        word_symbols: list[WordSymbols],
        word_freqs: list[int],
    ) -> tuple[Counter[Pair], PairIndices]:
        """Count pairs once and index the words containing each pair.

        Args:
            word_symbols (list[WordSymbols]): A list of tuples, where each tuple contains
                the symbols of a word.
            word_freqs (list[int]): A list of frequencies corresponding to each word in
                `word_symbols`.

        Returns:
            pair_counts (Counter[Pair]): A counter mapping each adjacent symbol pair to
                its frequency across all words.
            pair_word_indices (PairIndices): A dictionary mapping each adjacent symbol
                pair to a set of indices of words in `word_symbols` that contain that pair.

        Examples:
            >>> word_symbols = [('l', 'o', 'w'), ('l', 'o', 'w', 'e', 'r')]
            >>> word_freqs = [5, 2]
            >>> _init_pair_counts(word_symbols, word_freqs)
            pair_counts = Counter({
                ('l', 'o'): 7,
                ('o', 'w'): 7,
                ('w', 'e'): 2,
                ('e', 'r'): 2,
            })
            pair_word_indices = {
                ('l', 'o'): {0, 1},
                ('o', 'w'): {0, 1},
                ('w', 'e'): {0},
                ('e', 'r'): {0},
            }
        """
        # Count the frequency of each adjacent symbol pair across all words.
        pair_counts = Counter()
        pair_word_indices = defaultdict(set)

        z = zip(word_symbols, word_freqs, strict=True)
        for index, (symbols, freq) in enumerate(z):
            # We need to remove duplicates because a pair can occur multiple times
            # in a word, but we only want to count the word index once for each pair.
            word_pairs = set()

            for pair in it.pairwise(symbols):
                pair_counts[pair] += freq
                word_pairs.add(pair)

            # Mapping from each pair to the indices of words that contain that pair.
            for pair in word_pairs:
                pair_word_indices[pair].add(index)

        return pair_counts, dict(pair_word_indices)

    def _init_pair_freq_index(
        self,
        pair_counts: Counter[Pair],
    ) -> tuple[PairFreqBuckets, list[int]]:
        """Group pairs by frequency and build a max-frequency heap.

        Args:
            pair_counts (Counter[Pair]): A counter mapping each adjacent symbol pair to
                its frequency across all words.

        Returns:
            pair_freq_buckets (PairFreqBuckets): A dictionary mapping each frequency to a
                set of adjacent symbol pairs that have that frequency.
            freq_heap (list[int]): A max-heap of frequencies, allowing efficient retrieval
                of the highest frequency.

        Examples:
            >>> pair_counts = Counter({
                ('l', 'o'): 7,
                ('o', 'w'): 7,
                ('w', 'e'): 2,
                ('e', 'r'): 2,
            })
            >>> _init_pair_freq_index(pair_counts)
            pair_freq_buckets = {
                7: {('l', 'o'), ('o', 'w')},
                2: {('w', 'e'), ('e', 'r')},
            }
            freq_heap = [7, 2]  # Max-heap of frequencies
        """
        pair_freq_buckets = defaultdict(set)

        for pair, count in pair_counts.items():
            pair_freq_buckets[count].add(pair)

        if sys.version_info >= (3, 14):
            freq_heap = [count for count in pair_freq_buckets]
            heapq.heapify_max(freq_heap)
        else:
            freq_heap = [-count for count in pair_freq_buckets]
            heapq.heapify(freq_heap)

        return dict(pair_freq_buckets), freq_heap

    def _select_best_pair(
        self,
        pair_counts: Counter[Pair],
        pair_freq_buckets: PairFreqBuckets | None = None,
        freq_heap: list[int] | None = None,
    ) -> Pair | None:
        """Select the most frequent adjacent symbol pair that meets the minimum frequency
        requirement.

        Args:
            pair_counts (Counter[Pair]): A counter mapping adjacent symbol pairs to their
                frequencies.
            pair_freq_buckets (PairFreqBuckets | None, optional): A dictionary mapping
                frequencies to sets of adjacent symbol pairs with that frequency. If None,
                the function will compute the most frequent pair directly from `pair_counts`.
            freq_heap (list[int] | None, optional): A max-heap of frequencies, allowing
                efficient retrieval of the highest frequency. If None, the function will
                compute the most frequent pair directly from `pair_counts`.

        Returns:
            Pair | None: The most frequent adjacent symbol pair that meets the minimum frequency requirement, or None if no such pair exists.

        Examples:
            >>> pair_counts = Counter({
                ('l', 'o'): 7,
                ('o', 'w'): 7,
                ('w', 'e'): 2,
                ('e', 'r'): 2,
            })
            >>> _select_best_pair(pair_counts)
            ('l', 'o')
            >>> _select_best_pair(pair_counts, pair_freq_buckets, freq_heap)
            ('l', 'o')
        """
        if pair_freq_buckets is None or freq_heap is None:
            eligible_pairs = (
                (count, pair)
                for pair, count in pair_counts.items()
                if count >= self.min_frequency
            )
            return max(eligible_pairs, default=(0, None))[1]

        while freq_heap:
            if sys.version_info >= (3, 14):
                best_count = freq_heap[0]
                pairs = pair_freq_buckets.get(best_count)  # lazy delete check
                if pairs is None:
                    heapq.heappop_max(freq_heap)
                    continue
            else:
                best_count = -freq_heap[0]
                pairs = pair_freq_buckets.get(best_count)  # lazy delete check
                if pairs is None:
                    heapq.heappop(freq_heap)
                    continue

            if best_count < self.min_frequency:
                return None

            return max(pairs)

        return None

    def _merge_pair_and_update_pair_counts(
        self,
        index: int,
        pair: Pair,
        word_symbols: list[WordSymbols],
        word_freqs: list[int],
        pair_counts: Counter[Pair],
        pair_indices: PairIndices,
        pair_freq_buckets: PairFreqBuckets,
        freq_heap: list[int],
    ) -> None:
        """Merge a pair in one word and update only the pair counts that changed."""
        old_symbols = word_symbols[index]
        new_symbols = self._merge_pair(old_symbols, pair)

        # compares how many times the same pair appears in the old word and the new word.
        freq = word_freqs[index]
        old_pair_occ = Counter(it.pairwise(old_symbols))
        new_pair_occ = Counter(it.pairwise(new_symbols))

        for changed_pair in old_pair_occ.keys() | new_pair_occ.keys():
            old_occ = old_pair_occ[changed_pair]
            new_occ = new_pair_occ[changed_pair]
            delta = new_occ - old_occ

            if delta:  # only update counts if the pair occurrence changed
                self._update_pair_count(
                    changed_pair,
                    delta * freq,
                    pair_counts,
                    pair_freq_buckets,
                    freq_heap,
                )

            if new_occ:
                new_set = pair_indices.setdefault(changed_pair, set())
                new_set.add(index)
            else:
                indices = pair_indices.get(changed_pair)
                if indices is not None:
                    indices.discard(index)
                    if not indices:
                        del pair_indices[changed_pair]

        word_symbols[index] = new_symbols

    def _merge_pair(self, symbols: WordSymbols, pair: Pair) -> WordSymbols:
        """Merge the first occurrence of a symbol pair in a tuple of symbols.

        Args:
            symbols (WordSymbols): A tuple of symbols to merge.
            pair (Pair): A pair of symbols to merge.

        Returns:
            WordSymbols: A new tuple of symbols with the first occurrence of the pair merged.

        Examples:
            >>> symbols = ('l', 'o', 'w', 'e', 'r')
            >>> pair = ('l', 'o')
            >>> _merge_pair(symbols, pair)
            ('lo', 'w', 'e', 'r')
            >>> symbols = ('l', 'o', 'w', 'e', 'r')
            >>> pair = ('o', 'w')
            >>> _merge_pair(symbols, pair)
            ('l', 'ow', 'e', 'r')
        """
        index = 0
        merged = []

        while index < len(symbols):
            if index < len(symbols) - 1 and symbols[index : index + 2] == pair:
                merged.append(''.join(pair))
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        return tuple(merged)

    def _update_pair_count(
        self,
        pair: Pair,
        delta: int,
        pair_counts: Counter[Pair],
        pair_freq_buckets: PairFreqBuckets,
        freq_heap: list[int],
    ) -> None:
        """Update a pair count and its frequency-bucket membership.

        Args:
            pair (Pair): The adjacent symbol pair to update.
            delta (int): The change in frequency for the pair. Can be positive or negative.
            pair_counts (Counter[Pair]): A counter mapping adjacent symbol pairs to their
                frequencies.
            pair_freq_buckets (PairFreqBuckets): A dictionary mapping frequencies to sets
                of adjacent symbol pairs with that frequency.
            freq_heap (list[int]): A max-heap of frequencies, allowing efficient retrieval
                of the highest frequency.
        """
        # Get the old count of the pair, and update the frequency buckets accordingly.
        old_count = pair_counts[pair]

        # Only update the frequency buckets if the old count is non-zero, since we don't
        # want to remove a pair that doesn't exist. (Counter returns 0 for missing keys).
        if old_count > 0:
            old_bucket = pair_freq_buckets[old_count]
            old_bucket.remove(pair)  # Remove the pair from its old frequency bucket.

            # If the old bucket is now empty, remove it from the frequency buckets.
            if not old_bucket:
                del pair_freq_buckets[old_count]

        new_count = old_count + delta
        if new_count == 0:
            pair_counts.pop(pair, None)
            return

        pair_counts[pair] = new_count
        new_bucket = pair_freq_buckets.get(new_count)

        # If the new bucket does not exist, create it and add the new count to the frequency heap.
        if new_bucket is None:
            new_bucket = set()
            pair_freq_buckets[new_count] = new_bucket

            if sys.version_info >= (3, 14):
                heapq.heappush_max(freq_heap, new_count)
            else:
                heapq.heappush(freq_heap, -new_count)

        new_bucket.add(pair)

    def _save_result(self, vocab_tokens: list[str], merges: list[Pair]) -> None:
        """Save the trained vocabulary and merges to the model and tokenizer."""
        self.model.merges = list(merges)
        self.model._refresh_merge_ranks()

        vocab = {}
        used_ids = set()
        for token in self.special_tokens:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is None:
                raise KeyError(f'Special token {token!r} is not in the vocabulary.')
            if token_id in used_ids:
                raise ValueError(f'Duplicate special token ID: {token_id}.')

            vocab[token] = token_id
            used_ids.add(token_id)

        next_id = 0
        for token in vocab_tokens:
            if token in vocab:
                continue

            while next_id in used_ids:
                next_id += 1

            vocab[token] = next_id
            used_ids.add(next_id)
            next_id += 1

        self.tokenizer.vocab = vocab
        self.tokenizer.special_tokens = list(self.special_tokens)
