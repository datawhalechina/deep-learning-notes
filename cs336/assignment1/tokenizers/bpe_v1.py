import itertools as it
import time
from collections import Counter
from typing import Self, override

import datasets as ds
import dnnlpy.tokenizers as dltk
from common import ByteLevelDecoder, ByteLevelPreTokenizer

type Pair = tuple[str, str]
type WordSymbols = tuple[str, ...]

__all__ = [
    'BPETokenizerV1',
    'train_bpe_tinystories_v1',
]

NUM_STORIES = 10000
VOCAB_SIZE = 10000


class BPETokenizerV1(dltk.TraditionalTokenizer):
    """A naive byte-pair encoding tokenizer for teaching the BPE algorithm."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[Pair] | None = None,
        unk_token: str = '<unk>',
        special_tokens: list[str] | None = None,
    ):
        """Initialize the BPE tokenizer with vocabulary and merge rules.

        Args:
            vocab (dict[str, int], optional): A dictionary mapping tokens to their
                corresponding IDs.
            merges (list[Pair], optional): A list of token pairs to be merged.
            unk_token (str, default: '\\<unk>'): The token to use for unknown tokens.
            special_tokens (list[str], optional): A list of special tokens to be added to
                the vocabulary.
        """
        super().__init__(vocab, unk_token)
        self.merges = merges or []
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        self.pre_tokenizer = ByteLevelPreTokenizer()
        self.decoder = ByteLevelDecoder()

        if special_tokens is not None:
            self.add_special_tokens(special_tokens)

    @override
    def train(
        self,
        text: str | list[str],
        vocab_size: int = 100,
        min_frequency: int = 0,
        initial_alphabet: list[str] | None = None,
        unk_token: str = '<unk>',
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Train BPE by recounting every pair after every merge.

        Args:
            text (str | list[str]): The input text or a list of documents to train on.
            vocab_size (int, default: 100): The maximum size of the vocabulary.
            min_frequency (int, default: 0): The minimum frequency for a pair to be merged.
            initial_alphabet (list[str], optional): A list of initial alphabet characters
                to include in the vocabulary.
            unk_token (str, default: '\\<unk>'): The token to use for unknown tokens.
            special_tokens (list[str], optional): A list of special tokens to be added to
                the vocabulary.

        Returns:
            self (BPETokenizerV1): The trained tokenizer instance.

        Raises:
            AssertionError: If `vocab_size` is less than 1.
            AssertionError: If `min_frequency` is negative.

        Examples:
            >>> tokenizer = BPETokenizerV1()
            >>> tokenizer.train(['Hello world!'], vocab_size=10)
            >>> print(tokenizer)
            BPETokenizerV1(
                vocab_size=10,
                unk_token='<unk>',
                special_tokens=['<unk>'],
            )
        """
        if vocab_size < 1:
            raise AssertionError('`vocab_size` must be at least 1.')
        if min_frequency < 0:
            raise AssertionError('`min_frequency` must be non-negative.')

        if isinstance(text, str):
            text = [text]

        special_tokens = special_tokens or []
        special_tokens = list(dict.fromkeys([unk_token, *special_tokens]))
        word_counts = Counter()

        for story in text:
            for piece, is_special in self._split_special_tokens(story, special_tokens):
                if is_special:
                    continue

                for token in self.pre_tokenizer.pre_tokenize(piece):
                    word_counts[token] += 1

        sorted_word_counts = sorted(word_counts.items())
        word_symbols = [tuple(word) for word, _ in sorted_word_counts]
        word_freqs = [frequency for _, frequency in sorted_word_counts]

        alphabet = {char for word in word_counts for char in word}
        alphabet.update(self._prepare_initial_alphabet(initial_alphabet))

        vocab_tokens = list(special_tokens)
        for token in sorted(alphabet):
            if token not in vocab_tokens:
                vocab_tokens.append(token)

        merges = []
        while len(vocab_tokens) < vocab_size:
            pair_counts = self._count_pairs(word_symbols, word_freqs)
            best_pair = self._select_best_pair(pair_counts, min_frequency)
            if best_pair is None:
                break

            new_token = ''.join(best_pair)
            if new_token not in vocab_tokens:
                vocab_tokens.append(new_token)

            merges.append(best_pair)
            word_symbols = [
                self._merge_pair(symbols, best_pair) for symbols in word_symbols
            ]

        self.vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.unk_token = unk_token
        self.special_tokens = special_tokens

        return self

    @override
    def encode(self, text: str) -> list[int]:
        """Encode text into BPE token IDs without computing offsets.

        Args:
            text (str): The input text to be encoded.

        Returns:
            ids (list[int]): A list of token IDs corresponding to the input text.

        Raises:
            ValueError: If any of the tokens generated during encoding are not found in
                the vocabulary.

        Examples:
            >>> tokenizer = BPETokenizerV1()
            >>> tokenizer.train(['Hello world!'], vocab_size=10)
            >>> ids = tokenizer.encode('Hello world!')
            >>> print(ids)
            [0, 1, 2, 3]
        """
        ids = []

        for piece, is_special in self._split_special_tokens(text, self.special_tokens):
            if is_special:
                ids.append(self.token_to_id(piece))
                continue

            for token in self.pre_tokenizer.pre_tokenize(piece):
                symbols = tuple(token)

                while len(symbols) > 1:
                    pairs = [
                        pair
                        for pair in it.pairwise(symbols)
                        if pair in self.merge_ranks
                    ]
                    if not pairs:
                        break

                    best_pair = min(pairs, key=self.merge_ranks.__getitem__)
                    symbols = self._merge_pair(symbols, best_pair)

                ids.extend(self.token_to_id(symbol) for symbol in symbols)

        return ids

    @override
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode BPE token IDs back into text.

        Args:
            ids (list[int]): A list of token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding. If True, special tokens will be ignored in the output.

        Returns:
            text (str): The decoded text string.

        Raises:
            ValueError: If any of the provided token IDs are invalid or not found in the
                vocabulary.

        Examples:
            >>> tokenizer = BPETokenizerV1()
            >>> tokenizer.train(['Hello world!'], vocab_size=10)
            >>> ids = tokenizer.encode('Hello world!')
            >>> print(ids)
            [0, 1, 2, 3]
            >>> decoded_text = tokenizer.decode(ids)
            >>> print(decoded_text)
            'Hello world!'
        """
        if skip_special_tokens:
            special_ids = self.special_token_ids
        else:
            special_ids = set()

        tokens = [self.id_to_token(idx) for idx in ids if idx not in special_ids]
        return self.decoder.decode(tokens)

    def _prepare_initial_alphabet(
        self,
        initial_alphabet: list[str] | None,
    ) -> list[str]:
        """Keep the first character of every unique alphabet entry.

        Args:
            initial_alphabet (list[str] | None): A list of initial alphabet characters
                to include in the vocabulary.

        Returns:
            alphabet (list[str]): A list of unique first characters from the initial
                alphabet entries.

        Examples:
            >>> initial_alphabet = ['a', 'b', 'c', 'a', 'd']
            >>> alphabet = self._prepare_initial_alphabet(initial_alphabet)
            >>> print(alphabet)
            ['a', 'b', 'c', 'd']
        """
        if initial_alphabet is None:
            return []

        alphabet = []
        for token in initial_alphabet:
            if token and token[0] not in alphabet:
                alphabet.append(token[0])

        return alphabet

    def _split_special_tokens(
        self,
        text: str,
        special_tokens: list[str],
    ) -> list[tuple[str, bool]]:
        """Split text into ordinary pieces and atomic special tokens.

        Args:
            text (str): The input text to be split.
            special_tokens (list[str]): A list of special tokens to be treated as atomic.

        Returns:
            pieces: (list[tuple[str, bool]]): A list of tuples where each tuple contains
                a piece of text and a boolean indicating whether it is a special token.

        Examples:
            >>> text = 'Hello <unk> world!'
            >>> special_tokens = ['<unk>']
            >>> pieces = self._split_special_tokens(text, special_tokens)
            >>> print(pieces)
            [('Hello ', False), ('<unk>', True), (' world!', False)]
        """
        pieces = []
        cursor = 0

        while cursor < len(text):
            matches = (
                (index, -len(token), token)
                for token in special_tokens
                if token and (index := text.find(token, cursor)) >= 0
            )
            match = min(matches, default=None)

            if match is None:
                pieces.append((text[cursor:], False))
                break

            start, _, token = match
            if start > cursor:
                pieces.append((text[cursor:start], False))

            pieces.append((token, True))
            cursor = start + len(token)

        return pieces

    def _count_pairs(
        self,
        word_symbols: list[WordSymbols],
        word_freqs: list[int],
    ) -> Counter[Pair]:
        """Count all adjacent pairs by scanning every word.

        Args:
            word_symbols (list[WordSymbols]): A list of words represented as tuples of
                symbols.
            word_freqs (list[int]): A list of frequencies corresponding to each word.

        Returns:
            counter (Counter[Pair]): A counter mapping each pair of symbols to its frequency.

        Examples:
            >>> word_symbols = [
            ...     ('l', 'o', 'w'),
            ...     ('l', 'o', 'w'),
            ...     ('l', 'o', 'w', 'e', 'r'),
            ... ]
            >>> word_freqs = [5, 3, 2]
            >>> pair_counts = self._count_pairs(word_symbols, word_freqs)
            >>> print(pair_counts)
            Counter({
                ('l', 'o'): 10,
                ('o', 'w'): 10,
                ('w', 'e'): 2,
                ('e', 'r'): 2
            })
        """
        pair_counts = Counter()

        for symbols, freq in zip(word_symbols, word_freqs, strict=True):
            for pair in it.pairwise(symbols):
                pair_counts[pair] += freq

        return pair_counts

    def _select_best_pair(
        self,
        pair_counts: Counter[Pair],
        min_frequency: int,
    ) -> Pair | None:
        """Select the most frequent pair with a direct `max()` scan.

        Args:
            pair_counts (Counter[Pair]): A counter mapping each pair of symbols to
                its frequency.
            min_frequency (int): The minimum frequency for a pair to be considered.

        Returns:
            best_pair (Pair | None): The most frequent pair of symbols, or None if
                no pair meets the minimum frequency requirement.

        Examples:
            >>> pair_counts = Counter({
                ('l', 'o'): 10,
                ('o', 'w'): 10,
                ('w', 'e'): 2,
                ('e', 'r'): 2
            })
            >>> min_frequency = 3
            >>> best_pair = self._select_best_pair(pair_counts, min_frequency)
            >>> print(best_pair)
            ('l', 'o')
        """
        if not pair_counts:
            return None

        freq, pair = max((freq, pair) for pair, freq in pair_counts.items())
        if freq < min_frequency:
            return None

        return pair

    def _merge_pair(self, symbols: WordSymbols, pair: Pair) -> WordSymbols:
        """Merge every non-overlapping occurrence of one pair in a word.

        Args:
            symbols (WordSymbols): A tuple of symbols representing a word.
            pair (Pair): A tuple of two symbols to be merged.

        Returns:
            merged (WordSymbols): A new tuple of symbols with the specified pair merged.

        Examples:
            >>> symbols = ('l', 'o', 'w', 'e', 'r')
            >>> pair = ('l', 'o')
            >>> merged = self._merge_pair(symbols, pair)
            >>> print(merged)
            ('lo', 'w', 'e', 'r')
        """
        index = 0
        merged = []

        while index < len(symbols):
            if index + 1 < len(symbols) and symbols[index : index + 2] == pair:
                merged.append(''.join(pair))
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        return tuple(merged)


def train_bpe_tinystories_v1() -> BPETokenizerV1:
    """Train the BPE tokenizer v1 on the TinyStories dataset."""
    dataset = ds.load_dataset(
        'roneneldan/TinyStories',
        split=f'train[:{NUM_STORIES}]',
        num_proc=2,
    )
    assert len(dataset) == NUM_STORIES

    tokenizer = BPETokenizerV1()

    print('Training BPE tokenizer v1 on TinyStories...')
    start = time.perf_counter()
    tokenizer.train(
        dataset['text'],
        vocab_size=VOCAB_SIZE,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
    )
    end = time.perf_counter()
    print(f'Training completed in {end - start:.4f} seconds.')
    print(f'Tokenizer vocabulary size: {tokenizer.get_vocab_size()}.')
    print('LRU cache info:', dltk.utils.bytes_to_unicode.cache_info())

    return tokenizer


if __name__ == '__main__':
    train_bpe_tinystories_v1()
