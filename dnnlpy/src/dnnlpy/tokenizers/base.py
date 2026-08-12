from __future__ import annotations

import itertools as it
import os
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from typing import Self

from ..configtools import get_num_workers
from .utils import parallel_map

type Offset = tuple[int, int]

__all__ = [
    'Decoder',
    'Encoding',
    'Model',
    'Normalizer',
    'PostProcessor',
    'PreTokenizer',
    'Tokenizer',
    'TraditionalTokenizer',
    'Trainer',
]


class Normalizer(ABC):
    """Abstract base class for normalizers."""

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Normalize a string using the registered normalization method."""


class PreTokenizer(ABC):
    """Abstract base class for pre-tokenizers."""

    @abstractmethod
    def pre_tokenize(self, text: str) -> Iterator[tuple[str, Offset]]:
        """Pre-tokenize a string into a generator of (token, (start, end)) tuples."""

    def count_pre_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count pre-tokenized strings across an iterable of texts."""
        counts = Counter()

        for text in texts:
            for token, _ in self.pre_tokenize(text):
                counts[token] += 1

        return counts


class Model(ABC):
    """Abstract base class for tokenizer models."""

    unk_token: str
    vocab: dict[str, int]
    merges: list[tuple[str, str]]

    @abstractmethod
    def encode(
        self,
        tokenizer: Tokenizer,
        text: str,
    ) -> Encoding: ...

    @abstractmethod
    def decode(
        self,
        tokenizer: Tokenizer,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str: ...

    @abstractmethod
    def train_from_iterator(
        self,
        tokenizer: Tokenizer,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int = 100,
        min_frequency: int = 0,
        initial_alphabet: list[str] | None = None,
    ) -> None: ...

    @abstractmethod
    def save(self, tokenizer: Tokenizer, path: str | os.PathLike[str]) -> None:
        """Save the model data needed to restore a tokenizer."""

    @abstractmethod
    def load(self, tokenizer: Tokenizer, path: str | os.PathLike[str]) -> None:
        """Load model data into an existing tokenizer."""


class Trainer(ABC):
    """Abstract base class for tokenizer trainers."""

    @abstractmethod
    def train(self, texts: Iterable[str | Iterable[str]]) -> None:
        """Train the tokenizer model from an iterator of input texts."""


class Decoder(ABC):
    """Decoder protocol for decoding tokens back into text."""

    @abstractmethod
    def decode(self, tokens: list[str]) -> str:
        """Decode tokens back into text."""


class PostProcessor(ABC):
    """Abstract base class for post-processors."""

    @abstractmethod
    def process(self, encoding: Encoding) -> Encoding:
        """Process an encoding and return a new encoding."""


class Encoding:
    def __init__(
        self,
        ids: list[int],
        tokens: list[str],
        offsets: list[tuple[int, int]] | None = None,
        type_ids: list[int] | None = None,
        attention_mask: list[int] | None = None,
        special_tokens_mask: list[int] | None = None,
    ):
        """Initialize an Encoding object.

        Args:
            ids (list[int]): List of token IDs.
            tokens (list[str]): List of token strings.
            offsets (list[tuple[int, int]], optional): List of (start, end) offsets for
                each token. Defaults to None, which will create offsets of (0, 0).
            type_ids (list[int], optional): List of type IDs for each token. Defaults to
                None, which will create type IDs of 0.
            attention_mask (list[int], optional): List of attention mask values for each
                token. Defaults to None, which will create attention masks of 1.
            special_tokens_mask (list[int], optional): List of special token mask values
                for each token. Defaults to None, which will create special token masks of 0.
        """
        self.ids = ids
        self.tokens = tokens
        self.offsets = offsets or [(0, 0)] * len(ids)
        self.type_ids = type_ids or [0] * len(ids)
        self.attention_mask = attention_mask or [1] * len(ids)
        self.special_tokens_mask = special_tokens_mask or [0] * len(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return (
            f'Encoding('
            f'ids={self.ids}, '
            f'tokens={self.tokens}, '
            f'offsets={self.offsets}, '
            f'type_ids={self.type_ids}, '
            f'attention_mask={self.attention_mask}, '
            f'special_tokens_mask={self.special_tokens_mask}'
            f')'
        )


class TraditionalTokenizer(ABC):
    """Base class for traditional tokenizers that split text into words or characters.

    A tokenizer owns a vocabulary mapping string tokens to integer IDs and defines the
    common encode/decode interface. Subclasses decide how text is split into tokens,
    while this base class provides vocabulary lookup, special-token bookkeeping, and
    batch helpers.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        unk_token: str = '<unk>',
    ):
        """Create a tokenizer from an existing vocabulary.

        Args:
            vocab (dict[str, int]): Mapping from token strings to integer IDs.
            unk_token (str, default: '<unk>'): Token used when encoding unknown
                input tokens.

        Raises:
            KeyError: If `unk_token` is not present in `vocab`.
        """
        self._token_to_id = dict(vocab)
        self._refresh_id_lookup()

        if unk_token not in vocab:
            raise KeyError(f'Unknown token {unk_token!r} is not in vocab.')

        self.unk_token = unk_token
        self.special_tokens = [unk_token]

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    @vocab.setter
    def vocab(self, vocab: dict[str, int]) -> None:
        self.set_vocab(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def unk_id(self) -> int:
        unk_id = self.vocab.get(self.unk_token)
        if unk_id is None:
            raise KeyError('Missing [UNK] token from the vocabulary.')
        return unk_id

    @property
    def special_token_ids(self) -> set[int]:
        return {
            token_id
            for token in self.special_tokens
            if (token_id := self.vocab.get(token)) is not None
        }

    def __len__(self) -> int:
        return len(self.vocab)

    def extra_repr(self) -> str:
        return (
            f'vocab_size={self.vocab_size}, '
            f'unk_token={self.unk_token!r}, '
            f'special_tokens={self.special_tokens!r}'
        )

    def __repr__(self) -> str:
        extra = self.extra_repr()
        if extra:
            return f'{self.__class__.__name__}({extra})'
        return f'{self.__class__.__name__}()'

    def get_vocab(self) -> dict[str, int]:
        return self._token_to_id

    def set_vocab(self, vocab: dict[str, int]) -> None:
        self._token_to_id = vocab
        self._refresh_id_lookup()

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id)

    def id_to_token(self, index: int) -> str:
        if index not in self._id_to_token:
            raise KeyError(f'Unknown token ID: {index}.')
        return self._id_to_token[index]

    def lookup_indices(self, tokens: Sequence[str]) -> list[int]:
        """Map a Sequence of tokens to token IDs.

        Args:
            tokens (Sequence[str]): Tokens to look up.
        """
        return [self.token_to_id(token) for token in tokens]

    def lookup_tokens(self, indices: list[int]) -> list[str]:
        """Map a list of token IDs to tokens.

        Args:
            indices (list[int]): Token IDs to look up.
        """
        return [self.id_to_token(index) for index in indices]

    def add_tokens(self, tokens: list[str]) -> int:
        """Add new tokens to the tokenizer's vocabulary.

        Args:
            tokens (list[str]): A list of tokens to add.

        Returns:
            count (int): The number of new tokens added to the vocabulary.
        """
        count = 0

        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self._next_token_id()
                count += 1

        if count:
            self._refresh_id_lookup()

        return count

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Add tokens to the vocabulary and mark them as special.

        Existing vocabulary entries are not duplicated, but they are still
        marked as special. Special tokens are skipped by default during decode.

        Args:
            tokens (list[str]): Tokens to add or mark as special.

        Returns:
            The number of new vocabulary entries added.
        """
        count = self.add_tokens(tokens)

        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)

        return count

    def _next_token_id(self) -> int:
        """Get the next available token ID for a new vocabulary entry."""
        return max(self.vocab.values(), default=-1) + 1

    def _refresh_id_lookup(self) -> None:
        """Refresh the token-to-ID and ID-to-token mappings."""
        self._id_to_token = {idx: token for token, idx in self._token_to_id.items()}

    @classmethod
    @abstractmethod
    def train(cls, text: str | list[str], *args, **kwargs) -> Self:
        """Build a tokenizer from one text string or a list of text strings.

        Args:
            text (str | list[str]): Training corpus.
            *args: Additional tokenizer-specific positional arguments.
            **kwargs: Additional tokenizer-specific keyword arguments.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token IDs.

        Args:
            text (str): Text to encode.
        """

    def encode_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 1024,
    ) -> list[list[int]]:
        """Encode a batch of text strings.

        Args:
            texts (Sequence[str]): Text strings to encode.
            batch_size (int, default: 1024): Number of texts processed per batch.
        """
        return [
            self.encode(text)
            for batch in it.batched(texts, batch_size)
            for text in batch
        ]

    @abstractmethod
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into text.

        Args:
            ids (list[int]): Token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.
        """

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        skip_special_tokens: bool = True,
        batch_size: int = 1024,
    ) -> list[str]:
        """Decode a batch of token ID sequences.

        Args:
            batch_ids (list[list[int]]): Batch of token ID sequences to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.
            batch_size (int, default: 1024): Number of sequences processed per batch.
        """
        return [
            self.decode(ids, skip_special_tokens)
            for batch in it.batched(batch_ids, batch_size)
            for ids in batch
        ]


class Tokenizer:
    """Base class for tokenizers that provides methods for encoding and decoding text."""

    def __init__(
        self,
        model: Model,
        normalizer: Normalizer | None = None,
        pre_tokenizer: PreTokenizer | None = None,
        post_processor: PostProcessor | None = None,
        decoder: Decoder | None = None,
        num_workers: int | None = None,
    ):
        """Initialize a Tokenizer instance.

        This class is similar to the `tokenizers.Tokenizer` class from the Hugging Face
        `tokenizers` library, but it is implemented in pure Python and does not rely on
        Rust bindings.

        Args:
            model (Model): The tokenizer model to use for encoding and decoding.
            normalizer (Normalizer | None, optional): The normalizer to use for text
                normalization. Defaults to None.
            pre_tokenizer (PreTokenizer | None, optional): The pre-tokenizer to use for
                pre-tokenization. Defaults to None.
            post_processor (PostProcessor | None, optional): The post-processor to use
                for post-processing. Defaults to None.
            decoder (Decoder | None, optional): The decoder to use for decoding tokens
                back into text. Defaults to None.
            num_workers (int | None, optional): The number of worker threads to use for
                parallel processing. If None, defaults to the number of CPU cores. This
                function automatically choose to use multithreading or multiprocessing
                based on the presence of the GIL. Defaults to None.
        """
        self.model = model
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer
        self.post_processor = post_processor
        self.decoder = decoder
        self.num_workers = get_num_workers(num_workers)

        self.unk_token = model.unk_token
        self.special_tokens = [self.unk_token]

        self._refresh_id_lookup()

    @property
    def vocab(self) -> dict[str, int]:
        return self.get_vocab()

    @vocab.setter
    def vocab(self, vocab: dict[str, int]) -> None:
        self.set_vocab(vocab)

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    @property
    def unk_id(self) -> int:
        unk_id = self.token_to_id(self.unk_token)
        if unk_id is None:
            raise KeyError('Missing [UNK] token from the vocabulary.')
        return unk_id

    @property
    def special_token_ids(self) -> set[int]:
        return {
            token_id
            for token in self.special_tokens
            if (token_id := self.token_to_id(token)) is not None
        }

    def __len__(self) -> int:
        return self.vocab_size

    def extra_repr(self) -> str:
        return (
            f'vocab_size={self.vocab_size}, '
            f'unk_token={self.unk_token!r}, '
            f'special_tokens={self.special_tokens!r}'
        )

    def __repr__(self) -> str:
        extra = self.extra_repr()
        if extra:
            return f'{self.__class__.__name__}({extra})'
        return f'{self.__class__.__name__}()'

    def get_vocab(self) -> dict[str, int]:
        return self.model.vocab

    def set_vocab(self, vocab: dict[str, int]) -> None:
        self.model.vocab = vocab
        self._refresh_id_lookup()

    def get_vocab_size(self) -> int:
        return len(self.model.vocab)

    def token_to_id(self, token: str) -> int | None:
        return self._token_to_id.get(token)

    def id_to_token(self, index: int) -> str | None:
        return self._id_to_token.get(index)

    def lookup_indices(self, tokens: Sequence[str]) -> list[int | None]:
        """Convert a sequence of tokens to their corresponding token IDs.

        Args:
            tokens (Sequence[str]): A sequence of tokens to convert.

        Returns:
            list[int | None]: A list of token IDs corresponding to the input tokens.
                If a token is not found in the vocabulary, None is returned.
        """
        return [self.token_to_id(token) for token in tokens]

    def lookup_tokens(
        self,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Convert a sequence of token IDs to their corresponding tokens.

        Args:
            ids (Sequence[int]): A sequence of token IDs to convert.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during conversion.

        Returns:
            list[str]: A list of tokens corresponding to the input IDs.
        """
        if skip_special_tokens:
            special_ids = self.special_token_ids
        else:
            special_ids = set()

        tokens = []
        for index in ids:
            if skip_special_tokens and index in special_ids:
                continue

            token = self.id_to_token(index)
            if token is not None:
                tokens.append(token)

        return tokens

    def add_tokens(self, tokens: list[str]) -> int:
        """Add new tokens to the tokenizer's vocabulary.

        Args:
            tokens (list[str]): A list of tokens to add.

        Returns:
            count (int): The number of new tokens added to the vocabulary.
        """
        count = 0
        next_id = max(self.model.vocab.values(), default=-1) + 1

        for token in tokens:
            if token not in self.model.vocab:
                self.model.vocab[token] = next_id
                next_id, count = next_id + 1, count + 1

        if count:
            self._refresh_id_lookup()

        return count

    def add_special_tokens(self, tokens: list[str]) -> int:
        """Add special tokens to the tokenizer's vocabulary.

        This is the only API for registering special tokens. Tokens registered before
        training keep their IDs and are excluded from the training corpus. Tokens
        registered after training are appended without changing existing token IDs.

        Args:
            tokens (list[str]): A list of special tokens to add.

        Returns:
            count (int): The number of new special tokens added to the vocabulary.
        """
        count = self.add_tokens(tokens)

        for token in tokens:
            if token not in self.special_tokens:
                self.special_tokens.append(token)

        return count

    def encode(self, text: str) -> Encoding:
        """Encode a string into an `Encoding` object.

        Args:
            text (str): The input text.

        Returns:
            Encoding: An Encoding object containing token IDs, tokens, and offsets.

        Example:
            >>> tokenizer = Tokenizer(BPE())
            >>> encoding = tokenizer.encode('Hello, world!')
            >>> print(encoding.ids)
            >>> print(encoding.tokens)
        """
        return self.model.encode(self, text)

    def encode_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 1024,
    ) -> list[Encoding]:
        """Encode a batch of text strings.

        Args:
            texts (Sequence[str]): Input text strings to encode.
            batch_size (int, default: 1024): Number of texts processed by each worker task.

        Returns:
            Encodings corresponding to the input texts, in input order.
        """
        batches = it.batched(texts, batch_size)

        return list(parallel_map(self.encode, batches, self.num_workers))

    def decode(
        self, ids: Encoding | list[int], skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs back into text.

        Args:
            ids (list[int]): A list of token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding.

        Returns:
            str: The decoded string.

        Example:
            >>> tokenizer = Tokenizer(model=BPE())
            >>> text = tokenizer.decode([1, 2, 3])
            >>> print(text)
        """
        if isinstance(ids, Encoding):
            ids = ids.ids

        return self.model.decode(self, ids, skip_special_tokens)

    def decode_batch(
        self,
        batch_ids: list[list[int]],
        skip_special_tokens: bool = True,
        batch_size: int = 1024,
    ) -> list[str]:
        """Decode a batch of token ID sequences.

        Args:
            batch_ids (list[list[int]]): A list of token ID sequences to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding.
            batch_size (int, default: 1024): Number of sequences processed by each
                worker task.

        Returns:
            list[str]: A list of decoded strings corresponding to each input sequence.
        """
        batches = it.batched(batch_ids, batch_size)
        decode_fn = partial(self.decode, skip_special_tokens=skip_special_tokens)

        return list(parallel_map(decode_fn, batches, self.num_workers))

    def train_from_iterator(
        self,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int = 100,
        min_frequency: int = 0,
        initial_alphabet: list[str] | None = None,
    ) -> None:
        """Train the tokenizer model from an iterator.

        Register special tokens with :meth:`add_special_tokens`. Their existing IDs
        are preserved, and their occurrences are excluded from the training corpus.

        Args:
            texts (Iterable[str | Iterable[str]]): An iterable of input texts or
                a batch of input texts.
            vocab_size (int, default: 100): The desired vocabulary size.
            min_frequency (int, default: 0): The minimum frequency for a token to be
                included in the vocabulary.
            initial_alphabet (list[str] | None, default: None): A list of characters to
                include in the initial alphabet, even if they are not present in the
                training texts. If an entry has multiple characters, only the first
                character is used.
        """
        self.model.train_from_iterator(
            self,
            texts,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=initial_alphabet,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the tokenizer's model data as JSON."""
        self.model.save(self, path)

    def load(self, path: str | os.PathLike[str]) -> None:
        """Load model data into this tokenizer without replacing its components."""
        self.model.load(self, path)

    def _refresh_id_lookup(self) -> None:
        """Refresh the reverse mapping from IDs to tokens."""
        self._token_to_id = self.model.vocab
        self._id_to_token = {idx: token for token, idx in self.model.vocab.items()}

    def _split_special_tokens(
        self,
        text: str,
    ) -> Iterator[tuple[str, bool, Offset]]:
        """Split text into ordinary spans and atomic registered special tokens."""
        special_tokens = dict.fromkeys(token for token in self.special_tokens if token)
        special_tokens = tuple(special_tokens)

        cursor = 0
        while cursor < len(text):
            matches = (
                (index, -len(token), token)
                for token in special_tokens
                if (index := text.find(token, cursor)) >= 0
            )
            match = min(matches, default=None)

            if match is None:
                yield (text[cursor:], False, (cursor, len(text)))
                return

            start, _, token = match
            if start > cursor:
                yield (text[cursor:start], False, (cursor, start))

            end = start + len(token)
            yield (token, True, (start, end))
            cursor = end

    def _normalize(self, text: str) -> str:
        """Normalize a string with the registered normalizer."""
        if self.normalizer is None:
            return text
        else:
            return self.normalizer.normalize(text)

    def _pre_tokenize(self, text: str) -> Iterator[tuple[str, Offset]]:
        """Pre-tokenize a string with the registered pre-tokenizer."""
        if self.pre_tokenizer is None:
            yield (text, (0, len(text)))
        else:
            yield from self.pre_tokenizer.pre_tokenize(text)

    def _post_process(self, encoding: Encoding) -> Encoding:
        """Post-process an encoding with the registered post-processor."""
        if self.post_processor is None:
            return encoding
        else:
            return self.post_processor.process(encoding)

    def _decode(self, tokens: list[str]) -> str:
        """Decode a list of tokens into a string."""
        if self.decoder is None:
            return ''.join(tokens)
        else:
            return self.decoder.decode(tokens)

    def _count_pre_tokens(
        self,
        texts: Iterable[str],
        batch_size: int = 1024,
    ) -> Counter[str]:
        """Count pre-tokens in an iterable of texts using parallel processing."""
        batches = ((batch,) for batch in it.batched(texts, batch_size))

        if self.pre_tokenizer is None:
            count_fn = Counter
        else:
            count_fn = self.pre_tokenizer.count_pre_tokens

        word_counts = Counter()
        for counts in parallel_map(count_fn, batches, self.num_workers):
            word_counts.update(counts)

        return word_counts
