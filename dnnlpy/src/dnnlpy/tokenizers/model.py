import itertools as it
import json
import os
from collections.abc import Iterable, Sequence
from typing import override

from .base import Encoding, Model, Tokenizer
from .trainer import BPETrainer

type Pair = tuple[str, str]
type Offset = tuple[int, int]
type SymbolSpan = tuple[str, int, int]

__all__ = ['BPE']


class BPE(Model):
    """Creates a Byte-Pair Encoding (BPE) model for tokenization."""

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[Pair] | None = None,
        unk_token: str = '<unk>',
    ):
        """Initialize the BPE model with a vocabulary, merges, and an unknown token.

        Args:
            vocab (dict[str, int], optional): A dictionary mapping tokens to their IDs.
            merges (list[Pair], optional): A list of merge pairs for BPE.
            unk_token (str, default: '<unk>'): The token to use for unknown tokens.
        """
        self.vocab = vocab or {unk_token: 0}
        self.merges = merges or []
        self.unk_token = unk_token
        self._refresh_merge_ranks()

    def _refresh_merge_ranks(self) -> None:
        """Refresh the mapping of merge pairs to their ranks based on the current merges."""
        self._merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

    @override
    def save(self, tokenizer: Tokenizer, path: str | os.PathLike[str]) -> None:
        """Save the BPE data needed to restore an existing tokenizer."""
        data = {
            'version': 1,
            'model': 'BPE',
            'vocab': dict(tokenizer.vocab),
            'merges': [list(pair) for pair in self.merges],
            'unk_token': tokenizer.unk_token,
            'special_tokens': list(tokenizer.special_tokens),
        }

        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
            fp.write('\n')

    @override
    def load(self, tokenizer: Tokenizer, path: str | os.PathLike[str]) -> None:
        """Load BPE data without replacing the tokenizer's processing components."""
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        if not isinstance(data, dict):
            raise TypeError('Tokenizer JSON must contain a dict object.')

        version = data.get('version', 0)
        if version != 1:
            raise RuntimeError(f'Unsupported tokenizer version: {version!r}.')

        model = data.get('model', 'unknown')
        if model != 'BPE':
            raise RuntimeError(f'Expected a BPE model, got {model!r}.')

        vocab = data.get('vocab', {})
        try:
            vocab = {str(token): int(index) for token, index in vocab.items()}
        except (TypeError, ValueError):
            raise RuntimeError(
                'Vocabulary must be a dict of string keys and integer values.'
            )

        merges = data.get('merges', [])
        try:
            merges = [(str(pair[0]), str(pair[1])) for pair in merges]
        except (TypeError, ValueError):
            raise RuntimeError(
                'Merges must be a list of two-token pairs (lists or tuples).'
            )

        unk_token = data.get('unk_token', '<unk>')
        if not isinstance(unk_token, str) or unk_token not in vocab:
            raise RuntimeError('[UNK] token must be present in the vocabulary.')

        special_tokens = data.get('special_tokens', [])
        try:
            special_tokens = [str(token) for token in special_tokens]
        except (TypeError, ValueError):
            raise RuntimeError('Special tokens must be a list of strings.')

        if unk_token not in special_tokens:
            raise ValueError('[UNK] token must be marked as special.')

        self.vocab = vocab
        self.merges = merges
        self.unk_token = unk_token
        self._refresh_merge_ranks()

        tokenizer.vocab = vocab
        tokenizer.unk_token = unk_token
        tokenizer.special_tokens = list(special_tokens)

    @override
    def encode(
        self,
        tokenizer: Tokenizer,
        text: str,
    ) -> Encoding:
        """Encode the input text into a sequence of token IDs.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            text (str): The input text.

        Returns:
            Encoding: An Encoding object containing token IDs, tokens, and offsets.
        """
        tokens, offsets = self._prepare_input(tokenizer, text)

        ids = []
        output_tokens = []
        output_offsets = []

        for token, token_offset in zip(tokens, offsets, strict=True):
            for model_token, start, end in self._encode_token(token):
                output_token = model_token
                token_id = tokenizer.token_to_id(output_token)

                if token_id is None:
                    token_id = tokenizer.unk_id
                    output_token = self.unk_token

                ids.append(token_id)
                output_tokens.append(output_token)
                output_offsets.append((token_offset[0] + start, token_offset[0] + end))

        encoding = Encoding(
            ids=ids,
            tokens=output_tokens,
            offsets=output_offsets,
        )

        if tokenizer.post_processor is not None:
            return tokenizer._post_process(encoding)

        return encoding

    def _prepare_input(
        self,
        tokenizer: Tokenizer,
        text: str,
    ) -> tuple[list[str], list[Offset]]:
        """Prepare input tokens and offsets for encoding.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            text (str): The input text.

        Returns:
            tuple[list[str], list[Offset]]: A tuple containing the list of tokens and
                their corresponding offsets.

        """
        normalized = tokenizer._normalize(text)
        pieces = list(tokenizer._pre_tokenize(normalized))
        tokens = [piece for piece, _ in pieces]
        offsets = [offset for _, offset in pieces]
        return tokens, offsets

    def _encode_token(self, token: str) -> list[SymbolSpan]:
        """Encode a single token into subword units using BPE merges.

        Args:
            token (str): The input token to encode.

        Returns:
            list[SymbolSpan]: A list of tuples containing the subword units and their
                corresponding start and end offsets within the original token.

        Example:
            >>> vocab = {'l': 0, 'o': 1, 'w': 2, 'lo': 3, 'w</w>': 4}
            >>> merges = [('l', 'o'), ('lo', 'w</w>')]
            >>> bpe = BPE(vocab=vocab, merges=merges)
            >>> bpe._encode_token('low')
            [('lo', 0, 2), ('w</w>', 2, 5)]
        """
        symbols = [(char, index, index + 1) for index, char in enumerate(token)]

        while len(symbols) > 1:
            available_pairs = [
                (left[0], right[0])
                for left, right in it.pairwise(symbols)
                if (left[0], right[0]) in self._merge_ranks
            ]
            if not available_pairs:
                break

            best_pair = min(available_pairs, key=self._merge_ranks.__getitem__)
            symbols = self._merge_symbol_spans(symbols, best_pair)

        return symbols

    def _merge_symbol_spans(
        self,
        symbols: list[SymbolSpan],
        pair: Pair,
    ) -> list[SymbolSpan]:
        """Merge adjacent symbol spans based on the specified pair.

        Args:
            symbols (list[SymbolSpan]): A list of symbol spans to merge.
            pair (Pair): A tuple representing the pair of symbols to merge.

        Returns:
            list[SymbolSpan]: A new list of symbol spans after merging the specified pair.

        Example:
            >>> symbols = [('l', 0, 1), ('o', 1, 2), ('w</w>', 2, 5)]
            >>> pair = ('l', 'o')
            >>> bpe._merge_symbol_spans(symbols, pair)
            [('lo', 0, 2), ('w</w>', 2, 5)]
        """
        merged = []
        index = 0

        while index < len(symbols):
            current, start, _ = symbols[index]

            if index + 1 < len(symbols) and (current, symbols[index + 1][0]) == pair:
                *_, end = symbols[index + 1]
                merged.append((''.join(pair), start, end))
                index += 2
            else:
                merged.append(symbols[index])
                index += 1

        return merged

    @override
    def decode(
        self,
        tokenizer: Tokenizer,
        ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a sequence of token IDs back into a string.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            ids (Sequence[int]): A sequence of token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to skip special tokens
                during decoding.

        Returns:
            decoded (str): The decoded string.

        Example:
            >>> vocab = {'l': 0, 'o': 1, 'w': 2, 'lo': 3, 'w</w>': 4}
            >>> merges = [('l', 'o'), ('lo', 'w</w>')]
            >>> bpe = BPE(vocab=vocab, merges=merges)
            >>> tokenizer = Tokenizer(model=bpe)
            >>> tokenizer.add_special_tokens(['<unk>'])
            >>> tokenizer.train_from_iterator(['low', 'lower', 'newest', 'widest'])
            >>> ids = tokenizer.encode('lowest').ids
            >>> bpe.decode(tokenizer, ids)
            'lowest'
        """
        tokens = tokenizer.lookup_tokens(ids, skip_special_tokens)

        if tokenizer.decoder is not None:
            return tokenizer._decode(tokens)

        return ' '.join(tokens)

    @override
    def train_from_iterator(
        self,
        tokenizer: Tokenizer,
        texts: Iterable[str | Iterable[str]],
        vocab_size: int = 100,
        min_frequency: int = 0,
        initial_alphabet: list[str] | None = None,
    ) -> None:
        """Train the BPE model from an iterator of texts.

        Special tokens must be registered through ``Tokenizer.add_special_tokens``.
        Registered tokens keep their IDs and are excluded from the training corpus.

        Args:
            tokenizer (Tokenizer): The tokenizer instance.
            texts (Iterable[str | Iterable[str]]): An iterable of texts or pretokenized
                sequences.
            vocab_size (int, default: 100): The desired vocabulary size.
            min_frequency (int, default: 0): The minimum frequency for a token to be
                included in the vocabulary.
            initial_alphabet (list[str] | None, default: None): A list of characters to
                include in the initial alphabet, even if they are not present in the
                training texts. If an entry has multiple characters, only the first
                character is used.
        """
        trainer = BPETrainer(
            model=self,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=initial_alphabet,
        )
        trainer.train(texts)
