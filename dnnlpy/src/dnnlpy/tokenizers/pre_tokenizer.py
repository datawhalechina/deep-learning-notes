from collections import Counter
from collections.abc import Iterable, Iterator
from typing import override

import regex as re

from .base import PreTokenizer
from .utils import BYTES_TO_UNICODE, bytes_to_unicode

type Offset = tuple[int, int]

__all__ = [
    'ByteLevelPreTokenizer',
    'WhitespacePreTokenizer',
]

WHITESPACE = re.compile(
    r'\w+'
    r'|[^\w\s]+'
)
GPT2PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r'| ?\p{L}+'
    r'| ?\p{N}+'
    r'| ?[^\s\p{L}\p{N}]+'
    r'|\s+(?!\S)'
    r'|\s+'
)


class ByteLevelPreTokenizer(PreTokenizer):
    """Byte-level pre-tokenizer."""

    def __init__(
        self,
        add_prefix_space: bool = True,
        trim_offsets: bool = True,
        use_regex: bool = True,
    ):
        """Create a byte-level pre-tokenizer.

        Args:
            add_prefix_space (bool, default: False): Add one leading space when the input
                does not already start with whitespace.
            trim_offsets (bool, default: True): Trim leading and trailing whitespace offsets
                from produced tokens.
            use_regex (bool, default: True): Use regex to split text into tokens.
        """
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = trim_offsets
        self.use_regex = use_regex

    @staticmethod
    def alphabet() -> list[str]:
        return list(BYTES_TO_UNICODE.values())

    @override
    def pre_tokenize(self, text: str) -> Iterator[tuple[str, Offset]]:
        """Pre-tokenize text into byte-level tokens with offsets."""
        if not text:
            return

        original_length = len(text)
        add_prefix_space = self.add_prefix_space and not text[0].isspace()

        if add_prefix_space:
            text = f' {text}'

        # Shift offsets by 1 if a prefix space was added
        prefix_shift = int(add_prefix_space)

        if self.use_regex:
            pieces = (
                (matchs.group(), matchs.span())
                for matchs in GPT2PATTERN.finditer(text)
            )  # fmt: off
        else:
            pieces = [(text, (0, len(text)))]

        for token, (start, end) in pieces:
            offset_start = max(0, start - prefix_shift)
            offset_end = max(0, end - prefix_shift)

            offset_start = min(offset_start, original_length)
            offset_end = min(offset_end, original_length)

            token = bytes_to_unicode(token)
            yield (token, (offset_start, offset_end))

    @override
    def count_pre_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count byte-level tokens while converting each unique piece only once."""
        word_counts = Counter()

        for text in texts:
            if text and self.add_prefix_space and not text[0].isspace():
                text = f' {text}'

            if self.use_regex:
                word_counts.update(GPT2PATTERN.findall(text))
            else:
                word_counts[text] += 1

        return Counter(
            {
                bytes_to_unicode(token): frequency
                for token, frequency in word_counts.items()
            }
        )


class WhitespacePreTokenizer(PreTokenizer):
    """Split text into non-whitespace spans."""

    @override
    def pre_tokenize(self, text: str) -> Iterator[tuple[str, Offset]]:
        """Pre-tokenize text on whitespace."""
        if not text:
            return

        for matchs in WHITESPACE.finditer(text):
            yield (matchs.group(), matchs.span())

    @override
    def count_pre_tokens(self, texts: Iterable[str]) -> Counter[str]:
        """Count whitespace tokens while converting each unique piece only once."""
        word_counts = Counter()

        for text in texts:
            word_counts.update(WHITESPACE.findall(text))

        return word_counts
