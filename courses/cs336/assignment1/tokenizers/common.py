from collections.abc import Iterable
from typing import override

import dnnlpy.tokenizers as dltk
import regex as re

GPT2PATTERN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)"
    r'| ?\p{L}+'
    r'| ?\p{N}+'
    r'| ?[^\s\p{L}\p{N}]+'
    r'|\s+(?!\S)'
    r'|\s+'
)

__all__ = [
    'ByteLevelDecoder',
    'ByteLevelPreTokenizer',
]


class ByteLevelPreTokenizer(dltk.PreTokenizer):
    """A simple byte-level pre-tokenizer for BPE training."""

    @override
    def pre_tokenize(self, text: str) -> Iterable[str]:
        """Pre-tokenize the input text into byte-level Unicode tokens.

        Args:
            text (str): The input text to be pre-tokenized.

        Yields:
            toekn (str): Byte-level Unicode tokens extracted from the input text.

        Examples:
            >>> pre_tokenizer = ByteLevelPreTokenizer()
            >>> list(pre_tokenizer.pre_tokenize('Hello, 世界!'))
            ['Hello', ',', 'Ġä¸ĸçķĮ', '!']
        """
        for token in GPT2PATTERN.findall(text):
            yield dltk.bytes_to_unicode(token)

    @staticmethod
    def alphabet() -> list[str]:
        """Return the list of all possible byte-level Unicode characters.

        Returns:
            alphabet (list[str]): A list of all 256 byte-level Unicode characters.

        Examples:
            >>> alphabet = ByteLevelPreTokenizer.alphabet()
            >>> len(alphabet)
            256
            >>> alphabet[:5]
            ['!', '"', '#', '$', '%']
        """
        return list(dltk.BYTES_TO_UNICODE.values())


class ByteLevelDecoder(dltk.Decoder):
    """A simple byte-level decoder for BPE training."""

    @override
    def decode(self, tokens: list[str]) -> str:
        """Decode a list of byte-level Unicode tokens back into a string.

        Args:
            tokens (list[str]): A list of byte-level Unicode tokens to be decoded.

        Returns:
            text (str): The decoded string obtained from the input tokens.

        Examples:
            >>> decoder = ByteLevelDecoder()
            >>> tokens = ['Hello', ',', 'Ġä¸ĸçķĮ', '!']
            >>> decoder.decode(tokens)
            'Hello, 世界!'
        """
        text = ''.join(tokens)
        text = dltk.unicode_to_bytes(text)
        return text.decode('utf-8', errors='replace')
