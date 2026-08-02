from typing import override

from .base import Decoder
from .utils import unicode_to_bytes

__all__ = ['ByteLevelDecoder']


class ByteLevelDecoder(Decoder):
    """Decode byte-level tokens back into Unicode text."""

    @override
    def decode(self, tokens: list[str]) -> str:
        """Decode byte-level tokens.

        Args:
            tokens (list[str]): Tokens produced from UTF-8 bytes.
        """
        text = ''.join(tokens)
        text = unicode_to_bytes(text)
        return text.decode('utf-8', errors='replace')
