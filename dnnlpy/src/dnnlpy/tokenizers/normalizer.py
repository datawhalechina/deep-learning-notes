from typing import override

from .base import Normalizer
from .utils import bytes_to_unicode

__all__ = [
    'ByteLevelNormalizer',
    'LowercaseNormalizer',
    'StripNormalizer',
]


class ByteLevelNormalizer(Normalizer):
    """Normalize text into byte-level Unicode tokens."""

    @override
    def normalize(self, text: str) -> str:
        """Normalize text into byte-level Unicode text."""
        return bytes_to_unicode(text)


class LowercaseNormalizer(Normalizer):
    """Normalize text to lowercase."""

    @override
    def normalize(self, text: str) -> str:
        """Normalize text to lowercase."""
        return text.lower()


class StripNormalizer(Normalizer):
    """Strip leading and trailing whitespace."""

    def __init__(self, left: bool = True, right: bool = True):
        """Create a strip normalizer.

        Args:
            left (bool, default: True): Strip whitespace from the left side.
            right (bool, default: True): Strip whitespace from the right side.
        """
        self.left = left
        self.right = right

    @override
    def normalize(self, text: str) -> str:
        """Strip configured whitespace from text."""
        if self.left and self.right:
            return text.strip()
        if self.left:
            return text.lstrip()
        if self.right:
            return text.rstrip()
        return text
