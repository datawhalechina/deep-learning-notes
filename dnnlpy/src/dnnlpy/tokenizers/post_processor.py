from typing import override

from .base import Encoding, PostProcessor

type Offset = tuple[int, int]

__all__ = ['ByteLevelPostProcessor']


class ByteLevelPostProcessor(PostProcessor):
    """Byte-level post-processor."""

    def __init__(
        self,
        add_prefix_space: bool | None = None,
        trim_offsets: bool | None = None,
        use_regex: bool | None = None,
    ):
        """Create a byte-level post-processor.

        Args:
            add_prefix_space (bool | None, default: None): Accepted for
                compatibility with byte-level component configuration.
            trim_offsets (bool, default: True): Trim leading and trailing
                whitespace offsets from produced tokens.
            use_regex (bool | None, default: None): Accepted for compatibility
                with byte-level component configuration.
        """
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = True if trim_offsets is None else trim_offsets
        self.use_regex = use_regex

    def _trim_offset(self, token: str, offset: Offset) -> Offset:
        start, end = offset
        right = len(token) - len(token.rstrip(' Ġ'))
        return (start, max(start, end - right))

    @override
    def process(self, encoding: Encoding) -> Encoding:
        """Process a byte-level encoding."""
        offsets = list(encoding.offsets)

        if self.trim_offsets:
            offsets = [
                self._trim_offset(token, offset)
                for token, offset in zip(
                    encoding.tokens,
                    encoding.offsets,
                    strict=True,
                )
            ]

        return Encoding(
            ids=list(encoding.ids),
            tokens=list(encoding.tokens),
            offsets=offsets,
            type_ids=list(encoding.type_ids),
            attention_mask=list(encoding.attention_mask),
            special_tokens_mask=list(encoding.special_tokens_mask),
        )
