from typing import Self, override

from .base import TraditionalTokenizer

__all__ = [
    'CharacterTokenizer',
    'WordTokenizer',
]


class CharacterTokenizer(TraditionalTokenizer):
    """Tokenizer that treats each character as one token."""

    def __init__(self, vocab: dict[str, int], unk_token: str = '<unk>'):
        """Create a character tokenizer from an existing vocabulary.

        Args:
            vocab (dict[str, int]): Mapping from character tokens to IDs.
            unk_token (str, default: '<unk>'): Token used for unknown characters.
        """
        super().__init__(vocab, unk_token=unk_token)

    @override
    @classmethod
    def train(cls, text: str | list[str], unk_token: str = '<unk>') -> Self:
        """Build a character vocabulary from text.

        Args:
            text (str | list[str]): A single string or list of strings used as
                the training corpus.
            unk_token (str, default: '<unk>'): Token used for unknown characters.
        """
        if isinstance(text, str):
            text = [text]

        vocab_tokens = {ch for line in text for ch in line}
        vocab_tokens = [unk_token] + sorted(vocab_tokens - {unk_token})
        vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        return cls(vocab, unk_token)

    @override
    def encode(self, text: str) -> list[int]:
        """Encode text character by character into token IDs.

        Args:
            text (str): Text to encode.
        Returns:
            list[int]: List of token IDs corresponding to the characters in the
                input text.
        """
        return [self.token_to_id(ch) for ch in text]

    @override
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode character IDs into a string.

        Args:
            ids (list[int]): Character token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.

        Returns:
            str: Decoded string formed by concatenating characters corresponding
                to the input IDs.
        """
        if skip_special_tokens:
            special_tokens = set(self.special_tokens)
        else:
            special_tokens = set()

        tokens = []
        for i in ids:
            token = self.id_to_token(i)
            if token not in special_tokens:
                tokens.append(token)

        return ''.join(tokens)


class WordTokenizer(TraditionalTokenizer):
    """Tokenizer that splits text on whitespace-separated words."""

    def __init__(self, vocab: dict[str, int], unk_token: str = '<unk>'):
        """Create a word tokenizer from an existing vocabulary.

        Args:
            vocab (dict[str, int]): Mapping from word tokens to IDs.
            unk_token (str, default: '<unk>'): Token used for unknown words.
        """
        super().__init__(vocab, unk_token=unk_token)

    @override
    @classmethod
    def train(cls, text: str | list[str], unk_token: str = '<unk>') -> Self:
        """Build a word vocabulary from text.

        Args:
            text (str | list[str]): A single string or list of strings used as
                the training corpus.
            unk_token (str, default: '<unk>'): Token used for unknown words.
        """
        if isinstance(text, str):
            text = [text]

        vocab_tokens = {word for line in text for word in line.split()}
        vocab_tokens = [unk_token] + sorted(vocab_tokens - {unk_token})
        vocab = {token: idx for idx, token in enumerate(vocab_tokens)}
        return cls(vocab, unk_token)

    @override
    def encode(self, text: str) -> list[int]:
        """Encode whitespace-separated words into token IDs.

        Args:
            text (str): Text to encode.
        Returns:
            list[int]: List of token IDs corresponding to the words in the input text.
        """
        return [self.token_to_id(word) for word in text.split()]

    @override
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode word IDs into a space-separated string.

        Args:
            ids (list[int]): Word token IDs to decode.
            skip_special_tokens (bool, default: True): Whether to omit special
                tokens from output.

        Returns:
            str: Decoded string with words separated by spaces.
        """
        if skip_special_tokens:
            special_tokens = set(self.special_tokens)
        else:
            special_tokens = set()

        tokens = []
        for i in ids:
            token = self.id_to_token(i)
            if token not in special_tokens:
                tokens.append(token)

        return ' '.join(tokens)
