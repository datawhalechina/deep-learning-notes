import json
import os
from typing import cast

import tiktoken

import dnnlpy.tokenizers as dltk

from .adapters import get_tokenizer
from .conftest import FIXTURES_PATH

__all__ = [
    'test_address_matches_tiktoken',
    'test_address_roundtrip',
    'test_ascii_string_matches_tiktoken',
    'test_empty_matches_tiktoken',
    'test_encode_iterable_tinystories_matches_tiktoken',
    'test_encode_iterable_tinystories_sample_roundtrip',
    'test_encode_special_token_double_newline_non_whitespace',
    'test_encode_special_token_trailing_newlines',
    'test_german_matches_tiktoken',
    'test_german_roundtrip',
    'test_overlapping_special_tokens',
    'test_roundtrip_ascii_string',
    'test_roundtrip_empty',
    'test_roundtrip_single_character',
    'test_roundtrip_single_unicode_character',
    'test_roundtrip_unicode_string',
    'test_roundtrip_unicode_string_with_special_tokens',
    'test_single_character_matches_tiktoken',
    'test_single_unicode_character_matches_tiktoken',
    'test_tinystories_matches_tiktoken',
    'test_tinystories_sample_roundtrip',
    'test_unicode_string_matches_tiktoken',
    'test_unicode_string_with_special_tokens_matches_tiktoken',
]

VOCAB_PATH = FIXTURES_PATH / 'gpt2_vocab.json'
MERGES_PATH = FIXTURES_PATH / 'gpt2_merges.txt'


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike[str],
    merges_path: str | os.PathLike[str],
    special_tokens: list[str] | None = None,
) -> dltk.Tokenizer:
    """Get a BPE tokenizer from vocab and merges files.

    Args:
        vocab_path (str | os.PathLike[str]): Path to the vocab file.
        merges_path (str | os.PathLike[str]): Path to the merges file.
        special_tokens (list[str] | None): List of special tokens to add to the
            tokenizer.

    Returns:
        tokenizer (dltk.Tokenizer): A BPE tokenizer that uses the provided vocab,
            merges, and special tokens.
    """
    with open(vocab_path) as fp:
        vocab = json.load(fp)
        vocab = cast(dict[str, int], vocab)

    with open(merges_path) as fp:
        merges = [tuple(line.split()) for line in fp]
        merges = cast(list[tuple[str, str]], merges)

    return get_tokenizer(vocab, merges, special_tokens)


def test_roundtrip_empty():
    """Test that encoding and decoding an empty string returns an empty string."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = ''
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)

    assert test_string == decoded_string


def test_empty_matches_tiktoken():
    """Test that encoding and decoding an empty string matches tiktoken."""
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    test_string = ''
    acutal_ids = custom_tokenizer.encode(test_string)
    expected_ids = reference_tokenizer.encode(test_string)

    assert acutal_ids.ids == expected_ids

    tokenized_string = [custom_tokenizer.decode([x]) for x in acutal_ids.ids]
    assert tokenized_string == []

    assert custom_tokenizer.decode(acutal_ids.ids) == test_string
    assert reference_tokenizer.decode(expected_ids) == test_string


def test_roundtrip_single_character():
    """Test that encoding and decoding a single character returns the same character."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = 's'
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)

    assert test_string == decoded_string


def test_single_character_matches_tiktoken():
    """Test that encoding and decoding a single character matches tiktoken."""
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    test_string = 's'

    acutal_ids = custom_tokenizer.encode(test_string)
    expected_ids = reference_tokenizer.encode(test_string)

    assert acutal_ids.ids == expected_ids

    tokenized_string = [custom_tokenizer.decode([x]) for x in acutal_ids.ids]
    assert tokenized_string == ['s']

    assert custom_tokenizer.decode(acutal_ids) == test_string
    assert reference_tokenizer.decode(expected_ids) == test_string


def test_roundtrip_single_unicode_character():
    """Test that encoding and decoding a single unicode character returns the
    same character.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = '🙃'
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)

    assert test_string == decoded_string


def test_single_unicode_character_matches_tiktoken():
    """Test that encoding and decoding a single unicode character matches tiktoken."""
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    test_string = '🙃'

    actual_ids = custom_tokenizer.encode(test_string)
    expected_ids = reference_tokenizer.encode(test_string)

    assert actual_ids.ids == expected_ids
    assert custom_tokenizer.decode(actual_ids) == test_string
    assert reference_tokenizer.decode(expected_ids) == test_string


def test_roundtrip_ascii_string():
    """Test that encoding and decoding an ASCII string returns the same string."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = 'Hello, how are you?'
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)

    assert test_string == decoded_string


def test_ascii_string_matches_tiktoken():
    """Test that encoding and decoding an ASCII string matches tiktoken."""
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    test_string = 'Hello, how are you?'

    actual_ids = custom_tokenizer.encode(test_string)
    expected_ids = reference_tokenizer.encode(test_string)

    assert actual_ids.ids == expected_ids

    tokenized_string = [custom_tokenizer.decode([x]) for x in actual_ids.ids]
    assert tokenized_string == ['Hello', ',', ' how', ' are', ' you', '?']

    assert custom_tokenizer.decode(actual_ids) == test_string
    assert reference_tokenizer.decode(expected_ids) == test_string


def test_roundtrip_unicode_string():
    """Test that encoding and decoding a unicode string returns the same string."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    test_string = 'Héllò hôw are ü? 🙃'
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)

    assert test_string == decoded_string


def test_unicode_string_matches_tiktoken():
    """Test that encoding and decoding a unicode string matches tiktoken."""
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    test_string = 'Héllò hôw are ü? 🙃'

    actual_id = custom_tokenizer.encode(test_string)
    expected_ids = reference_tokenizer.encode(test_string)

    assert actual_id.ids == expected_ids

    assert custom_tokenizer.decode(actual_id) == test_string
    assert reference_tokenizer.decode(expected_ids) == test_string


def test_roundtrip_unicode_string_with_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )

    test_string = 'Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>'
    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [
        tokenizer.decode([x], skip_special_tokens=False) for x in encoded_ids.ids
    ]

    # Ensure the special <|endoftext|> token is preserved
    assert tokenized_string.count('<|endoftext|>') == 3

    decoded_string = tokenizer.decode(encoded_ids, skip_special_tokens=False)
    assert test_string == decoded_string


def test_unicode_string_with_special_tokens_matches_tiktoken():
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    test_string = 'Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>'

    expected_ids = reference_tokenizer.encode(
        test_string,
        allowed_special={'<|endoftext|>'},
    )
    actual_ids = custom_tokenizer.encode(test_string)

    assert actual_ids.ids == expected_ids
    assert custom_tokenizer.decode(actual_ids, skip_special_tokens=False) == test_string
    assert reference_tokenizer.decode(expected_ids) == test_string


def test_overlapping_special_tokens():
    """Test that overlapping special tokens are handled correctly."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>', '<|endoftext|><|endoftext|>'],
    )

    test_string = 'Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>'

    encoded_ids = tokenizer.encode(test_string)
    tokenized_string = [
        tokenizer.decode([x], skip_special_tokens=False) for x in encoded_ids.ids
    ]

    # Ensure the double <|endoftext|><|endoftext|> is preserved as a single token
    assert tokenized_string.count('<|endoftext|>') == 1
    assert tokenized_string.count('<|endoftext|><|endoftext|>') == 1
    assert tokenizer.decode(encoded_ids, skip_special_tokens=False) == test_string


def test_address_roundtrip():
    """Test that encoding and decoding the address.txt fixture returns the same string."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    with open(FIXTURES_PATH / 'address.txt') as fp:
        corpus_contents = fp.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_address_matches_tiktoken():
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    corpus_path = FIXTURES_PATH / 'address.txt'
    with open(corpus_path) as fp:
        corpus_contents = fp.read()

    actual_ids = custom_tokenizer.encode(corpus_contents)
    expected_ids = reference_tokenizer.encode(corpus_contents)

    assert actual_ids.ids == expected_ids
    assert custom_tokenizer.decode(actual_ids) == corpus_contents
    assert reference_tokenizer.decode(expected_ids) == corpus_contents


def test_german_roundtrip():
    """Test that encoding and decoding the german.txt fixture returns the same string."""
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    with open(FIXTURES_PATH / 'german.txt') as fp:
        corpus_contents = fp.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_german_matches_tiktoken():
    """Test that encoding and decoding the german.txt fixture matches tiktoken."""
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    corpus_path = FIXTURES_PATH / 'german.txt'
    with open(corpus_path) as fp:
        corpus_contents = fp.read()

    actual_ids = custom_tokenizer.encode(corpus_contents)
    expected_ids = reference_tokenizer.encode(corpus_contents)

    assert actual_ids.ids == expected_ids
    assert custom_tokenizer.decode(actual_ids) == corpus_contents
    assert reference_tokenizer.decode(expected_ids) == corpus_contents


def test_tinystories_sample_roundtrip():
    """Test that encoding and decoding the tinystories_sample.txt fixture returns
    the same string.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    with open(FIXTURES_PATH / 'tinystories_sample.txt') as fp:
        corpus_contents = fp.read()

    ids = tokenizer.encode(corpus_contents)
    assert tokenizer.decode(ids) == corpus_contents


def test_tinystories_matches_tiktoken():
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    corpus_path = FIXTURES_PATH / 'tinystories_sample.txt'
    with open(corpus_path) as fp:
        corpus_contents = fp.read()

    actual_ids = custom_tokenizer.encode(corpus_contents)
    expected_ids = reference_tokenizer.encode(
        corpus_contents,
        allowed_special={'<|endoftext|>'},
    )

    assert actual_ids.ids == expected_ids
    assert (
        custom_tokenizer.decode(actual_ids, skip_special_tokens=False)
        == corpus_contents
    )
    assert reference_tokenizer.decode(expected_ids) == corpus_contents


def test_encode_special_token_trailing_newlines():
    """Test that encoding and decoding a string with a special token and trailing newlines
    returns the same string.
    """
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    corpus_path = FIXTURES_PATH / 'special_token_trailing_newlines.txt'
    with open(corpus_path) as fp:
        corpus_contents = fp.read()

    actual_ids = custom_tokenizer.encode(corpus_contents)
    expected_ids = reference_tokenizer.encode(
        corpus_contents,
        allowed_special={'<|endoftext|>'},
    )

    assert actual_ids.ids == expected_ids
    assert (
        custom_tokenizer.decode(actual_ids, skip_special_tokens=False)
        == corpus_contents
    )
    assert reference_tokenizer.decode(expected_ids) == corpus_contents


def test_encode_special_token_double_newline_non_whitespace():
    """Test that encoding and decoding a string with a special token and double newlines
    returns the same string.
    """
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    corpus_path = FIXTURES_PATH / 'special_token_double_newlines_non_whitespace.txt'
    with open(corpus_path) as fp:
        corpus_contents = fp.read()

    actual_ids = custom_tokenizer.encode(corpus_contents)
    expected_ids = reference_tokenizer.encode(
        corpus_contents,
        allowed_special={'<|endoftext|>'},
    )

    assert actual_ids.ids == expected_ids
    assert (
        custom_tokenizer.decode(actual_ids, skip_special_tokens=False)
        == corpus_contents
    )
    assert reference_tokenizer.decode(expected_ids) == corpus_contents


def test_encode_iterable_tinystories_sample_roundtrip():
    """Test that encoding and decoding the tinystories_sample.txt fixture using
    encode_iterable returns the same string.
    """
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    encoded_text = []

    with open(FIXTURES_PATH / 'tinystories_sample.txt') as fp:
        for line in fp:
            encoded_line = tokenizer.encode(line)
            encoded_text.extend(encoded_line.ids)

    with open(FIXTURES_PATH / 'tinystories_sample.txt') as fp:
        corpus_contents = fp.read()

    assert tokenizer.decode(encoded_text, skip_special_tokens=False) == corpus_contents


def test_encode_iterable_tinystories_matches_tiktoken():
    """Test that encoding and decoding the tinystories_sample.txt fixture using
    encode_iterable matches tiktoken.
    """
    custom_tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=['<|endoftext|>'],
    )
    reference_tokenizer = tiktoken.get_encoding('gpt2')

    corpus_path = FIXTURES_PATH / 'tinystories_sample.txt'
    with open(corpus_path) as fp:
        corpus_contents = fp.read()

    expected_ids = reference_tokenizer.encode(
        corpus_contents,
        allowed_special={'<|endoftext|>'},
    )

    actual_ids = []
    with open(FIXTURES_PATH / 'tinystories_sample.txt') as fp:
        for line in fp:
            encoded_line = custom_tokenizer.encode(line)
            actual_ids.extend(encoded_line.ids)

    assert actual_ids == expected_ids
    assert (
        custom_tokenizer.decode(actual_ids, skip_special_tokens=False)
        == corpus_contents
    )
    assert reference_tokenizer.decode(expected_ids) == corpus_contents
