import pytest

import dnnlpy.tokenizers as dltk


@pytest.mark.parametrize(
    ('tokenizer_cls', 'text', 'expected_vocab'),
    [
        (
            dltk.CharacterTokenizer,
            ['deep', 'learning'],
            {
                '<unk>': 0,
                'a': 1,
                'd': 2,
                'e': 3,
                'g': 4,
                'i': 5,
                'l': 6,
                'n': 7,
                'p': 8,
                'r': 9,
            },
        ),
        (
            dltk.WordTokenizer,
            ['deep learning', 'deep networks'],
            {'<unk>': 0, 'deep': 1, 'learning': 2, 'networks': 3},
        ),
    ],
)
def test_train_builds_sorted_vocab_from_multiple_texts(
    tokenizer_cls: dltk.TraditionalTokenizer,
    text: str | list[str],
    expected_vocab: dict[str, int],
):
    tokenizer = tokenizer_cls.train(text)

    assert tokenizer.vocab == expected_vocab


def test_character_tokenizer_encodes_and_decodes_characters():
    tokenizer = dltk.CharacterTokenizer.train('deep')

    ids = tokenizer.encode('peed')

    assert ids == [tokenizer.vocab[token] for token in 'peed']
    assert tokenizer.decode(ids) == 'peed'


def test_word_tokenizer_encodes_and_decodes_whitespace_separated_words():
    tokenizer = dltk.WordTokenizer.train('deep learning')

    ids = tokenizer.encode('  deep\tlearning  ')

    assert ids == [tokenizer.vocab['deep'], tokenizer.vocab['learning']]
    assert tokenizer.decode(ids) == 'deep learning'


@pytest.mark.parametrize(
    ('tokenizer', 'text', 'expected'),
    [
        (dltk.CharacterTokenizer.train('deep'), 'x', ''),
        (dltk.WordTokenizer.train('deep learning'), 'unknown', ''),
    ],
)
def test_unknown_tokens_use_unk_token(
    tokenizer: dltk.TraditionalTokenizer, text: str, expected: str
):
    ids = tokenizer.encode(text)

    assert ids == [tokenizer.unk_id]
    assert tokenizer.decode(ids) == expected
    assert tokenizer.decode(ids, skip_special_tokens=False) == '<unk>'


@pytest.mark.parametrize(
    ('tokenizer', 'text', 'expected_without_special', 'expected_with_special'),
    [
        (dltk.CharacterTokenizer.train('ab'), 'a^b', 'ab', 'a^b'),
        (
            dltk.WordTokenizer.train('deep learning'),
            'deep <eos>',
            'deep',
            'deep <eos>',
        ),
    ],
)
def test_decode_can_include_or_skip_added_special_tokens(
    tokenizer: dltk.TraditionalTokenizer,
    text: str,
    expected_without_special: str,
    expected_with_special: str,
):
    special_token = '^' if isinstance(tokenizer, dltk.CharacterTokenizer) else '<eos>'
    tokenizer.add_special_tokens([special_token])
    ids = tokenizer.encode(text)

    assert tokenizer.decode(ids) == expected_without_special
    assert tokenizer.decode(ids, skip_special_tokens=False) == expected_with_special


@pytest.mark.parametrize(
    ('tokenizer', 'tokens'),
    [
        (dltk.CharacterTokenizer.train('ab'), ['^', '$']),
        (dltk.WordTokenizer.train('deep learning'), ['<mask>', '<eos>']),
    ],
)
def test_add_tokens_extends_vocab_without_marking_tokens_as_special(
    tokenizer: dltk.TraditionalTokenizer,
    tokens: list[str],
):
    original_vocab_size = tokenizer.vocab_size

    assert tokenizer.add_tokens(tokens) == len(tokens)
    assert tokenizer.add_tokens(tokens) == 0
    assert tokenizer.vocab_size == original_vocab_size + len(tokens)
    assert tokenizer.lookup_tokens(tokenizer.lookup_indices(tokens)) == tokens
    assert not tokenizer.special_token_ids.intersection(
        tokenizer.lookup_indices(tokens)
    )


def test_add_tokens_uses_next_available_sparse_vocab_id():
    tokenizer = dltk.CharacterTokenizer({'<unk>': 3, 'a': 7})

    assert tokenizer.add_tokens(['b']) == 1
    assert tokenizer.vocab == {'<unk>': 3, 'a': 7, 'b': 8}
    assert tokenizer.token_to_id('b') == 8
    assert tokenizer.id_to_token(8) == 'b'


def test_unk_id_and_special_token_ids_follow_vocab_updates():
    tokenizer = dltk.CharacterTokenizer({'<unk>': 0, 'a': 2})

    assert tokenizer.unk_id == 0
    assert tokenizer.special_token_ids == {0}

    tokenizer.add_special_tokens(['a'])
    assert tokenizer.special_token_ids == {0, 2}

    tokenizer.set_vocab({'<unk>': 4, 'a': 7})
    assert tokenizer.unk_id == 4
    assert tokenizer.special_token_ids == {4, 7}


@pytest.mark.parametrize(
    'tokenizer_class', [dltk.CharacterTokenizer, dltk.WordTokenizer]
)
def test_constructor_requires_unk_token_in_vocab(
    tokenizer_class: type[dltk.TraditionalTokenizer],
):
    with pytest.raises(KeyError, match='Unknown token'):
        tokenizer_class({'known': 0})
