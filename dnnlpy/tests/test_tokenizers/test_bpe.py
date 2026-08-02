import json
from typing import cast

from tokenizers import Tokenizer
from tokenizers.models import BPE

import dnnlpy.tokenizers as dltk


def test_bpe_encode_matches_hf_tokenizers():
    vocab = {
        '<unk>': 0,
        'l': 1,
        'o': 2,
        'w': 3,
        'lo': 4,
        'low': 5,
        'e': 6,
        'r': 7,
        'er': 8,
        'lower': 9,
    }
    merges = [('l', 'o'), ('lo', 'w'), ('e', 'r'), ('low', 'er')]

    tokenizer = dltk.Tokenizer(dltk.BPE(vocab, merges), num_workers=1)
    hf_tokenizer = Tokenizer(BPE(vocab, merges, unk_token='<unk>'))

    text = 'lower'

    actual = tokenizer.encode(text)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)

    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_bpe_encode_unknown_piece_matches_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]

    tokenizer = dltk.Tokenizer(dltk.BPE(vocab, merges), num_workers=1)
    hf_tokenizer = Tokenizer(BPE(vocab, merges, unk_token='<unk>'))

    text = 'lowx'

    actual = tokenizer.encode(text)
    expected = hf_tokenizer.encode(text, add_special_tokens=False)

    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_bpe_decode_matches_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]

    tokenizer = dltk.Tokenizer(dltk.BPE(vocab, merges), num_workers=1)
    hf_tokenizer = Tokenizer(BPE(vocab, merges, unk_token='<unk>'))

    ids = [5, 0]

    actual = tokenizer.decode(ids, skip_special_tokens=False)
    expected = hf_tokenizer.decode(ids, skip_special_tokens=False)
    assert actual == expected


def test_bpe_batch_encode_and_decode_match_hf_tokenizers():
    vocab = {'<unk>': 0, 'l': 1, 'o': 2, 'w': 3, 'lo': 4, 'low': 5}
    merges = [('l', 'o'), ('lo', 'w')]

    tokenizer = dltk.Tokenizer(dltk.BPE(vocab, merges), num_workers=1)
    hf_tokenizer = Tokenizer(BPE(vocab, merges, unk_token='<unk>'))
    texts = ['low', 'lowx']

    actual = tokenizer.encode_batch(texts, batch_size=1)
    expected = hf_tokenizer.encode_batch(texts, add_special_tokens=False)

    flag1 = [encoding.tokens for encoding in actual]
    flag2 = [encoding.tokens for encoding in expected]
    assert flag1 == flag2

    flag1 = tokenizer.decode_batch(
        [encoding.ids for encoding in actual],
        skip_special_tokens=False,
        batch_size=1,
    )
    flag2 = hf_tokenizer.decode_batch(
        [encoding.ids for encoding in expected],
        skip_special_tokens=False,
    )
    assert flag1 == flag2


def test_bpe_train_from_iterator_accepts_batches():
    tokenizer = dltk.Tokenizer(
        dltk.BPE(),
        normalizer=dltk.StripNormalizer(),
        num_workers=1,
    )
    texts = [[' low ', ' lower '], [' lowest ']]

    tokenizer.train_from_iterator(texts, vocab_size=20)

    assert 'low' in tokenizer.vocab
    assert 'lower' in tokenizer.vocab


def test_bpe_train_from_iterator_includes_initial_alphabet():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)

    tokenizer.train_from_iterator(['aaa'], vocab_size=10, initial_alphabet=['z'])

    assert 'z' in tokenizer.vocab


def test_bpe_train_from_iterator_uses_first_initial_alphabet_character():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)

    tokenizer.train_from_iterator(['aaa'], vocab_size=10, initial_alphabet=['xyz'])

    assert 'x' in tokenizer.vocab
    assert 'xyz' not in tokenizer.vocab


def test_bpe_training_preserves_preadded_special_token_ids_and_excludes_them():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)
    tokenizer.add_tokens(['ordinary'])
    tokenizer.add_special_tokens(['<eos>', '<pad>'])
    special_tokens = tokenizer.special_tokens

    special_ids = {}
    for token in special_tokens:
        special_ids[token] = tokenizer.token_to_id(token)

    tokenizer.train_from_iterator(
        ['ab<eos>cd<pad>ef'],
        vocab_size=12,
    )

    model = cast(dltk.BPE, tokenizer.model)
    ordinary_tokens = set(tokenizer.vocab) - set(special_tokens)

    for token in special_tokens:
        assert tokenizer.token_to_id(token) == special_ids[token]

    assert all('<' not in token for token in ordinary_tokens)
    assert all('>' not in token for token in ordinary_tokens)
    assert all('<' not in ''.join(pair) for pair in model.merges)
    assert all('>' not in ''.join(pair) for pair in model.merges)
    assert ('ab', 'cd') not in model.merges
    assert ('cd', 'ef') not in model.merges


def test_bpe_add_special_tokens_after_training_preserves_existing_ids():
    tokenizer = dltk.Tokenizer(dltk.BPE(), num_workers=1)
    tokenizer.train_from_iterator(['abab'], vocab_size=6)

    model = cast(dltk.BPE, tokenizer.model)
    vocab = tokenizer.vocab
    merges = list(model.merges)
    special_tokens = tokenizer.special_tokens
    expected_id = max(vocab.values()) + 1

    tokenizer.add_special_tokens(['<eos>'])

    assert tokenizer.token_to_id('<eos>') == expected_id

    for token, token_id in vocab.items():
        assert tokenizer.token_to_id(token) == token_id

    assert model.merges == merges

    tokenizer.train_from_iterator(['cd<eos>cd'], vocab_size=6)

    assert tokenizer.token_to_id('<eos>') == expected_id

    for token in vocab:
        if token not in special_tokens:
            assert '<' not in token and '>' not in token


def test_bpe_training_ignores_empty_texts():
    tokenizer = dltk.Tokenizer(
        dltk.BPE(),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(),
        num_workers=1,
    )

    tokenizer.train_from_iterator(['', 'low lower', ''], vocab_size=20)

    model = cast(dltk.BPE, tokenizer.model)

    assert model.merges


def test_bpe_save_and_load_restore_data_without_replacing_components(tmp_path):
    tokenizer = dltk.Tokenizer(
        dltk.BPE(),
        normalizer=dltk.StripNormalizer(left=True, right=False),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(
            add_prefix_space=False,
            trim_offsets=False,
        ),
        post_processor=dltk.ByteLevelPostProcessor(trim_offsets=False),
        decoder=dltk.ByteLevelDecoder(),
        num_workers=1,
    )
    tokenizer.add_special_tokens(['<pad>'])
    pad_id = tokenizer.token_to_id('<pad>')

    tokenizer.train_from_iterator(
        ['Hello world', 'Hello tokenizer'],
        vocab_size=50,
    )

    assert tokenizer.token_to_id('<pad>') == pad_id

    path = tmp_path / 'tokenizer.json'
    tokenizer.save(path)

    data = json.loads(path.read_text(encoding='utf-8'))
    model = cast(dltk.BPE, tokenizer.model)

    assert set(data) == {
        'version',
        'model',
        'vocab',
        'merges',
        'unk_token',
        'special_tokens',
    }
    assert data['vocab'] == tokenizer.vocab
    assert data['merges'] == [list(pair) for pair in model.merges]

    restored = dltk.Tokenizer(
        dltk.BPE(),
        normalizer=dltk.StripNormalizer(left=True, right=False),
        pre_tokenizer=dltk.ByteLevelPreTokenizer(
            add_prefix_space=False,
            trim_offsets=False,
        ),
        post_processor=dltk.ByteLevelPostProcessor(trim_offsets=False),
        decoder=dltk.ByteLevelDecoder(),
        num_workers=1,
    )

    restored.load(path)
    restored_model = cast(dltk.BPE, restored.model)

    assert restored.vocab == tokenizer.vocab
    assert restored_model.vocab == tokenizer.vocab
    assert restored_model.merges == model.merges
    assert restored.unk_token == tokenizer.unk_token
    assert restored.special_tokens == tokenizer.special_tokens

    text = ' Hello tokenizer '
    actual = restored.encode(text)
    expected = tokenizer.encode(text)

    assert actual.ids == expected.ids
    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets
    assert restored.decode(actual.ids) == tokenizer.decode(expected.ids)
