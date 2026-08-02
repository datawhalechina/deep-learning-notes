from tokenizers.decoders import ByteLevel

import dnnlpy.tokenizers as dltk


def test_byte_level_decoder_matches_hf_tokenizers():
    decoder = dltk.ByteLevelDecoder()
    hf_decoder = ByteLevel()

    tokens = ['Once', 'Ġupon', 'Ġa', 'Ġtime', '!']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected


def test_byte_level_decoder_restores_unicode_text():
    decoder = dltk.ByteLevelDecoder()
    hf_decoder = ByteLevel()

    tokens = ['CafÃ©', 'ĠðŁĺģ']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected


def test_byte_level_decoder_restores_leading_space():
    decoder = dltk.ByteLevelDecoder()
    hf_decoder = ByteLevel()

    tokens = ['ĠHello']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected


def test_byte_level_decoder_concatenates_tokens_without_separator():
    decoder = dltk.ByteLevelDecoder()
    hf_decoder = ByteLevel()

    tokens = ['hello', 'world']

    actual = decoder.decode(tokens)
    expected = hf_decoder.decode(tokens)
    assert actual == expected
