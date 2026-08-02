from tokenizers.normalizers import ByteLevel, Lowercase, Strip

import dnnlpy.tokenizers as dltk


def test_byte_level_normalizer_matches_hf_tokenizers():
    normalizer = dltk.ByteLevelNormalizer()
    hf_normalizer = ByteLevel()

    text = 'Café 😁'

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_lowercase_normalizer_matches_hf_tokenizers():
    normalizer = dltk.LowercaseNormalizer()
    hf_normalizer = Lowercase()

    text = 'Hello WORLD'

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_matches_hf_tokenizers():
    normalizer = dltk.StripNormalizer()
    hf_normalizer = Strip()

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_can_strip_left_only():
    normalizer = dltk.StripNormalizer(left=True, right=False)
    hf_normalizer = Strip(left=True, right=False)

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_can_strip_right_only():
    normalizer = dltk.StripNormalizer(left=False, right=True)
    hf_normalizer = Strip(left=False, right=True)

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected


def test_strip_normalizer_can_disable_stripping():
    normalizer = dltk.StripNormalizer(left=False, right=False)
    hf_normalizer = Strip(left=False, right=False)

    text = '  Hello  '

    actual = normalizer.normalize(text)
    expected = hf_normalizer.normalize_str(text)
    assert actual == expected
