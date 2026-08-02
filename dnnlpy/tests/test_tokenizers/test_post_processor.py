import json

from tokenizers import Encoding
from tokenizers.processors import ByteLevel

import dnnlpy.tokenizers as dltk


def _hf_encoding(encoding: dltk.Encoding) -> Encoding:
    hf_encoding = Encoding()
    hf_encoding.__setstate__(
        json.dumps(
            {
                'ids': encoding.ids,
                'type_ids': encoding.type_ids,
                'tokens': encoding.tokens,
                'words': list(range(len(encoding.ids))),
                'offsets': encoding.offsets,
                'special_tokens_mask': encoding.special_tokens_mask,
                'attention_mask': encoding.attention_mask,
                'overflowing': [],
                'sequence_ranges': {},
            }
        ).encode(),
    )
    return hf_encoding


def test_byte_level_post_processor_accepts_byte_level_options():
    post_processor = dltk.ByteLevelPostProcessor()
    hf_post_processor = ByteLevel()

    assert post_processor.add_prefix_space is None
    assert post_processor.trim_offsets == hf_post_processor.trim_offsets
    assert post_processor.use_regex is None


def test_byte_level_post_processor_can_disable_offset_trimming():
    encoding = dltk.Encoding(ids=[1], tokens=[' token '], offsets=[(0, 7)])
    post_processor = dltk.ByteLevelPostProcessor(trim_offsets=False)
    hf_post_processor = ByteLevel(trim_offsets=False)

    actual = post_processor.process(encoding)
    expected = hf_post_processor.process(_hf_encoding(encoding))

    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets


def test_byte_level_post_processor_trims_offsets_by_default():
    encoding = dltk.Encoding(ids=[1], tokens=[' token '], offsets=[(0, 7)])
    post_processor = dltk.ByteLevelPostProcessor()
    hf_post_processor = ByteLevel()

    actual = post_processor.process(encoding)
    expected = hf_post_processor.process(_hf_encoding(encoding))

    assert actual.tokens == expected.tokens
    assert actual.offsets == expected.offsets
