import heapq
import sys

from .base import (
    Decoder as Decoder,
    Encoding as Encoding,
    Model as Model,
    Normalizer as Normalizer,
    PostProcessor as PostProcessor,
    PreTokenizer as PreTokenizer,
    Tokenizer as Tokenizer,
    TraditionalTokenizer as TraditionalTokenizer,
    Trainer as Trainer,
)
from .decoder import ByteLevelDecoder as ByteLevelDecoder
from .model import BPE as BPE
from .normalizer import (
    ByteLevelNormalizer as ByteLevelNormalizer,
    LowercaseNormalizer as LowercaseNormalizer,
    StripNormalizer as StripNormalizer,
)
from .post_processor import ByteLevelPostProcessor as ByteLevelPostProcessor
from .pre_tokenizer import (
    ByteLevelPreTokenizer as ByteLevelPreTokenizer,
    WhitespacePreTokenizer as WhitespacePreTokenizer,
)
from .traditional import (
    CharacterTokenizer as CharacterTokenizer,
    WordTokenizer as WordTokenizer,
)
from .trainer import BPETrainer as BPETrainer
from .utils import (
    bytes_to_unicode as bytes_to_unicode,
    parallel_map as parallel_map,
    unicode_to_bytes as unicode_to_bytes,
)

if sys.version_info < (3, 14):

    def patch_heapq_for_max_heap():
        """Patch the `heapq` module to add support for max heaps in Python < 3.14."""

        def _heappush_max[T](heap: list[T], item: T) -> None:
            heap.append(item)
            heapq._siftdown_max(heap, 0, len(heap) - 1)

        heapq._heappush_max = _heappush_max

        for attr in dir(heapq):
            if attr.startswith('_heap') and attr.endswith('_max'):
                setattr(heapq, attr[1:], getattr(heapq, attr))

        for func in (
            '_heappush_max',
            '_heappop_max',
            '_heapreplace_max',
            '_heapify_max',
        ):
            if not hasattr(heapq, func):
                raise ImportError(
                    f'Failed to patch heapq for max heap support: {func} not found.'
                )

    patch_heapq_for_max_heap()
    del patch_heapq_for_max_heap
