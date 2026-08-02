import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache, partial

from ..configtools import get_num_workers, has_gil

__all__ = [
    'bytes_to_unicode',
    'parallel_map',
    'unicode_to_bytes',
]


def _bytes_to_unicode() -> dict[int, str]:
    """Create a mapping from byte values (0-255) to Unicode characters."""
    byte_encoder = {
        byte: chr(codepoint)
        for byte, codepoint in [
            *[(byte, byte) for byte in range(ord('!'), ord('~') + 1)],
            *[(byte, byte) for byte in range(ord('¡'), ord('¬') + 1)],
            *[(byte, byte) for byte in range(ord('®'), ord('ÿ') + 1)],
        ]
    }

    next_codepoint = 256
    for byte in range(256):
        if byte not in byte_encoder:
            byte_encoder[byte] = chr(next_codepoint)
            next_codepoint += 1

    return byte_encoder


BYTES_TO_UNICODE = _bytes_to_unicode()
UNICODE_TO_BYTES = {char: byte for byte, char in BYTES_TO_UNICODE.items()}


@lru_cache(maxsize=100_000)
def bytes_to_unicode(text: str) -> str:
    return ''.join(BYTES_TO_UNICODE[byte] for byte in text.encode('utf-8'))


@lru_cache(maxsize=100_000)
def unicode_to_bytes(text: str) -> bytes:
    return bytes(UNICODE_TO_BYTES[char] for char in text)


def _batch_map[T, R](func: Callable[[T], R], values: Iterable[T]) -> list[R]:
    return [func(value) for value in values]


def parallel_map[T, R](
    func: Callable[[T], R],
    batches: Iterable[Iterable[T]],
    num_workers: int | None = None,
    buffersize: int | None = None,
) -> Iterator[R]:
    """Map a function over batches of values using threads or processes.

    Args:
        func (Callable): A callable that takes one value and returns a result.
        batches (Iterable): An iterable of batches. Each batch is processed as one
            worker task, and its results are yielded individually in input order.
        num_workers (int, default: 1): The number of worker threads to use for
            parallel processing.
        buffersize (int | None, optional): The maximum number of results to buffer
            before yielding. If None, defaults to `num_workers * 4`.
    """
    num_workers = get_num_workers(num_workers)

    if buffersize is None:
        buffersize = num_workers * 4

    if num_workers <= 1:
        for batch in batches:
            for value in batch:
                yield func(value)
    else:
        if has_gil():
            executor = ProcessPoolExecutor(num_workers)
        else:
            executor = ThreadPoolExecutor(num_workers)

        with executor:
            batch_map = partial(_batch_map, func)

            if sys.version_info >= (3, 14):
                results = executor.map(batch_map, batches, buffersize=buffersize)
            else:
                results = _buffered_map(executor, batch_map, batches, buffersize)

            for result in results:
                yield from result


# TODO: Remove this function when Python 3.14 is the minimum supported version.
def _buffered_map[T, R](
    executor: Executor,
    func: Callable[[T], R],
    values: Iterable[T],
    buffersize: int,
) -> Iterator[R]:
    """Map lazily while keeping at most buffer_size pending futures.

    This function is used by Python < 3.14, where Executor.map does not support
    `buffersize` argument.
    """
    if buffersize < 1:
        raise AssertionError('`buffer_size` must be at least 1.')

    iterator = iter(values)
    pending = deque()

    for _ in range(buffersize):
        try:
            value = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(func, value))

    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            value = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(func, value))
