import math
from collections import Counter

import pytest
import torch
from torch.testing import assert_close

from .adapters import run_get_batch

__all__ = ['test_get_batch']


def test_get_batch():
    """Test the `get_batch` function for correctness and randomness.

    This test checks that the `get_batch` function correctly samples input sequences and
    their corresponding labels from a given dataset. It also verifies that the sampling is
    random and that the starting indices of the sampled sequences are uniformly distributed
    across the possible range of starting indices.
    """
    dataset = torch.arange(100)
    context_length = 7
    batch_size = 32
    device = 'cpu'

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000

    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        assert_close(x[:, 1:], y[:, :-1])

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0

    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size)
        * (1 / num_possible_starting_indices)
        * (1 - (1 / num_possible_starting_indices))
    )

    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise AssertionError(
                f'Starting index {starting_index} occurs {count} times, but expected '
                f'at least {occurrences_lower_bound}.'
            )
        if count > occurrences_upper_bound:
            raise AssertionError(
                f'Starting index {starting_index} occurs {count} times, but expected '
                f'at most {occurrences_upper_bound}.'
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device='cuda:99',
        )

        if torch.cuda.is_available():
            assert 'CUDA error' in str(excinfo.value)
        else:
            assert 'Torch not compiled with CUDA enabled' in str(excinfo.value)
