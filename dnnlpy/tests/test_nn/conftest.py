import pytest


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def src_len() -> int:
    return 4


@pytest.fixture
def tgt_len() -> int:
    return 8


@pytest.fixture
def d_model() -> int:
    return 6


@pytest.fixture
def num_heads() -> int:
    return 2


@pytest.fixture
def head_dim(d_model: int, num_heads: int) -> int:
    return d_model // num_heads


@pytest.fixture
def key_dim() -> int:
    return 6


@pytest.fixture
def value_dim() -> int:
    return 8
