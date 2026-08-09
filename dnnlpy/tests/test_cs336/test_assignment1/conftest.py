import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.testing import assert_close

__all__ = [
    'FIXTURES_PATH',
    'SNAPSHOT_PATH',
    'SnapShot',
]

FIXTURES_PATH = Path(__file__).parent / 'fixtures'
SNAPSHOT_PATH = Path(__file__).parent / 'snapshots'


class SnapShot:
    """Snapshot testing utility for tensors saved with `torch.save`."""

    def __init__(
        self,
        snapshot_dir: str | os.PathLike[str] = 'snapshots',
        test_name: str | None = None,
        exact_match: bool = False,
        force_update: bool = False,
    ):
        """Initialize a Snapshot instance.

        Args:
            snapshot_dir (str | os.PathLike[str]): Directory to store snapshot files.
            test_name (str | None): Default test name for snapshot files. If None, must
                be provided when calling `assert_match`.
            exact_match (bool): If True, requires exact match of tensors. Otherwise,
                allows for approximate matching with specified tolerances.
            force_update (bool): If True, updates the snapshot file with the actual tensor
                instead of asserting a match. This is useful for updating snapshots when
                the expected output changes.
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.test_name = test_name
        self.exact_match = exact_match
        self.force_update = force_update

    def _get_snapshot_path(self, test_name: str) -> Path:
        """Get the path to the snapshot file for a given test name."""
        return self.snapshot_dir / f'{test_name}.pt'

    def assert_close(
        self,
        actual: Tensor,
        rtol: float = 1e-4,
        atol: float = 1e-2,
        test_name: str | None = None,
        force_update: bool | None = None,
    ) -> None:
        """Assert that `actual` matches its saved tensor snapshot.

        Args:
            actual (Tensor): The tensor to compare against the snapshot.
            rtol (float): Relative tolerance for approximate matching. Ignored if
                `exact_match` is True.
            atol (float): Absolute tolerance for approximate matching. Ignored if
                `exact_match` is True.
            test_name (str | None): The name of the test for which to retrieve the
                snapshot. If None, uses the default test name provided in init.
            force_update (bool | None): If True, updates the snapshot file with the
                actual tensor instead of asserting a match. If None, uses the default
                value provided during initialization.
        """
        if force_update is None:
            force_update = self.force_update

        if self.exact_match:
            rtol = atol = 0

        if test_name is None:
            test_name = self.test_name
        if test_name is None:
            raise RuntimeError('Test name must be provided or set as default.')

        snapshot_path = self._get_snapshot_path(test_name)
        if force_update:
            torch.save(actual.detach().cpu(), snapshot_path)
            return

        expected = torch.load(snapshot_path, map_location='cpu', weights_only=True)
        assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> SnapShot:
    """Fixture to provide a Snapshot instance for snapshot testing."""
    exact_match = request.config.getoption('--snapshot-exact', default=False)
    if exact_match is None:
        exact_match = False

    return SnapShot(
        snapshot_dir=Path(__file__).parent / 'snapshots',
        test_name=request.node.name,
        exact_match=exact_match,
    )


@pytest.fixture
def ts_state_dict() -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Fixture to load the state dict and config for TorchScript tests."""
    ts_path = FIXTURES_PATH / 'ts_tests'
    state_dict = torch.load(ts_path / 'model.pt', map_location='cpu')
    state_dict = cast(dict[str, Tensor], state_dict)

    with open(ts_path / 'model_config.json') as fp:
        config = json.load(fp)
        config = cast(dict[str, Any], config)

    return state_dict, config


@pytest.fixture
def n_layers() -> int:
    return 3


@pytest.fixture
def vocab_size() -> int:
    return 10_000


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def n_queries() -> int:
    return 12


@pytest.fixture
def n_keys() -> int:
    return 16


@pytest.fixture
def n_heads() -> int:
    return 4


@pytest.fixture
def d_head() -> int:
    return 16


@pytest.fixture
def d_model(n_heads: int, d_head: int) -> int:
    return n_heads * d_head


@pytest.fixture
def d_ff() -> int:
    return 128


@pytest.fixture
def query(batch_size: int, n_queries: int, d_model: int) -> Tensor:
    torch.manual_seed(1)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def key(batch_size: int, n_keys: int, d_model: int) -> Tensor:
    torch.manual_seed(2)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def value(batch_size: int, n_keys: int, d_model: int) -> Tensor:
    torch.manual_seed(3)
    return torch.randn(batch_size, n_keys, d_model)


@pytest.fixture
def embeddings(batch_size: int, n_queries: int, d_model: int) -> Tensor:
    torch.manual_seed(4)
    return torch.randn(batch_size, n_queries, d_model)


@pytest.fixture
def attn_mask(batch_size: int, n_queries: int, n_keys: int) -> Tensor:
    torch.manual_seed(5)
    return torch.randn(batch_size, n_queries, n_keys) > 0.5


@pytest.fixture
def theta() -> float:
    return 10000.0


@pytest.fixture
def token_ids(batch_size: int, n_queries: int) -> Tensor:
    torch.manual_seed(6)
    return torch.randint(0, 10_000, (batch_size, n_queries))


@pytest.fixture
def pos_ids(n_queries: int) -> Tensor:
    return torch.arange(0, n_queries)
