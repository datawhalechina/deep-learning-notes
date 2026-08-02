import os
import sys
from pathlib import Path

import pytest

os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(PACKAGE_ROOT))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--slow',
        action='store_true',
        default=False,
        help='run tests marked as slow',
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('markers', 'slow: marks tests as slow')


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip slow tests unless pytest was invoked with --slow."""
    if config.getoption('--slow'):
        return

    skip_slow = pytest.mark.skip(reason='pass --slow to run this test')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
