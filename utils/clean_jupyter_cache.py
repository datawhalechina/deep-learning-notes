"""Delete .jupyter_cache directories from the repository."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ('zh', 'en', 'cs336')


def main():
    for path in PATH:
        folder = ROOT / path

        if not folder.is_dir():
            continue

        for cache in folder.rglob('.jupyter_cache'):
            if cache.is_dir():
                print(f'Deleting: {cache}', flush=True)
                shutil.rmtree(cache)


if __name__ == '__main__':
    main()
