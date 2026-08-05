"""Clean up *_files directories in the current directory and its subdirectories."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for cache_dir in ROOT.rglob('*_files'):
        if cache_dir.is_dir():
            print(f'Deleting: {cache_dir.name}', flush=True)
            shutil.rmtree(cache_dir)


if __name__ == '__main__':
    main()
