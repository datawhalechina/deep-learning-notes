"""Delete rendered *_files directories from the repository."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for dir in ROOT.rglob('*_files'):
        if dir.is_dir():
            print(f'Deleting: {dir.name}.', flush=True)
            shutil.rmtree(dir)


if __name__ == '__main__':
    main()
