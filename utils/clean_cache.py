"""Delete rendered *_files directories from the repository."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANG = ('zh', 'en')


def main():
    for lang in LANG:
        lang_dir = ROOT / lang

        if not lang_dir.is_dir():
            continue

        for dir in lang_dir.rglob('*_files'):
            if dir.is_dir():
                print(f'Deleting: {dir.name}.', flush=True)
                shutil.rmtree(dir)


if __name__ == '__main__':
    main()
