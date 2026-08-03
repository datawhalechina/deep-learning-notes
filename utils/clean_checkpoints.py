import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {'models'}


def main():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in EXCLUDED_DIRS]
        directory = Path(dirpath)

        for filename in filenames:
            p = directory / filename

            if p.suffix.lower() in {'.pt', '.pth'}:
                print(f'Deleting: {p.name}', flush=True)
                p.unlink()


if __name__ == '__main__':
    main()
