"""Delete PyTorch checkpoints outside model directories."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {'models'}
CHECKPOINT_SUFFIXES = {'.pt', '.pth'}


def main() -> None:
    for directory, dirnames, filenames in ROOT.walk():
        dirnames[:] = [dirname for dirname in dirnames if dirname not in EXCLUDED_DIRS]

        for filename in filenames:
            path = directory / filename
            if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                print(f'Deleting: {path.name}', flush=True)
                path.unlink()


if __name__ == '__main__':
    main()
