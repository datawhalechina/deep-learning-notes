"""Delete PyTorch checkpoints outside model directories."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOTS = (ROOT / 'zh', ROOT / 'en')
EXCLUDED_DIRS = {'models'}
CHECKPOINT_SUFFIXES = {'.pt', '.pth'}


def main():
    for checkpoint_root in CHECKPOINT_ROOTS:
        for dir, dirnames, filenames in checkpoint_root.walk():
            dirnames[:] = [name for name in dirnames if name not in EXCLUDED_DIRS]

            for filename in filenames:
                path = dir / filename
                if path.suffix.lower() in CHECKPOINT_SUFFIXES:
                    print(f'Deleting: {path.name}', flush=True)
                    path.unlink()


if __name__ == '__main__':
    main()
