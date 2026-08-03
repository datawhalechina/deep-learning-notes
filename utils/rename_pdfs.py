"""Move rendered Typst PDFs into the top-level _typst directory."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPST_DIR = ROOT / '_typst'

PDFS = {
    'zh': 'deep-learning-notes-zh.pdf',
    'en': 'deep-learning-notes-en.pdf',
}


def move_pdf(language: str, target_name: str) -> None:
    """Move one language PDF to _typst with its final filename."""
    language_dir = TYPST_DIR / language
    target = TYPST_DIR / target_name
    if not language_dir.exists():
        path = language_dir.relative_to(ROOT)
        print(f'Skipped missing directory: {path}.', flush=True)
        return

    candidates = [
        language_dir / 'deep-learning-notes.pdf',
        language_dir / target_name,
    ]

    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        if target.exists():
            path = target.relative_to(ROOT)
            print(f'Target already exists: {path}.', flush=True)
            return

        path = ', '.join(str(path.relative_to(ROOT)) for path in candidates)
        raise FileNotFoundError(f'Expected one of: {path}')

    if target.exists():
        target.unlink()
    source.replace(target)

    src_path = source.relative_to(ROOT)
    tgt_path = target.relative_to(ROOT)
    print(f'Moved {src_path} -> {tgt_path}.', flush=True)


def remove_language_dirs() -> None:
    """Remove Typst language directories after PDFs are moved."""
    for language in PDFS:
        language_dir = TYPST_DIR / language
        if language_dir.exists():
            shutil.rmtree(language_dir)
            print(f'Deleted {language_dir.relative_to(ROOT)}.', flush=True)


def main() -> None:
    for language, target_name in PDFS.items():
        move_pdf(language, target_name)
    remove_language_dirs()


if __name__ == '__main__':
    main()
