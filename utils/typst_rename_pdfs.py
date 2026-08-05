"""Move rendered Typst PDFs into the top-level _typst directory."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPST_DIR = ROOT / '_typst'

PDF_NAMES = {
    'zh': 'deep-learning-notes-zh.pdf',
    'en': 'deep-learning-notes-en.pdf',
}


def move_pdf(language: str, target_name: str) -> None:
    language_dir = TYPST_DIR / language
    target = TYPST_DIR / target_name

    if not language_dir.exists():
        path = language_dir.relative_to(ROOT)
        print(f'Skipped missing directory: {path}.', flush=True)
        return

    source = language_dir / 'deep-learning-notes.pdf'
    target.unlink(missing_ok=True)
    source.replace(target)

    src_path = source.relative_to(ROOT)
    tgt_path = target.relative_to(ROOT)
    print(f'Moved {src_path} -> {tgt_path}.', flush=True)


def remove_language_dirs() -> None:
    for language in PDF_NAMES:
        language_dir = TYPST_DIR / language
        if language_dir.exists():
            shutil.rmtree(language_dir)
            print(f'Deleted {language_dir.relative_to(ROOT)}.', flush=True)


def main() -> None:
    for language, target_name in PDF_NAMES.items():
        move_pdf(language, target_name)
    remove_language_dirs()


if __name__ == '__main__':
    main()
