"""Move rendered Typst PDFs into the top-level _typst directory."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TYPST_DIR = ROOT / '_typst'

PDF_NAMES = {
    'zh': 'deep-learning-notes-zh.pdf',
    'en': 'deep-learning-notes-en.pdf',
}


def find_source(language_dir: Path, target_name: str) -> Path | None:
    """Find the rendered PDF without depending on the checkout directory name."""
    candidates = (
        language_dir / 'deep-learning-notes.pdf',
        language_dir / target_name,
        language_dir / f'{ROOT.name}.pdf',
    )
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate

    rendered_pdfs = list(language_dir.glob('*.pdf'))
    if len(rendered_pdfs) == 1:
        return rendered_pdfs[0]
    if not rendered_pdfs:
        return None

    paths = ', '.join(str(path.relative_to(ROOT)) for path in rendered_pdfs)
    raise RuntimeError(f'Ambiguous rendered PDFs: {paths}.')


def move_pdf(language: str, target_name: str) -> None:
    """Move the rendered PDF for a given language to the top-level _typst directory."""
    language_dir = TYPST_DIR / language
    target = TYPST_DIR / target_name

    if not language_dir.exists():
        path = language_dir.relative_to(ROOT)
        print(f'Skipped missing directory: {path}.', flush=True)
        return

    source = find_source(language_dir, target_name)
    if source is None:
        if target.is_file():
            path = target.relative_to(ROOT)
            print(f'Target already exists: {path}.', flush=True)
            return

        path = language_dir.relative_to(ROOT)
        print(f'Skipped missing PDF: {path}.', flush=True)
        return

    target.unlink(missing_ok=True)
    source.replace(target)

    src_path = source.relative_to(ROOT)
    tgt_path = target.relative_to(ROOT)
    print(f'Moved {src_path} -> {tgt_path}.', flush=True)


def remove_language_dirs() -> None:
    """Remove the language directories after moving the PDFs."""
    for language in PDF_NAMES:
        language_dir = TYPST_DIR / language
        if language_dir.exists():
            shutil.rmtree(language_dir)
            print(f'Deleted {language_dir.relative_to(ROOT)}.', flush=True)


def main():
    """Move rendered Typst PDFs into the top-level _typst directory."""
    for language, target in PDF_NAMES.items():
        move_pdf(language, target)
    remove_language_dirs()


if __name__ == '__main__':
    main()
