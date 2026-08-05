"""Remove embedded notebook attachments and point figure links at files."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / '_jupyter'
NOTEBOOK_DIRS = [GENERATED_ROOT / 'zh', GENERATED_ROOT / 'en']

ATTACHMENT_PREFIX = 'attachment:figures/'
FIGURE_PREFIX = 'figures/'


def clean_text(text: str) -> tuple[str, bool]:
    """Remove attachment references from a string and replace them with figure links."""
    cleaned = text.replace(ATTACHMENT_PREFIX, FIGURE_PREFIX)
    changed = cleaned != text
    return cleaned, changed


def clean_source(source: Any) -> tuple[Any, bool]:
    """Clean the source of a notebook cell, which can be a string or a list of strings."""
    if isinstance(source, str):
        return clean_text(source)

    if not isinstance(source, list):
        return source, False

    cleaned = []
    changed = False

    for src in source:
        if isinstance(src, str):
            src, src_changed = clean_text(src)
            changed = changed or src_changed
        cleaned.append(src)

    return cleaned, changed


def clean_notebook(path: Path) -> bool:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    changed = False

    for cell in notebook.get('cells', []):
        if not isinstance(cell, dict):
            continue

        if 'attachments' in cell:
            del cell['attachments']
            changed = True

        src, src_changed = clean_source(cell.get('source'))

        if src_changed:
            cell['source'] = src
            changed = True

    if not changed:
        return False

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return True


def main():
    """Clean all notebooks in the specified directories."""
    notebooks = sorted(path for dir in NOTEBOOK_DIRS for path in dir.rglob('*.ipynb'))
    for notebook in notebooks:
        if clean_notebook(notebook):
            print(f'Cleaned {notebook.relative_to(ROOT).name}.', flush=True)


if __name__ == '__main__':
    main()
