"""Remove embedded notebook attachments and point figure links at files."""

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / '_jupyter'
NOTEBOOK_DIRS = [GENERATED_ROOT / 'zh', GENERATED_ROOT / 'en']
ATTACHMENT_PREFIX = 'attachment:figures/'
FIGURE_PREFIX = 'figures/'


def clean_text(text: str) -> tuple[str, int]:
    replacements = text.count(ATTACHMENT_PREFIX)
    return text.replace(ATTACHMENT_PREFIX, FIGURE_PREFIX), replacements


def clean_source(source: Any) -> tuple[Any, int]:
    if isinstance(source, str):
        return clean_text(source)

    if not isinstance(source, list):
        return source, 0

    cleaned = []
    replacements = 0
    for item in source:
        if isinstance(item, str):
            item, count = clean_text(item)
            replacements += count
        cleaned.append(item)

    return cleaned, replacements


def clean_notebook(path: Path) -> tuple[bool, int, int]:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    removed_attachments = 0
    replaced_references = 0

    for cell in notebook.get('cells', []):
        if not isinstance(cell, dict):
            continue

        if 'attachments' in cell:
            removed_attachments += 1
            del cell['attachments']

        source, replacements = clean_source(cell.get('source'))
        if replacements:
            cell['source'] = source
            replaced_references += replacements

    changed = bool(removed_attachments or replaced_references)
    if not changed:
        return False, 0, 0

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return True, removed_attachments, replaced_references


def main() -> None:
    changed_files = 0
    removed_attachments = 0
    replaced_references = 0

    notebooks = sorted(
        path for directory in NOTEBOOK_DIRS for path in directory.rglob('*.ipynb')
    )

    for notebook in notebooks:
        changed, attachments, references = clean_notebook(notebook)
        if changed:
            changed_files += 1
            removed_attachments += attachments
            replaced_references += references
            print(f'Cleaning {notebook.name}', flush=True)

    print(
        'Cleaned '
        f'{changed_files} file(s), removed {removed_attachments} attachment block(s), '
        f'and rewrote {replaced_references} figure reference(s).',
        flush=True,
    )


if __name__ == '__main__':
    main()
