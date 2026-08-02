"""Remove embedded notebook attachments and point figure links at files."""

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ATTACHMENT_PREFIX = 'attachment:figures/'
FIGURE_PREFIX = 'figures/'


def find_notebooks(paths: list[Path]) -> list[Path]:
    notebooks = set()

    for path in paths:
        if path.is_file() and path.suffix == '.ipynb':
            notebooks.add(path)
        elif path.is_dir():
            notebooks.update(p for p in path.rglob('*.ipynb') if p.is_file())

    return sorted(notebooks)


def clean_source(source: Any) -> tuple[Any, int]:
    replacements = 0

    if isinstance(source, str):
        replacements = source.count(ATTACHMENT_PREFIX)
        return source.replace(ATTACHMENT_PREFIX, FIGURE_PREFIX), replacements

    if isinstance(source, list):
        cleaned = []
        for item in source:
            if isinstance(item, str):
                replacements += item.count(ATTACHMENT_PREFIX)
                cleaned.append(item.replace(ATTACHMENT_PREFIX, FIGURE_PREFIX))
            else:
                cleaned.append(item)
        return cleaned, replacements

    return source, replacements


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

    changed = removed_attachments > 0 or replaced_references > 0
    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    return changed, removed_attachments, replaced_references


def default_paths() -> list[Path]:
    generated = ROOT / '_jupyter'
    if generated.exists():
        return [generated]

    return [ROOT / 'zh', ROOT / 'en']


def parse_args() -> argparse.Namespace:
    description = 'Remove ipynb attachments and rewrite attachment figure URLs.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'paths',
        nargs='*',
        type=Path,
        help='Notebook files or directories to clean. Defaults to _jupyter.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path if path.is_absolute() else ROOT / path for path in args.paths]
    if not paths:
        paths = default_paths()

    notebooks = find_notebooks(paths)
    changed_files = 0
    removed_attachments = 0
    replaced_references = 0

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
