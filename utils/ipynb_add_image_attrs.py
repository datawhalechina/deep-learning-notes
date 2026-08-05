"""Copy QMD image sizes into generated Jupyter Notebooks."""

import html
import json
import re
from pathlib import Path
from typing import Any

type Records = dict[str, dict[str, str]]

ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / '_jupyter'
QMD_DIRS = [ROOT / 'zh', ROOT / 'en']
IMAGE_SUFFIXES = {'.gif', '.jpeg', '.jpg', '.png', '.svg', '.webp'}

MARKDOWN_IMAGE_RE = re.compile(
    r'!\[.*?\]\((?P<src>[^)\s]+)[^)]*\)'
    r'\{(?P<attrs>[^}]*)\}'
)
SIZE_RE = re.compile(
    r'\b(?P<name>width|height)\s*=\s*'
    r"(?P<quote>[\"']?)(?P<value>[^\"'\s}]+)(?P=quote)"
)
IMG_TAG_RE = re.compile(
    r'<img\b[^>]*?/?>',
    re.IGNORECASE,
)
SRC_RE = re.compile(
    r"\bsrc\s*=\s*(?P<quote>[\"'])(?P<src>.*?)(?P=quote)",
    re.IGNORECASE,
)


def read_image_sizes(path: Path) -> Records:
    """Read image width and height attributes from a QMD file."""
    records = {}
    text = path.read_text(encoding='utf-8')

    for image in MARKDOWN_IMAGE_RE.finditer(text):
        src = image['src']
        if Path(src).suffix.lower() not in IMAGE_SUFFIXES:
            continue

        sizes = {
            size['name']: size['value'] for size in SIZE_RE.finditer(image['attrs'])
        }
        if sizes:
            records[src] = sizes

    return records


def add_image_sizes(text: str, records: Records) -> tuple[str, bool]:
    """Add missing image size attributes to HTML img tags."""
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed

        tag = match.group(0)
        src_match = SRC_RE.search(tag)

        if src_match is None:
            return tag

        src = html.unescape(src_match['src'])
        sizes = records.get(src)

        if not sizes:
            return tag

        additions = [
            f'{name}="{html.escape(value)}"'
            for name, value in sizes.items()
            if re.search(rf'\b{re.escape(name)}\s*=', tag, re.IGNORECASE) is None
        ]

        if not additions:
            return tag

        changed = True

        stripped_tag = tag.rstrip()
        self_closing = stripped_tag.endswith('/>')

        if self_closing:
            opening = stripped_tag[:-2].rstrip()
            closing = ' />'
        else:
            opening = stripped_tag[:-1].rstrip()
            closing = '>'

        return f'{opening} {" ".join(additions)}{closing}'

    updated_text = IMG_TAG_RE.sub(replace, text)
    return updated_text, changed


def update_source(source: Any, records: Records) -> tuple[Any, bool]:
    """Update image tags in a notebook cell source."""
    if isinstance(source, str):
        return add_image_sizes(source, records)

    if not isinstance(source, list):
        return source, False

    updated = []
    changed = False

    for item in source:
        if isinstance(item, str):
            item, item_changed = add_image_sizes(item, records)

            if item_changed:
                changed = True

        updated.append(item)

    return updated, changed


def update_notebook(path: Path, records: Records) -> bool:
    """Update image sizes in a generated notebook."""
    notebook = json.loads(path.read_text(encoding='utf-8'))
    changed = False

    for cell in notebook.get('cells', []):
        if not isinstance(cell, dict):
            continue

        src, src_changed = update_source(cell.get('source'), records)

        if src_changed:
            cell['source'] = src
            changed = True

    if changed:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    return changed


def main():
    """Update generated notebooks using image sizes from QMD files."""
    files = sorted(path for dir in QMD_DIRS for path in dir.rglob('*.qmd'))

    for path in files:
        records = read_image_sizes(path)

        if not records:
            continue

        relative_path = path.relative_to(ROOT)
        notebook_path = GENERATED_ROOT / relative_path.with_suffix('.ipynb')

        if not notebook_path.exists():
            print(
                f'Skipped missing notebook: {notebook_path.relative_to(ROOT).name}.',
                flush=True,
            )
            continue

        if update_notebook(notebook_path, records):
            print(f'Updated {notebook_path.relative_to(ROOT).name}.', flush=True)


if __name__ == '__main__':
    main()
