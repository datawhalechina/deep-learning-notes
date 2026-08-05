"""Copy QMD image sizes into generated notebooks."""

import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / '_jupyter'
QMD_DIRS = [ROOT / 'zh', ROOT / 'en']
IMAGE_SUFFIXES = {'.gif', '.jpeg', '.jpg', '.png', '.svg', '.webp'}

MARKDOWN_IMAGE_RE = re.compile(r'!\[.*?\]\((?P<src>[^)\s]+)[^)]*\)\{(?P<attrs>[^}]*)\}')
SIZE_RE = re.compile(
    r'\b(?P<name>width|height)\s*=\s*'
    r'(?P<quote>["\']?)(?P<value>[^"\'\s}]+)(?P=quote)'
)
IMG_TAG_RE = re.compile(r'<img\b[^>]*?/?>', re.IGNORECASE)
SRC_RE = re.compile(
    r'\bsrc\s*=\s*(?P<quote>["\'])(?P<src>.*?)(?P=quote)', re.IGNORECASE
)


def read_image_sizes(path: Path) -> dict[str, dict[str, str]]:
    sizes_by_src = {}
    text = path.read_text(encoding='utf-8')

    for image in MARKDOWN_IMAGE_RE.finditer(text):
        src = image['src']
        if Path(src).suffix.lower() not in IMAGE_SUFFIXES:
            continue

        sizes = {
            size['name']: size['value'] for size in SIZE_RE.finditer(image['attrs'])
        }
        if sizes:
            sizes_by_src[src] = sizes

    return sizes_by_src


def add_sizes(text: str, sizes_by_src: dict[str, dict[str, str]]) -> tuple[str, int]:
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements

        tag = match.group(0)
        src_match = SRC_RE.search(tag)
        if src_match is None:
            return tag

        sizes = sizes_by_src.get(html.unescape(src_match['src']))
        if not sizes:
            return tag

        additions = [
            f'{name}="{html.escape(value, quote=True)}"'
            for name, value in sizes.items()
            if re.search(rf'\b{name}\s*=', tag, re.IGNORECASE) is None
        ]
        if not additions:
            return tag

        replacements += 1
        self_closing = tag.rstrip().endswith('/>')
        opening = tag.rstrip()[: -2 if self_closing else -1].rstrip()
        closing = ' />' if self_closing else '>'
        return f'{opening} {" ".join(additions)}{closing}'

    return IMG_TAG_RE.sub(replace, text), replacements


def update_source(
    source: Any, sizes_by_src: dict[str, dict[str, str]]
) -> tuple[Any, int]:
    if isinstance(source, str):
        return add_sizes(source, sizes_by_src)

    if not isinstance(source, list):
        return source, 0

    updated = []
    replacements = 0
    for item in source:
        if isinstance(item, str):
            item, count = add_sizes(item, sizes_by_src)
            replacements += count
        updated.append(item)

    return updated, replacements


def update_notebook(path: Path, sizes_by_src: dict[str, dict[str, str]]) -> int:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    replacements = 0

    for cell in notebook.get('cells', []):
        if not isinstance(cell, dict):
            continue

        source, count = update_source(cell.get('source'), sizes_by_src)
        if count:
            cell['source'] = source
            replacements += count

    if replacements:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    return replacements


def main() -> None:
    changed_files = 0
    total_replacements = 0

    qmd_files = sorted(
        path for directory in QMD_DIRS for path in directory.rglob('*.qmd')
    )
    for qmd_path in qmd_files:
        sizes_by_src = read_image_sizes(qmd_path)
        if not sizes_by_src:
            continue

        notebook_path = GENERATED_ROOT / qmd_path.relative_to(ROOT).with_suffix(
            '.ipynb'
        )
        if not notebook_path.exists():
            print(
                f'Skipped missing notebook: {notebook_path.relative_to(ROOT)}',
                flush=True,
            )
            continue

        replacements = update_notebook(notebook_path, sizes_by_src)
        if replacements:
            changed_files += 1
            total_replacements += replacements
            print(
                f'Updated {notebook_path.relative_to(ROOT)} with '
                f'{replacements} image size(s).',
                flush=True,
            )

    print(
        f'Updated {changed_files} notebook(s), '
        f'added sizes to {total_replacements} image(s).',
        flush=True,
    )


if __name__ == '__main__':
    main()
