"""Copy Quarto image attributes from qmd files into generated notebooks."""

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
GENERATED_ROOT = ROOT / '_jupyter'
IMAGE_EXTENSIONS = {'.gif', '.jpeg', '.jpg', '.png', '.svg', '.webp'}
IMG_TAG_RE = re.compile(r'<img\b(?P<attrs>[^>]*)/?>', re.IGNORECASE)
HTML_ATTR_RE = re.compile(
    r'(?P<name>[A-Za-z_:][\w:.-]*)(?:\s*=\s*(?P<quote>["\'])(?P<value>.*?)\2)?',
    re.DOTALL,
)


def iter_qmd_files(paths: list[Path]) -> list[Path]:
    qmd_files: set[Path] = set()

    for path in paths:
        if path.is_file() and path.suffix == '.qmd':
            qmd_files.add(path)
        elif path.is_dir():
            qmd_files.update(p for p in path.rglob('*.qmd') if p.is_file())

    return sorted(qmd_files)


def parse_bracketed(
    text: str, start: int, open_char: str, close_char: str
) -> tuple[str, int] | None:
    depth = 0
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == '\\':
            escaped = True
            continue
        if char == open_char:
            depth += 1
            continue
        if char == close_char:
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1

    return None


def iter_markdown_images(text: str) -> list[tuple[str, str]]:
    images: list[tuple[str, str]] = []
    index = 0

    while True:
        marker = text.find('![', index)
        if marker == -1:
            break

        label = parse_bracketed(text, marker + 1, '[', ']')
        if label is None:
            index = marker + 2
            continue

        after_label = label[1]
        if after_label >= len(text) or text[after_label] != '(':
            index = after_label
            continue

        target = parse_bracketed(text, after_label, '(', ')')
        if target is None:
            index = after_label + 1
            continue

        after_target = target[1]
        if after_target >= len(text) or text[after_target] != '{':
            index = after_target
            continue

        attrs = parse_bracketed(text, after_target, '{', '}')
        if attrs is None:
            index = after_target + 1
            continue

        src = target[0].strip().split()[0]
        images.append((src, attrs[0]))
        index = attrs[1]

    return images


def split_attr_tokens(attrs: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None

    for char in attrs.strip():
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue
        if char.isspace():
            if current:
                tokens.append(''.join(current))
                current = []
            continue
        current.append(char)

    if current:
        tokens.append(''.join(current))

    return tokens


def parse_qmd_attrs(attrs: str) -> dict[str, str]:
    parsed: dict[str, str] = {}

    for token in split_attr_tokens(attrs):
        if token.startswith(('.', '#')) or '=' not in token:
            continue

        key, value = token.split('=', 1)
        key = key.strip()
        if not key or key.startswith('fig-'):
            continue

        parsed[key] = value.strip().strip('"\'')

    return parsed


def image_attrs_for_qmd(path: Path) -> dict[str, dict[str, str]]:
    attrs_by_src: dict[str, dict[str, str]] = {}
    text = path.read_text(encoding='utf-8')

    for src, attrs in iter_markdown_images(text):
        parsed = parse_qmd_attrs(attrs)
        if parsed and Path(src).suffix.lower() in IMAGE_EXTENSIONS:
            attrs_by_src[src] = parsed

    return attrs_by_src


def notebook_for_qmd(path: Path) -> Path:
    relative = path.relative_to(ROOT)
    return GENERATED_ROOT / relative.with_suffix('.ipynb')


def parse_html_attrs(attrs: str) -> dict[str, str]:
    parsed: dict[str, str] = {}

    for match in HTML_ATTR_RE.finditer(attrs):
        name = match.group('name')
        value = match.group('value')
        parsed[name.lower()] = value if value is not None else ''

    return parsed


def add_attrs_to_img_tag(
    match: re.Match[str], attrs_by_src: dict[str, dict[str, str]]
) -> str:
    attr_text = match.group('attrs')
    existing = parse_html_attrs(attr_text)
    src = html.unescape(existing.get('src', ''))
    image_attrs = attrs_by_src.get(src)
    if not image_attrs:
        return match.group(0)

    additions = [
        f'{key}="{html.escape(value, quote=True)}"'
        for key, value in image_attrs.items()
        if key.lower() not in existing
    ]
    if not additions:
        return match.group(0)

    self_closing = match.group(0).rstrip().endswith('/>')
    if self_closing:
        attr_text = attr_text.rstrip()
        if attr_text.endswith('/'):
            attr_text = attr_text[:-1].rstrip()

    close = ' />' if self_closing else '>'
    return f'<img{attr_text} {" ".join(additions)}{close}'


def update_source(
    source: Any, attrs_by_src: dict[str, dict[str, str]]
) -> tuple[Any, int]:
    replacements = 0

    def replace(text: str) -> str:
        nonlocal replacements

        def replace_tag(match: re.Match[str]) -> str:
            nonlocal replacements
            updated = add_attrs_to_img_tag(match, attrs_by_src)
            if updated != match.group(0):
                replacements += 1
            return updated

        return IMG_TAG_RE.sub(replace_tag, text)

    if isinstance(source, str):
        return replace(source), replacements

    if isinstance(source, list):
        return [
            replace(item) if isinstance(item, str) else item for item in source
        ], replacements

    return source, replacements


def update_notebook(
    path: Path, attrs_by_src: dict[str, dict[str, str]]
) -> tuple[bool, int]:
    notebook = json.loads(path.read_text(encoding='utf-8'))
    replacements = 0

    for cell in notebook.get('cells', []):
        if not isinstance(cell, dict):
            continue

        source, count = update_source(cell.get('source'), attrs_by_src)
        if count:
            cell['source'] = source
            replacements += count

    if replacements:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )

    return replacements > 0, replacements


def default_paths() -> list[Path]:
    return [ROOT / 'zh']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Copy qmd image width/height attributes into generated ipynb HTML.',
    )
    parser.add_argument(
        'paths',
        nargs='*',
        type=Path,
        help='Qmd files or directories to read. Defaults to zh.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [path if path.is_absolute() else ROOT / path for path in args.paths]
    if not paths:
        paths = default_paths()

    changed_files = 0
    total_replacements = 0

    for qmd_path in iter_qmd_files(paths):
        attrs_by_src = image_attrs_for_qmd(qmd_path)
        if not attrs_by_src:
            continue

        notebook_path = notebook_for_qmd(qmd_path)
        if not notebook_path.exists():
            print(
                f'Skipped missing notebook: {notebook_path.relative_to(ROOT)}',
                flush=True,
            )
            continue

        changed, replacements = update_notebook(notebook_path, attrs_by_src)
        if changed:
            changed_files += 1
            total_replacements += replacements
            print(
                f'Updated {notebook_path.relative_to(ROOT)} with {replacements} image attribute(s).',
                flush=True,
            )

    print(
        f'Updated {changed_files} notebook(s), added attributes to {total_replacements} image(s).',
        flush=True,
    )


if __name__ == '__main__':
    main()
