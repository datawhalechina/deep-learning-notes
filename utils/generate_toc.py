"""Generate English and Chinese TOC files from the Quarto sidebars."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUARTO_CONFIG = ROOT / '_quarto-html.yml'
TITLE = 'Deep Learning Notes'

FRONT_MATTER_TITLE_RE = re.compile(r'^title:\s*(?P<title>.+?)\s*$')
NUMBER_RE = re.compile(r'ch(?P<chapter>\d+)\.(?P<section>\d+)')


def strip_yaml_string(value: str) -> str:
    """Remove simple single or double quotes around a YAML scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def sort_key(path: Path) -> tuple[int, int, str]:
    """Sort chapter files by numeric chapter.section when possible."""
    match = NUMBER_RE.search(path.stem)
    if match:
        return (int(match['chapter']), int(match['section']), path.name)
    return (10**9, 10**9, path.name)


def read_qmd_title(path: Path) -> str:
    """Read the front-matter title from a QMD file."""
    in_front_matter = False

    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip() == '---':
            if in_front_matter:
                break
            in_front_matter = True
            continue

        if in_front_matter:
            match = FRONT_MATTER_TITLE_RE.match(line)
            if match:
                return strip_yaml_string(match['title'])

    raise ValueError(f'No front-matter title found in {path}')


def read_chapters(language: str) -> list[tuple[str, list[Path]]]:
    """Extract chapter names and their QMD files from _quarto.yml."""
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding='utf-8'))
    chapters: list[tuple[str, list[Path]]] = []

    sidebars = config['website']['sidebar']
    sidebar = next(item for item in sidebars if item.get('id') == language)

    for part in sidebar['contents']:
        for chapter in part.get('contents', []):
            chapter_title = chapter['section']
            pattern = chapter['contents']
            files = sorted(ROOT.glob(pattern), key=sort_key)
            chapters.append((chapter_title, files))

    return chapters


def build_toc(chapters: list[tuple[str, list[Path]]]) -> str:
    """Build the Markdown table of contents."""
    lines = [f'# {TITLE}', '']

    for chapter_title, files in chapters:
        lines.append(f'## {chapter_title}')
        lines.append('')

        for path in files:
            lines.append(f'- {read_qmd_title(path)}')

        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def main() -> None:
    outputs = {
        'en': ROOT / 'en' / 'README.md',
        'zh': ROOT / 'zh' / 'README.md',
    }

    for language, output in outputs.items():
        output.write_text(build_toc(read_chapters(language)), encoding='utf-8')

    print('TOC files generated successfully.', flush=True)


if __name__ == '__main__':
    main()
