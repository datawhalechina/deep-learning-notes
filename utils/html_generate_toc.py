"""Generate English and Chinese TOC files from the Quarto sidebars."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
QUARTO_CONFIG = ROOT / '_quarto-html.yml'
TITLE = 'Deep Learning Notes'
LANGUAGES = ('en', 'zh')

FRONT_MATTER_TITLE_RE = re.compile(r'^title:\s*(?P<title>.+?)\s*$')
NUMBER_RE = re.compile(r'ch(?P<chapter>\d+)\.(?P<section>\d+)')


def sort_key(path: Path) -> tuple[int, int, int, str]:
    """Sort QMD files by chapter and section numbers."""
    match = NUMBER_RE.search(path.stem)
    if match:
        return (0, int(match['chapter']), int(match['section']), path.name)
    return (1, 0, 0, path.name)


def read_qmd_title(path: Path) -> str:
    """Read the title from the front-matter of a QMD file."""
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
                return match['title'].strip().strip('"\'')

    raise RuntimeError(f'No front-matter title found in {path}.')


def read_chapters(language: str) -> list[tuple[str, list[Path]]]:
    """Read chapters and their QMD files from the Quarto sidebar configuration."""
    config = yaml.safe_load(QUARTO_CONFIG.read_text(encoding='utf-8'))
    chapters = []

    sidebars = config['website']['sidebar']
    sidebar = next(item for item in sidebars if item.get('id') == language)

    for part in sidebar['contents']:
        for chapter in part.get('contents', []):
            title = chapter['section']
            pattern = chapter['contents']
            files = sorted(ROOT.glob(pattern), key=sort_key)
            chapters.append((title, files))

    return chapters


def build_toc(chapters: list[tuple[str, list[Path]]]) -> str:
    """Build a Markdown TOC from chapters and their QMD files."""
    lines = [f'# {TITLE}', '']

    for chapter_title, files in chapters:
        lines.append(f'## {chapter_title}')
        lines.append('')

        for path in files:
            lines.append(f'- {read_qmd_title(path)}')

        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def main():
    """Generate TOC files for English and Chinese."""
    for language in LANGUAGES:
        chapters = read_chapters(language)
        book_toc = build_toc(chapters)

        markdown = ROOT / language / 'README.md'
        markdown.write_text(book_toc, encoding='utf-8')

    print('TOC files generated successfully.', flush=True)


if __name__ == '__main__':
    main()
