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


def sort_key(path: Path) -> tuple[int, int, str]:
    match = NUMBER_RE.search(path.stem)
    if match:
        return (int(match['chapter']), int(match['section']), path.name)
    return (10**9, 10**9, path.name)


def read_qmd_title(path: Path) -> str:
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

    raise ValueError(f'No front-matter title found in {path}')


def read_chapters(language: str) -> list[tuple[str, list[Path]]]:
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
    lines = [f'# {TITLE}', '']

    for chapter_title, files in chapters:
        lines.append(f'## {chapter_title}')
        lines.append('')

        for path in files:
            lines.append(f'- {read_qmd_title(path)}')

        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def main() -> None:
    for language in LANGUAGES:
        output = ROOT / language / 'README.md'
        output.write_text(build_toc(read_chapters(language)), encoding='utf-8')

    print('TOC files generated successfully.', flush=True)


if __name__ == '__main__':
    main()
