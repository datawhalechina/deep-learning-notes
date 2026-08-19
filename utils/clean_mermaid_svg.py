"""Remove unused Font Awesome data from a Mermaid-exported SVG."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG_ROOTS = (ROOT / 'zh', ROOT / 'en', ROOT / 'cs336')

STYLE_RE = re.compile(r'(<style\b[^>]*>)(.*?)(</style>)', re.IGNORECASE | re.DOTALL)
MERMAID_MARKER_RE = re.compile(r'#diagram-graph\s*\{', re.IGNORECASE)
FA6_RE = re.compile(r'Font Awesome(?: Free)? 6', re.IGNORECASE)

# Detect real Font Awesome usage outside the <style> block.
FA_CLASS_RE = re.compile(
    r'class\s*=\s*["\'][^"\']*(?:^|\s)(?:fa|fas|far|fab|fal|fat|fad|fa-[\w-]+)(?:\s|$)[^"\']*["\']',
    re.IGNORECASE,
)
FA_FONT_RE = re.compile(r'font-family\s*:\s*[^;"\']*Font\s*Awesome', re.IGNORECASE)


def human_size(n: int) -> str:
    """Format a byte count as a human-readable size."""
    units = ['B', 'KB', 'MB', 'GB']
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f'{value:.1f} {unit}'
        value /= 1024
    return f'{n} B'


def clean_svg(svg: str) -> tuple[str, int]:
    """Remove unused Font Awesome CSS from Mermaid SVG text."""
    match = STYLE_RE.search(svg)
    if not match:
        raise RuntimeError('No <style>...</style> block found.')

    style_open, css, style_close = match.groups()

    # Inspect only the actual SVG markup.
    svg_body = svg[: match.start()] + svg[match.end() :]

    if FA_CLASS_RE.search(svg_body) or FA_FONT_RE.search(svg_body):
        raise RuntimeError(
            'Font Awesome is actually used by SVG elements, so automatic removal was aborted.'
        )

    marker = MERMAID_MARKER_RE.search(css)
    if not marker:
        raise RuntimeError("Could not find Mermaid CSS marker '#diagram-graph {'.")

    removable_prefix = css[: marker.start()]

    # Extra safety check.
    if 'Font Awesome' not in removable_prefix and 'FontAwesome' not in removable_prefix:
        raise RuntimeError(
            "The CSS before '#diagram-graph {' does not look like Font Awesome; aborting."
        )

    # Keep Mermaid's CSS exactly as-is from #diagram-graph onward.
    cleaned_css = css[marker.start() :]

    cleaned = (
        svg[: match.start()]
        + style_open
        + cleaned_css
        + style_close
        + svg[match.end() :]
    )

    removed = len(svg.encode('utf-8')) - len(cleaned.encode('utf-8'))
    return cleaned, removed


def main() -> None:
    """Remove unused Font Awesome 6 data from SVG files."""
    for svg_root in SVG_ROOTS:
        for path in sorted(svg_root.rglob('*.svg')):
            original = path.read_text(encoding='utf-8')

            if not FA6_RE.search(original):
                continue

            cleaned, removed = clean_svg(original)
            path.write_text(cleaned, encoding='utf-8')

            relative_path = path.relative_to(ROOT)
            print(
                f'Cleaned {relative_path.name}: removed {human_size(removed)}.',
                flush=True,
            )


if __name__ == '__main__':
    main()
