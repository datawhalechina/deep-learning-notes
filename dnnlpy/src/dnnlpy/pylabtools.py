# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false

# TODO: Remove this when the minimum supported Python version is 3.14.
# See <PEP 649> and <PEP 749> for more details.
from __future__ import annotations

import struct
from binascii import b2a_base64
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

type FigureFormat = Literal['png', 'retina', 'highdpi', 'jpg', 'jpeg', 'svg', 'pdf']

__all__ = ['set_matplotlib_format']


def _get_ipython_shell() -> InteractiveShell:
    """Return the active IPython shell for inline figure configuration."""
    try:
        from IPython.core.getipython import get_ipython
    except ImportError as err:
        raise ImportError(
            'set_matplotlib_format() function requires IPython. '
            'Install it with `pip install ipython` if you want to use this function.'
        ) from err

    shell = get_ipython()
    if shell is None:
        raise RuntimeError(
            'set_matplotlib_format() function must be called from an IPython environment.'
        )
    return shell


def set_matplotlib_format(fmt: FigureFormat, dpi_ratio: int = 3) -> None:
    """Select figure formats for the inline backend.

    Args:
        fmt (Literal['png', 'retina', 'highdpi', 'jpeg', 'svg', 'pdf']):
            One figure format to enable: {'png', 'retina', 'highdpi', 'jpeg', 'svg', 'pdf'}.
        dpi_ratio (int, default: 3):
            The dpi ratio to use for 'highdpi' format. Default is 3, which means
            3 times the normal dpi. Only used when fmt is 'highdpi'.
    """
    shell = _get_ipython_shell()
    set_figure_format(shell, fmt, dpi_ratio)


def read_png_header(data: bytes) -> tuple[int, int]:
    """read the (width, height) from a png header."""
    ihdr = data.index(b'IHDR')
    return struct.unpack('>ii', data[ihdr + 4 : ihdr + 12])


def print_figure(
    fig: Figure,
    fmt: str = 'png',
    bbox_inches: str = 'tight',
    dpi_ratio: int = 3,
    base64: bool = False,
) -> Any:
    """Print a figure to an image, and return the resulting file data."""
    if not fig.axes and not fig.lines:
        return

    dpi = fig.dpi
    if fmt == 'highdpi':
        dpi = dpi * dpi_ratio
        fmt = 'png'

    # build keyword args
    kwargs = {
        'format': fmt,
        'facecolor': fig.get_facecolor(),
        'edgecolor': fig.get_edgecolor(),
        'dpi': dpi,
        'bbox_inches': bbox_inches,
    }

    bytes_io = BytesIO()
    if fig.canvas is None:
        fig.set_canvas(FigureCanvasBase(fig))

    fig.canvas.print_figure(bytes_io, **kwargs)
    data = bytes_io.getvalue()
    if fmt == 'svg':
        data = data.decode('utf-8')
    elif base64:
        data = b2a_base64(data, newline=False).decode('ascii')
    return data


def highdpi_figure(fig: Figure, dpi_ratio: int = 3, base64: bool = False) -> Any:
    """format a figure as a high-dpi png."""
    pngdata = print_figure(fig, fmt='highdpi', dpi_ratio=dpi_ratio)
    if pngdata is None:
        return
    width, height = read_png_header(pngdata)
    metadata = {'width': width // dpi_ratio, 'height': height // dpi_ratio}
    if base64:
        pngdata = b2a_base64(pngdata, newline=False).decode('ascii')
    return pngdata, metadata


def set_figure_format(shell: InteractiveShell, fmt: str, dpi_ratio: int = 3) -> None:
    """Set figure format for the inline backend.

    Args:
        shell (InteractiveShell):
            The main IPython instance.
        fmt (str):
            One figure format to enable: {'png', 'retina', 'highdpi', 'jpeg', 'svg', 'pdf'}.
        dpi_ratio (int, default: 3):
            The dpi ratio to use for 'highdpi' format. Default is 3, which means
            3 times the normal dpi. Only used when fmt is 'highdpi'.
    """
    svg_formatter = shell.display_formatter.formatters['image/svg+xml']
    png_formatter = shell.display_formatter.formatters['image/png']
    jpg_formatter = shell.display_formatter.formatters['image/jpeg']
    pdf_formatter = shell.display_formatter.formatters['application/pdf']

    for fig in shell.display_formatter.formatters.values():
        fig.pop(Figure, None)

    if fmt == 'png':
        png_printer = partial(print_figure, fmt='png', base64=True)
        png_formatter.for_type(Figure, png_printer)
    elif fmt == 'retina':
        retina_printer = partial(highdpi_figure, dpi_ratio=2, base64=True)
        png_formatter.for_type(Figure, retina_printer)
    elif fmt == 'highdpi':
        highdpi_printer = partial(highdpi_figure, dpi_ratio=dpi_ratio, base64=True)
        png_formatter.for_type(Figure, highdpi_printer)
    elif fmt in ('jpg', 'jpeg'):
        jpg_printer = partial(print_figure, fmt='jpg', base64=True)
        jpg_formatter.for_type(Figure, jpg_printer)
    elif fmt == 'svg':
        svg_printer = partial(print_figure, fmt='svg')
        svg_formatter.for_type(Figure, svg_printer)
    elif fmt == 'pdf':
        pdf_printer = partial(print_figure, fmt='pdf', base64=True)
        pdf_formatter.for_type(Figure, pdf_printer)
    else:
        supported = {'png', 'retina', 'highdpi', 'jpeg', 'svg', 'pdf'}
        raise NotImplementedError(
            f"Unsupported figure format '{fmt}'. Supported formats are: {supported}."
        )
