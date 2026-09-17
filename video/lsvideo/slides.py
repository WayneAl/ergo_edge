from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pymupdf

from .model import H, W, VideoError


def ensure_pngs(pdf: Path, pages: Iterable[int], out_dir: Path) -> None:
    """Render each deck page to <out_dir>/<n:02d>.png at 1920x1080, re-rendering when the PDF is newer."""
    if not pdf.exists():
        raise VideoError(f"{pdf} missing — run make pdf")
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(pdf)
    try:
        for n in pages:
            if not 1 <= n <= doc.page_count:
                raise VideoError(f"slide {n}: the deck has {doc.page_count} pages")
            png = out_dir / f"{n:02d}.png"
            if png.exists() and png.stat().st_mtime >= pdf.stat().st_mtime:
                continue
            page = doc[n - 1]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(W / page.rect.width, H / page.rect.height))
            if (pix.width, pix.height) != (W, H):
                raise VideoError(f"slide {n}: rendered {pix.width}x{pix.height}, expected {W}x{H} — "
                                 f"the deck page must be 16:9")
            pix.save(png)
    finally:
        doc.close()


def edge_color(png: Path) -> str:
    """The slide's background colour, from its top-left corner, as ffmpeg's 0xRRGGBB."""
    r, g, b = pymupdf.Pixmap(str(png)).pixel(2, 2)[:3]
    return f"0x{r:02X}{g:02X}{b:02X}"
