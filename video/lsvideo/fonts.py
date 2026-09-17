from __future__ import annotations

from pathlib import Path

from fontTools.ttLib import TTFont

from .model import VideoError


def ensure_ttf(src_dir: Path, dst_dir: Path) -> Path:
    """libass cannot open woff2; convert each deck font to TTF once."""
    srcs = sorted(src_dir.glob("*.woff2"))
    if not srcs:
        raise VideoError(f"{src_dir}: no .woff2 fonts to convert for the captions")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in srcs:
        dst = dst_dir / f"{src.stem}.ttf"
        if dst.exists():
            continue
        font = TTFont(src)
        font.flavor = None
        tmp = dst.with_suffix(".ttf.tmp")
        font.save(tmp)
        tmp.replace(dst)
    return dst_dir
