"""Render every PDF page to deck/dist/png/NN.png for visual review (`make png`)."""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

DECK = Path(__file__).resolve().parent.parent
PDF = DECK / "dist" / "proposal.pdf"
OUT = DECK / "dist" / "png"


def main() -> int:
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    doc = fitz.open(PDF)
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        pix.save(OUT / f"{i:02d}.png")
    print(f"{doc.page_count} pages -> {OUT} at {scale:g}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
