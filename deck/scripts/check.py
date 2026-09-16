"""Deck checks. Run via `make check` (uv supplies pymupdf).

Fails loud on: page count != <section class="slide"> count (overflow), > 20 body pages,
PDF > 15 MB, CJK glyphs in the HTML, leftover TODO markers, missing font files.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz  # pymupdf

DECK = Path(__file__).resolve().parent.parent
HTML = DECK / "slides.html"
PDF = DECK / "dist" / "proposal.pdf"
MAX_BODY = 20
MAX_BYTES = 15 * 1024 * 1024


def main() -> int:
    errors: list[str] = []
    html = HTML.read_text(encoding="utf-8")
    sections = re.findall(r'<section class="slide([^"]*)"', html)
    n_slides = len(sections)
    n_body = sum(1 for cls in sections if "appendix" not in cls)

    if not PDF.exists():
        errors.append(f"missing {PDF}; run `make pdf` first")
    else:
        doc = fitz.open(PDF)
        if doc.page_count != n_slides:
            errors.append(f"page count {doc.page_count} != {n_slides} slides (a slide overflowed)")
        size = PDF.stat().st_size
        if size > MAX_BYTES:
            errors.append(f"PDF is {size/1e6:.1f} MB > 15 MB")
        pdf_text = "".join(p.get_text() for p in doc)
        if re.search(r"[぀-ヿ㐀-鿿]", pdf_text):
            errors.append("CJK glyphs found in the PDF text (contest requires English)")
        print(f"pdf: {doc.page_count} pages, {size/1e6:.2f} MB")

    if n_body > MAX_BODY:
        errors.append(f"{n_body} body slides > {MAX_BODY}")
    if re.search(r"[぀-ヿ㐀-鿿]", html):
        errors.append("CJK characters in slides.html")
    todos = [m.start() for m in re.finditer(r"TODO", html)]
    if todos:
        lines = sorted({html.count("\n", 0, i) + 1 for i in todos})
        errors.append(f"TODO markers left in slides.html at lines {lines}")

    fonts_css = (DECK / "fonts.css").read_text()
    for rel in re.findall(r"url\('([^']+)'\)", fonts_css):
        if not (DECK / rel).exists():
            errors.append(f"font file missing: {rel}")

    print(f"slides: {n_slides} total, {n_body} body")
    for e in errors:
        print("FAIL:", e)
    print("OK" if not errors else f"{len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
