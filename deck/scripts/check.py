"""Deck checks. Run via `make check` (uv supplies pymupdf).

Fails loud on: page count != <section class="slide"> count, content past the safe area (slides
clip with overflow:hidden, so this is measured in headless Chrome), > 20 body pages, PDF > 15 MB,
CJK glyphs, leftover TODO or TEAM_NAME/MEMBER_NAME placeholders, missing font files.
"""
from __future__ import annotations

import html as htmllib
import os
import re
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz

DECK = Path(__file__).resolve().parent.parent
HTML = DECK / "slides.html"
PDF = DECK / "dist" / "proposal.pdf"
MAX_BODY = 20
MAX_BYTES = 15 * 1024 * 1024
SAFE_BOTTOM = 1000  # px from slide top; the footer sits below
SAFE_RIGHT = 1824.5  # 96 px right margin
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

PROBE = """
<script>
window.addEventListener('load', () => document.fonts.ready.then(() => {
  const out = [];
  document.querySelectorAll('section.slide').forEach((s, i) => {
    const sr = s.getBoundingClientRect();
    const foot = s.querySelector('.foot');
    let maxB = 0, maxR = 0, worst = '';
    s.querySelectorAll('*').forEach(el => {
      if (foot && (el === foot || foot.contains(el))) return;
      if (el.closest('svg.defs') || el.closest('marker')) return;
      const r = el.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return;
      const b = r.bottom - sr.top, rr = r.right - sr.left;
      if (b > maxB) { maxB = b; worst = el.tagName + ' "' + (el.textContent || '').trim().slice(0, 40) + '"'; }
      if (rr > maxR) maxR = rr;
    });
    out.push([i + 1, Math.round(maxB), Math.round(maxR), worst].join('\\t'));
  });
  const pre = document.createElement('pre'); pre.id = 'probe-out'; pre.textContent = out.join('\\n');
  document.body.appendChild(pre);
}));
</script>
"""


def overflow_errors(html: str) -> list[str]:
    probe = DECK / "dist" / "probe.html"
    probe.parent.mkdir(exist_ok=True)
    page = html.replace("<head>", f'<head><base href="file://{DECK}/">', 1).replace("</body>", PROBE + "</body>", 1)
    probe.write_text(page, encoding="utf-8")
    dom = subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=5000",
         "--window-size=1920,1080", "--dump-dom", f"file://{probe}"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    m = re.search(r'<pre id="probe-out">(.*?)</pre>', dom, re.S)
    if not m:
        return ["overflow probe produced no output"]
    errors = []
    rows = re.findall(r"^(\d+)\t(\d+)\t(\d+)\t(.*)$", htmllib.unescape(m.group(1)), re.M)
    if not rows:
        return ["overflow probe output unparseable"]
    for idx, bottom, right, worst in rows:
        if float(bottom) > SAFE_BOTTOM or float(right) > SAFE_RIGHT:
            errors.append(f"slide {idx} past safe area (bottom {bottom}, right {right}): {worst}")
    return errors


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

    placeholders = sorted(set(re.findall(r"TEAM_NAME|MEMBER_NAME", html)))
    if placeholders:
        errors.append(f"placeholders left: {', '.join(placeholders)}")
    errors += overflow_errors(html)

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
