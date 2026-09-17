CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
DECK := deck
DIST := $(DECK)/dist
PDF := $(DIST)/proposal.pdf
PY := uv run --no-project --with pymupdf python
VIDEO_PY := uv run --no-project --python 3.12 --with 'edge-tts>=7.2' --with pymupdf --with fonttools --with brotli python

.PHONY: pdf png check all clean video-test video

all: pdf check png

pdf: $(PDF)

$(PDF): $(DECK)/slides.html $(DECK)/deck.css $(DECK)/fonts.css
	@mkdir -p $(DIST)
	"$(CHROME)" --headless=new --disable-gpu --no-pdf-header-footer \
	  --print-to-pdf="$(abspath $(PDF))" "file://$(abspath $(DECK)/slides.html)" 2>/dev/null
	@ls -la $(PDF)

check: $(PDF)
	$(PY) $(DECK)/scripts/check.py

png: $(PDF)
	$(PY) $(DECK)/scripts/png.py 0.5

clean:
	rm -rf $(DIST)

video-test:
	uv run --no-project --python 3.12 --with pytest python -m pytest video/tests -q

video: $(PDF)
	$(VIDEO_PY) video/build.py $(if $(DRAFT),--draft)
