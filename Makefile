CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
DECK := deck
DIST := $(DECK)/dist
PDF := $(DIST)/proposal.pdf
PY := uv run --with pymupdf python

.PHONY: pdf png check all clean

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
