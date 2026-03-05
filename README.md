# PDF Page -> SVG Pipeline (UV)

This repo contains a minimal UV-based pipeline to extract selected pages from a PDF and convert them to SVG.

## Pipeline Part 1

Inputs from user:
- PDF file path
- Pages to extract

Output:
- SVG files for the selected pages only

## Pipeline Part 2

Inputs from user:
- PDF file path
- Pages to extract

Output:
- `highlights.json` with token-level metadata per selected page:
  - `page`
  - `line`
  - `sentence`
  - `tokenId`
  - `token` (Devanagari, preserving svaras when present in PDF text layer)
  - `kind` (`word` or `punct`)
  - `x`, `y`, `w`, `h` (bbox in pt from PDF text layer)

Important:
- Part 2 uses PDF text-layer extraction (`pdftotext -bbox-layout`), but only from the same relevant region used by SVG extraction:
  - same selected pages
  - same header/footer trim window (`top_*`, `bottom_*`)

## Prerequisites

- Python 3.11+ with `uv`
- Poppler tools available on PATH:
  - `pdfinfo`
  - `pdftocairo`

macOS:

```bash
brew install poppler
```

## Setup

```bash
uv venv
uv sync
```

## Run

Use the script entrypoint:

```bash
uv run pdf-pages-to-svg \
  --pdf data/ganapatiaccent.pdf \
  --pages 1,3-4 \
  --out out/ganapati/pages \
  --prefix page
```

Output files:

- `out/ganapati/pages/page-001.svg`
- `out/ganapati/pages/page-003.svg`
- `out/ganapati/pages/page-004.svg`

## Run With Input Config (Recommended)

Create a JSON input file (example at `data/extract_config.example.json`):

```json
{
  "pdf": "data/ganapatiaccent.pdf",
  "pages": "1,3-4",
  "out": "out/ganapati/pages",
  "prefix": "page",
  "trim": {
    "top_ratio": 0.08,
    "bottom_ratio": 0.06
  }
}
```

Then run:

```bash
uv run pdf-pages-to-svg --config data/extract_config.example.json
```

CLI flags can override config values:

```bash
uv run pdf-pages-to-svg \
  --config data/extract_config.example.json \
  --pages 2-4
```

## Page Selection Format

- Single pages: `--pages 2,5,9`
- Ranges: `--pages 1-4`
- Mixed: `--pages 1,3,6-8`

## Header/Footer Removal

Use trim options to remove header/footer regions from output SVGs:

- `--trim-top-ratio 0.08` trims top 8% of page height
- `--trim-bottom-ratio 0.06` trims bottom 6% of page height
- `--trim-top-px` and `--trim-bottom-px` are absolute alternatives

Ratio options override pixel options when both are set.

## Build highlights.json

```bash
uv run python scripts/build_highlights_json.py \
  --pdf data/ganapatiaccent.pdf \
  --pages 2,3,4 \
  --trim-top-ratio 0.08 \
  --trim-bottom-ratio 0.06 \
  --out out/ganapati/highlights.json
```

Equivalent entrypoint:

```bash
uv run build-highlights-json \
  --pdf data/ganapatiaccent.pdf \
  --pages 2,3,4 \
  --trim-top-ratio 0.08 \
  --trim-bottom-ratio 0.06 \
  --out out/ganapati/highlights.json
```

## Clean highlights.json (drop lines by page)

Use page/line drop rules:

```bash
uv run python scripts/clean_highlights_json.py \
  --in out/ganapati/highlights.json \
  --out out/ganapati/highlights_cleaned.json \
  --drop 2:1-3
```

Equivalent entrypoint:

```bash
uv run clean-highlights-json \
  --in out/ganapati/highlights.json \
  --out out/ganapati/highlights_cleaned.json \
  --drop 2:1-3
```

## Extract features from highlights_cleaned.json

Create `extracted_features.json` with per-token features:
1. `medianWordWidthLine`: median word width over the token's line (punctuation excluded)
2. `medianWordWidthSentence`: median word width over the token's sentence (punctuation excluded)
3. `distanceFromPrevWord`: distance from nearest previous word token; punctuation between words is included via geometric span. Tokens with no previous word are filled in second pass using first-pass global average.
   - when punctuation exists between words, distance is explicitly computed as:
     - gap(prev word -> punctuation) + punctuation width(s) + gap(last punctuation -> current token)

```bash
uv run python scripts/extract_highlight_features.py \
  --in out/ganapati/highlights_cleaned.json \
  --out out/ganapati/extracted_features.json
```

Equivalent entrypoint:

```bash
uv run extract-highlight-features \
  --in out/ganapati/highlights_cleaned.json \
  --out out/ganapati/extracted_features.json
```
