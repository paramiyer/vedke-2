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

For ASR audio preparation from YouTube:
- `yt-dlp`
- `ffmpeg`

macOS:

```bash
brew install yt-dlp ffmpeg
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

## Prepare ASR audio from YouTube

Download best available audio from a YouTube URL, then generate:
- `data/<experiment>/input/audio.mp3`
- `data/<experiment>/input/audio.16k.wav` (mono, 16kHz)
- `data/<experiment>/input/yt_link.txt`

```bash
uv run prepare-asr-audio \
  --experiment ganapati_trial_01 \
  --url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --overwrite
```

Notes:
- The tool requires both `yt-dlp` and `ffmpeg` on PATH.

## Sarvam STT CLI

Set your API key in `.env` (repo root):

```bash
SARVAM_API_KEY="your_sarvam_key"
```

Run end-to-end Sarvam batch job for one audio file:

```bash
uv run run-sarvam-stt \
  --experiment ganapati_trial_01 \
  --with-timestamps \
  --model saaras:v3 \
  --mode translit \
  --language-code sa-IN
```

Outputs:
- `data/<experiment>/out/status.json`
- downloaded Sarvam output files (for example `data/<experiment>/out/0.json`)
- `data/<experiment>/out/manifest.json`

## Agent 4: PDF Truth (Token-Level)

Extract token-level PDF truth with fields:
- `tokenId`
- `token`
- `kind` (`word` or `punc`)
- `x`, `y`, `w`, `h`

```bash
uv run run-agent4-pdf-truth \
  --experiment ganapati_trial_01 \
  --pdf data/ganapatiaccent.pdf \
  --pages 2,3,4 \
  --trim-top-ratio 0.08 \
  --trim-bottom-ratio 0.06
```

Output:
- `data/<experiment>/out/pdf_tokens.json`

## Run One Build Stage With Guardrails

```bash
uv run run-build-stage \
  --stage source_intake \
  --experiment ganapati_trial_01 \
  --youtube-url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --pdf data/ganapatiaccent.pdf \
  --pages 2,3,4
```

Available stages:
- `source_intake`
- `asr_preparation`
- `pdf_truth`
- `alignment_config`
- `quality_gates`
