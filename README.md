# vedke-2: Sanskrit Audio/OCR Alignment Workspace

An experiment-driven pipeline for aligning Sanskrit PDF text (OCR) with audio transcriptions (ASR), with a karaoke-style interactive viewer.

## Overview

Primary workflow:
1. Create an experiment slug
2. Prepare audio from YouTube (`source_intake` + `asr_preparation`)
3. Run Sarvam STT to get word-level timestamps
4. Build OCR PDF truth from a Sanskrit PDF (`pdf_truth`)
5. Run LLM alignment to map OCR tokens to audio timestamps (`alignment_config`)
6. Review and correct alignment in the browser UI
7. View the result as a karaoke-style playback

All run data lives under `data/<experiment>/input/` and `data/<experiment>/out/`.

---

## Prerequisites

- Python 3.11+ with [`uv`](https://github.com/astral-sh/uv)
- [Poppler](https://poppler.freedesktop.org/) tools on PATH (`pdfinfo`, `pdftocairo`, `pdftotext`)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) on PATH
- `yt-dlp` and `ffmpeg` for audio preparation
- Node.js 18+ for the frontend server

macOS:

```bash
brew install poppler tesseract yt-dlp ffmpeg
```

---

## Setup

```bash
uv venv
uv sync
```

Frontend:

```bash
cd frontend
npm install
```

---

## Environment

Create a `.env` at the repo root:

```bash
SARVAM_API_KEY="your_sarvam_key"
OPENAI_API_KEY="your_openai_key"
OPENAI_ALIGNMENT_MODEL="gpt-4.1"   # or gpt-4.1-mini
```

---

## Run the Frontend UI

```bash
cd frontend
node server.mjs   # API server on http://localhost:8787
npm run dev       # Vite dev server (or serve frontend/dist)
```

The UI has two tabs:
- **Build** — run pipeline stages, review alignment, run feedback loops
- **View** — karaoke playback of aligned tokens over page images

---

## Pipeline Stages

Run any stage via:

```bash
uv run run-build-stage \
  --stage <stage_name> \
  --experiment <experiment_slug> \
  [stage-specific flags]
```

### `source_intake`

Downloads audio from YouTube and records the source PDF path.

```bash
uv run run-build-stage \
  --stage source_intake \
  --experiment ganapati_trial_01 \
  --youtube-url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --pdf data/ganapatiaccent.pdf \
  --pages 2,3,4
```

### `asr_preparation`

Converts audio to 16kHz mono WAV and runs Sarvam STT batch job.

```bash
uv run run-build-stage \
  --stage asr_preparation \
  --experiment ganapati_trial_01
```

Outputs under `data/<experiment>/out/`:
- `0.json` — Sarvam word-level timestamps
- `manifest.json`, `status.json`

### `pdf_truth`

Runs Tesseract OCR over the selected PDF pages and produces cleaned token data.

```bash
uv run run-build-stage \
  --stage pdf_truth \
  --experiment ganapati_trial_01 \
  --pdf data/ganapatiaccent.pdf \
  --pages 2,3,4
```

Outputs:
- `check_extraction.json` — raw OCR artifact
- `pdf_tokens.json` — token-level data (tokenId, token, kind, x, y, w, h)
- `pdf_tokens_cleaned.json` — after cleanup rules

### `alignment_config`

Calls OpenAI to align OCR tokens with Sarvam ASR timestamps. Requires an anchor token.

```bash
uv run run-build-stage \
  --stage alignment_config \
  --experiment ganapati_trial_01
```

Or run directly with explicit anchor:

```bash
uv run run-alignment-llm \
  --experiment ganapati_trial_01 \
  --anchor-token <tokenId>
```

Outputs:
- `pdf_tokens_enriched_with_timestamps.json`
- `pdf_tokens_segment_mapping_review.csv`
- `pdf_tokens_left_out_non_punctuation.csv`
- `alignment_llm_manifest.json`
- `alignment_config.json`

Python guardrails applied post-LLM:
- Anchor validation
- Monotonic timestamp check and auto-repair
- Ineligible-token timestamp stripping

**Manual override**: the UI provides a guided flow to run the alignment prompt manually in ChatGPT, upload outputs, and validate them via `scripts/validate_alignment_manual.py` before committing.

### `quality_gates`

Runs guardrail checks across all prior stage outputs.

```bash
uv run run-build-stage \
  --stage quality_gates \
  --experiment ganapati_trial_01
```

---

## Alignment Review (UI)

After `alignment_config` completes, the Build tab shows:

- **Segment mapping table** — alignment CSV with kept/left-out token counts per segment
- **Alignment Review Editor** — drag tokens between `kept` and `left-out` columns per segment; set per-segment time reduction; click **Realign** to rerun floor-and-redistribute timestamp algorithm
- **Feedback box** — free-text corrections for multihop LLM re-alignment

---

## Karaoke View (UI)

The View tab plays back aligned tokens over scanned page images, highlighting the active token in sync with the audio. Requires `alignment_config` to be complete for the experiment.

---

## Individual Script Entrypoints

| Entrypoint | Script |
|---|---|
| `pdf-pages-to-svg` | `scripts/pdf_pages_to_svg.py` |
| `build-highlights-json` | `scripts/build_highlights_json.py` |
| `clean-highlights-json` | `scripts/clean_highlights_json.py` |
| `extract-highlight-features` | `scripts/extract_highlight_features.py` |
| `prepare-asr-audio` | `scripts/prepare_asr_audio.py` |
| `run-sarvam-stt` | `scripts/run_sarvam_stt.py` |
| `run-build-stage` | `scripts/stage_orchestrator.py` |
| `run-alignment-llm` | `scripts/run_alignment_llm.py` |

### PDF to SVG (legacy utility)

```bash
uv run pdf-pages-to-svg \
  --pdf data/ganapatiaccent.pdf \
  --pages 1,3-4 \
  --out out/ganapati/pages \
  --prefix page \
  --trim-top-ratio 0.08 \
  --trim-bottom-ratio 0.06
```

### Prepare ASR audio from YouTube (standalone)

```bash
uv run prepare-asr-audio \
  --experiment ganapati_trial_01 \
  --url "https://www.youtube.com/watch?v=<VIDEO_ID>" \
  --overwrite
```

### Sarvam STT (standalone)

```bash
uv run run-sarvam-stt \
  --experiment ganapati_trial_01 \
  --with-timestamps \
  --model saaras:v3 \
  --mode translit \
  --language-code sa-IN
```

---

## Page Selection Format

- Single pages: `--pages 2,5,9`
- Ranges: `--pages 1-4`
- Mixed: `--pages 1,3,6-8`

---

## Data Layout

```
data/<experiment>/
  input/
    yt_link.txt
    audio.mp3
    audio.16k.wav
  out/
    0.json                                  # Sarvam ASR output
    manifest.json
    status.json
    check_extraction.json                   # raw OCR
    pdf_tokens.json
    pdf_tokens_cleaned.json
    pdf_tokens_enriched_with_timestamps.json
    pdf_tokens_segment_mapping_review.csv
    pdf_tokens_left_out_non_punctuation.csv
    alignment_llm_manifest.json
    alignment_config.json
    source_intake_manifest.json
    quality_gates.json
```
