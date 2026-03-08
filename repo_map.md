# REPO_MAP

## 1) Repository Purpose

`vedke-2` is now an experiment-driven Sanskrit audio/text orchestration workspace.

Primary workflow:
1. Create an experiment slug
2. Prepare audio from YouTube
3. Run Sarvam STT
4. Build PDF truth artifacts
5. Align and review

All active run data follows:
- `data/<experiment>/input/`
- `data/<experiment>/out/`

Legacy outputs/notebooks are moved under `archive/` and excluded from git.

---

## 2) Core Runtime Structure

### Active Data Layout

- `data/<experiment>/input/`
  - `yt_link.txt`
  - `audio.mp3`
  - `audio.16k.wav`
  - source PDF reference path (tracked by workflow/UI)

- `data/<experiment>/out/`
  - Sarvam outputs (e.g. `0.json`)
  - `status.json`
  - `manifest.json`
  - Agent 4 PDF truth artifact: `pdf_tokens.json`
    - token-level only
    - `kind` in `{word, punc}`
    - bbox coordinates `x,y,w,h`
  - stage guardrail artifacts (`source_intake_manifest.json`, `alignment_config.json`, `quality_gates.json`)

### Archived Layout

- `archive/`
  - previous notebooks, prior `out/` data, and historical artifacts

---

## 3) Key Scripts (Current)

### ASR Input Preparation

- `scripts/prepare_asr_audio.py`
  - Downloads YouTube audio using `yt-dlp`
  - Converts to 16k mono WAV using `ffmpeg`
  - Writes artifacts to `data/<experiment>/input/`
  - CLI:
    - `--experiment`
    - `--url`
    - `--overwrite`

### Sarvam STT Pipeline

- `scripts/run_sarvam_stt.py`
  - Loads `SARVAM_API_KEY` from env / `.env`
  - Creates/reuses Sarvam job
  - Uploads `data/<experiment>/input/audio.mp3`
  - Starts job, polls status, downloads outputs to `data/<experiment>/out/`
  - CLI:
    - `--experiment`
    - `--audio` (optional override)
    - `--job-id` (optional)
    - `--model`
    - `--mode`
    - `--language-code`
    - `--with-timestamps`

### PDF / Text Truth Utilities

- `scripts/agent4_pdf_truth.py`
  - Agent 4 implementation for PDF truth extraction
  - Writes `data/<experiment>/out/pdf_tokens.json`
  - Keeps only token-level fields needed for truth alignment:
    - `tokenId`, `token`, `kind`, `x`, `y`, `w`, `h`

- `scripts/pdf_pages_to_svg.py`
- `scripts/build_highlights_json.py`
- `scripts/clean_highlights_json.py`
- `scripts/extract_highlight_features.py`
- `scripts/build_sentences.py`
- `scripts/build_sentence_inputs.py`
- `scripts/sentence_segmenter.py`
- `scripts/tune_segmenter.py`

These continue to power extraction, cleanup, sentence artifacts, and segmentation logic.

### Stage Runner + Guardrails

- `scripts/stage_orchestrator.py`
  - Implements per-stage execution for current Build flow (`run this stage`)
  - Enforces guardrails before each stage
  - Stages:
    - `source_intake`
    - `asr_preparation`
    - `pdf_truth`
    - `alignment_config`
    - `quality_gates`

---

## 4) Frontend (React)

- `frontend/src/App.tsx`
  - Build/View tabs
  - Build tab includes:
    - mandatory inputs
    - advanced parameters
    - experiment-folder flow awareness
    - PDF folder finder UI

- `frontend/src/styles.css`
  - Build tab layout and controls

---

## 5) Project Entry Points

Defined in `pyproject.toml`:
- `pdf-pages-to-svg`
- `build-highlights-json`
- `clean-highlights-json`
- `extract-highlight-features`
- `prepare-asr-audio`
- `run-sarvam-stt`
- `run-agent4-pdf-truth`
- `run-build-stage`

---

## 6) Config / Safety

- `.env`
  - contains `SARVAM_API_KEY`

- `.gitignore`
  - ignores `.env*`
  - ignores `archive/`

---

## 7) Notes for Future Work

- Build tab currently simulates stage outputs for UI orchestration preview.
- Stage orchestration CLI + guardrails now exist in `scripts/stage_orchestrator.py`.
