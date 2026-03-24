# REPO_MAP

## 1) Repository Purpose

`vedke-2` is an experiment-driven Sanskrit audio/OCR alignment workspace.

Primary workflow:
1. Create an experiment slug
2. Prepare audio from YouTube
3. Run Sarvam STT
4. Build OCR PDF truth artifacts
5. Run LLM alignment + review + feedback loop

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
  - OCR PDF truth artifacts:
    - `check_extraction.json`
    - `pdf_tokens.json`
    - `pdf_tokens_cleaned.json`
  - Alignment artifacts:
    - `pdf_tokens_enriched_with_timestamps.json`
    - `pdf_tokens_segment_mapping_review.csv`
    - `pdf_tokens_left_out_non_punctuation.csv`
    - `alignment_llm_manifest.json`
    - `alignment_llm_raw_response.json` (debug)
    - `alignment_config.json`
  - stage guardrail artifacts (`source_intake_manifest.json`, `quality_gates.json`)

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

- `scripts/build_ocr_check_extraction.py`
  - OCR-first PDF truth extraction (Tesseract TSV flow)
  - Implements extraction cleanup rules that drive `check_extraction.json`
  - Produces token-level OCR fields used downstream in Stage 4

- `scripts/agent4_pdf_truth.py`
  - Legacy/auxiliary PDF extraction path

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
  - `pdf_truth` now runs OCR extraction and writes both `check_extraction.json` and `pdf_tokens*.json`
  - `alignment_config` calls `scripts/run_alignment_llm.py` and writes Stage 4 artifacts

- `scripts/run_alignment_llm.py`
  - Calls OpenAI chat completions with OCR + audio payload prompt
  - Inputs:
    - `data/<experiment>/out/0.json`
    - `data/<experiment>/out/pdf_tokens.json`
    - `--anchor-token` (required user input; first OCR anchor tokenId from `pdf_tokens.json` that maps to first token in `timestamps.words[0]`)
  - Outputs:
    - `pdf_tokens_enriched_with_timestamps.json`
    - `pdf_tokens_segment_mapping_review.csv`
    - `pdf_tokens_left_out_non_punctuation.csv`
    - `alignment_llm_manifest.json`
  - Post-output Python guardrails:
    - anchor guardrail
    - monotonic timestamp guardrail
    - ineligible-token timestamp stripping
    - monotonic auto-repair where safe

---

## 4) Frontend (React)

- `frontend/src/App.tsx`
  - Build/View tabs
  - Build tab includes:
    - mandatory inputs
    - advanced parameters
    - experiment-folder flow awareness
    - PDF folder finder UI
    - Stage 4 async execution and polling
    - Stage 4 manual override popup:
      - fetches exact prompt text
      - guides user to run prompt in ChatGPT with `0.json` + `pdf_tokens.json`
      - accepts uploaded downloaded outputs
      - validates uploads via backend guardrails before marking Stage 4 complete
    - alignment review table from `pdf_tokens_segment_mapping_review.csv`
    - left-out tokens table from `pdf_tokens_left_out_non_punctuation.csv`
    - reviewer feedback box for multihop alignment correction

- `frontend/src/styles.css`
  - Build tab layout and controls

- `frontend/server.mjs`
  - Local API server on `http://localhost:8787`
  - Stage endpoints:
    - `POST /api/run-stage`
    - `POST /api/run-stage-async` (currently used for Stage 4)
    - `GET /api/job-status`
    - `GET /api/stage-artifacts`
    - `POST /api/alignment-feedback`
    - `POST /api/alignment-manual-prompt`
    - `POST /api/alignment-manual-import`
  - Prefers `.venv/bin/python3` when present
  - Logs stage command start/end and subprocess stdout/stderr

- `scripts/validate_alignment_manual.py`
  - Validates manual ChatGPT alignment uploads
  - Inputs:
    - uploaded enriched JSON
    - uploaded review CSV
    - anchor token + experiment
  - Applies Python guardrails on uploaded output and emits recommendations
  - On pass, commits final Stage 4 artifacts into `data/<experiment>/out/`

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
- `run-alignment-llm`

---

## 6) Config / Safety

- `.env`
  - contains:
    - `SARVAM_API_KEY`
    - `OPENAI_API_KEY`
    - `OPENAI_ALIGNMENT_MODEL` (e.g. `gpt-5.1` or `gpt-4.1-mini`)

- `.gitignore`
  - ignores `.env*`
  - ignores `archive/`

---

## 7) Notes for Future Work

- Stage 4 can run long; frontend uses async job polling.
- Alignment quality is prompt-driven, but output safety/consistency is enforced in Python guardrail functions.
