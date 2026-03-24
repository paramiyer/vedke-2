# Logic Notes (ASR + PDF Truth + Deterministic Alignment)

## 1) ASR Stage (Sarvam Batch API)

### Request contract
Use Sarvam batch endpoints in this order:
1. `POST /speech-to-text/job/v1` (create job)
2. `POST /speech-to-text/job/v1/upload-files`
3. `PUT` to `upload_urls[file].file_url`
4. `POST /speech-to-text/job/v1/{job_id}/start`
5. `GET /speech-to-text/job/v1/{job_id}/status` (poll)
6. `POST /speech-to-text/job/v1/download-files`
7. `GET` each `download_urls[file].file_url`

### Default params used
- `model: saaras:v3`
- `mode: verbatim`
- `language_code: sa-IN`
- `with_timestamps: true`
- `num_speakers: 1`

### Artifacts produced
- `data/<experiment>/out/0.json` (Sarvam output)
- `data/<experiment>/out/status.json`
- `data/<experiment>/out/manifest.json`
- `data/<experiment>/out/audio_cleaned_tokens.json`

### Guardrail
Before building `audio_cleaned_tokens.json`, fail if:
- `0.json.timestamps.words` is null or empty

---

## 2) `audio_cleaned_tokens.json` logic

Input: `0.json.timestamps.words[i]`, `start_time_seconds[i]`, `end_time_seconds[i]`

For each segment:
1. Split `timestamps.words[i]` by whitespace to segment tokens.
2. Compute segment duration:
   - `segment_ms = round(end_s*1000) - round(start_s*1000)`
3. Allocate token durations proportionally by token character length.
4. Chain times:
   - first token starts at segment start
   - each next token starts at previous token end
5. Emit per-token fields:
   - `token_id`, `token`, `char_count`, `start_ms`, `end_ms`, etc.

### Example
Segment text:
`ओम् भद्रं कर्णेभिश्रृणुजामदेवाः भद्रं पश्येमाक्षभिर्यजत्राः`
with `8390 -> 22560 ms`:
- `ओम्` (len 3) -> `8390-9163`
- `भद्रं` (len 5) -> `9163-10451`
- `कर्णेभिश्रृणुजामदेवाः` (len 21) -> `10451-15862`
- `भद्रं` (len 5) -> `15862-17150`
- `पश्येमाक्षभिर्यजत्राः` (len 21) -> `17150-22560`

---

## 3) PDF Truth extraction (Stage 4 OCR)

Primary output:
- `pdf_tokens.json` with token-level bbox fields:
  - `tokenId`, `token`, `kind`, `x`, `y`, `w`, `h`

Raw OCR artifact:
- `check_extraction.json`

Cleaned output:
- `pdf_tokens_cleaned.json` with:
  - only `kind: word`
  - remove Latin/English tokens
  - keep Devanagari tokens only
  - add `char_count`

### Path handling
For PDF truth stage, user PDF path is resolved and copied to:
- `data/<experiment>/input/source.pdf`
Then extraction runs on this local copy.
Stage 4 now applies OCR cleanup rules from `build_ocr_check_extraction.py`:
- skip bracketed spans
- skip underlined lines
- split dandas (`।`, `॥`)
- add token metadata (`char_count`, confidence)

---

## 4) OCR sample extraction rules (`check_extraction.json`)

Script:
- `scripts/build_ocr_check_extraction.py`

Current sample config:
- pages: `8,9,10,11,12`
- trim: top `0.08`, bottom `0.06`
- OCR: `tesseract_tsv`, language `san+script/Devanagari`

Post-processing rules:
1. Skip text in brackets (and bracket chars).
2. Skip underlined lines (single/double underline heuristic).
3. Split dandas as separate tokens:
   - `।`, `॥`
4. Add:
   - `kind` (`word` or `punc`)
   - `char_count`

---

## 5) Deterministic alignment approach (current plan)

Goal: align ASR tokens (from `0.json` / `audio_cleaned_tokens.json`) to PDF/OCR tokens.

### Anchor strategy
For each ASR segment (`timestamps.words[i]`):
1. Use the **last token** of that segment as anchor.
2. Search forward only in PDF token stream (monotonic index).
3. Normalize before matching:
   - Devanagari-only normalization
   - equivalent handling for variants (example: `ओम्` ≈ `ओं`)
4. Prefer:
   - earliest high-confidence single-token match
   - else short span merge (2 tokens) if score improves
5. Mark low-confidence anchors for review.

### Example anchors from current files
1. ASR 2nd segment last token:
   - target: `वृद्धश्रवाः`
   - matched exactly in OCR tokens (score `1.0`)

2. ASR 3rd segment last token:
   - target: `ओम्`
   - OCR practical equivalent found: `ओं` (`P0008_T0043`)
   - should be treated as equivalent in normalization rules

---

## 6) Practical caveats

1. If PDF has non-Unicode Devanagari fonts (`pdffonts` shows `uni=no`), `pdftotext` can produce gibberish.
2. In that case, use OCR fallback (`tesseract`) for better token extraction.
3. OCR introduces spelling noise; alignment must allow fuzzy matching + merge/split.
