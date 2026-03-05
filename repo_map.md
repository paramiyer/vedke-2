# REPO_MAP

This file is the retrieval index for `vedke-2`.  
Use it to locate exact code elements quickly before opening full files.

## 1) Repository Purpose

`vedke-2` is a focused PDF page extraction pipeline:
- Input: one PDF + selected pages (CLI or JSON config)
- Processing: page export to SVG via Poppler + optional header/footer trimming
- Output: deterministic `page-XXX.svg` files under `out/...`

The pipeline is built for `uv` workflows and currently centers on one extractor command.

## 2) High-Level Data Flow

### Part 1 (Current Implemented Stage)

Take two user inputs:
- PDF file path
- Pages to extract

Produce:
- SVG files for the selected pages only

1. Parse CLI args and/or config JSON
2. Validate required external tools (`pdfinfo`, `pdftocairo`)
3. Resolve page count and validate page selection
4. Convert each selected page to SVG (`pdftocairo -svg`)
5. Normalize extensionless output naming if needed
6. Crop top/bottom regions (header/footer removal)
7. Write final SVG files in output directory

### Part 2 (Highlights Extraction Stage)

Take two user inputs:
- PDF file path
- Pages to extract

Produce:
- `highlights.json` with token-level data for selected pages:
  - `page`
  - `line`
  - `sentence`
  - `tokenId`
  - `token` (Devanagari with svaras when available in PDF text layer)
  - `kind` (`word` or `punct`)
  - bbox fields: `x`, `y`, `w`, `h`

Extraction details:
1. Run `pdftotext -bbox-layout` to get XML text-layer bboxes
2. Parse page -> line -> word structure
3. Keep only selected pages
4. Apply same trim window used in SVG extraction (top/bottom by px or ratio) to exclude header/footer
5. Build deterministic token IDs (`P####_L####_S####_T####`)
6. Infer `kind`:
   - punctuation/digits -> `punct`
   - otherwise -> `word`
7. Increment sentence counter after terminal punctuation (`।`, `॥`, `.`, `!`, `?`)
8. Write `highlights.json` with extraction metadata (`engine`, `units`, `sourceFilter`, `fields`, `notes`)

### Part 3 (Highlights Cleanup Stage)

Take user inputs:
- input highlights file path
- page/line drop rules

Produce:
- `highlights_cleaned.json`

Cleanup details:
1. Parse drop rules in format `<page>:<lineSpec>`
2. Filter token records whose `line` belongs to dropped lines for that page
3. Preserve remaining records unchanged (no line renumbering)
4. Write cleaned payload and record `dropped_lines` under extraction/sourceFilter

### Part 4 (Feature Extraction Stage)

Take user inputs:
- input `highlights_cleaned.json` path
- output path for `extracted_features.json`

Produce:
- `extracted_features.json` with feature values for each token

Features per token:
1. `medianWordWidthLine`
   - median `w` over all word tokens in same page+line
   - punctuation excluded
2. `medianWordWidthSentence`
   - median `w` over all word tokens in same page+sentence
   - punctuation excluded
3. `distanceFromPrevWord`
   - for each token, distance from nearest previous word token in line order
   - punctuation between words is explicitly accounted as:
     - gap(prev word -> punctuation) + punctuation width(s) + gap(last punctuation -> current token)
   - first-pass placeholder when no previous word
   - second-pass fill with global average from first-pass observed distances
4. `distanceToNextWord`
   - symmetric rule toward next word token in the same line
   - same punctuation-aware decomposition
   - first-pass placeholder when no next word
   - second-pass fill with global average from first-pass observed distances
5. `normalizedXInLine`
   - `(x - min_x_line) / (max_x_line - min_x_line)`
6. `normalizedWidthInLine`
   - `w / medianWordWidthLine`
7. `isLineStart`, `isLineEnd`
8. `isSentenceStart`, `isSentenceEnd`
9. `prevWordWidth`, `nextWordWidth`
10. `tokenLengthChars`
11. `hasSvaraMarks`
12. `punctuationDensityLine`
13. `lineWordCount`, `sentenceWordCount`
14. `yRankInPage`, `yRankInPageNormalized`
15. `boxArea`, `aspectRatio`

Worked examples (from current `out/ganapati/extracted_features.json`):

- Example A: `P0002_L0004_S0001_T0009`
  - token: `देवाः`
  - previous word: `P0002_L0004_S0001_T0008` (`श‍ृणु॒याम॑`)
  - no punctuation in between
  - computed:
    - `x_current - (x_prev_word + w_prev_word)`
    - `136.017 - (97.507 + 32.294) = 6.216`
  - `distanceFromPrevWord = 6.216`

- Example B: `P0002_L0004_S0002_T0011`
  - token: `भ॒द्रं`
  - previous word: `P0002_L0004_S0001_T0009` (`देवाः`)
  - punctuation between: `P0002_L0004_S0001_T0010` (`।`)
  - explicit punctuation-aware decomposition:
    - gap(prev word -> punct) = `157.689 - (136.017 + 18.716) = 2.956`
    - punct width = `5.551`
    - gap(punct -> current) = `166.195 - (157.689 + 5.551) = 2.955`
    - total = `2.956 + 5.551 + 2.955 = 11.462`
  - `distanceFromPrevWord = 11.462`

- Example C: `P0002_L0005_S0003_T0014`
  - token: `स्थि॒रैरङ्गै᳚स्तुष्टु॒वाꣳ`
  - first token in line 5, so no previous word in line
  - first pass stores placeholder
  - second pass fills with global average from observed distances:
    - `placeholderFillValuePrev = 4.388166`
    - stored `distanceFromPrevWord = 4.388` (rounded)

## 3) Root Files

- `pyproject.toml`
  - Project metadata and entrypoint registration.
  - Console script:
    - `pdf-pages-to-svg = "scripts.pdf_pages_to_svg:main"`
  - Hatch wheel packaging currently ships `scripts/`.

- `README.md`
  - Operator instructions for setup and usage.
  - Documents:
    - direct CLI usage
    - JSON config usage
    - page selection syntax
    - trim semantics (ratio vs px)

- `uv.lock`
  - UV lockfile for environment reproducibility.

- `repo_map.md` (this file)
  - Retrieval map for code navigation.

- `agent.md`
  - Working rules for agents (retrieval-first behavior).

## 4) Runtime Input / Output Locations

- `data/ganapatiaccent.pdf`
  - Source PDF currently used in local runs.

- `data/extract_config.example.json`
  - Example JSON config for extractor inputs.
  - Demonstrates:
    - `pdf`
    - `pages`
    - `out`
    - `prefix`
    - `trim.top_ratio`
    - `trim.bottom_ratio`

- `out/ganapati/pages/page-002.svg`
- `out/ganapati/pages/page-003.svg`
- `out/ganapati/pages/page-004.svg`
  - Current generated artifacts from pipeline runs.

## 5) Primary Implementation Module

- `scripts/pdf_pages_to_svg.py`
  - Canonical extractor implementation used by console entrypoint.
  - Core responsibilities:
    - configuration merge (CLI + JSON)
    - page parsing and validation
    - tool checks
    - page conversion
    - SVG crop transform

### Key Types

- `ConversionPlan`
  - Immutable dataclass containing execution inputs:
    - `pdf_path`
    - `output_dir`
    - `pages`
    - `prefix`
    - trim parameters (`*_px`, `*_ratio`)

### Key Functions (by role)

- Command utilities:
  - `_run(cmd)`
  - `_require_tool(name, hint)`
  - `_page_count(pdf_path)`

- Page parsing:
  - `_parse_page_spec(spec, max_page)`
  - `_parse_pages(raw, max_page)`

- Config and value resolution:
  - `_load_config(path)`
  - `_resolve_value(cli_value, config_value, default_value)`
  - `_build_plan(args)`

- SVG geometry helpers:
  - `_parse_css_length(value)`
  - `_format_length(value, unit)`
  - `_detect_namespace(tag)`
  - `_qname(tag, namespace)`

- Cropping and conversion:
  - `_crop_svg(svg_path, page, plan)`
    - Reads `viewBox`
    - Computes top/bottom trim via px or ratio
    - Rewrites `viewBox` and `height`
    - Wraps SVG content in clipped group
  - `_convert_page_to_svg(plan, page)`
    - Clears stale output (`.svg` and extensionless output)
    - Executes `pdftocairo`
    - Renames extensionless output to `.svg`
    - Applies crop stage

- CLI surface:
  - `build_parser()`
    - Options:
      - `--config`
      - `--pdf`
      - `--pages`
      - `--out`
      - `--prefix`
      - `--trim-top-px`
      - `--trim-bottom-px`
      - `--trim-top-ratio`
      - `--trim-bottom-ratio`
  - `main(argv=None)`

- `scripts/build_highlights_json.py`
  - Highlights extraction pipeline from PDF text layer to token-level JSON.
  - Core responsibilities:
    - page selection parsing and validation
    - config + CLI merge (supports shared trim config)
    - bbox XML extraction via `pdftotext -bbox-layout`
    - trim-window filtering aligned to SVG extraction
    - page/line/token parsing
    - sentence assignment from punctuation
    - token kind inference (`word` vs `punct`)
    - deterministic token id generation

### Key Types (Highlights)

- `WordBox`
  - `page`, `line`, `text`, `x`, `y`, `w`, `h`

### Key Functions (Highlights)

- `_extract_word_boxes(xml_text, selected_pages)`
- `build_highlights_payload(pdf_path, pages, word_boxes)`
- `_is_punct(text)`
- `_parse_pages(raw, max_page)`
- `_parse_ratio(value, field)`
- `_parse_px(value, field)`
- `_load_config(path)`
- `_resolve_value(cli_value, config_value, default_value)`
- `_parse_page_spec(spec, max_page)`
- `build_parser()`
- `main(argv=None)`

- `scripts/clean_highlights_json.py`
  - Line-drop cleanup pipeline for generated highlights.
  - Core responsibilities:
    - parse page/line drop rules
    - filter per-page records by `line`
    - write `highlights_cleaned.json`
    - annotate payload metadata with `dropped_lines`

### Key Functions (Cleanup)

- `_parse_line_spec(spec)`
- `_parse_drop_rule(rule)`
- `_build_drop_map(rules)`
- `clean_highlights(payload, drop_map)`
- `build_parser()`
- `main(argv=None)`

- `scripts/extract_highlight_features.py`
  - Feature extraction pipeline over cleaned highlights.
  - Core responsibilities:
    - compute line-level word width medians
    - compute sentence-level word width medians
    - compute two-pass distance-to-previous-word and distance-to-next-word features
    - compute token geometry/context/start-end/svara features
    - write enriched `extracted_features.json`

### Key Functions (Feature Extraction)

- `_distance_between(records, start_idx, end_idx, line)`
- `_has_svara_marks(token)`
- `_group_word_width_medians(pages)`
- `_extract_features(payload)`
- `build_parser()`
- `main(argv=None)`

## 6) Additional Package Copies (Non-Canonical Right Now)

These exist but are not the active packaged entrypoint path today:

- `src/vedke2/pdf_pages_to_svg.py`
- `vedke2/pdf_pages_to_svg.py`

They are near-duplicate implementations and can cause drift/confusion if edited independently.  
For now, treat `scripts/pdf_pages_to_svg.py` as source-of-truth.

## 7) Retrieval Playbook (Fast Code Navigation)

When you need a specific behavior, jump directly:

- CLI argument behavior:
  - search: `build_parser`
- Config merge precedence:
  - search: `_resolve_value`, `_build_plan`
- Page list parsing:
  - search: `_parse_page_spec`, `_parse_pages`
- Header/footer trimming:
  - search: `_crop_svg`
- Actual conversion command:
  - search: `_convert_page_to_svg`, `pdftocairo`
- Extensionless file handling:
  - search: `extensionless`
- Output naming:
  - search: `base_name = f"{plan.prefix}-{page:03d}"`

Recommended terminal queries:

```bash
rg -n "def _crop_svg|def _convert_page_to_svg|def _build_plan" scripts/pdf_pages_to_svg.py
rg -n "trim-top|trim-bottom|--pages|--config" scripts/pdf_pages_to_svg.py README.md
sed -n '1,260p' scripts/pdf_pages_to_svg.py
rg -n "build_highlights|_extract_word_boxes|tokenId|sentence|kind" scripts/build_highlights_json.py
sed -n '1,260p' scripts/build_highlights_json.py
rg -n "drop|clean_highlights|lineSpec|dropped_lines" scripts/clean_highlights_json.py
sed -n '1,240p' scripts/clean_highlights_json.py
rg -n "distanceToNextWord|normalizedXInLine|hasSvaraMarks|placeholderFillValuePrev|placeholderFillValueNext|_extract_features" scripts/extract_highlight_features.py
sed -n '1,260p' scripts/extract_highlight_features.py
```

## 8) Known Constraints / Assumptions

- Requires Poppler binaries on PATH:
  - `pdfinfo`
  - `pdftocairo`
- Trim values are interpreted in SVG viewBox units (or ratios of page height).
- Ratio trim overrides pixel trim when both are provided.
- Extraction expects text/graphics PDF pages but does not parse text content; it only converts and crops.

## 9) Typical Operations

- Regenerate selected pages:
  - `uv run python scripts/pdf_pages_to_svg.py --pdf ... --pages 2,3,4 --out ...`
- Config-driven run:
  - `uv run python scripts/pdf_pages_to_svg.py --config data/extract_config.example.json`
- Build highlights:
  - `uv run python scripts/build_highlights_json.py --pdf ... --pages 2,3,4 --out out/ganapati/highlights.json`
- Clean highlights (drop lines):
  - `uv run python scripts/clean_highlights_json.py --in out/ganapati/highlights.json --out out/ganapati/highlights_cleaned.json --drop 2:1-3`
- Extract token features:
  - `uv run python scripts/extract_highlight_features.py --in out/ganapati/highlights_cleaned.json --out out/ganapati/extracted_features.json`
- Clean and rebuild:
  - remove `out/` then rerun extractor with desired params.
