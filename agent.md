# Agent Operating Notes

## Core Rule

Use `repo_map.md` as the primary retrieval index before reading code files.

Do not parse entire code files by default.  
First locate exact code elements (functions, classes, flags, paths), then open only those ranges.

## Retrieval-First Workflow (RAG Style)

1. Open `repo_map.md` and identify:
   - likely file(s)
   - exact function/class/section names
   - related CLI flags or data contracts
2. Use targeted search (`rg`) to pinpoint symbols/lines.
3. Read minimal slices with `sed -n` around matching lines.
4. Edit only the necessary file(s) and function blocks.
5. Validate with minimal commands focused on changed behavior.

## Required Navigation Pattern

Use this pattern for investigation:

```bash
rg -n "<symbol-or-flag>" repo_map.md
rg -n "<symbol-or-flag>" <candidate-file>
sed -n '<start>,<end>p' <candidate-file>
```

Examples:

```bash
rg -n "_crop_svg|_build_plan|build_parser" repo_map.md scripts/pdf_pages_to_svg.py
rg -n "--trim-top|--pages|--config" scripts/pdf_pages_to_svg.py README.md
```

## Scope and Accuracy Rules

- Prefer exact symbol references from `repo_map.md` over exploratory full-file scans.
- Only broaden search when the map is stale/missing data.
- If map and code disagree, update `repo_map.md` after code changes.
- Treat `scripts/pdf_pages_to_svg.py` as current source-of-truth unless project packaging changes.

## Output Expectations

- When explaining code, reference precise file paths and symbol names.
- When making changes, state:
  - what symbol changed
  - why it changed
  - how it was validated

