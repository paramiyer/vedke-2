from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_line_spec(spec: str) -> set[int]:
    lines: set[int] = set()
    for raw_chunk in spec.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if start > end:
                raise ValueError(f"Invalid line range '{chunk}': start must be <= end")
            for value in range(start, end + 1):
                if value < 1:
                    raise ValueError(f"Line numbers must be >= 1, got {value}")
                lines.add(value)
            continue
        value = int(chunk)
        if value < 1:
            raise ValueError(f"Line numbers must be >= 1, got {value}")
        lines.add(value)
    if not lines:
        raise ValueError("Empty line spec. Use values like '1-3,7'.")
    return lines


def _parse_drop_rule(rule: str) -> tuple[int, set[int]]:
    if ":" not in rule:
        raise ValueError(f"Invalid drop rule '{rule}'. Expected format '<page>:<lineSpec>'")
    page_s, line_spec = rule.split(":", 1)
    page = int(page_s.strip())
    if page < 1:
        raise ValueError(f"Page number must be >= 1, got {page}")
    return page, _parse_line_spec(line_spec.strip())


def _build_drop_map(rules: list[str]) -> dict[int, set[int]]:
    drop_map: dict[int, set[int]] = {}
    for rule in rules:
        page, lines = _parse_drop_rule(rule)
        drop_map.setdefault(page, set()).update(lines)
    return drop_map


def clean_highlights(payload: dict, drop_map: dict[int, set[int]]) -> tuple[dict, dict[int, int]]:
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("Invalid highlights payload: 'pages' must be an object")

    removed_count: dict[int, int] = {}
    cleaned_pages: dict[str, list[dict]] = {}

    for page_key, records in pages.items():
        if not isinstance(records, list):
            raise ValueError(f"Invalid page payload for page '{page_key}': expected array")
        page_num = int(page_key)
        drop_lines = drop_map.get(page_num, set())

        kept: list[dict] = []
        removed = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            line = int(record.get("line", -1))
            if line in drop_lines:
                removed += 1
                continue
            kept.append(record)

        cleaned_pages[page_key] = kept
        if removed:
            removed_count[page_num] = removed

    cleaned = dict(payload)
    cleaned["pages"] = cleaned_pages
    extraction = cleaned.get("extraction")
    if isinstance(extraction, dict):
        source_filter = extraction.get("sourceFilter")
        if not isinstance(source_filter, dict):
            source_filter = {}
            extraction["sourceFilter"] = source_filter
        source_filter["dropped_lines"] = {str(page): sorted(lines) for page, lines in sorted(drop_map.items())}

    return cleaned, removed_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clean-highlights-json",
        description="Drop specified page/line ranges from highlights.json and write highlights_cleaned.json.",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Path to input highlights.json")
    parser.add_argument("--out", required=True, help="Path to output highlights_cleaned.json")
    parser.add_argument(
        "--drop",
        action="append",
        required=True,
        help="Drop rule in format '<page>:<lineSpec>', e.g. --drop 2:1-3 --drop 3:1",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input highlights file not found: {input_path}")

    drop_map = _build_drop_map(list(args.drop))
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    cleaned, removed_count = clean_highlights(payload, drop_map=drop_map)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output: {out_path}")
    for page, lines in sorted(drop_map.items()):
        print(f"drop page {page}: lines {','.join(str(x) for x in sorted(lines))}")
    for page, removed in sorted(removed_count.items()):
        print(f"removed page {page}: {removed} token(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
