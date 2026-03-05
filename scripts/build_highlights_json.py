from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SENTENCE_END = {"।", "॥", ".", "!", "?"}


@dataclass(frozen=True)
class WordBox:
    page: int
    line: int
    text: str
    x: float
    y: float
    w: float
    h: float


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _require_tool(name: str, hint: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required tool '{name}'. {hint}")


def _page_count(pdf_path: Path) -> int:
    result = _run(["pdfinfo", str(pdf_path)])
    if result.returncode != 0:
        raise RuntimeError(f"pdfinfo failed for {pdf_path}:\n{result.stderr.strip()}")
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not parse page count from pdfinfo output for {pdf_path}")


def _parse_page_spec(spec: str, max_page: int) -> list[int]:
    pages: set[int] = set()
    for raw_chunk in spec.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if start > end:
                raise ValueError(f"Invalid range '{chunk}': start must be <= end")
            pages.update(range(start, end + 1))
            continue
        pages.add(int(chunk))
    if not pages:
        raise ValueError("No pages selected. Use --pages like '2,3,5-7'.")
    invalid = sorted(page for page in pages if page < 1 or page > max_page)
    if invalid:
        raise ValueError(f"Pages out of bounds (1..{max_page}): {invalid}")
    return sorted(pages)


def _parse_pages(raw: object, max_page: int) -> list[int]:
    if isinstance(raw, str):
        return _parse_page_spec(raw, max_page=max_page)
    if isinstance(raw, list):
        pages: set[int] = set()
        for item in raw:
            if not isinstance(item, int):
                raise ValueError("Config 'pages' list must contain integers only")
            pages.add(item)
        if not pages:
            raise ValueError("Config 'pages' list is empty")
        invalid = sorted(page for page in pages if page < 1 or page > max_page)
        if invalid:
            raise ValueError(f"Pages out of bounds (1..{max_page}): {invalid}")
        return sorted(pages)
    raise ValueError("Pages must be either a string like '2,3-4' or an integer list")


def _parse_ratio(value: object, field: str) -> float | None:
    if value is None:
        return None
    ratio = float(value)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"{field} must be in [0, 1)")
    return ratio


def _parse_px(value: object, field: str) -> float | None:
    if value is None:
        return None
    px = float(value)
    if px < 0:
        raise ValueError(f"{field} must be >= 0")
    return px


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object")
    return payload


def _resolve_value(cli_value: object, config_value: object, default_value: object) -> object:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default_value


def _is_punct(text: str) -> bool:
    if not text:
        return True
    if text in {"।", "॥"}:
        return True
    if text.isdigit():
        return True
    for ch in text:
        if ch.isspace():
            continue
        if ch in {"।", "॥"}:
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        return False
    return True


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _split_trailing_sentence_punct(text: str) -> tuple[str, str | None]:
    if text.endswith("॥") and text != "॥":
        return text[:-1].rstrip(), "॥"
    if text.endswith("।") and text != "।":
        return text[:-1].rstrip(), "।"
    return text, None


def _extract_word_boxes(
    xml_text: str,
    selected_pages: set[int],
    trim_top_px: float | None,
    trim_bottom_px: float | None,
    trim_top_ratio: float | None,
    trim_bottom_ratio: float | None,
) -> list[WordBox]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"Could not parse pdftotext XML: {exc}") from exc

    out: list[WordBox] = []
    page_idx = 0
    for page in root.iter():
        if _local_name(page.tag) != "page":
            continue
        page_idx += 1
        if page_idx not in selected_pages:
            continue

        page_height = float(page.attrib.get("height", "0"))
        top_trim = trim_top_px if trim_top_px is not None else 0.0
        bottom_trim = trim_bottom_px if trim_bottom_px is not None else 0.0
        if trim_top_ratio is not None:
            top_trim = page_height * trim_top_ratio
        if trim_bottom_ratio is not None:
            bottom_trim = page_height * trim_bottom_ratio
        visible_y_min = top_trim
        visible_y_max = max(visible_y_min, page_height - bottom_trim)

        line_idx = 0
        for line in page.iter():
            if _local_name(line.tag) != "line":
                continue
            line_words: list[WordBox] = []
            for word in line.iter():
                if _local_name(word.tag) != "word":
                    continue
                text = (word.text or "").strip()
                if not text:
                    continue
                x_min = float(word.attrib["xMin"])
                y_min = float(word.attrib["yMin"])
                x_max = float(word.attrib["xMax"])
                y_max = float(word.attrib["yMax"])
                center_y = (y_min + y_max) / 2.0
                if center_y < visible_y_min or center_y > visible_y_max:
                    continue
                line_words.append(
                    WordBox(
                        page=page_idx,
                        line=0,  # set after validating line has content
                        text=text,
                        x=x_min,
                        y=y_min,
                        w=max(0.0, x_max - x_min),
                        h=max(0.0, y_max - y_min),
                    )
                )
            if not line_words:
                continue
            line_idx += 1
            for item in line_words:
                out.append(
                    WordBox(
                        page=item.page,
                        line=line_idx,
                        text=item.text,
                        x=item.x,
                        y=item.y,
                        w=item.w,
                        h=item.h,
                    )
                )
    return out


def build_highlights_payload(
    pdf_path: Path,
    pages: list[int],
    word_boxes: list[WordBox],
    trim_top_px: float | None,
    trim_bottom_px: float | None,
    trim_top_ratio: float | None,
    trim_bottom_ratio: float | None,
) -> dict:
    by_page: dict[int, list[WordBox]] = {}
    for box in word_boxes:
        by_page.setdefault(box.page, []).append(box)

    payload_pages: dict[str, list[dict[str, object]]] = {}
    for page in pages:
        items = by_page.get(page, [])
        sentence_idx = 1
        token_idx = 0
        page_records: list[dict[str, object]] = []
        for item in items:
            text_core, trailing_punct = _split_trailing_sentence_punct(item.text)
            emit_tokens: list[tuple[str, float, float]] = []
            if text_core:
                emit_tokens.append((text_core, float(item.x), float(item.w)))
            if trailing_punct is not None:
                # Emit trailing danda/double-danda as a standalone punctuation token.
                # Reuse the line box with a zero-width anchor at token end to preserve order deterministically.
                emit_tokens.append((trailing_punct, float(item.x) + float(item.w), 0.0))
            if not emit_tokens:
                emit_tokens = [(item.text, float(item.x), float(item.w))]

            for token_text, token_x, token_w in emit_tokens:
                token_idx += 1
                kind = "punct" if _is_punct(token_text) else "word"
                token_id = f"P{page:04d}_L{item.line:04d}_S{sentence_idx:04d}_T{token_idx:04d}"
                page_records.append(
                    {
                        "tokenId": token_id,
                        "page": page,
                        "line": item.line,
                        "sentence": sentence_idx,
                        "token": token_text,
                        "kind": kind,
                        "x": round(token_x, 3),
                        "y": round(item.y, 3),
                        "w": round(token_w, 3),
                        "h": round(item.h, 3),
                    }
                )
                if kind == "punct" and token_text in SENTENCE_END:
                    sentence_idx += 1
        payload_pages[str(page)] = page_records

    return {
        "sourcePdf": str(pdf_path),
        "extraction": {
            "engine": "pdftotext -bbox-layout",
            "units": "pt",
            "sourceFilter": {
                "pages": pages,
                "trim": {
                    "top_px": trim_top_px,
                    "bottom_px": trim_bottom_px,
                    "top_ratio": trim_top_ratio,
                    "bottom_ratio": trim_bottom_ratio,
                },
            },
            "fields": [
                "tokenId",
                "page",
                "line",
                "sentence",
                "token",
                "kind",
                "x",
                "y",
                "w",
                "h",
            ],
            "notes": [
                "token preserves Devanagari from PDF text layer (including svara marks when present)",
                "sentence increments after danda/terminal punctuation",
                "kind is word or punct",
            ],
        },
        "pages": payload_pages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-highlights-json",
        description="Build token-level highlights.json (page/line/sentence/token/bbox) from PDF text-layer boxes.",
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file for extractor inputs.")
    parser.add_argument("--pdf", default=None, help="Path to source PDF file.")
    parser.add_argument("--pages", default=None, help="Comma-separated pages/ranges, e.g. '2,3,5-7'.")
    parser.add_argument(
        "--out",
        default="out/ganapati/highlights.json",
        help="Output path for highlights JSON (default: out/ganapati/highlights.json).",
    )
    parser.add_argument(
        "--trim-top-px",
        type=float,
        default=None,
        help="Trim this many units from top of each page in PDF bbox space.",
    )
    parser.add_argument(
        "--trim-bottom-px",
        type=float,
        default=None,
        help="Trim this many units from bottom of each page in PDF bbox space.",
    )
    parser.add_argument(
        "--trim-top-ratio",
        type=float,
        default=None,
        help="Trim top ratio of page height (0 <= ratio < 1). Overrides --trim-top-px.",
    )
    parser.add_argument(
        "--trim-bottom-ratio",
        type=float,
        default=None,
        help="Trim bottom ratio of page height (0 <= ratio < 1). Overrides --trim-bottom-px.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config: dict = {}
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = _load_config(config_path)

    trim_cfg = config.get("trim", {})
    if trim_cfg is not None and not isinstance(trim_cfg, dict):
        raise ValueError("Config field 'trim' must be an object when provided")

    pdf_value = _resolve_value(args.pdf, config.get("pdf"), None)
    if pdf_value is None:
        raise ValueError("Missing PDF input. Provide --pdf or config 'pdf'.")
    pdf_path = Path(str(pdf_value)).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out_value = _resolve_value(args.out, config.get("highlights_out"), args.out)
    out_path = Path(str(out_value)).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _require_tool("pdfinfo", "Install poppler (macOS: `brew install poppler`).")
    _require_tool("pdftotext", "Install poppler (macOS: `brew install poppler`).")

    max_page = _page_count(pdf_path)
    pages_value = _resolve_value(args.pages, config.get("pages"), None)
    if pages_value is None:
        raise ValueError("Missing page selection. Provide --pages or config 'pages'.")
    pages = _parse_pages(pages_value, max_page=max_page)

    trim_top_px = _parse_px(_resolve_value(args.trim_top_px, trim_cfg.get("top_px"), None), "top_px")
    trim_bottom_px = _parse_px(_resolve_value(args.trim_bottom_px, trim_cfg.get("bottom_px"), None), "bottom_px")
    trim_top_ratio = _parse_ratio(_resolve_value(args.trim_top_ratio, trim_cfg.get("top_ratio"), None), "top_ratio")
    trim_bottom_ratio = _parse_ratio(
        _resolve_value(args.trim_bottom_ratio, trim_cfg.get("bottom_ratio"), None),
        "bottom_ratio",
    )

    result = _run(["pdftotext", "-bbox-layout", str(pdf_path), "-"])
    if result.returncode != 0:
        raise RuntimeError(f"pdftotext failed for {pdf_path}:\n{result.stderr.strip()}")

    word_boxes = _extract_word_boxes(
        result.stdout,
        selected_pages=set(pages),
        trim_top_px=trim_top_px,
        trim_bottom_px=trim_bottom_px,
        trim_top_ratio=trim_top_ratio,
        trim_bottom_ratio=trim_bottom_ratio,
    )
    payload = build_highlights_payload(
        pdf_path=pdf_path,
        pages=pages,
        word_boxes=word_boxes,
        trim_top_px=trim_top_px,
        trim_bottom_px=trim_bottom_px,
        trim_top_ratio=trim_top_ratio,
        trim_bottom_ratio=trim_bottom_ratio,
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"PDF: {pdf_path}")
    print(f"Pages: {','.join(str(p) for p in pages)}")
    print(
        "Trim: "
        f"top_px={trim_top_px}, bottom_px={trim_bottom_px}, "
        f"top_ratio={trim_top_ratio}, bottom_ratio={trim_bottom_ratio}"
    )
    print(f"Output: {out_path}")
    for p in pages:
        print(f"page {p}: {len(payload['pages'].get(str(p), []))} token(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
