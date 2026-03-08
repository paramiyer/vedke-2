from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.build_highlights_json import (
    _extract_word_boxes,
    _is_punct,
    _page_count,
    _parse_pages,
    _parse_px,
    _parse_ratio,
    _require_tool,
    _run,
    _split_trailing_sentence_punct,
)


def _slugify(value: str) -> str:
    chars: list[str] = []
    for ch in value.strip().lower():
        if ch.isalnum():
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    slug = "".join(chars).strip("_")
    if not slug:
        raise ValueError("--experiment must contain at least one alphanumeric character")
    return slug


def _resolve_pdf_path(raw_pdf: str) -> Path:
    candidate = Path(raw_pdf).expanduser()
    if candidate.exists():
        return candidate.resolve()

    # Common UX case: only filename is provided for a file in Downloads.
    if candidate.parent == Path("."):
        downloads_candidate = (Path.home() / "Downloads" / candidate.name).expanduser()
        if downloads_candidate.exists():
            return downloads_candidate.resolve()

    # Handle relative "Downloads/..." provided from UI.
    parts = candidate.parts
    if parts and parts[0].lower() == "downloads":
        downloads_candidate = (Path.home() / Path(*parts)).expanduser()
        if downloads_candidate.exists():
            return downloads_candidate.resolve()

    return candidate.resolve()


def build_pdf_truth_tokens_payload(
    pdf_path: Path,
    pages: list[int],
    word_boxes: list[dict] | list[object],
    trim_top_px: float | None,
    trim_bottom_px: float | None,
    trim_top_ratio: float | None,
    trim_bottom_ratio: float | None,
) -> dict:
    by_page: dict[int, list[object]] = {}
    for box in word_boxes:
        page = int(getattr(box, "page"))
        by_page.setdefault(page, []).append(box)

    payload_pages: dict[str, list[dict[str, object]]] = {}
    for page in pages:
        items = by_page.get(page, [])
        token_idx = 0
        page_records: list[dict[str, object]] = []
        for item in items:
            text = str(getattr(item, "text"))
            x = float(getattr(item, "x"))
            y = float(getattr(item, "y"))
            w = float(getattr(item, "w"))
            h = float(getattr(item, "h"))
            line = int(getattr(item, "line"))
            text_core, trailing_punct = _split_trailing_sentence_punct(text)
            emit_tokens: list[tuple[str, float, float]] = []
            if text_core:
                emit_tokens.append((text_core, x, w))
            if trailing_punct is not None:
                emit_tokens.append((trailing_punct, x + w, 0.0))
            if not emit_tokens:
                emit_tokens = [(text, x, w)]

            for token_text, token_x, token_w in emit_tokens:
                token_idx += 1
                kind = "punc" if _is_punct(token_text) else "word"
                token_id = f"P{page:04d}_L{line:04d}_T{token_idx:04d}"
                page_records.append(
                    {
                        "tokenId": token_id,
                        "token": token_text,
                        "kind": kind,
                        "x": round(token_x, 3),
                        "y": round(y, 3),
                        "w": round(token_w, 3),
                        "h": round(h, 3),
                    }
                )
        payload_pages[str(page)] = page_records

    return {
        "sourcePdf": str(pdf_path),
        "extraction": {
            "agent": "agent4_pdf_truth",
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
            "fields": ["tokenId", "token", "kind", "x", "y", "w", "h"],
            "notes": ["kind is word or punc", "token-level PDF truth with bbox coordinates"],
        },
        "pages": payload_pages,
    }


def _is_devanagari_token(token: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in token)


def _has_latin(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token))


def build_pdf_truth_tokens_cleaned_payload(full_payload: dict) -> dict:
    cleaned_pages: dict[str, list[dict[str, object]]] = {}
    pages = full_payload.get("pages", {})
    if isinstance(pages, dict):
        for page, records in pages.items():
            out: list[dict[str, object]] = []
            if isinstance(records, list):
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    if str(rec.get("kind")) != "word":
                        continue
                    token = str(rec.get("token", "")).strip()
                    if not token:
                        continue
                    if _has_latin(token):
                        continue
                    if not _is_devanagari_token(token):
                        continue
                    out.append({"token": token, "kind": "word", "char_count": len(token)})
            cleaned_pages[str(page)] = out

    return {
        "sourcePdf": full_payload.get("sourcePdf"),
        "extraction": {
            "agent": "agent4_pdf_truth",
            "artifact": "pdf_tokens_cleaned",
            "filters": ["kind=word", "drop_latin", "keep_devanagari_only"],
            "fields": ["token", "kind", "char_count"],
        },
        "pages": cleaned_pages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-agent4-pdf-truth",
        description="Agent 4: Extract token-level PDF truth with kind=word|punc and x/y/w/h into data/<experiment>/out/pdf_tokens.json",
    )
    parser.add_argument("--experiment", required=True, help="Experiment name used as folder slug under data/")
    parser.add_argument("--pdf", required=True, help="Source PDF path")
    parser.add_argument("--pages", required=True, help="Pages selection, e.g. 2,3,4")
    parser.add_argument("--trim-top-px", type=float, default=None)
    parser.add_argument("--trim-bottom-px", type=float, default=None)
    parser.add_argument("--trim-top-ratio", type=float, default=None)
    parser.add_argument("--trim-bottom-ratio", type=float, default=None)
    parser.add_argument("--out", default=None, help="Optional output file override")
    parser.add_argument("--cleaned-out", default=None, help="Optional cleaned output file override")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    slug = _slugify(args.experiment)
    pdf_path = _resolve_pdf_path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_path = Path(args.out).expanduser().resolve() if args.out else (Path("data") / slug / "out" / "pdf_tokens.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_out_path = (
        Path(args.cleaned_out).expanduser().resolve()
        if args.cleaned_out
        else (Path("data") / slug / "out" / "pdf_tokens_cleaned.json").resolve()
    )
    cleaned_out_path.parent.mkdir(parents=True, exist_ok=True)

    _require_tool("pdfinfo", "Install poppler (macOS: `brew install poppler`).")
    _require_tool("pdftotext", "Install poppler (macOS: `brew install poppler`).")

    max_page = _page_count(pdf_path)
    pages = _parse_pages(args.pages, max_page=max_page)
    trim_top_px = _parse_px(args.trim_top_px, "top_px")
    trim_bottom_px = _parse_px(args.trim_bottom_px, "bottom_px")
    trim_top_ratio = _parse_ratio(args.trim_top_ratio, "top_ratio")
    trim_bottom_ratio = _parse_ratio(args.trim_bottom_ratio, "bottom_ratio")

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

    payload = build_pdf_truth_tokens_payload(
        pdf_path=pdf_path,
        pages=pages,
        word_boxes=word_boxes,
        trim_top_px=trim_top_px,
        trim_bottom_px=trim_bottom_px,
        trim_top_ratio=trim_top_ratio,
        trim_bottom_ratio=trim_bottom_ratio,
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cleaned_payload = build_pdf_truth_tokens_cleaned_payload(payload)
    cleaned_out_path.write_text(json.dumps(cleaned_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"experiment={slug}")
    print(f"pdf={pdf_path}")
    print(f"pages={','.join(str(x) for x in pages)}")
    print(f"output={out_path}")
    print(f"cleaned_output={cleaned_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
