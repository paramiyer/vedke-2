from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

OPEN_BRACKETS = set("([{<（［｛")
CLOSE_BRACKETS = set(")]}>）］｝")
DANDAS = {"।", "॥"}
PUNC_CHARS = set(".,;:!?-—()[]{}<>/\\'\"`|।॥")


@dataclass(frozen=True)
class RawToken:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    block_num: int
    par_num: int
    line_num: int


def _parse_pages(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return sorted(set(out))


def _is_punc(token: str) -> bool:
    if token in DANDAS:
        return True
    return bool(token) and all(ch in PUNC_CHARS for ch in token)


def _split_dandas(token: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    for ch in token:
        if ch in DANDAS:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(ch)
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [x for x in out if x.strip()]


def _strip_bracketed_parts(token: str, depth: int) -> tuple[list[str], int]:
    out: list[str] = []
    buf: list[str] = []
    d = depth
    for ch in token:
        if ch in OPEN_BRACKETS:
            if d == 0 and buf:
                out.append("".join(buf))
                buf = []
            d += 1
            continue
        if ch in CLOSE_BRACKETS:
            if d > 0:
                d -= 1
            continue
        if d == 0:
            buf.append(ch)
    if buf and d == 0:
        out.append("".join(buf))
    return [x for x in out if x.strip()], d


def _line_has_underline(crop_img: Image.Image, x0: int, y0: int, x1: int, y1: int) -> bool:
    # Heuristic: look for a dark horizontal line immediately below the full OCR line bbox.
    gray = crop_img.convert("L")
    W, H = gray.size
    lx0 = max(0, x0)
    lx1 = min(W, x1)
    ly0 = max(0, y1 + 1)
    line_h = max(1, y1 - y0)
    ly1 = min(H, y1 + max(3, int(round(line_h * 0.25))))
    if lx1 <= lx0 or ly1 <= ly0:
        return False
    pix = gray.load()
    row_scores = []
    for yy in range(ly0, ly1):
        dark = 0
        total = lx1 - lx0
        for xx in range(lx0, lx1):
            if pix[xx, yy] < 90:
                dark += 1
        row_scores.append(dark / total if total else 0.0)
    if not row_scores:
        return False
    return max(row_scores) > 0.28


def _ocr_page_tokens(img_path: Path, lang: str) -> list[RawToken]:
    tsv_cmd = [
        "tesseract",
        str(img_path),
        "stdout",
        "-l",
        lang,
        "--psm",
        "6",
        "tsv",
    ]
    tsv = subprocess.run(tsv_cmd, capture_output=True, text=True)
    if tsv.returncode != 0:
        raise RuntimeError(tsv.stderr)

    lines = tsv.stdout.splitlines()
    out: list[RawToken] = []
    if not lines:
        return out

    header = lines[0].split("\t")
    idx = {k: i for i, k in enumerate(header)}
    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) < len(header):
            cols += [""] * (len(header) - len(cols))
        text = cols[idx.get("text", -1)].strip() if idx.get("text", -1) >= 0 else ""
        if not text:
            continue
        conf_raw = cols[idx.get("conf", -1)] if idx.get("conf", -1) >= 0 else "-1"
        try:
            conf = float(conf_raw)
        except ValueError:
            conf = -1.0
        if conf < 0:
            continue
        left = int(cols[idx["left"]]) if cols[idx["left"]].strip() else 0
        top_px = int(cols[idx["top"]]) if cols[idx["top"]].strip() else 0
        width = int(cols[idx["width"]]) if cols[idx["width"]].strip() else 0
        height = int(cols[idx["height"]]) if cols[idx["height"]].strip() else 0
        block_num = int(cols[idx["block_num"]]) if "block_num" in idx and cols[idx["block_num"]].strip() else 0
        par_num = int(cols[idx["par_num"]]) if "par_num" in idx and cols[idx["par_num"]].strip() else 0
        line_num = int(cols[idx["line_num"]]) if "line_num" in idx and cols[idx["line_num"]].strip() else 0
        out.append(
            RawToken(
                text=text,
                x=left,
                y=top_px,
                w=width,
                h=height,
                conf=conf,
                block_num=block_num,
                par_num=par_num,
                line_num=line_num,
            )
        )
    return out


def build_extraction(
    pdf: Path,
    pages: list[int],
    trim_top_ratio: float,
    trim_bottom_ratio: float,
    lang: str,
    skip_brackets: bool,
    skip_underlined: bool,
) -> dict:
    results: dict[str, dict] = {}

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        prefix = tdp / "page"
        cmd = [
            "pdftoppm",
            "-f",
            str(pages[0]),
            "-l",
            str(pages[-1]),
            "-r",
            "300",
            "-png",
            str(pdf),
            str(prefix),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr)

        images = sorted(tdp.glob("page-*.png"))
        if len(images) != len(pages):
            raise RuntimeError(f"Expected {len(pages)} images, got {len(images)}")

        for page_no, img_path in zip(pages, images):
            im = Image.open(img_path)
            w, h = im.size
            top_trim = int(round(h * trim_top_ratio))
            bottom_keep = int(round(h * (1.0 - trim_bottom_ratio)))
            cropped = im.crop((0, top_trim, w, bottom_keep))
            crop_path = tdp / f"crop-{page_no}.png"
            cropped.save(crop_path)

            raw_tokens = _ocr_page_tokens(crop_path, lang=lang)
            line_boxes: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
            for rt in raw_tokens:
                k = (rt.block_num, rt.par_num, rt.line_num)
                x0, y0, x1, y1 = rt.x, rt.y, rt.x + rt.w, rt.y + rt.h
                if k in line_boxes:
                    a0, b0, a1, b1 = line_boxes[k]
                    line_boxes[k] = (min(a0, x0), min(b0, y0), max(a1, x1), max(b1, y1))
                else:
                    line_boxes[k] = (x0, y0, x1, y1)

            underlined_lines: set[tuple[int, int, int]] = set()
            for k, (x0, y0, x1, y1) in line_boxes.items():
                if _line_has_underline(cropped, x0, y0, x1, y1):
                    underlined_lines.add(k)

            page_tokens: list[dict] = []
            bracket_depth = 0
            token_id = 0
            for rt in raw_tokens:
                line_key = (rt.block_num, rt.par_num, rt.line_num)
                underlined = line_key in underlined_lines
                if skip_underlined and underlined:
                    continue

                parts = _split_dandas(rt.text)
                final_parts: list[str] = []
                for part in parts:
                    if skip_brackets:
                        kept, bracket_depth = _strip_bracketed_parts(part, bracket_depth)
                        final_parts.extend(kept)
                    else:
                        final_parts.append(part)

                for part in final_parts:
                    tok = part.strip()
                    if not tok:
                        continue
                    token_id += 1
                    kind = "punc" if _is_punc(tok) else "word"
                    page_tokens.append(
                        {
                            "tokenId": f"P{page_no:04d}_T{token_id:04d}",
                            "token": tok,
                            "kind": kind,
                            "char_count": len(tok),
                            "x": rt.x,
                            "y": rt.y + top_trim,
                            "w": rt.w,
                            "h": rt.h,
                            "conf": round(rt.conf, 2),
                            "underlined": underlined,
                        }
                    )

            results[str(page_no)] = {
                "page": page_no,
                "image_size_px": {"w": w, "h": h},
                "trim_px": {"top": top_trim, "bottom": h - bottom_keep},
                "token_count": len(page_tokens),
                "tokens": page_tokens,
            }

    return {
        "source_pdf": str(pdf),
        "method": "tesseract_tsv",
        "language": lang,
        "pages": pages,
        "trim": {"top_ratio": trim_top_ratio, "bottom_ratio": trim_bottom_ratio},
        "postprocess": {
            "skip_brackets": skip_brackets,
            "skip_underlined": skip_underlined,
            "split_dandas": True,
            "with_kind": True,
            "with_char_count": True,
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OCR sample extraction with token cleanup rules")
    p.add_argument("--pdf", required=True)
    p.add_argument("--pages", required=True, help="e.g. 8,9,10,11,12")
    p.add_argument("--trim-top-ratio", type=float, default=0.08)
    p.add_argument("--trim-bottom-ratio", type=float, default=0.06)
    p.add_argument("--lang", default="san+hin")
    p.add_argument("--out", default="check_extraction.json")
    p.add_argument("--skip-brackets", action="store_true")
    p.add_argument("--skip-underlined", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pdf = Path(args.pdf).expanduser().resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    pages = _parse_pages(args.pages)
    payload = build_extraction(
        pdf=pdf,
        pages=pages,
        trim_top_ratio=float(args.trim_top_ratio),
        trim_bottom_ratio=float(args.trim_bottom_ratio),
        lang=args.lang,
        skip_brackets=bool(args.skip_brackets),
        skip_underlined=bool(args.skip_underlined),
    )
    out = Path(args.out).expanduser().resolve()
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
