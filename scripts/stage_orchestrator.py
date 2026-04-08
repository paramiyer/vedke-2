from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

STAGES = ("source_intake", "asr_preparation", "pdf_truth", "alignment_config", "quality_gates")
DEFAULT_OCR_LANG = "san+script/Devanagari"
DEFAULT_TOP_TRIM_RATIO = 0.08
DEFAULT_BOTTOM_TRIM_RATIO = 0.06


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


@dataclass(frozen=True)
class BuildPaths:
    experiment: str
    root: Path
    input_dir: Path
    out_dir: Path


def build_paths(experiment: str) -> BuildPaths:
    slug = _slugify(experiment)
    root = (Path("data") / slug).resolve()
    return BuildPaths(
        experiment=slug,
        root=root,
        input_dir=(root / "input"),
        out_dir=(root / "out"),
    )


def _resolve_user_pdf_path(raw_pdf: str) -> Path:
    raw = raw_pdf.strip()
    candidate = Path(raw).expanduser()
    attempts: list[Path] = []

    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.append(Path.cwd() / candidate)
        attempts.append(Path.home() / candidate)
        attempts.append(candidate)
        attempts.append(Path.home() / "Downloads" / candidate.name)

    # If user passed "Downloads/foo.pdf", map to ~/Downloads/foo.pdf.
    parts = candidate.parts
    if parts and parts[0].lower() == "downloads":
        attempts.append(Path.home() / Path(*parts[1:]))

    for p in attempts:
        p = p.expanduser()
        if p.exists() and p.is_file():
            return p.resolve()

    attempted = ", ".join(str(p) for p in attempts)
    raise FileNotFoundError(f"PDF not found for '{raw_pdf}'. Tried: {attempted}")


def _copy_pdf_to_input(paths: BuildPaths, raw_pdf: str) -> Path:
    src = _resolve_user_pdf_path(raw_pdf)
    dst = paths.input_dir / "source.pdf"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    (paths.input_dir / "source_pdf_local.txt").write_text(str(dst) + "\n", encoding="utf-8")
    return dst


def _is_devanagari_token(token: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in token)


def _has_latin(token: str) -> bool:
    return bool(re.search(r"[A-Za-z]", token))


def _ocr_payload_to_pdf_tokens_payload(ocr_payload: dict) -> dict:
    pages = [int(p) for p in ocr_payload.get("pages", [])]
    payload_pages: dict[str, list[dict[str, object]]] = {}
    # Canonical page dimensions in OCR source space (300 DPI pixel coordinates).
    # These are the authoritative reference dimensions for all token x/y/w/h
    # values stored in this artifact.  Downstream consumers MUST use these
    # dimensions (not computed max-token-extent) when mapping token coordinates
    # to any display coordinate space.
    page_image_sizes: dict[str, dict[str, int]] = {}
    results = ocr_payload.get("results", {})
    for page in pages:
        page_block = results.get(str(page), {}) if isinstance(results, dict) else {}
        tokens = page_block.get("tokens", []) if isinstance(page_block, dict) else []
        page_records: list[dict[str, object]] = []
        if isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                page_records.append(
                    {
                        "tokenId": str(token.get("tokenId", "")),
                        "token": str(token.get("token", "")),
                        "kind": str(token.get("kind", "word")),
                        "x": token.get("x", 0),
                        "y": token.get("y", 0),
                        "w": token.get("w", 0),
                        "h": token.get("h", 0),
                        "char_count": token.get("char_count"),
                        "conf": token.get("conf"),
                        "underlined": token.get("underlined", False),
                    }
                )
        payload_pages[str(page)] = page_records
        # Carry image_size_px from OCR results into the tokens artifact.
        if isinstance(page_block, dict):
            img_size = page_block.get("image_size_px")
            if isinstance(img_size, dict):
                w_px = img_size.get("w")
                h_px = img_size.get("h")
                if isinstance(w_px, int) and isinstance(h_px, int) and w_px > 0 and h_px > 0:
                    page_image_sizes[str(page)] = {"w": w_px, "h": h_px}

    return {
        "sourcePdf": str(ocr_payload.get("source_pdf", "")),
        "extraction": {
            "agent": "stage4_ocr_truth",
            "engine": str(ocr_payload.get("method", "tesseract_tsv")),
            "units": "px",
            "language": ocr_payload.get("language"),
            "sourceFilter": {
                "pages": pages,
                "trim": ocr_payload.get("trim", {}),
            },
            "postprocess": ocr_payload.get("postprocess", {}),
            "fields": ["tokenId", "token", "kind", "x", "y", "w", "h"],
            "notes": [
                "Built from OCR extraction with bracket and underline cleanup rules.",
                "Additional token metadata includes char_count, conf, and underlined.",
                "pageImageSizes contains canonical page dimensions in OCR source space (px).",
                "All token x/y/w/h coordinates are in this same pixel coordinate space.",
            ],
        },
        "pageImageSizes": page_image_sizes,
        "pages": payload_pages,
    }


def _build_cleaned_pdf_tokens_payload(full_payload: dict) -> dict:
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
            "agent": "stage4_ocr_truth",
            "artifact": "pdf_tokens_cleaned",
            "filters": ["kind=word", "drop_latin", "keep_devanagari_only"],
            "fields": ["token", "kind", "char_count"],
        },
        "pages": cleaned_pages,
    }


def run_ocr_pdf_truth(
    *,
    pdf_path: Path,
    pages: str,
    top_trim_ratio: float | None,
    bottom_trim_ratio: float | None,
    out_path: Path,
    cleaned_out_path: Path,
) -> None:
    try:
        from scripts.build_ocr_check_extraction import _parse_pages as _parse_ocr_pages
        from scripts.build_ocr_check_extraction import build_extraction as build_ocr_extraction
    except ModuleNotFoundError as exc:
        if exc.name == "PIL":
            raise RuntimeError("Missing dependency: Pillow (PIL). Run `uv sync` (recommended) or `python3 -m pip install pillow`.") from exc
        raise

    page_list = _parse_ocr_pages(pages)
    if not page_list:
        raise ValueError("pdf_truth requires at least one page")
    # Stage 3 produces frozen karaoke page images alongside OCR artifacts.
    # These 300 DPI PNG images define the canonical coordinate space that all
    # downstream token bounding boxes reference.
    karaoke_pages_dir = out_path.parent / "karaoke_pages"
    ocr_payload = build_ocr_extraction(
        pdf=pdf_path,
        pages=page_list,
        trim_top_ratio=top_trim_ratio if top_trim_ratio is not None else DEFAULT_TOP_TRIM_RATIO,
        trim_bottom_ratio=bottom_trim_ratio if bottom_trim_ratio is not None else DEFAULT_BOTTOM_TRIM_RATIO,
        lang=DEFAULT_OCR_LANG,
        skip_brackets=True,
        skip_underlined=True,
        page_images_dir=karaoke_pages_dir,
    )
    # Preserve the raw OCR artifact for debugging and tuning.
    check_extraction_path = out_path.parent / "check_extraction.json"
    check_extraction_path.write_text(json.dumps(ocr_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = _ocr_payload_to_pdf_tokens_payload(ocr_payload)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cleaned_payload = _build_cleaned_pdf_tokens_payload(payload)
    cleaned_out_path.write_text(json.dumps(cleaned_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def guardrails_for_stage(stage: str, paths: BuildPaths) -> list[str]:
    if stage not in STAGES:
        return [f"unknown stage: {stage}"]
    errors: list[str] = []
    if stage in {"asr_preparation", "pdf_truth", "alignment_config", "quality_gates"}:
        if not paths.input_dir.exists():
            errors.append(f"missing input dir: {paths.input_dir}")
    if stage in {"alignment_config", "quality_gates"}:
        if not paths.out_dir.exists():
            errors.append(f"missing out dir: {paths.out_dir}")
    if stage in {"asr_preparation", "pdf_truth", "alignment_config", "quality_gates"}:
        if not (paths.input_dir / "yt_link.txt").exists():
            errors.append("missing yt_link.txt in input dir")
    if stage in {"asr_preparation", "alignment_config", "quality_gates"}:
        if not (paths.input_dir / "audio.mp3").exists():
            errors.append("missing audio.mp3 in input dir")
    if stage in {"alignment_config", "quality_gates"}:
        if not (paths.out_dir / "0.json").exists():
            errors.append("missing sarvam output 0.json in out dir")
    if stage == "quality_gates":
        if not (paths.out_dir / "pdf_tokens.json").exists():
            errors.append("missing pdf_tokens.json in out dir")
    return errors


def run_stage(
    stage: str,
    *,
    experiment: str,
    youtube_url: str | None = None,
    pdf_path: str | None = None,
    pages: str | None = None,
    ai_model: str = "saaras:v3",
    language_code: str = "sa-IN",
    top_trim_ratio: float | None = None,
    bottom_trim_ratio: float | None = None,
    min_confidence: float = 0.78,
    max_edit_cost: float = 0.32,
    anchor_token: str | None = None,
    anchor_segment: int = 0,
) -> Path:
    paths = build_paths(experiment)
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    if stage == "source_intake":
        if not youtube_url:
            raise ValueError("source_intake requires youtube_url")
        if not pdf_path:
            raise ValueError("source_intake requires pdf_path")
        if not pages:
            raise ValueError("source_intake requires pages")
        (paths.input_dir / "yt_link.txt").write_text(youtube_url.strip() + "\n", encoding="utf-8")
        (paths.input_dir / "source_pdf.txt").write_text(pdf_path.strip() + "\n", encoding="utf-8")
        (paths.input_dir / "pages.txt").write_text(pages.strip() + "\n", encoding="utf-8")
        manifest = {
            "stage": stage,
            "experiment": paths.experiment,
            "input_dir": str(paths.input_dir),
            "out_dir": str(paths.out_dir),
            "source_pdf": pdf_path,
            "pages": pages,
        }
        out = paths.out_dir / "source_intake_manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    errors = guardrails_for_stage(stage, paths)
    if errors:
        raise RuntimeError("guardrails failed: " + "; ".join(errors))

    if stage == "asr_preparation":
        manifest = {
            "stage": stage,
            "experiment": paths.experiment,
            "audio": str(paths.input_dir / "audio.mp3"),
            "model": ai_model,
            "language_code": language_code,
            "note": "ASR execution is handled by run-sarvam-stt; this stage validates prerequisites.",
        }
        out = paths.out_dir / "asr_preparation_manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    if stage == "pdf_truth":
        local_pdf_path: Path | None = None
        if not pdf_path:
            source_pdf = (paths.input_dir / "source_pdf.txt")
            if source_pdf.exists():
                pdf_path = source_pdf.read_text(encoding="utf-8").strip()
        if not pages:
            source_pages = (paths.input_dir / "pages.txt")
            if source_pages.exists():
                pages = source_pages.read_text(encoding="utf-8").strip()
        if not pdf_path or not pages:
            raise ValueError("pdf_truth requires pdf_path and pages (or source files from source_intake)")
        local_pdf_path = _copy_pdf_to_input(paths, pdf_path)
        out_path = paths.out_dir / "pdf_tokens.json"
        cleaned_out_path = paths.out_dir / "pdf_tokens_cleaned.json"
        run_ocr_pdf_truth(
            pdf_path=local_pdf_path,
            pages=pages,
            top_trim_ratio=top_trim_ratio,
            bottom_trim_ratio=bottom_trim_ratio,
            out_path=out_path,
            cleaned_out_path=cleaned_out_path,
        )
        return paths.out_dir / "pdf_tokens.json"

    if stage == "alignment_config":
        from scripts.run_alignment_llm import main as run_alignment_llm

        if not str(anchor_token or "").strip():
            raise ValueError("alignment_config requires --anchor-token")
        run_alignment_llm(["--experiment", paths.experiment, "--anchor-token", anchor_token, "--anchor-segment", str(anchor_segment)])
        llm_manifest_path = paths.out_dir / "alignment_llm_manifest.json"
        llm_manifest: dict[str, object] = {}
        if llm_manifest_path.exists():
            loaded = json.loads(llm_manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                llm_manifest = loaded
        payload = {
            "stage": stage,
            "experiment": paths.experiment,
            "strategy": "llm_alignment_with_review_loop",
            "thresholds": {"minConfidence": min_confidence, "maxEditCost": max_edit_cost},
            "inputs": {"sarvam": str(paths.out_dir / "0.json"), "pdf_tokens": str(paths.out_dir / "pdf_tokens.json")},
            "outputs": {
                "enriched_pdf_tokens": str(paths.out_dir / "pdf_tokens_enriched_with_timestamps.json"),
                "review_csv": str(paths.out_dir / "pdf_tokens_segment_mapping_review.csv"),
            },
            "summary": llm_manifest.get("summary", {}),
        }
        out = paths.out_dir / "alignment_config.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    if stage == "quality_gates":
        payload = {
            "stage": stage,
            "experiment": paths.experiment,
            "required_artifacts": [
                str(paths.input_dir / "audio.mp3"),
                str(paths.out_dir / "0.json"),
                str(paths.out_dir / "pdf_tokens.json"),
                str(paths.out_dir / "alignment_config.json"),
            ],
            "gates": {
                "minConfidence": min_confidence,
                "maxEditCost": max_edit_cost,
                "routeOnFailure": "human_review",
            },
        }
        out = paths.out_dir / "quality_gates.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    raise ValueError(f"unknown stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run-build-stage", description="Run one build stage with guardrails.")
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--experiment", required=True)
    p.add_argument("--youtube-url", default=None)
    p.add_argument("--pdf", default=None)
    p.add_argument("--pages", default=None)
    p.add_argument("--ai-model", default="saaras:v3")
    p.add_argument("--language-code", default="sa-IN")
    p.add_argument("--trim-top-ratio", type=float, default=None)
    p.add_argument("--trim-bottom-ratio", type=float, default=None)
    p.add_argument("--min-confidence", type=float, default=0.78)
    p.add_argument("--max-edit-cost", type=float, default=0.32)
    p.add_argument("--anchor-token", default=None)
    p.add_argument("--anchor-segment", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out = run_stage(
        args.stage,
        experiment=args.experiment,
        youtube_url=args.youtube_url,
        pdf_path=args.pdf,
        pages=args.pages,
        ai_model=args.ai_model,
        language_code=args.language_code,
        top_trim_ratio=args.trim_top_ratio,
        bottom_trim_ratio=args.trim_bottom_ratio,
        min_confidence=args.min_confidence,
        max_edit_cost=args.max_edit_cost,
        anchor_token=args.anchor_token,
        anchor_segment=args.anchor_segment,
    )
    print(f"stage={args.stage}")
    print(f"output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
