from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts.run_alignment_llm import (
    _apply_output_guardrails_in_place,
    _extract_enriched_payload,
    _extract_first_segment_start,
    _left_out_rate_after_anchor,
    _load_dotenv,
    _slugify,
    _validate_enriched_payload,
    _write_left_out_csv,
)


def _normalize_token_timestamp_fields_in_place(enriched: dict[str, Any]) -> int:
    pages = enriched.get("pages")
    if not isinstance(pages, dict):
        return 0
    changed = 0
    for recs in pages.values():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if "start_time_seconds" not in rec and "audio_start_s" in rec:
                rec["start_time_seconds"] = rec.get("audio_start_s")
                changed += 1
            if "end_time_seconds" not in rec and "audio_end_s" in rec:
                rec["end_time_seconds"] = rec.get("audio_end_s")
                changed += 1
            if "timing_segment_index" not in rec and "matched_segment_index" in rec:
                rec["timing_segment_index"] = rec.get("matched_segment_index")
                changed += 1
    return changed


def _parse_review_csv(path: Path) -> list[dict[str, str]]:
    required_fields = {
        "segment_index",
        "audio_text",
        "audio_start_s",
        "audio_end_s",
        "coarse_start_token_id",
        "coarse_end_token_id",
        "kept_token_ids",
        "kept_tokens_text",
        "dropped_candidate_token_ids",
        "dropped_candidate_tokens_text",
        "previous_segment_end_token_id",
        "status",
        "coverage_ratio_estimate",
        "left_out_eligible_count_within_span",
        "notes",
    }
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = sorted(required_fields - header)
        if missing:
            raise RuntimeError(f"review csv missing required columns: {', '.join(missing)}")
        for row in reader:
            rows.append({k: str(v or "") for k, v in row.items()})
    if not rows:
        raise RuntimeError("review csv has no rows")
    return rows


def _count_stamped_tokens(enriched: dict[str, Any]) -> int:
    pages = enriched.get("pages")
    if not isinstance(pages, dict):
        return 0
    n = 0
    for recs in pages.values():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if "start_time_seconds" in rec and "end_time_seconds" in rec:
                n += 1
    return n


def _build_recommendations(
    *,
    fixes: dict[str, int],
    left_out_rate: float,
    stamped_tokens: int,
    review_row_count: int,
) -> list[str]:
    recs: list[str] = []
    if fixes.get("anchor_fixed", 0) > 0:
        recs.append(
            "Anchor token needed correction. In manual prompt output, explicitly stamp anchor in segment 0 at first audio start."
        )
    if fixes.get("monotonic_fixed", 0) > 0:
        recs.append(
            "Detected non-monotonic timestamps. Ask model to keep globally non-decreasing start/end times across OCR reading order."
        )
    if left_out_rate > 0.05:
        recs.append(
            f"Left-out rate is {left_out_rate:.2%}. Improve coarse span cleanup and compound-run matching to increase justified coverage."
        )
    if stamped_tokens < 50:
        recs.append("Very low stamped-token count. Verify files used in chat are the correct 0.json + pdf_tokens.json for this experiment.")
    if review_row_count < 5:
        recs.append("Review CSV has very few rows. Ensure model returns one row per audio segment.")
    return recs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="validate-alignment-manual")
    p.add_argument("--experiment", required=True)
    p.add_argument("--anchor-token", required=True)
    p.add_argument("--enriched-json", required=True)
    p.add_argument("--review-csv", required=True)
    p.add_argument("--left-out-csv", default=None)
    p.add_argument("--commit", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv(Path(".env").resolve())

    slug = _slugify(args.experiment)
    out_dir = (Path("data") / slug / "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_dir / "0.json"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio source file: {audio_path}")
    audio_payload = json.loads(audio_path.read_text(encoding="utf-8"))
    if not isinstance(audio_payload, dict):
        raise RuntimeError("Invalid 0.json payload")
    first_audio_start = _extract_first_segment_start(audio_payload)

    enriched_source = Path(args.enriched_json).resolve()
    review_source = Path(args.review_csv).resolve()
    left_out_source = Path(args.left_out_csv).resolve() if args.left_out_csv else None
    if not enriched_source.exists():
        raise FileNotFoundError(f"Missing uploaded enriched json: {enriched_source}")
    if not review_source.exists():
        raise FileNotFoundError(f"Missing uploaded review csv: {review_source}")

    uploaded_payload = json.loads(enriched_source.read_text(encoding="utf-8"))
    if not isinstance(uploaded_payload, dict):
        raise RuntimeError("Uploaded enriched json is not an object")
    enriched = _extract_enriched_payload(uploaded_payload)
    if not isinstance(enriched, dict):
        raise RuntimeError("Uploaded enriched json missing enriched payload/pages")
    normalized_fields = _normalize_token_timestamp_fields_in_place(enriched)

    fixes = _apply_output_guardrails_in_place(
        enriched,
        anchor_token_id=str(args.anchor_token).strip(),
        expected_first_start=first_audio_start,
    )
    ok, err = _validate_enriched_payload(
        enriched,
        anchor_token_id=str(args.anchor_token).strip(),
        expected_first_start=first_audio_start,
    )
    if not ok:
        raise RuntimeError(err or "guardrail validation failed")

    review_rows = _parse_review_csv(review_source)
    stamped_tokens = _count_stamped_tokens(enriched)
    left_out_rate = _left_out_rate_after_anchor(enriched, str(args.anchor_token).strip())
    recommendations = _build_recommendations(
        fixes=fixes,
        left_out_rate=left_out_rate,
        stamped_tokens=stamped_tokens,
        review_row_count=len(review_rows),
    )

    outputs: dict[str, str] = {}
    if args.commit:
        enriched_path = out_dir / "pdf_tokens_enriched_with_timestamps.json"
        review_path = out_dir / "pdf_tokens_segment_mapping_review.csv"
        left_out_path = out_dir / "pdf_tokens_left_out_non_punctuation.csv"
        manifest_path = out_dir / "alignment_llm_manifest.json"

        enriched_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        review_path.write_text(review_source.read_text(encoding="utf-8"), encoding="utf-8")
        if left_out_source and left_out_source.exists():
            left_out_path.write_text(left_out_source.read_text(encoding="utf-8"), encoding="utf-8")
            left_out_count = sum(1 for _ in left_out_path.read_text(encoding="utf-8").splitlines()[1:] if _.strip())
        else:
            left_out_count = _write_left_out_csv(left_out_path, enriched)

        manifest = {
            "experiment": slug,
            "mode": "manual_upload",
            "anchor_token": str(args.anchor_token).strip(),
            "outputs": {
                "enriched_json": str(enriched_path),
                "review_csv": str(review_path),
                "left_out_csv": str(left_out_path),
            },
            "summary": {
                "total_segments_processed": len(review_rows),
                "left_out_non_punctuation_count": left_out_count,
                "left_out_rate_after_first_anchor": left_out_rate,
                "stamped_tokens": stamped_tokens,
            },
            "guardrail_fixes": fixes,
            "normalized_timestamp_fields": normalized_fields,
            "recommendations": recommendations,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs = {
            "enriched_json": str(enriched_path),
            "review_csv": str(review_path),
            "left_out_csv": str(left_out_path),
            "manifest": str(manifest_path),
        }

    print(
        json.dumps(
            {
                "ok": True,
                "experiment": slug,
                "guardrail_fixes": fixes,
                "stamped_tokens": stamped_tokens,
                "left_out_rate_after_first_anchor": left_out_rate,
                "review_rows": len(review_rows),
                "recommendations": recommendations,
                "outputs": outputs,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
