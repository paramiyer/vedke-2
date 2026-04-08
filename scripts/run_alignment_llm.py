from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

ALIGNMENT_RUNNER_VERSION = "2026-03-20-anchor-retry-v2"


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


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


def _write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "segment_index",
        "audio_text",
        "kept_tokens_text",
        "audio_start_s",
        "audio_end_s",
        "coarse_start_token_id",
        "coarse_end_token_id",
        "kept_token_ids",
        "dropped_candidate_token_ids",
        "dropped_candidate_tokens_text",
        "previous_segment_end_token_id",
        "skipped_token_ids",
        "match_ratio",
        "manual_fallback",
        "status",
        "coverage_ratio_estimate",
        "left_out_eligible_count_within_span",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out: dict[str, Any] = {}
            for k in fields:
                val = row.get(k, "")
                if isinstance(val, list):
                    out[k] = "|".join(str(x) for x in val)
                else:
                    out[k] = val
            writer.writerow(out)


def _token_text_map(enriched: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pages = enriched.get("pages")
    if not isinstance(pages, dict):
        return mapping
    for recs in pages.values():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            token_id = str(rec.get("tokenId", "")).strip()
            if not token_id:
                continue
            mapping[token_id] = str(rec.get("token", "")).strip()
    return mapping


def _normalize_token_id_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if "|" in s:
            return [x.strip() for x in s.split("|") if x.strip()]
        if "," in s:
            return [x.strip() for x in s.split(",") if x.strip()]
        return [s]
    return []


def _augment_review_rows_with_token_text(
    rows: list[dict[str, Any]], enriched: dict[str, Any]
) -> list[dict[str, Any]]:
    token_map = _token_text_map(enriched)
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        kept_ids = _normalize_token_id_list(item.get("kept_token_ids", []))
        if kept_ids and not item.get("kept_tokens_text"):
            item["kept_tokens_text"] = [token_map.get(tid, "") for tid in kept_ids if token_map.get(tid, "")]
        dropped_ids = _normalize_token_id_list(item.get("dropped_candidate_token_ids", []))
        if dropped_ids and not item.get("dropped_candidate_tokens_text"):
            item["dropped_candidate_tokens_text"] = [token_map.get(tid, "") for tid in dropped_ids if token_map.get(tid, "")]
        out_rows.append(item)
    return out_rows


def _extract_anchor_segment_start(audio_payload: dict[str, Any], anchor_segment_index: int) -> float:
    ts = audio_payload.get("timestamps") if isinstance(audio_payload, dict) else None
    starts = ts.get("start_time_seconds", []) if isinstance(ts, dict) else []
    if not isinstance(starts, list) or not starts:
        raise RuntimeError("0.json missing timestamps.start_time_seconds")
    if anchor_segment_index < 0 or anchor_segment_index >= len(starts):
        raise RuntimeError(
            f"anchor_segment_index={anchor_segment_index} is out of range "
            f"(0.json has {len(starts)} segments)"
        )
    try:
        return float(starts[anchor_segment_index])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid timestamp at segment {anchor_segment_index} in 0.json") from exc


def _assert_anchor_guardrail(
    enriched: dict[str, Any], anchor_token_id: str, expected_first_start: float,
    anchor_segment_index: int = 0, tolerance_s: float = 1.5
) -> None:
    pages = enriched.get("pages")
    if not isinstance(pages, dict):
        raise RuntimeError("enriched payload missing pages object")
    anchor = None
    for recs in pages.values():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if isinstance(rec, dict) and str(rec.get("tokenId")) == anchor_token_id:
                anchor = rec
                break
        if anchor is not None:
            break
    if anchor is None:
        raise RuntimeError(f"Guardrail failed: anchor token {anchor_token_id} not found in enriched payload")
    if "start_time_seconds" not in anchor or "end_time_seconds" not in anchor:
        raise RuntimeError(f"Guardrail failed: anchor token {anchor_token_id} is unstamped")
    seg_idx = anchor.get("timing_segment_index")
    try:
        seg_idx_int = int(seg_idx)
    except (TypeError, ValueError):
        raise RuntimeError("Guardrail failed: anchor token has invalid timing_segment_index") from None
    if seg_idx_int != anchor_segment_index:
        raise RuntimeError(
            f"Guardrail failed: anchor token {anchor_token_id} mapped to segment {seg_idx_int}, expected segment {anchor_segment_index}"
        )
    try:
        st = float(anchor.get("start_time_seconds"))
    except (TypeError, ValueError):
        raise RuntimeError("Guardrail failed: anchor token has invalid start_time_seconds") from None
    if abs(st - expected_first_start) > tolerance_s:
        raise RuntimeError(
            f"Guardrail failed: anchor start_time_seconds={st:.3f} is far from first audio start={expected_first_start:.3f}"
        )


def _assert_monotonicity_guardrail(enriched: dict[str, Any]) -> None:
    pages = enriched.get("pages")
    if not isinstance(pages, dict):
        raise RuntimeError("enriched payload missing pages object")

    def page_key(x: str) -> tuple[int, str]:
        try:
            return (0, f"{int(x):08d}")
        except ValueError:
            return (1, x)

    stamped: list[tuple[str, float, float]] = []
    for page in sorted(pages.keys(), key=page_key):
        recs = pages.get(page)
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if "start_time_seconds" not in rec or "end_time_seconds" not in rec:
                continue
            token_id = str(rec.get("tokenId", ""))
            try:
                st = float(rec.get("start_time_seconds"))
                en = float(rec.get("end_time_seconds"))
            except (TypeError, ValueError):
                raise RuntimeError(f"Guardrail failed: invalid timestamp values for token {token_id}") from None
            if en < st:
                raise RuntimeError(f"Guardrail failed: end_time < start_time for token {token_id}")
            stamped.append((token_id, st, en))

    prev_start = None
    prev_end = None
    prev_token = None
    for token_id, st, en in stamped:
        if prev_start is not None and st < prev_start:
            raise RuntimeError(
                f"Guardrail failed: non-monotonic start time at token {token_id} ({st}) after {prev_token} ({prev_start})"
            )
        if prev_end is not None and en < prev_end:
            raise RuntimeError(
                f"Guardrail failed: non-monotonic end time at token {token_id} ({en}) after {prev_token} ({prev_end})"
            )
        prev_start = st
        prev_end = en
        prev_token = token_id


def _build_prompt(
    ocr_json_text: str,
    audio_json_text: str,
    anchor_token_id: str,
    anchor_segment_index: int,
    feedback: str | None,
    *,
    include_embedded_inputs: bool,
) -> str:
    feedback_block = ""
    if feedback and feedback.strip():
        feedback_block = f"""
Additional reviewer feedback for this run:
{feedback.strip()}
"""
    base = f"""
You are running a MANUAL OVERRIDE for an existing prompt-based audio-to-OCR alignment pipeline.

This prompt was generated by an application so that a user can manually send it to GPT together with the required files, obtain the outputs, and upload those outputs back into the app.

You must operate within this existing workflow exactly as defined here.

IMPORTANT:
- The user is manually attaching the required files to this GPT run.
- You must read the attached files directly.
- You must produce the required output files exactly as specified.
- Do NOT redesign the interface.
- Do NOT switch to another approach.
- Do NOT ask for a different workflow.
- Do NOT ask again for inputs the app already provides.
- Do NOT return a conceptual explanation instead of the required output artifacts.

Your task is to perform the alignment robustly within the current prompt-driven workflow and generate the specified output files for re-upload into the app.

==================================================
CONTEXT ALREADY PROVIDED BY THE APP / USER
==================================================

The application already prepares this manual override correctly.

The user will attach:
- `0.json`
- `pdf_tokens.json`

The application also supplies:
- `first_anchor_token_id`

Use these directly.
Do NOT ask for them again.
Do NOT rename them.
Do NOT redesign how they are passed.

For this run:
- `first_anchor_token_id = "{anchor_token_id}"`
- `anchor_segment_index = {anchor_segment_index}`

==================================================
OBJECTIVE
==================================================

Read:
- `0.json`
- `pdf_tokens.json`

Use:
- `first_anchor_token_id`

Write:
- `pdf_tokens_enriched_with_timestamps.json`
- `pdf_tokens_segment_mapping_review.csv`
- `pdf_tokens_left_out_non_punctuation.csv`

The output must:
1. append timestamps to aligned OCR tokens
2. produce a segment-level mapping review CSV
3. produce a CSV of eligible OCR tokens that were left unstamped

==================================================
INPUT SCHEMA
==================================================

1. `0.json`
Use:
- `timestamps.words`
- `timestamps.start_time_seconds`
- `timestamps.end_time_seconds`

The i-th segment is:
- `audio_text = timestamps.words[i]`
- `audio_start_s = timestamps.start_time_seconds[i]`
- `audio_end_s = timestamps.end_time_seconds[i]`

2. `pdf_tokens.json`
OCR source of truth.
Grouped by page.
Tokens may contain:
- `tokenId`
- `token`
- `kind`
- `w`

OCR tokens are already in reading order across pages.

3. `first_anchor_token_id`
Already supplied by the application.
This OCR token is the first OCR-matchable token within audio segment {anchor_segment_index}.
Audio words spoken before this OCR token in segment {anchor_segment_index} (if any) have no matching OCR entry and must be skipped.
Audio segments 0 through {anchor_segment_index - 1} (if any) have no matching OCR tokens at all.

==================================================
NON-NEGOTIABLE RULES
==================================================

1. Use `pdf_tokens.json` as the OCR source of truth.
2. Preserve monotonic forward order only.
3. Never go backward in OCR.
4. Never assume the next segment starts at previous OCR end + 1.
5. For segment {anchor_segment_index}, use `first_anchor_token_id` as the OCR start anchor. Do not assign OCR tokens to any segment before segment {anchor_segment_index}.
6. For later segments, always search forward beyond the previous matched OCR end.
7. Only OCR tokens with `kind == "word"` are eligible for matching and timestamp distribution.
8. Exclude from matching:
   - punctuation
   - roman-only tokens
   - numeric-only tokens
   - obvious OCR noise
9. Do NOT stamp every OCR token inside a coarse span.
10. Only stamp OCR tokens that survive cleanup.
11. Some OCR text may not exist in audio; such OCR tokens must remain unstamped.
12. Coarse OCR boundaries are provisional only.
13. Final timestamping must happen only after token-level cleanup inside the span.
14. Do not hallucinate matches.
15. If a clean alignment cannot be justified, mark fallback instead of guessing.
16. You must actively try to keep the eligible left-out rate below 2% after the first anchor, but only by justified alignment, not by indiscriminate stamping.
17. Differences caused by compounded words, split words, sandhi-like joins, OCR merges, OCR splits, or spelling variation between OCR and audio must be treated as expected alignment phenomena, not as automatic mismatches.

==================================================
PRIMARY OPTIMIZATION TARGET
==================================================

The main target is:

- minimize the number of eligible OCR word tokens left unstamped after the first anchor
- keep the left-out rate below 2% whenever the evidence can support it
- treat a left-out rate materially above 2% as a failure unless the review clearly documents why conservative fallback was necessary

But this is NOT a license to stamp unrelated OCR text.

The correct tradeoff is:
- maximize justified OCR token coverage
- while preserving monotonicity
- while avoiding unsupported keeps
- while respecting compound-run logic
- while using phonetic and sound-based similarity when lexical form differs because of OCR distortion or compounding

This means:
- the leave-out rate is the primary optimization target
- the effective keep threshold is derived adaptively from the local score distribution and alignment coherence
- do NOT hard-code one global threshold as an absolute rule
- when multiple plausible interpretations exist, prefer the one that yields higher justified coverage without causing drift

==================================================
ANTI-DRIFT GUARDRAILS
==================================================

You MUST follow stages in this exact order for every segment:

Stage A: read audio segment
Stage B: find coarse OCR start
Stage C: find coarse OCR end
Stage D: clean up tokens inside the coarse span
Stage E: distribute timestamps
Stage F: record review state

Do not skip stages.
Do not reorder stages.
Do not perform timestamp distribution before cleanup is complete.

==================================================
NORMALIZATION RULES
==================================================

Create a comparison normalization used ONLY for matching/scoring.
Never overwrite original OCR token text.

For both audio text and OCR token text:
- lowercase if meaningful
- strip surrounding punctuation, danda-like separators, and hyphen artifacts for comparison
- remove zero-width or joiner artifacts
- normalize repeated spaces
- normalize mild OCR variants conservatively
- preserve original token boundaries in OCR
- use normalization only for scoring, not for output replacement

In addition to lexical normalization, create a phonetic comparison normalization used ONLY for similarity scoring when exact surface forms differ.

Phonetic comparison must:
- compare approximate sound patterns, not just exact character sequences
- help recover matches where one side is compounded and the other is split
- help recover matches where OCR dropped, merged, or distorted characters but the spoken sound sequence is still close
- help recover matches across mild orthographic variation that preserves approximate pronunciation
- never be used as a license to match unrelated text that merely sounds vaguely similar

Examples of what phonetic comparison is for:
- one OCR token may correspond to multiple spoken words because the OCR merged them
- multiple OCR tokens may correspond to one spoken word because the OCR split it
- OCR may distort a token while preserving much of its sound pattern
- the audio may contain an isolated form while OCR contains a compounded form, or vice versa

Important:
- A single OCR token may correspond to multiple spoken words due to OCR compounding
- A spoken word may correspond to multiple OCR tokens due to OCR fragmentation
- Therefore scoring must support:
  - 1:1
  - 1:many
  - many:1
  - short compound runs
- Matching must consider both lexical similarity and phonetic/sound-pattern similarity
- When lexical similarity is mediocre but phonetic continuity inside a compound run is strong, the token should remain a serious keep candidate

==================================================
COMPOUND / PHONETIC MATCHING STRATEGY
==================================================

This pipeline must explicitly try to rescue justified matches that would otherwise be left out because of compound differences.

When comparing OCR tokens to spoken units, evaluate all of the following:
- exact token-to-token similarity
- normalized substring overlap
- compound-to-subsequence similarity
- merged-token-to-multiword similarity
- split-token-to-singleword similarity
- approximate phonetic similarity
- neighbor-supported phonetic continuity across short runs

You must explicitly test whether an apparent mismatch is actually one of these cases:
- OCR compounded multiple spoken words into one token
- OCR split one spoken unit across multiple tokens
- OCR inserted or removed minor characters without destroying the sound pattern
- audio uses separated words while OCR preserves a joined form
- audio uses a joined form while OCR preserves separated tokens

When the evidence supports one of those cases:
- prefer keeping the relevant token or token run
- prefer a justified compound interpretation over leaving eligible tokens unstamped
- allow borderline lexical scores to be rescued by strong phonetic and neighbor coherence

Do not require exact spelling equality when:
- the sound sequence is close
- the neighboring tokens also line up
- the monotonic path remains clean
- the interpretation reduces left-outs without causing spillover

==================================================
ELIGIBILITY FILTER FOR OCR TOKENS
==================================================

Only tokens with `kind == "word"` can be considered.

Mark an OCR token as NOT ELIGIBLE for matching/timestamping if any of these hold:
- token is punctuation or separator
- token is purely numeric
- token is roman-only / ASCII-letter-only
- token is an obvious section number or verse number
- token is obvious header/title text not represented in the audio at that point
- token is obvious OCR garbage with no plausible match signal

Do NOT exclude a token merely because OCR confidence is low.
Low-confidence Indic-script tokens may still be valid if supported by neighboring alignment coherence, compound behavior, or phonetic similarity.

==================================================
SEGMENT PROCESSING
==================================================

--------------------------------------------------
Stage A: Read audio segment
--------------------------------------------------

For segment i:
- read `audio_text`, `audio_start_s`, `audio_end_s`
- tokenize `audio_text` into spoken units for alignment
- keep original spoken text and normalized spoken units
- compute `segment_duration = audio_end_s - audio_start_s`

Audio tokenization rules:
- split on whitespace
- also allow sub-token comparison for long compounds
- allow grouped spans of adjacent spoken units for compound comparison
- keep original token order
- do not reorder or deduplicate

--------------------------------------------------
Stage B: Find coarse OCR start
--------------------------------------------------

For segment {anchor_segment_index}:
- coarse start must begin at `first_anchor_token_id`
- do not use earlier OCR tokens before that anchor

For segments i > 0:
- search forward only
- start search strictly after the previous segment’s matched OCR end token
- do NOT assume exact adjacency
- allow a forward gap because some OCR tokens may be absent in audio or left unstamped

To find coarse start:
- inspect a forward window of eligible OCR tokens
- compare the beginning of the audio segment against candidate OCR runs
- prefer starts that maximize:
  - normalized lexical overlap
  - prefix consistency
  - local continuity with the previous segment
  - plausible compound-run behavior
  - plausible phonetic continuity
- penalize starts that require skipping obvious stronger matches nearer the front
- penalize starts that begin inside a clearly better candidate run

Important:
- coarse start is provisional
- it is only the beginning of the candidate span, not the final stamped set

--------------------------------------------------
Stage C: Find coarse OCR end
--------------------------------------------------

From the coarse start, scan forward only to find a provisional coarse end.

Goal:
- find an OCR span likely to contain most or all of the audio segment’s spoken content
- include plausible compound tokens when justified
- include plausible phonetically-supported mismatches when justified
- do not over-extend into unrelated downstream text

Evaluate candidate end points using:
- cumulative coverage of audio units
- suffix consistency with the segment ending
- local coherence of neighboring OCR tokens
- compound plausibility
- phonetic continuity
- spillover risk into the next segment
- whether the span contains a plausible dense alignment core

A coarse span may include extra OCR tokens.
That is allowed.
Those extras must be removed in Stage D before timestamping.

--------------------------------------------------
Stage D: Clean up tokens inside the coarse span
--------------------------------------------------

This is the most important stage.

Inside the coarse span, determine exactly which eligible OCR tokens are justified keeps.

You must NOT keep tokens merely because they lie between coarse start and coarse end.

For each eligible OCR token in the coarse span, compute a local keep score using:
- normalized lexical similarity to one or more nearby spoken units
- phonetic similarity to one or more nearby spoken units
- compound compatibility:
  - token may absorb neighboring spoken units if the OCR merged them
  - token may represent part of a spoken unit if OCR split it
- neighbor coherence:
  - does keeping this token improve monotonic alignment continuity?
  - do adjacent kept tokens form a plausible run?
  - do adjacent kept tokens preserve phonetic flow across the run?
- segment-level coverage contribution:
  - does keeping this token reduce unexplained audio content?
  - does keeping this token reduce eligible left-outs without introducing obvious mismatch?
- overreach penalty:
  - does keeping this token require stretching beyond plausible audio support?

Then derive the effective keep boundary adaptively from the local score distribution.

Rules:
- do NOT use a single fixed global keep threshold
- do NOT keep isolated low-support tokens unless they are rescued by strong neighboring coherence
- do keep low-to-medium lexical score tokens when they sit inside a strong compound run and improve justified coverage
- do keep low-to-medium lexical score tokens when phonetic similarity and neighbor coherence strongly support them
- do drop tokens that are inside the coarse span but have weak support and poor coherence
- if two neighboring OCR tokens are both plausible and together explain one spoken unit better than either alone, keep both
- if one OCR token explains multiple spoken units better than fragmented alternatives, keep that compound token
- if one candidate has lower exact lexical similarity but much better compound and phonetic coherence, prefer that candidate
- if a token is ambiguous and unsupported, drop it

Compound-run logic is mandatory.
Phonetic rescue logic is mandatory.

Adaptive keep interpretation:
- derive a local strong core of clearly supported tokens
- then expand to neighboring borderline tokens only when monotonic compound coherence and/or phonetic continuity support them
- explicitly try to rescue justified tokens that would otherwise become left-outs due to compounding differences
- stop expansion when evidence drops and spillover risk rises

If the span cannot be cleaned confidently:
- keep only the high-confidence core
- mark fallback / partial coverage in review
- do not guess the rest

Important:
- A left-out rate above 2% should trigger more aggressive but still justified rescue attempts inside Stage D before giving up
- rescue attempts must focus on compound interpretation, grouped spoken-unit comparison, grouped OCR-token comparison, and phonetic similarity
- do not accept a high left-out rate merely because exact token spelling did not match

--------------------------------------------------
Stage E: Distribute timestamps
--------------------------------------------------

Only after Stage D produces the final kept OCR tokens for the segment:
- assign timestamps to the kept OCR tokens only
- leave all dropped or ineligible tokens unstamped

Timestamp rules:
- timestamps must lie within `[audio_start_s, audio_end_s]`
- timestamps must be monotonic non-decreasing across kept OCR tokens
- segment-local timestamps should respect the order of kept OCR tokens
- distribution should be proportional to support when possible:
  - longer compound tokens or tokens covering more spoken material may receive proportionally larger spans
  - otherwise use normalized character/coverage weighting
- if a kept OCR token corresponds to multiple spoken units, give it the combined time share
- if multiple kept OCR tokens jointly cover one spoken unit, divide that time share proportionally and monotonically
- never assign timestamps to punctuation
- never assign timestamps to dropped eligible tokens

Each stamped OCR token MUST have EXACTLY these two timestamp fields written into the token object:
- `start_time_seconds`  ← the token's start time in seconds (float)
- `end_time_seconds`    ← the token's end time in seconds (float)

Do NOT use `audio_start_s` or `audio_end_s` as token-level field names. Those are segment-level variable names used in earlier stages. The token-level fields must always be `start_time_seconds` and `end_time_seconds`.

Do not invent unsupported fine precision.
But do maintain strict monotonic order.

--------------------------------------------------
Stage F: Record review state
--------------------------------------------------

For every audio segment, record one row in `pdf_tokens_segment_mapping_review.csv`.

Required columns:
- `segment_index`
- `audio_start_s`
- `audio_end_s`
- `audio_text`
- `coarse_start_token_id`
- `coarse_end_token_id`
- `kept_token_ids`
- `kept_tokens_text`
- `dropped_candidate_token_ids`
- `dropped_candidate_tokens_text`
- `previous_segment_end_token_id`
- `status`
- `coverage_ratio_estimate`
- `left_out_eligible_count_within_span`
- `notes`

`status` should be one of:
- `ok`
- `partial`
- `fallback`

Guidance:
- `ok` = strong coherent mapping
- `partial` = usable but some eligible content left unstamped
- `fallback` = only a conservative core could be justified

`notes` should mention reasons such as:
- `compound_run_kept`
- `compound_run_partial`
- `phonetic_rescue_kept`
- `phonetic_rescue_partial`
- `ocr_noise_dropped`
- `coarse_span_trimmed`
- `weak_suffix_support`
- `spillover_prevented`
- `segment_start_from_anchor`
- `search_started_after_previous_end`
- `left_out_rate_pressure`

Also record summary notes whenever:
- compound handling materially reduced left-outs
- phonetic matching materially reduced left-outs
- the segment required rescue logic to keep the overall left-out rate below 2%
- the segment could not be fully rescued without risking hallucinated alignment

==================================================
LEFT-OUT TOKEN CSV
==================================================

Create `pdf_tokens_left_out_non_punctuation.csv` containing OCR tokens that:
- are after the first anchor position
- have `kind == "word"`
- are eligible for matching
- received no timestamps
- are not punctuation, numeric, roman-only, or otherwise ineligible

Required columns:
- `tokenId`
- `page`
- `token`
- `normalized_token`
- `reason_left_out`
- `nearest_segment_index`
- `within_coarse_span_of_segment`
- `candidate_support_level`

`reason_left_out` should be one of:
- `no_justified_match`
- `dropped_in_cleanup`
- `outside_any_supported_span`
- `weak_compound_evidence`
- `weak_phonetic_evidence`
- `spillover_risk`
- `ambiguous_ocr_noise`

`candidate_support_level` should be one of:
- `low`
- `medium`
- `high`

Important:
- this CSV is for eligible unstamped OCR words only
- do not include punctuation
- do not include numeric markers
- do not include pre-anchor header/title material
- do not include intentionally ineligible tokens
- keep this file as small as possible through justified compound and phonetic rescue, with a target overall left-out rate below 2%

==================================================
OUTPUT JSON REQUIREMENTS
==================================================

Write `pdf_tokens_enriched_with_timestamps.json` preserving the original structure of `pdf_tokens.json` as much as possible.

For each OCR token:
- preserve original fields
- append timestamp fields only when justified
- optionally append review metadata fields such as:
  - `is_eligible`
  - `is_matched`
  - `matched_segment_index`
  - `match_status`
  - `match_notes`

Do not remove existing fields.

Matched eligible OCR word tokens should have timestamps.
Unmatched or ineligible tokens should not be given fabricated timestamps.

==================================================
SEARCH AND SCORING STRATEGY
==================================================

Use a forward-only sliding candidate strategy.

For each segment:
1. build normalized spoken units
2. build phonetic representations or sound-pattern representations of spoken units
3. open a forward OCR candidate window after the allowed search start
4. generate candidate coarse starts
5. for each start, evaluate plausible coarse ends
6. score candidate spans by:
   - spoken coverage
   - OCR run coherence
   - compound plausibility
   - phonetic similarity / sound-pattern continuity
   - suffix fit
   - low spillover
   - justified left-out reduction
7. choose the best provisional span
8. clean it aggressively but conservatively in Stage D
9. timestamp only the kept OCR tokens

Scoring principles:
- exact match is good but not required
- near-match with OCR distortion is acceptable
- coherent compound runs are often better than isolated exact matches
- phonetic similarity can rescue a weak lexical match when neighborhood coherence is strong
- a token with mediocre individual similarity can still be correct if it sits inside a high-coherence run
- an isolated token with decent lexical similarity may still be wrong if it breaks monotonic continuity or causes spillover
- among multiple monotonic candidates, prefer the candidate that produces the lowest justified left-out count

==================================================
WHAT TO AVOID
==================================================

Do NOT:
- ask again for `0.json`, `pdf_tokens.json`, or `first_anchor_token_id`
- redesign the interface
- switch to a different algorithm family
- stamp full coarse spans blindly
- force every audio word to have an OCR token
- force every OCR token to have an audio word
- go backward in OCR
- use previous end + 1 as a hard next start
- prefer aggressive stamping over justified stamping
- reject compound tokens just because they are long or contain hyphen artifacts
- reject phonetically close candidates merely because exact spelling differs
- keep title/header or verse-number tokens as spoken content
- hallucinate mappings when support is weak

==================================================
SUCCESS CRITERIA
==================================================

A successful run does all of the following:
- starts segment {anchor_segment_index} exactly at `first_anchor_token_id`
- stays monotonic and forward-only
- uses coarse spans only provisionally
- performs token-level cleanup before timestamping
- preserves compound-run matches where justified
- uses phonetic rescue where justified
- leaves unsupported OCR text unstamped
- minimizes eligible left-out OCR tokens after the anchor
- keeps the overall eligible left-out rate below 2% whenever support allows
- treats a materially higher left-out rate as a sign that compound and phonetic rescue was insufficiently explored
- records conservative fallback instead of guessing when evidence is weak
- writes all three required outputs

==================================================
FINAL EXECUTION INSTRUCTION
==================================================

Using the files attached to this GPT run:

- read `0.json`
- read `pdf_tokens.json`
- use `first_anchor_token_id`

Produce these output files:
- `pdf_tokens_enriched_with_timestamps.json`
- `pdf_tokens_segment_mapping_review.csv`
- `pdf_tokens_left_out_non_punctuation.csv`

Follow the exact stage order:
A -> B -> C -> D -> E -> F

Remember:
- cleanup happens before timestamp distribution
- keeping the eligible left-out rate below 2% is a hard optimization target
- you must explicitly try compound-rescue and phonetic-rescue strategies before accepting left-outs
- coverage is the optimization target, but only when justified
- when uncertain, prefer conservative partial/fallback over fabricated stamping
- return the required output artifacts for upload back into the application
{feedback_block}
"""
    if not include_embedded_inputs:
        return base
    return (
        base
        + f"""

Input file `pdf_tokens.json` content:
{ocr_json_text}

Input file `0.json` content:
{audio_json_text}

Return ONLY a JSON object with this exact top-level shape:
{{
  "summary": {{
    "total_segments_processed": <int>,
    "strong_matches": <int>,
    "approximate_matches": <int>,
    "fallback_matches": <int>,
    "left_out_non_punctuation_count": <int>,
    "left_out_rate_after_first_anchor": <float>,
    "refinement_pass_used": <bool>
  }},
  "enriched_pdf_tokens": <object>,
  "review_rows": [<objects with csv columns>]
}}
"""
    )


def _is_roman_only_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    return bool(re.fullmatch(r"[A-Za-z]+", t))


def _is_numeric_only_token(token: str) -> bool:
    t = token.strip()
    if not t:
        return False
    return bool(re.fullmatch(r"[0-9०-९]+", t))


def _eligible_non_punctuation_token(rec: dict[str, Any]) -> bool:
    if str(rec.get("kind", "")).strip() != "word":
        return False
    token = str(rec.get("token", "")).strip()
    if not token:
        return False
    if _is_roman_only_token(token):
        return False
    if _is_numeric_only_token(token):
        return False
    return True


def _iter_tokens_in_reading_order(enriched: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    pages = enriched.get("pages", {})
    if not isinstance(pages, dict):
        return []

    def page_key(x: str) -> tuple[int, str]:
        try:
            return (0, f"{int(x):08d}")
        except ValueError:
            return (1, x)

    out: list[tuple[str, dict[str, Any]]] = []
    for page in sorted(pages.keys(), key=page_key):
        recs = pages.get(page, [])
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if isinstance(rec, dict):
                out.append((str(page), rec))
    return out


def _write_left_out_csv(path: Path, enriched: dict[str, Any]) -> int:
    fields = ["page", "tokenId", "token", "kind", "w"]
    rows: list[dict[str, Any]] = []
    for page, rec in _iter_tokens_in_reading_order(enriched):
        if not _eligible_non_punctuation_token(rec):
            continue
        if "start_time_seconds" in rec or "end_time_seconds" in rec:
            continue
        rows.append(
            {
                "page": page,
                "tokenId": rec.get("tokenId", ""),
                "token": rec.get("token", ""),
                "kind": rec.get("kind", ""),
                "w": rec.get("w", ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _left_out_rate_after_anchor(enriched: dict[str, Any], anchor_token_id: str) -> float:
    ordered = _iter_tokens_in_reading_order(enriched)
    anchor_idx = -1
    for i, (_, rec) in enumerate(ordered):
        if str(rec.get("tokenId", "")) == anchor_token_id:
            anchor_idx = i
            break
    if anchor_idx < 0:
        return 0.0
    eligible = 0
    left_out = 0
    for _, rec in ordered[anchor_idx:]:
        if not _eligible_non_punctuation_token(rec):
            continue
        eligible += 1
        if "start_time_seconds" not in rec or "end_time_seconds" not in rec:
            left_out += 1
    if eligible == 0:
        return 0.0
    return round(left_out / eligible, 6)


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_list_of_dicts(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
    return out


def _extract_enriched_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    keys = [
        "enriched_pdf_tokens",
        "pdf_tokens_enriched_with_timestamps",
        "enriched_with_timestamps",
        "enriched",
    ]
    for key in keys:
        cand = _coerce_dict(payload.get(key))
        if cand is not None:
            return cand
    # Some models may return the enriched object at top-level directly.
    if isinstance(payload.get("pages"), dict):
        return payload
    return None


def _extract_review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "review_rows",
        "segment_mapping_review_rows",
        "pdf_tokens_segment_mapping_review",
    ]
    for key in keys:
        cand = _coerce_list_of_dicts(payload.get(key))
        if cand is not None:
            return cand
    return []


def _validate_enriched_payload(
    enriched: dict[str, Any], anchor_token_id: str, expected_first_start: float, anchor_segment_index: int = 0
) -> tuple[bool, str]:
    try:
        _assert_anchor_guardrail(enriched, anchor_token_id=anchor_token_id, expected_first_start=expected_first_start, anchor_segment_index=anchor_segment_index)
        _assert_monotonicity_guardrail(enriched)
        return True, ""
    except RuntimeError as exc:
        return False, str(exc)


def _repair_monotonic_timestamps_in_place(enriched: dict[str, Any]) -> int:
    changed = 0
    prev_start: float | None = None
    prev_end: float | None = None
    for _, rec in _iter_tokens_in_reading_order(enriched):
        if "start_time_seconds" not in rec or "end_time_seconds" not in rec:
            continue
        try:
            st = float(rec.get("start_time_seconds"))
            en = float(rec.get("end_time_seconds"))
        except (TypeError, ValueError):
            continue
        new_st = st
        new_en = en
        if prev_start is not None and new_st < prev_start:
            new_st = prev_start
        if prev_end is not None and new_st < prev_end:
            new_st = prev_end
        if new_en < new_st:
            new_en = new_st
        if prev_end is not None and new_en < prev_end:
            new_en = prev_end
        if new_st != st:
            rec["start_time_seconds"] = round(new_st, 3)
            changed += 1
        if new_en != en:
            rec["end_time_seconds"] = round(new_en, 3)
            changed += 1
        prev_start = float(rec.get("start_time_seconds"))
        prev_end = float(rec.get("end_time_seconds"))
    return changed


def _find_token_record(enriched: dict[str, Any], token_id: str) -> dict[str, Any] | None:
    pages = enriched.get("pages")
    if not isinstance(pages, dict):
        return None
    for recs in pages.values():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if isinstance(rec, dict) and str(rec.get("tokenId", "")) == token_id:
                return rec
    return None


def _normalize_token_timestamp_field_names_in_place(enriched: dict[str, Any]) -> int:
    """Rename audio_start_s/audio_end_s → start_time_seconds/end_time_seconds on token objects.

    Some model versions write the segment-level variable names (audio_start_s, audio_end_s)
    into token objects instead of the canonical pipeline field names. This normalizer fixes
    that in place so all downstream guardrails and the karaoke renderer see consistent names.
    Returns number of tokens corrected.
    """
    corrected = 0
    for _, rec in _iter_tokens_in_reading_order(enriched):
        if not isinstance(rec, dict):
            continue
        changed = False
        if "audio_start_s" in rec and "start_time_seconds" not in rec:
            rec["start_time_seconds"] = rec.pop("audio_start_s")
            changed = True
        elif "audio_start_s" in rec:
            rec.pop("audio_start_s")
            changed = True
        if "audio_end_s" in rec and "end_time_seconds" not in rec:
            rec["end_time_seconds"] = rec.pop("audio_end_s")
            changed = True
        elif "audio_end_s" in rec:
            rec.pop("audio_end_s")
            changed = True
        if "matched_segment_index" in rec and "timing_segment_index" not in rec:
            rec["timing_segment_index"] = rec["matched_segment_index"]
            changed = True
        if changed:
            corrected += 1
    return corrected


def _strip_ineligible_timestamps_in_place(enriched: dict[str, Any]) -> int:
    changed = 0
    for _, rec in _iter_tokens_in_reading_order(enriched):
        if not _eligible_non_punctuation_token(rec):
            had = False
            for key in (
                "start_time_seconds",
                "end_time_seconds",
                "timing_segment_index",
                "timing_match_ratio",
                "timing_manual_fallback",
            ):
                if key in rec:
                    had = True
                    rec.pop(key, None)
            if had:
                changed += 1
    return changed


def _ensure_anchor_stamped_in_place(enriched: dict[str, Any], anchor_token_id: str, expected_first_start: float, anchor_segment_index: int = 0) -> int:
    rec = _find_token_record(enriched, anchor_token_id)
    if rec is None:
        raise RuntimeError(f"Guardrail failed: anchor token {anchor_token_id} not found in enriched payload")
    changed = 0
    had_start = "start_time_seconds" in rec
    had_end = "end_time_seconds" in rec
    if not (had_start and had_end):
        rec["start_time_seconds"] = round(float(expected_first_start), 3)
        rec["end_time_seconds"] = round(float(expected_first_start) + 0.001, 3)
        rec["timing_segment_index"] = anchor_segment_index
        rec.setdefault("timing_match_ratio", 0.0)
        rec["timing_manual_fallback"] = True
        changed += 1
    else:
        try:
            st = float(rec.get("start_time_seconds"))
            en = float(rec.get("end_time_seconds"))
        except (TypeError, ValueError):
            st = float(expected_first_start)
            en = st + 0.001
            changed += 1
        if abs(st - float(expected_first_start)) > 1.5:
            dur = max(0.001, en - st)
            st = float(expected_first_start)
            en = st + dur
            changed += 1
        if en < st:
            en = st + 0.001
            changed += 1
        rec["start_time_seconds"] = round(st, 3)
        rec["end_time_seconds"] = round(en, 3)
    try:
        seg = int(rec.get("timing_segment_index"))
    except (TypeError, ValueError):
        seg = anchor_segment_index
    if seg != anchor_segment_index or "timing_segment_index" not in rec:
        rec["timing_segment_index"] = anchor_segment_index
        changed += 1
    rec.setdefault("timing_match_ratio", rec.get("timing_match_ratio", 0.0))
    rec.setdefault("timing_manual_fallback", rec.get("timing_manual_fallback", False))
    return changed


def _apply_output_guardrails_in_place(enriched: dict[str, Any], anchor_token_id: str, expected_first_start: float, anchor_segment_index: int = 0) -> dict[str, int]:
    pages = enriched.get("pages")
    if not isinstance(pages, dict) or not pages:
        raise RuntimeError("enriched payload missing pages object")

    field_names_normalized = _normalize_token_timestamp_field_names_in_place(enriched)
    ineligible_unstamped = _strip_ineligible_timestamps_in_place(enriched)
    anchor_fixed = _ensure_anchor_stamped_in_place(
        enriched, anchor_token_id=anchor_token_id, expected_first_start=expected_first_start, anchor_segment_index=anchor_segment_index
    )
    monotonic_fixed = _repair_monotonic_timestamps_in_place(enriched)
    ok, err = _validate_enriched_payload(enriched, anchor_token_id=anchor_token_id, expected_first_start=expected_first_start, anchor_segment_index=anchor_segment_index)
    if not ok:
        raise RuntimeError(err or "guardrail validation failed after repair")
    return {
        "field_names_normalized": field_names_normalized,
        "ineligible_unstamped": ineligible_unstamped,
        "anchor_fixed": anchor_fixed,
        "monotonic_fixed": monotonic_fixed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-alignment-llm",
        description="Run Stage 4 LLM alignment from out/0.json + out/pdf_tokens.json and write enriched JSON + review CSV.",
    )
    parser.add_argument("--experiment", required=True, help="Experiment name used as folder slug under data/")
    parser.add_argument("--model", default=None, help="OpenAI model name override")
    parser.add_argument("--anchor-token", required=True, help="First mapped OCR anchor tokenId (required)")
    parser.add_argument("--anchor-segment", type=int, default=0, help="Audio segment index (0-based) that the anchor token belongs to (default: 0)")
    parser.add_argument("--feedback", default=None, help="Optional reviewer feedback for iterative correction")
    parser.add_argument("--max-completion-tokens", type=int, default=120000, help="Max completion tokens for model output")
    parser.add_argument("--emit-prompt-only", action="store_true", help="Print resolved prompt and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_dotenv(Path(".env").resolve())
    model = args.model or os.getenv("OPENAI_ALIGNMENT_MODEL", "gpt-5.1").strip() or "gpt-5.1"

    slug = _slugify(args.experiment)
    out_dir = (Path("data") / slug / "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = out_dir / "0.json"
    ocr_path = out_dir / "pdf_tokens.json"
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio source file: {audio_path}")
    if not ocr_path.exists():
        raise FileNotFoundError(f"Missing OCR source file: {ocr_path}")

    audio_text = audio_path.read_text(encoding="utf-8")
    ocr_text = ocr_path.read_text(encoding="utf-8")
    audio_payload = json.loads(audio_text)
    if not isinstance(audio_payload, dict):
        raise RuntimeError("Invalid 0.json payload")
    anchor_segment_index = max(0, int(args.anchor_segment or 0))
    first_audio_start = _extract_anchor_segment_start(audio_payload, anchor_segment_index)
    anchor_token_id = str(args.anchor_token or "").strip()
    if not anchor_token_id:
        raise RuntimeError("Missing required --anchor-token for alignment run")
    prompt = _build_prompt(
        ocr_json_text=ocr_text,
        audio_json_text=audio_text,
        anchor_token_id=anchor_token_id,
        anchor_segment_index=anchor_segment_index,
        feedback=args.feedback,
        include_embedded_inputs=not args.emit_prompt_only,
    )
    if args.emit_prompt_only:
        print(prompt)
        return 0

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Set it in environment or .env")
    print("[alignment] start")
    print(f"[alignment] runner_version={ALIGNMENT_RUNNER_VERSION}")
    print(f"[alignment] experiment={slug}")
    print(f"[alignment] model={model}")
    print(f"[alignment] anchor_token={anchor_token_id}")
    print(f"[alignment] anchor_segment_index={anchor_segment_index}")
    print(f"[alignment] audio_path={audio_path} bytes={len(audio_text.encode('utf-8'))}")
    print(f"[alignment] ocr_path={ocr_path} bytes={len(ocr_text.encode('utf-8'))}")
    print("[alignment] prompt:begin")
    print(prompt)
    print("[alignment] prompt:end")

    client = OpenAI(api_key=api_key)
    payload: dict[str, Any] | None = None
    enriched: dict[str, Any] | None = None
    review_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    raw_response_path = out_dir / "alignment_llm_raw_response.json"
    last_error = ""
    for attempt in range(1, 4):
        messages = [{"role": "system", "content": "You are a strict JSON generator for OCR/audio alignment output."}]
        if attempt == 1:
            messages.append({"role": "user", "content": prompt})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        prompt
                        + "\n\nIMPORTANT RETRY REQUIREMENT:\n"
                        + "- Return full non-empty enriched OCR JSON with pages.\n"
                        + "- Do not return empty {} for enriched payload.\n"
                        + "- Do not return zeroed summary unless inputs are actually empty.\n"
                        + f"- Previous attempt failed validation: {last_error or 'unknown'}\n"
                        + f"- Anchor token `{anchor_token_id}` MUST be stamped in segment {anchor_segment_index}.\n"
                        + f"- Anchor start_time_seconds must be close to {first_audio_start:.3f}.\n"
                    ),
                }
            )

        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            max_completion_tokens=int(args.max_completion_tokens),
            messages=messages,
        )
        content = response.choices[0].message.content
        if not content:
            continue
        print("[alignment] response:begin")
        print(content)
        print("[alignment] response:end")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            continue
        raw_response_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        enriched_try = _extract_enriched_payload(parsed)
        if not (isinstance(enriched_try, dict) and isinstance(enriched_try.get("pages"), dict) and len(enriched_try["pages"]) > 0):
            last_error = "empty/invalid enriched payload"
            print(f"[alignment] retrying due to {last_error} (attempt={attempt})")
            continue

        try:
            fixes = _apply_output_guardrails_in_place(
                enriched_try, anchor_token_id=anchor_token_id, expected_first_start=first_audio_start, anchor_segment_index=anchor_segment_index
            )
        except RuntimeError as exc:
            last_error = str(exc)
            print(f"[alignment] retrying due to {last_error} (attempt={attempt})")
            continue
        print(
            "[alignment] output-guardrails "
            f"ineligible_unstamped={fixes['ineligible_unstamped']} "
            f"anchor_fixed={fixes['anchor_fixed']} "
            f"monotonic_fixed={fixes['monotonic_fixed']}"
        )

        payload = parsed
        enriched = enriched_try
        summary_try = parsed.get("summary", {})
        summary = summary_try if isinstance(summary_try, dict) else {}
        review_rows = _extract_review_rows(parsed)
        break

    if payload is None:
        raise RuntimeError(
            f"Model returned invalid enriched payload after retries (last_error={last_error or 'unknown'}). "
            f"Inspect debug file: {raw_response_path}"
        )

    # Backfill timing_segment_index from review_rows for any stamped token missing it.
    # The LLM sometimes writes start_time_seconds/end_time_seconds correctly but omits
    # timing_segment_index; review_rows is the authoritative source for segment membership.
    for rrow in (r for r in review_rows if isinstance(r, dict)):
        seg_idx = int(rrow.get("segment_index", 0))
        kept_str = str(rrow.get("kept_token_ids", ""))
        for tid in (s.strip() for s in kept_str.split(";") if s.strip()):
            rec = _find_token_record(enriched, tid)
            if rec is not None and "timing_segment_index" not in rec:
                rec["timing_segment_index"] = seg_idx

    # Backfill is_eligible on every token so the Review Editor can identify left-out tokens.
    # The LLM does not write this field; we compute it from the same eligibility function
    # used everywhere else in the pipeline.
    for _, rec in _iter_tokens_in_reading_order(enriched):
        if "is_eligible" not in rec:
            rec["is_eligible"] = _eligible_non_punctuation_token(rec)

    enriched_path = out_dir / "pdf_tokens_enriched_with_timestamps.json"
    review_csv_path = out_dir / "pdf_tokens_segment_mapping_review.csv"
    left_out_csv_path = out_dir / "pdf_tokens_left_out_non_punctuation.csv"
    enriched_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_rows_augmented = _augment_review_rows_with_token_text(
        [row for row in review_rows if isinstance(row, dict)],
        enriched,
    )
    _write_review_csv(review_csv_path, review_rows_augmented)
    left_out_count = _write_left_out_csv(left_out_csv_path, enriched)
    left_out_rate = _left_out_rate_after_anchor(enriched, anchor_token_id)
    summary_obj = dict(summary) if isinstance(summary, dict) else {}
    summary_obj.setdefault("left_out_non_punctuation_count", left_out_count)
    summary_obj.setdefault("left_out_rate_after_first_anchor", left_out_rate)
    summary_obj.setdefault("refinement_pass_used", False)

    manifest = {
        "experiment": slug,
        "model": model,
        "anchor_token": anchor_token_id,
        "anchor_segment_index": anchor_segment_index,
        "inputs": {"audio": str(audio_path), "ocr": str(ocr_path)},
        "outputs": {
            "enriched_json": str(enriched_path),
            "review_csv": str(review_csv_path),
            "left_out_csv": str(left_out_csv_path),
        },
        "summary": summary_obj,
        "feedback": args.feedback or "",
    }
    manifest_path = out_dir / "alignment_llm_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"experiment={slug}")
    print(f"model={model}")
    print(f"enriched_json={enriched_path}")
    print(f"review_csv={review_csv_path}")
    print(f"left_out_csv={left_out_csv_path}")
    print(f"total_segments_processed={summary_obj.get('total_segments_processed')}")
    print(f"strong_matches={summary_obj.get('strong_matches')}")
    print(f"approximate_matches={summary_obj.get('approximate_matches')}")
    print(f"fallback_matches={summary_obj.get('fallback_matches')}")
    print(f"left_out_non_punctuation_count={summary_obj.get('left_out_non_punctuation_count')}")
    print(f"left_out_rate_after_first_anchor={summary_obj.get('left_out_rate_after_first_anchor')}")
    print(f"refinement_pass_used={summary_obj.get('refinement_pass_used')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
