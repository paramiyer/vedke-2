from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
import unicodedata


def _is_word(record: dict) -> bool:
    return str(record.get("kind", "")) == "word"


def _round_or_none(value: float | None, ndigits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, ndigits)


def _distance_between(records: list[dict], start_idx: int, end_idx: int, line: int) -> float:
    start = records[start_idx]
    end = records[end_idx]
    cursor = float(start.get("x", 0.0)) + float(start.get("w", 0.0))
    total = 0.0

    for mid in range(start_idx + 1, end_idx):
        rec = records[mid]
        if int(rec.get("line", -1)) != line:
            continue
        if str(rec.get("kind", "")) != "punct":
            continue
        punct_x = float(rec.get("x", 0.0))
        punct_w = float(rec.get("w", 0.0))
        total += max(0.0, punct_x - cursor)
        total += punct_w
        cursor = punct_x + punct_w

    end_x = float(end.get("x", 0.0))
    total += max(0.0, end_x - cursor)
    return total


def _has_svara_marks(token: str) -> bool:
    direct_marks = {"॒", "॑", "᳚"}
    if any(ch in direct_marks for ch in token):
        return True
    for ch in token:
        name = unicodedata.name(ch, "")
        if "VEDIC" in name:
            return True
    return False


def _group_word_width_medians(
    pages: dict[str, list[dict]],
) -> tuple[dict[tuple[int, int], float | None], dict[tuple[int, int], float | None]]:
    by_line: dict[tuple[int, int], list[float]] = defaultdict(list)
    by_sentence: dict[tuple[int, int], list[float]] = defaultdict(list)

    for page_key, records in pages.items():
        page = int(page_key)
        for record in records:
            if not isinstance(record, dict):
                continue
            if not _is_word(record):
                continue
            line = int(record.get("line", -1))
            sentence = int(record.get("sentence", -1))
            width = float(record.get("w", 0.0))
            if line >= 1:
                by_line[(page, line)].append(width)
            if sentence >= 1:
                by_sentence[(page, sentence)].append(width)

    line_medians: dict[tuple[int, int], float | None] = {}
    for key, values in by_line.items():
        line_medians[key] = float(median(values)) if values else None

    sentence_medians: dict[tuple[int, int], float | None] = {}
    for key, values in by_sentence.items():
        sentence_medians[key] = float(median(values)) if values else None

    return line_medians, sentence_medians


def _extract_features(payload: dict) -> dict:
    pages = payload.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("Invalid highlights payload: 'pages' must be an object")

    line_medians, sentence_medians = _group_word_width_medians(pages)

    out_pages: dict[str, list[dict]] = {}
    prev_pass_observed: list[float] = []
    next_pass_observed: list[float] = []

    # First pass per page for distances and contextual indices.
    page_runtime: dict[str, dict[str, object]] = {}
    for page_key, records in pages.items():
        if not isinstance(records, list):
            raise ValueError(f"Invalid page payload for page '{page_key}': expected array")

        line_to_indices: dict[int, list[int]] = defaultdict(list)
        sentence_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                continue
            line = int(rec.get("line", -1))
            sentence = int(rec.get("sentence", -1))
            if line >= 1:
                line_to_indices[line].append(idx)
            if sentence >= 1:
                sentence_to_indices[sentence].append(idx)

        prev_word_idx_map: dict[int, int | None] = {}
        next_word_idx_map: dict[int, int | None] = {}
        dist_prev_raw: dict[int, float | None] = {}
        dist_next_raw: dict[int, float | None] = {}
        prev_word_width_raw: dict[int, float | None] = {}
        next_word_width_raw: dict[int, float | None] = {}

        for line, indices in line_to_indices.items():
            word_indices = [i for i in indices if _is_word(records[i])]
            for i, idx in enumerate(indices):
                prev_candidates = [widx for widx in word_indices if widx < idx]
                next_candidates = [widx for widx in word_indices if widx > idx]
                prev_idx = prev_candidates[-1] if prev_candidates else None
                next_idx = next_candidates[0] if next_candidates else None
                prev_word_idx_map[idx] = prev_idx
                next_word_idx_map[idx] = next_idx

                prev_word_width_raw[idx] = float(records[prev_idx].get("w", 0.0)) if prev_idx is not None else None
                next_word_width_raw[idx] = float(records[next_idx].get("w", 0.0)) if next_idx is not None else None

                if prev_idx is None:
                    dist_prev_raw[idx] = None
                else:
                    val = _distance_between(records, prev_idx, idx, line=line)
                    dist_prev_raw[idx] = val
                    prev_pass_observed.append(val)

                if next_idx is None:
                    dist_next_raw[idx] = None
                else:
                    val = _distance_between(records, idx, next_idx, line=line)
                    dist_next_raw[idx] = val
                    next_pass_observed.append(val)

        page_runtime[page_key] = {
            "line_to_indices": line_to_indices,
            "sentence_to_indices": sentence_to_indices,
            "prev_word_idx_map": prev_word_idx_map,
            "next_word_idx_map": next_word_idx_map,
            "dist_prev_raw": dist_prev_raw,
            "dist_next_raw": dist_next_raw,
            "prev_word_width_raw": prev_word_width_raw,
            "next_word_width_raw": next_word_width_raw,
        }

    placeholder_fill_prev = sum(prev_pass_observed) / len(prev_pass_observed) if prev_pass_observed else 0.0
    placeholder_fill_next = sum(next_pass_observed) / len(next_pass_observed) if next_pass_observed else 0.0

    # Second pass: write all features.
    for page_key, records in pages.items():
        page = int(page_key)
        rt = page_runtime[page_key]
        line_to_indices: dict[int, list[int]] = rt["line_to_indices"]  # type: ignore[assignment]
        sentence_to_indices: dict[int, list[int]] = rt["sentence_to_indices"]  # type: ignore[assignment]
        dist_prev_raw: dict[int, float | None] = rt["dist_prev_raw"]  # type: ignore[assignment]
        dist_next_raw: dict[int, float | None] = rt["dist_next_raw"]  # type: ignore[assignment]
        prev_word_width_raw: dict[int, float | None] = rt["prev_word_width_raw"]  # type: ignore[assignment]
        next_word_width_raw: dict[int, float | None] = rt["next_word_width_raw"]  # type: ignore[assignment]

        # Per-line stats used by multiple features.
        line_token_count: dict[int, int] = {}
        line_word_count: dict[int, int] = {}
        line_punct_count: dict[int, int] = {}
        line_min_x: dict[int, float] = {}
        line_max_x: dict[int, float] = {}
        for line, indices in line_to_indices.items():
            tokens = [records[i] for i in indices]
            line_token_count[line] = len(tokens)
            line_word_count[line] = sum(1 for t in tokens if _is_word(t))
            line_punct_count[line] = sum(1 for t in tokens if str(t.get("kind", "")) == "punct")
            x_values = [float(t.get("x", 0.0)) for t in tokens]
            line_min_x[line] = min(x_values) if x_values else 0.0
            line_max_x[line] = max(x_values) if x_values else 0.0

        sentence_word_count: dict[int, int] = {}
        for sentence, indices in sentence_to_indices.items():
            tokens = [records[i] for i in indices]
            sentence_word_count[sentence] = sum(1 for t in tokens if _is_word(t))

        # Line rank on page.
        ordered_lines = sorted(line_to_indices.keys())
        line_rank: dict[int, int] = {line: idx + 1 for idx, line in enumerate(ordered_lines)}
        line_count = len(ordered_lines)

        page_out: list[dict] = []
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            line = int(record.get("line", -1))
            sentence = int(record.get("sentence", -1))
            line_median = line_medians.get((page, line))
            sentence_median = sentence_medians.get((page, sentence))

            raw_prev = dist_prev_raw.get(idx)
            raw_next = dist_next_raw.get(idx)
            final_prev = placeholder_fill_prev if raw_prev is None else raw_prev
            final_next = placeholder_fill_next if raw_next is None else raw_next

            token = str(record.get("token", ""))
            w = float(record.get("w", 0.0))
            h = float(record.get("h", 0.0))
            x = float(record.get("x", 0.0))

            line_min = line_min_x.get(line, x)
            line_max = line_max_x.get(line, x)
            span = line_max - line_min
            normalized_x = 0.0 if span <= 0 else (x - line_min) / span

            normalized_width_line = None
            if line_median is not None and line_median > 0:
                normalized_width_line = w / line_median

            line_indices = line_to_indices.get(line, [])
            sentence_indices = sentence_to_indices.get(sentence, [])
            is_line_start = bool(line_indices and line_indices[0] == idx)
            is_line_end = bool(line_indices and line_indices[-1] == idx)
            is_sentence_start = bool(sentence_indices and sentence_indices[0] == idx)
            is_sentence_end = bool(sentence_indices and sentence_indices[-1] == idx)

            density = 0.0
            token_count = line_token_count.get(line, 0)
            if token_count > 0:
                density = line_punct_count.get(line, 0) / token_count

            if line_count <= 1:
                y_rank_norm = 0.0
            else:
                y_rank_norm = (line_rank.get(line, 1) - 1) / (line_count - 1)

            enriched = dict(record)
            enriched["medianWordWidthLine"] = _round_or_none(line_median, 3)
            enriched["medianWordWidthSentence"] = _round_or_none(sentence_median, 3)
            enriched["distanceFromPrevWord"] = round(final_prev, 3)
            enriched["distanceToNextWord"] = round(final_next, 3)
            enriched["normalizedXInLine"] = round(normalized_x, 6)
            enriched["normalizedWidthInLine"] = _round_or_none(normalized_width_line, 6)
            enriched["isLineStart"] = is_line_start
            enriched["isLineEnd"] = is_line_end
            enriched["isSentenceStart"] = is_sentence_start
            enriched["isSentenceEnd"] = is_sentence_end
            enriched["prevWordWidth"] = _round_or_none(prev_word_width_raw.get(idx), 3)
            enriched["nextWordWidth"] = _round_or_none(next_word_width_raw.get(idx), 3)
            enriched["tokenLengthChars"] = len(token)
            enriched["hasSvaraMarks"] = _has_svara_marks(token)
            enriched["punctuationDensityLine"] = round(density, 6)
            enriched["lineWordCount"] = int(line_word_count.get(line, 0))
            enriched["sentenceWordCount"] = int(sentence_word_count.get(sentence, 0))
            enriched["yRankInPage"] = int(line_rank.get(line, 1))
            enriched["yRankInPageNormalized"] = round(y_rank_norm, 6)
            enriched["boxArea"] = round(w * h, 3)
            enriched["aspectRatio"] = None if h <= 0 else round(w / h, 6)
            page_out.append(enriched)
        out_pages[page_key] = page_out

    out_payload = dict(payload)
    out_payload["pages"] = out_pages
    out_payload["featureExtraction"] = {
        "features": [
            "medianWordWidthLine",
            "medianWordWidthSentence",
            "distanceFromPrevWord",
            "distanceToNextWord",
            "normalizedXInLine",
            "normalizedWidthInLine",
            "isLineStart",
            "isLineEnd",
            "isSentenceStart",
            "isSentenceEnd",
            "prevWordWidth",
            "nextWordWidth",
            "tokenLengthChars",
            "hasSvaraMarks",
            "punctuationDensityLine",
            "lineWordCount",
            "sentenceWordCount",
            "yRankInPage",
            "yRankInPageNormalized",
            "boxArea",
            "aspectRatio",
        ],
        "rules": {
            "line_and_sentence_median": "word tokens only (punctuation excluded)",
            "distanceFromPrevWord": "distance from nearest previous word token on same line; punctuation handled explicitly as gap+punctWidth+gap",
            "distanceToNextWord": "distance to nearest next word token on same line; punctuation handled explicitly as gap+punctWidth+gap",
            "distance_placeholders": "first pass uses placeholder for missing prev/next word, second pass fills with global averages from observed distances",
            "normalizedWidthInLine": "token width divided by line median word width",
        },
        "placeholderFillValuePrev": round(placeholder_fill_prev, 6),
        "placeholderFillValueNext": round(placeholder_fill_next, 6),
    }
    return out_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-highlight-features",
        description="Extract token-level features from highlights_cleaned.json into extracted_features.json.",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Path to input highlights_cleaned.json")
    parser.add_argument("--out", required=True, help="Path to output extracted_features.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input highlights file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    extracted = _extract_features(payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(extracted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Output: {out_path}")
    feature_meta = extracted.get("featureExtraction", {})
    print(f"Placeholder fill prev: {feature_meta.get('placeholderFillValuePrev')}")
    print(f"Placeholder fill next: {feature_meta.get('placeholderFillValueNext')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
