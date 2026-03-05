from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_PARAMS = {
    "pause_bonus_danda": 1.2,
    "pause_bonus_double_danda": 1.5,
    "alpha_svara": 0.15,
    "min_seg_ms": 600,
    "max_seg_ms": 12000,
    "boundary_silence_window_ms": 200,
    "duration_prior_strength": 0.8,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compute_weight(sentence: dict[str, Any], alpha_svara: float, bonuses: dict[str, float]) -> float:
    base = float(sentence["widthSumWords"])
    svara_ratio = float(sentence["svaraRatio"])
    svara_factor = 1.0 + alpha_svara * svara_ratio
    ends_with = sentence["endsWith"]

    punct_factor = 1.0
    if ends_with == "॥":
        punct_factor = bonuses["pause_bonus_double_danda"]
    elif ends_with == "।":
        punct_factor = bonuses["pause_bonus_danda"]

    return max(1e-6, base * svara_factor * punct_factor)


def _get_with_default(
    source: dict[str, Any],
    key: str,
    default: Any,
    missing_fields_counts: dict[str, int],
) -> Any:
    if key not in source or source.get(key) is None:
        missing_fields_counts[key] += 1
        return default
    return source[key]


def build_sentence_inputs(sentences: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(sentences, list) or not sentences:
        raise ValueError("Input sentences.json must be a non-empty JSON list")

    for idx, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            raise ValueError(f"Sentence at index {idx} must be a JSON object")
        if not sentence.get("sentenceId"):
            raise ValueError(f"Missing sentenceId at index {idx}")

    missing_fields_counts: dict[str, int] = {
        "page": 0,
        "line": 0,
        "sentence": 0,
        "text": 0,
        "wordCount": 0,
        "punctCount": 0,
        "svaraCount": 0,
        "svaraRatio": 0,
        "widthSumWords": 0,
        "widthSumAll": 0,
        "endsWith": 0,
        "endPunctTokenId": 0,
    }
    endswith_counts = {"danda": 0, "double_danda": 0, "none": 0}

    normalized: list[dict[str, Any]] = []
    bonuses = {
        "pause_bonus_danda": float(DEFAULT_PARAMS["pause_bonus_danda"]),
        "pause_bonus_double_danda": float(DEFAULT_PARAMS["pause_bonus_double_danda"]),
    }
    alpha_svara = float(DEFAULT_PARAMS["alpha_svara"])

    for idx, source in enumerate(sentences):
        page = int(_get_with_default(source, "page", 0, missing_fields_counts))
        line = int(_get_with_default(source, "line", 0, missing_fields_counts))
        sentence_num = int(_get_with_default(source, "sentence", 0, missing_fields_counts))
        text = str(_get_with_default(source, "text", "", missing_fields_counts))

        word_count = int(_get_with_default(source, "wordCount", 0, missing_fields_counts))
        punct_count = int(_get_with_default(source, "punctCount", 0, missing_fields_counts))
        svara_count = int(_get_with_default(source, "svaraCount", 0, missing_fields_counts))
        svara_ratio = float(_get_with_default(source, "svaraRatio", 0.0, missing_fields_counts))

        width_sum_words = float(_get_with_default(source, "widthSumWords", 1.0, missing_fields_counts))
        width_sum_all = float(_get_with_default(source, "widthSumAll", width_sum_words, missing_fields_counts))

        raw_ends_with = _get_with_default(source, "endsWith", None, missing_fields_counts)
        ends_with = raw_ends_with if raw_ends_with in {"।", "॥"} else None

        raw_end_punct_token_id = _get_with_default(source, "endPunctTokenId", None, missing_fields_counts)
        end_punct_token_id = str(raw_end_punct_token_id) if raw_end_punct_token_id is not None else None
        if ends_with is None:
            end_punct_token_id = None

        record: dict[str, Any] = {
            "idx": idx,
            "sentenceId": str(source["sentenceId"]),
            "page": page,
            "line": line,
            "sentence": sentence_num,
            "text": text,
            "wordCount": word_count,
            "punctCount": punct_count,
            "svaraCount": svara_count,
            "svaraRatio": svara_ratio,
            "widthSumWords": width_sum_words,
            "widthSumAll": width_sum_all,
            "endsWith": ends_with,
            "endPunctTokenId": end_punct_token_id,
        }

        if record["wordCount"] < 0:
            raise ValueError(f"wordCount must be >= 0 for {record['sentenceId']}")
        if record["widthSumWords"] < 0:
            raise ValueError(f"widthSumWords must be >= 0 for {record['sentenceId']}")

        weight = compute_weight(record, alpha_svara=alpha_svara, bonuses=bonuses)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid weight for {record['sentenceId']}: {weight}")
        record["weight"] = weight

        if ends_with == "।":
            endswith_counts["danda"] += 1
        elif ends_with == "॥":
            endswith_counts["double_danda"] += 1
        else:
            endswith_counts["none"] += 1

        normalized.append(record)

    k = len(normalized)
    expected_indices = list(range(k))
    observed_indices = [int(item["idx"]) for item in normalized]
    if observed_indices != expected_indices:
        raise ValueError("idx must be contiguous from 0..K-1")
    if k != (max(observed_indices) + 1 if observed_indices else 0):
        raise ValueError("K must equal max(idx)+1")

    weights = [float(item["weight"]) for item in normalized]
    weight_stats = {
        "sum": float(sum(weights)),
        "min": float(min(weights)),
        "median": float(median(weights)),
        "max": float(max(weights)),
    }

    out = {
        "target_segments": k,
        "sentences": normalized,
        "params": dict(DEFAULT_PARAMS),
        "qc": {
            "K": k,
            "missing_fields_counts": missing_fields_counts,
            "endswith_counts": endswith_counts,
            "weight_stats": weight_stats,
        },
    }
    if out["target_segments"] != k:
        raise ValueError("target_segments must equal K")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build-sentence-inputs",
        description="Build deterministic sentence_inputs.json from sentences.json",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Path to sentences.json")
    parser.add_argument("--out", dest="output_path", required=True, help="Path to sentence_inputs.json")
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = load_json(input_path)
    result = build_sentence_inputs(payload)

    out_path = Path(args.output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    k = int(result["qc"]["K"])
    stats = result["qc"]["weight_stats"]
    missing_width = int(result["qc"]["missing_fields_counts"]["widthSumWords"])
    print(
        f"K={k} weights: sum={stats['sum']:.6f} min={stats['min']:.6f} "
        f"median={stats['median']:.6f} max={stats['max']:.6f} missing_widthSumWords={missing_width}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
