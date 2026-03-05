from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any


TOKEN_ID_RE = re.compile(r"^P(\d+)_L(\d+)_S(\d+)_T(\d+)$")
DEFAULT_NO_SPACE_BEFORE_PUNCT = {"।", "॥", ",", ".", ":", ";", "?"}
DEFAULT_SVARA_CHARS = {"॑", "॒", "᳚", "ꣳ"}


def parse_token_id(token_id: str) -> tuple[int, int, int, int] | None:
    match = TOKEN_ID_RE.fullmatch(token_id)
    if not match:
        return None
    page, line, sentence, token_index = (int(part) for part in match.groups())
    return page, line, sentence, token_index


def sort_tokens(tokens: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    parse_fail_token_ids: list[str] = []

    def sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
        token_id = str(record.get("tokenId", ""))
        parsed = parse_token_id(token_id)
        if parsed is not None:
            page, line, sentence, token_index = parsed
            return (0, page, line, sentence, token_index)
        parse_fail_token_ids.append(token_id)
        page = int(record.get("page", 0))
        line = int(record.get("line", 0))
        sentence = int(record.get("sentence", 0))
        x = float(record.get("x", 0.0))
        y = float(record.get("y", 0.0))
        return (1, page, line, sentence, x, y, token_id)

    sorted_records = sorted(tokens, key=sort_key)
    return sorted_records, parse_fail_token_ids


def has_svara(
    token_str: str,
    svara_chars: set[str] | None = None,
    enable_vedic_mark_fallback: bool = True,
) -> bool:
    configured = svara_chars if svara_chars is not None else DEFAULT_SVARA_CHARS
    if any(ch in configured for ch in token_str):
        return True
    if not enable_vedic_mark_fallback:
        return False
    for ch in token_str:
        if not unicodedata.category(ch).startswith("M"):
            continue
        # U+1CD0..U+1CFF is the Vedic Extensions block.
        if 0x1CD0 <= ord(ch) <= 0x1CFF:
            return True
    return False


def build_sentence_obj(
    page: int,
    line: int,
    sentence: int,
    group_tokens: list[dict[str, Any]],
    no_space_before_punct: set[str] | None = None,
    svara_chars: set[str] | None = None,
) -> dict[str, Any]:
    punct_no_space = no_space_before_punct if no_space_before_punct is not None else DEFAULT_NO_SPACE_BEFORE_PUNCT
    token_ids: list[str] = []
    tokens: list[str] = []
    word_count = 0
    punct_count = 0
    svara_count = 0
    width_sum_words = 0.0
    width_sum_all = 0.0
    x0 = float("inf")
    y0 = float("inf")
    x1 = float("-inf")
    y1 = float("-inf")

    for item in group_tokens:
        token_id = str(item.get("tokenId", ""))
        token = str(item.get("token", ""))
        kind = str(item.get("kind", ""))
        x = float(item.get("x", 0.0))
        y = float(item.get("y", 0.0))
        w = float(item.get("w", 0.0))
        h = float(item.get("h", 0.0))

        token_ids.append(token_id)
        tokens.append(token)
        width_sum_all += w

        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x + w)
        y1 = max(y1, y + h)

        if kind == "word":
            word_count += 1
            width_sum_words += w
            if has_svara(token, svara_chars=svara_chars):
                svara_count += 1
        elif kind == "punct":
            punct_count += 1

    text_parts: list[str] = []
    for item in group_tokens:
        token = str(item.get("token", ""))
        kind = str(item.get("kind", ""))
        if not text_parts:
            text_parts.append(token)
            continue
        if kind == "punct" and token in punct_no_space:
            text_parts[-1] = text_parts[-1] + token
        else:
            text_parts.append(token)

    text = " ".join(text_parts)
    last_token = group_tokens[-1]
    last_kind = str(last_token.get("kind", ""))
    ends_with = str(last_token.get("token", "")) if last_kind == "punct" else None
    end_punct_token_id = str(last_token.get("tokenId", "")) if last_kind == "punct" else None

    return {
        "sentenceId": f"P{page:04d}_L{line:04d}_S{sentence:04d}",
        "page": page,
        "line": line,
        "sentence": sentence,
        "tokenIds": token_ids,
        "tokens": tokens,
        "text": text,
        "endsWith": ends_with,
        "endPunctTokenId": end_punct_token_id,
        "wordCount": word_count,
        "punctCount": punct_count,
        "svaraCount": svara_count,
        "svaraRatio": 0.0 if word_count == 0 else (svara_count / word_count),
        "widthSumWords": width_sum_words,
        "widthSumAll": width_sum_all,
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    }


def build_sentences(
    tokens: list[dict[str, Any]],
    no_space_before_punct: set[str] | None = None,
    svara_chars: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    sorted_tokens, parse_fail_token_ids = sort_tokens(tokens)
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)

    for record in sorted_tokens:
        token_id = str(record.get("tokenId", ""))
        parsed = parse_token_id(token_id)
        if parsed is not None:
            page, line, sentence, _token_index = parsed
        else:
            page = int(record.get("page", 0))
            line = int(record.get("line", 0))
            sentence = int(record.get("sentence", 0))
        groups[(page, line, sentence)].append(record)

    out: list[dict[str, Any]] = []
    for page, line, sentence in sorted(groups.keys()):
        group_tokens = groups[(page, line, sentence)]
        last_token_index: int | None = None
        for item in group_tokens:
            token_id = str(item.get("tokenId", ""))
            parsed = parse_token_id(token_id)
            if parsed is None:
                continue
            token_index = parsed[3]
            if last_token_index is not None and token_index <= last_token_index:
                raise ValueError(
                    f"Non-increasing token index in sentence P{page:04d}_L{line:04d}_S{sentence:04d}: {token_id}"
                )
            last_token_index = token_index

        out.append(
            build_sentence_obj(
                page=page,
                line=line,
                sentence=sentence,
                group_tokens=group_tokens,
                no_space_before_punct=no_space_before_punct,
                svara_chars=svara_chars,
            )
        )
    return out, parse_fail_token_ids


def _parse_set_csv(raw: str | None, default: set[str]) -> set[str]:
    if raw is None:
        return set(default)
    values = [item.strip() for item in raw.split(",")]
    return {item for item in values if item}


def _extract_tokens_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        pages = payload.get("pages")
        if not isinstance(pages, dict):
            raise ValueError("Input JSON object must contain a 'pages' object with token arrays")
        out: list[dict[str, Any]] = []
        for page_key in sorted(pages.keys(), key=lambda key: int(key) if str(key).isdigit() else str(key)):
            records = pages.get(page_key)
            if not isinstance(records, list):
                continue
            page_from_key: int | None = int(page_key) if str(page_key).isdigit() else None
            for item in records:
                if not isinstance(item, dict):
                    continue
                if "page" in item or page_from_key is None:
                    out.append(item)
                else:
                    enriched = dict(item)
                    enriched["page"] = page_from_key
                    out.append(enriched)
        return out
    raise ValueError("Input must be either a JSON array of token objects or an object with a 'pages' map")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-sentences",
        description="Build sentence-level JSON from token-level highlights_cleaned.json",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="Path to highlights_cleaned.json")
    parser.add_argument("--out-dir", default=None, help="Output directory for sentences.json and qc_report.json")
    parser.add_argument(
        "--no-space-before-punct",
        default=None,
        help="Comma-separated punctuation tokens that should not have a preceding space",
    )
    parser.add_argument(
        "--svara-chars",
        default=None,
        help="Comma-separated explicit svara chars (default: ॑,॒,᳚,ꣳ)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences_path = out_dir / "sentences.json"
    qc_path = out_dir / "qc_report.json"

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    tokens = _extract_tokens_from_payload(payload)

    token_ids = [str(item.get("tokenId", "")) for item in tokens]
    duplicate_ids = sorted([token_id for token_id, count in Counter(token_ids).items() if count > 1])

    no_space_before_punct = _parse_set_csv(args.no_space_before_punct, DEFAULT_NO_SPACE_BEFORE_PUNCT)
    svara_chars = _parse_set_csv(args.svara_chars, DEFAULT_SVARA_CHARS)

    qc_report: dict[str, Any] = {
        "sentences_missing_end_punct": [],
        "parse_fail_tokenIds": [],
        "duplicate_tokenIds": duplicate_ids[:50],
    }
    if duplicate_ids:
        qc_path.write_text(json.dumps(qc_report, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        raise ValueError(f"Duplicate tokenIds found: {len(duplicate_ids)}")

    sentences, parse_fail_token_ids = build_sentences(
        tokens=tokens,
        no_space_before_punct=no_space_before_punct,
        svara_chars=svara_chars,
    )
    qc_report["parse_fail_tokenIds"] = parse_fail_token_ids[:50]
    qc_report["sentences_missing_end_punct"] = [s["sentenceId"] for s in sentences if s["endsWith"] is None]

    if parse_fail_token_ids:
        print(f"WARNING: tokenId parse failures: {len(parse_fail_token_ids)}")

    n_tokens = len(tokens)
    n_sentences = len(sentences)
    n_pages = len({int(s["page"]) for s in sentences})
    print(f"n_tokens={n_tokens}, n_sentences={n_sentences}, n_pages={n_pages}")
    print("top_10_sentences_by_wordCount:")
    for sentence in sorted(sentences, key=lambda item: (-int(item["wordCount"]), str(item["sentenceId"])))[:10]:
        print(f"- {sentence['sentenceId']}: wordCount={sentence['wordCount']}")
    print(f"sentences_with_endsWith_null={len(qc_report['sentences_missing_end_punct'])}")

    sentences_path.write_text(json.dumps(sentences, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    qc_path.write_text(json.dumps(qc_report, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"sentences_json={sentences_path}")
    print(f"qc_report_json={qc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
