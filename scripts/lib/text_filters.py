from __future__ import annotations

import re
from typing import Any

ASCII_RE = re.compile(r"[A-Za-z0-9@:/._-]")
EMAIL_RE = re.compile(r"@[^\s]*\.")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _is_numeric_marker(text: str) -> bool:
    stripped = text.strip()
    for ch in ["।", "॥", " ", "\t", "\n", ",", ".", ":", ";", "-"]:
        stripped = stripped.replace(ch, "")
    return bool(stripped) and stripped.isdigit()


def filter_sentences(sentences: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    filtered_sentences: list[dict[str, Any]] = []
    filtered_out: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []

    for original_idx, sentence in enumerate(sentences):
        text = str(sentence.get("text", ""))
        n_chars = max(1, len(text))
        ascii_ratio = sum(1 for ch in text if ASCII_RE.match(ch) is not None) / n_chars
        skip = False
        reasons: list[str] = []

        if ascii_ratio >= 0.25:
            skip = True
            reasons.append("ASCII_RATIO_GE_0_25")

        if EMAIL_RE.search(text) or URL_RE.search(text):
            skip = True
            reasons.append("EMAIL_OR_URL")

        devanagari_present = any("\u0900" <= ch <= "\u097F" for ch in text)
        has_alpha = any(ch.isalpha() for ch in text)
        if (not devanagari_present) and has_alpha:
            skip = True
            reasons.append("NO_DEVANAGARI_WITH_ALPHA")

        if _is_numeric_marker(text):
            skip = True
            reasons.append("NUMERIC_ONLY_MARKER")

        if skip:
            filtered_out.append(
                {
                    "original_idx": original_idx,
                    "sentenceId": sentence.get("sentenceId"),
                    "text": text,
                    "reasons": reasons,
                    "ascii_ratio": round(ascii_ratio, 6),
                }
            )
            continue

        filtered_idx = len(filtered_sentences)
        filtered_sentences.append(sentence)
        mapping.append(
            {
                "filtered_idx": filtered_idx,
                "original_idx": original_idx,
                "sentenceId": sentence.get("sentenceId"),
            }
        )

    return filtered_sentences, filtered_out, mapping
