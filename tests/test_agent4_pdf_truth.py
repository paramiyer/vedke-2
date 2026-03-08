from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.agent4_pdf_truth import build_pdf_truth_tokens_cleaned_payload, build_pdf_truth_tokens_payload


@dataclass(frozen=True)
class _Box:
    page: int
    line: int
    text: str
    x: float
    y: float
    w: float
    h: float


def test_build_pdf_truth_tokens_payload_word_and_punc_split() -> None:
    boxes = [
        _Box(page=2, line=1, text="देवाः।", x=10.0, y=20.0, w=30.0, h=8.0),
        _Box(page=2, line=1, text="ॐ", x=45.0, y=20.0, w=8.0, h=8.0),
    ]
    payload = build_pdf_truth_tokens_payload(
        pdf_path=Path("/tmp/fake.pdf"),
        pages=[2],
        word_boxes=boxes,
        trim_top_px=None,
        trim_bottom_px=None,
        trim_top_ratio=0.08,
        trim_bottom_ratio=0.06,
    )
    records = payload["pages"]["2"]
    assert len(records) == 3
    assert records[0]["kind"] == "word"
    assert records[1]["kind"] == "punc"
    assert records[1]["token"] == "।"
    assert records[1]["x"] == 40.0
    assert records[2]["kind"] == "word"


def test_payload_fields_contract() -> None:
    boxes = [_Box(page=1, line=3, text="॥", x=1.0, y=2.0, w=3.0, h=4.0)]
    payload = build_pdf_truth_tokens_payload(
        pdf_path=Path("/tmp/fake.pdf"),
        pages=[1],
        word_boxes=boxes,
        trim_top_px=0.0,
        trim_bottom_px=0.0,
        trim_top_ratio=None,
        trim_bottom_ratio=None,
    )
    assert payload["extraction"]["fields"] == ["tokenId", "token", "kind", "x", "y", "w", "h"]
    rec = payload["pages"]["1"][0]
    assert set(rec.keys()) == {"tokenId", "token", "kind", "x", "y", "w", "h"}
    assert rec["kind"] == "punc"


def test_cleaned_payload_keeps_only_devanagari_words_with_char_count() -> None:
    full_payload = {
        "sourcePdf": "/tmp/fake.pdf",
        "pages": {
            "2": [
                {"token": "नमः", "kind": "word"},
                {"token": "Ganesha", "kind": "word"},
                {"token": "।", "kind": "punc"},
                {"token": "शांति", "kind": "word"},
            ]
        },
    }
    cleaned = build_pdf_truth_tokens_cleaned_payload(full_payload)
    rows = cleaned["pages"]["2"]
    assert rows == [
        {"token": "नमः", "kind": "word", "char_count": 3},
        {"token": "शांति", "kind": "word", "char_count": 5},
    ]
