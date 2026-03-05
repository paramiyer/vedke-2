from __future__ import annotations

from scripts.build_sentences import (
    _extract_tokens_from_payload,
    build_sentences,
    build_sentence_obj,
    parse_token_id,
    sort_tokens,
)


def test_tokenid_parse_and_sort() -> None:
    assert parse_token_id("P0002_L0004_S0001_T0005") == (2, 4, 1, 5)
    assert parse_token_id("bad-id") is None

    tokens = [
        {"tokenId": "P0001_L0001_S0001_T0003", "page": 1, "line": 1, "sentence": 1, "token": "।", "kind": "punct", "x": 2.0, "y": 1.0, "w": 0.5, "h": 1.0},
        {"tokenId": "BAD_ID", "page": 1, "line": 1, "sentence": 2, "token": "fallback", "kind": "word", "x": 1.0, "y": 2.0, "w": 2.0, "h": 1.0},
        {"tokenId": "P0001_L0001_S0001_T0001", "page": 1, "line": 1, "sentence": 1, "token": "अग्निम्", "kind": "word", "x": 0.0, "y": 1.0, "w": 1.0, "h": 1.0},
        {"tokenId": "P0001_L0001_S0001_T0002", "page": 1, "line": 1, "sentence": 1, "token": "ईळे", "kind": "word", "x": 1.0, "y": 1.0, "w": 1.0, "h": 1.0},
    ]

    sorted_tokens, parse_fails = sort_tokens(tokens)
    assert [t["tokenId"] for t in sorted_tokens] == [
        "P0001_L0001_S0001_T0001",
        "P0001_L0001_S0001_T0002",
        "P0001_L0001_S0001_T0003",
        "BAD_ID",
    ]
    assert parse_fails == ["BAD_ID"]


def test_grouping_and_counts() -> None:
    tokens = [
        {"tokenId": "P0001_L0001_S0001_T0001", "page": 1, "line": 1, "sentence": 1, "token": "अ॑ग्निम्", "kind": "word", "x": 0.0, "y": 0.0, "w": 5.0, "h": 2.0},
        {"tokenId": "P0001_L0001_S0001_T0002", "page": 1, "line": 1, "sentence": 1, "token": "ईळे", "kind": "word", "x": 6.0, "y": 0.0, "w": 4.0, "h": 2.0},
        {"tokenId": "P0001_L0001_S0001_T0003", "page": 1, "line": 1, "sentence": 1, "token": "पुरोहितम्", "kind": "word", "x": 11.0, "y": 0.0, "w": 6.0, "h": 2.0},
        {"tokenId": "P0001_L0001_S0001_T0004", "page": 1, "line": 1, "sentence": 1, "token": "।", "kind": "punct", "x": 18.0, "y": 0.0, "w": 1.0, "h": 2.0},
        {"tokenId": "P0001_L0001_S0002_T0005", "page": 1, "line": 1, "sentence": 2, "token": "यज्ञस्य", "kind": "word", "x": 0.0, "y": 4.0, "w": 5.0, "h": 2.0},
        {"tokenId": "P0001_L0001_S0002_T0006", "page": 1, "line": 1, "sentence": 2, "token": "देवम्॒", "kind": "word", "x": 6.0, "y": 4.0, "w": 4.0, "h": 2.0},
        {"tokenId": "P0001_L0001_S0002_T0007", "page": 1, "line": 1, "sentence": 2, "token": "॥", "kind": "punct", "x": 11.0, "y": 4.0, "w": 1.0, "h": 2.0},
        {"tokenId": "P0001_L0002_S0001_T0001", "page": 1, "line": 2, "sentence": 1, "token": "नमः", "kind": "word", "x": 1.0, "y": 10.0, "w": 3.0, "h": 2.0},
        {"tokenId": "P0001_L0002_S0001_T0002", "page": 1, "line": 2, "sentence": 1, "token": ",", "kind": "punct", "x": 5.0, "y": 10.0, "w": 1.0, "h": 2.0},
        {"tokenId": "P0001_L0002_S0001_T0003", "page": 1, "line": 2, "sentence": 1, "token": "शिवाय", "kind": "word", "x": 7.0, "y": 10.0, "w": 4.0, "h": 2.0},
    ]

    sentences, parse_fails = build_sentences(tokens)
    assert parse_fails == []
    assert len(sentences) == 3

    first = sentences[0]
    assert first["sentenceId"] == "P0001_L0001_S0001"
    assert first["endPunctTokenId"] == "P0001_L0001_S0001_T0004"
    assert first["svaraCount"] == 1
    assert first["bbox"] == {"x0": 0.0, "y0": 0.0, "x1": 19.0, "y1": 2.0}

    second = sentences[1]
    assert second["endPunctTokenId"] == "P0001_L0001_S0002_T0007"
    assert second["svaraCount"] == 1

    third = sentences[2]
    assert third["endsWith"] is None
    assert third["endPunctTokenId"] is None


def test_text_joining_punct_spacing() -> None:
    group = [
        {"tokenId": "P0001_L0001_S0001_T0001", "token": "अग्निम्", "kind": "word", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        {"tokenId": "P0001_L0001_S0001_T0002", "token": "।", "kind": "punct", "x": 1.1, "y": 0.0, "w": 0.2, "h": 1.0},
        {"tokenId": "P0001_L0001_S0001_T0003", "token": "॥", "kind": "punct", "x": 1.4, "y": 0.0, "w": 0.2, "h": 1.0},
        {"tokenId": "P0001_L0001_S0001_T0004", "token": "नमः", "kind": "word", "x": 2.0, "y": 0.0, "w": 1.0, "h": 1.0},
    ]
    sentence = build_sentence_obj(page=1, line=1, sentence=1, group_tokens=group)
    assert sentence["text"] == "अग्निम्।॥ नमः"
    assert " ।" not in sentence["text"]
    assert " ॥" not in sentence["text"]


def test_extract_tokens_from_pages_payload() -> None:
    payload = {
        "sourcePdf": "x.pdf",
        "pages": {
            "2": [
                {
                    "tokenId": "P0002_L0001_S0001_T0001",
                    "line": 1,
                    "sentence": 1,
                    "token": "ॐ",
                    "kind": "word",
                    "x": 1.0,
                    "y": 2.0,
                    "w": 3.0,
                    "h": 4.0,
                }
            ]
        },
    }
    tokens = _extract_tokens_from_payload(payload)
    assert len(tokens) == 1
    assert tokens[0]["page"] == 2
