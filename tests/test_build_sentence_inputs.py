from __future__ import annotations

import pytest

from scripts.build_sentence_inputs import build_sentence_inputs


def test_build_sentence_inputs_weights_and_qc() -> None:
    sentences = [
        {
            "sentenceId": "P0001_L0001_S0001",
            "page": 1,
            "line": 1,
            "sentence": 1,
            "text": "अग्निम्।",
            "wordCount": 3,
            "punctCount": 1,
            "svaraCount": 1,
            "svaraRatio": 0.2,
            "widthSumWords": 10.0,
            "widthSumAll": 11.0,
            "endsWith": "।",
            "endPunctTokenId": "P0001_L0001_S0001_T0004",
        },
        {
            "sentenceId": "P0001_L0001_S0002",
            "page": 1,
            "line": 1,
            "sentence": 2,
            "text": "इन्द्रम्॥",
            "wordCount": 2,
            "punctCount": 1,
            "svaraCount": 1,
            "svaraRatio": 0.5,
            "widthSumWords": 8.0,
            "widthSumAll": 9.0,
            "endsWith": "॥",
            "endPunctTokenId": "P0001_L0001_S0002_T0007",
        },
        {
            "sentenceId": "P0001_L0002_S0001",
            "page": 1,
            "line": 2,
            "sentence": 1,
            "text": "नमः",
            "wordCount": 1,
            "punctCount": 0,
            "svaraCount": 0,
            "endsWith": None,
            "endPunctTokenId": None,
        },
    ]

    out = build_sentence_inputs(sentences)
    assert out["target_segments"] == 3
    assert [item["idx"] for item in out["sentences"]] == [0, 1, 2]

    assert out["sentences"][0]["weight"] == pytest.approx(12.36, abs=1e-6)
    assert out["sentences"][1]["weight"] == pytest.approx(12.9, abs=1e-6)
    assert out["sentences"][2]["weight"] == pytest.approx(1.0, abs=1e-6)

    assert out["sentences"][0]["endsWith"] == "।"
    assert out["sentences"][0]["endPunctTokenId"] == "P0001_L0001_S0001_T0004"
    assert out["sentences"][1]["endsWith"] == "॥"
    assert out["sentences"][2]["endsWith"] is None

    assert out["qc"]["K"] == 3
    assert out["qc"]["missing_fields_counts"]["widthSumWords"] == 1
    assert out["qc"]["missing_fields_counts"]["svaraRatio"] == 1
    assert out["qc"]["endswith_counts"] == {"danda": 1, "double_danda": 1, "none": 1}
    assert out["qc"]["weight_stats"]["sum"] == pytest.approx(26.26, abs=1e-6)
    assert out["qc"]["weight_stats"]["median"] == pytest.approx(12.36, abs=1e-6)


def test_requires_sentence_id() -> None:
    with pytest.raises(ValueError, match="Missing sentenceId"):
        build_sentence_inputs([{"page": 1}])
