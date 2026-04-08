from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import stage_orchestrator as so


@pytest.fixture()
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data"


def test_source_intake_stage_creates_structure(data_root: Path) -> None:
    out = so.run_stage(
        "source_intake",
        experiment="Ganesh Trial 01",
        youtube_url="https://youtu.be/x",
        pdf_path="data/ganapatiaccent.pdf",
        pages="2,3,4",
    )
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["experiment"] == "ganesh_trial_01"
    assert (data_root / "ganesh_trial_01" / "input" / "yt_link.txt").exists()
    assert (data_root / "ganesh_trial_01" / "input" / "source_pdf.txt").exists()


def test_asr_stage_guardrail_requires_audio(data_root: Path) -> None:
    so.run_stage(
        "source_intake",
        experiment="exp1",
        youtube_url="https://youtu.be/x",
        pdf_path="doc.pdf",
        pages="1",
    )
    with pytest.raises(RuntimeError, match="missing audio.mp3"):
        so.run_stage("asr_preparation", experiment="exp1")


def test_pdf_truth_stage_invokes_ocr_builder(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf_src = data_root.parent / "source.pdf"
    pdf_src.write_bytes(b"%PDF-1.4\n%fake\n")
    so.run_stage(
        "source_intake",
        experiment="exp2",
        youtube_url="https://youtu.be/x",
        pdf_path=str(pdf_src),
        pages="2,3",
    )
    (data_root / "exp2" / "input" / "audio.mp3").write_bytes(b"x")

    called: dict[str, str | float | None] = {}

    def _fake_ocr_runner(
        *,
        pdf_path: Path,
        pages: str,
        top_trim_ratio: float | None,
        bottom_trim_ratio: float | None,
        out_path: Path,
        cleaned_out_path: Path,
    ) -> None:
        called["pdf_path"] = str(pdf_path)
        called["pages"] = pages
        called["top_trim_ratio"] = top_trim_ratio
        called["bottom_trim_ratio"] = bottom_trim_ratio
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('{"pages":{}}', encoding="utf-8")
        cleaned_out_path.write_text('{"pages":{}}', encoding="utf-8")

    monkeypatch.setattr(so, "run_ocr_pdf_truth", _fake_ocr_runner)
    out = so.run_stage("pdf_truth", experiment="exp2", top_trim_ratio=0.08, bottom_trim_ratio=0.06)
    assert out.name == "pdf_tokens.json"
    assert out.exists()
    assert called["pages"] == "2,3"
    assert str(called["pdf_path"]).endswith("data/exp2/input/source.pdf")


def test_alignment_config_stage_writes_file(data_root: Path) -> None:
    exp = data_root / "exp3"
    (exp / "input").mkdir(parents=True, exist_ok=True)
    (exp / "out").mkdir(parents=True, exist_ok=True)
    (exp / "input" / "yt_link.txt").write_text("https://youtu.be/x\n", encoding="utf-8")
    (exp / "input" / "audio.mp3").write_bytes(b"x")
    (exp / "out" / "0.json").write_text("{}", encoding="utf-8")
    out = so.run_stage("alignment_config", experiment="exp3", min_confidence=0.8, max_edit_cost=0.25)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["thresholds"] == {"minConfidence": 0.8, "maxEditCost": 0.25}


def test_quality_gates_stage_requires_pdf_tokens(data_root: Path) -> None:
    exp = data_root / "exp4"
    (exp / "input").mkdir(parents=True, exist_ok=True)
    (exp / "out").mkdir(parents=True, exist_ok=True)
    (exp / "input" / "yt_link.txt").write_text("https://youtu.be/x\n", encoding="utf-8")
    (exp / "input" / "audio.mp3").write_bytes(b"x")
    (exp / "out" / "0.json").write_text("{}", encoding="utf-8")
    (exp / "out" / "alignment_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing pdf_tokens.json"):
        so.run_stage("quality_gates", experiment="exp4")


# ---------------------------------------------------------------------------
# Coordinate-space / pageImageSizes tests
# ---------------------------------------------------------------------------

def _make_ocr_payload(pages: list[int], img_w: int = 2550, img_h: int = 3300) -> dict[str, Any]:
    """Build a minimal OCR payload matching the shape produced by build_extraction()."""
    results: dict[str, Any] = {}
    for page in pages:
        results[str(page)] = {
            "page": page,
            "image_size_px": {"w": img_w, "h": img_h},
            "trim_px": {"top": int(img_h * 0.08), "bottom": int(img_h * 0.06)},
            "token_count": 2,
            "tokens": [
                {
                    "tokenId": f"P{page:04d}_T0001",
                    "token": "नमस्ते",
                    "kind": "word",
                    "char_count": 6,
                    "x": 300,
                    "y": 400,
                    "w": 200,
                    "h": 80,
                    "conf": 95.0,
                    "underlined": False,
                },
                {
                    "tokenId": f"P{page:04d}_T0002",
                    "token": "।",
                    "kind": "punc",
                    "char_count": 1,
                    "x": 510,
                    "y": 400,
                    "w": 20,
                    "h": 80,
                    "conf": 90.0,
                    "underlined": False,
                },
            ],
        }
    return {
        "source_pdf": "/path/to/test.pdf",
        "method": "tesseract_tsv",
        "language": "san+script/Devanagari",
        "pages": pages,
        "trim": {"top_ratio": 0.08, "bottom_ratio": 0.06},
        "postprocess": {"skip_brackets": True, "skip_underlined": True},
        "results": results,
    }


def test_ocr_payload_to_pdf_tokens_includes_page_image_sizes() -> None:
    """pageImageSizes must be emitted at the top level with correct dimensions."""
    ocr_payload = _make_ocr_payload(pages=[8, 9], img_w=2550, img_h=3300)
    result = so._ocr_payload_to_pdf_tokens_payload(ocr_payload)

    assert "pageImageSizes" in result, "pageImageSizes missing from pdf_tokens payload"
    sizes = result["pageImageSizes"]
    assert sizes["8"] == {"w": 2550, "h": 3300}
    assert sizes["9"] == {"w": 2550, "h": 3300}


def test_ocr_payload_page_image_sizes_match_token_coordinate_space() -> None:
    """Token bboxes must be strictly inside the declared pageImageSizes bounds.

    This verifies that coordinates are consistent: a highlight placed at any
    token (x, y, w, h) must not overflow the declared page dimensions.
    """
    ocr_payload = _make_ocr_payload(pages=[8], img_w=2550, img_h=3300)
    result = so._ocr_payload_to_pdf_tokens_payload(ocr_payload)

    page_w = result["pageImageSizes"]["8"]["w"]
    page_h = result["pageImageSizes"]["8"]["h"]
    for rec in result["pages"]["8"]:
        assert rec["x"] >= 0
        assert rec["y"] >= 0
        assert rec["x"] + rec["w"] <= page_w, (
            f"Token {rec['tokenId']} right edge {rec['x'] + rec['w']} exceeds page width {page_w}"
        )
        assert rec["y"] + rec["h"] <= page_h, (
            f"Token {rec['tokenId']} bottom edge {rec['y'] + rec['h']} exceeds page height {page_h}"
        )


def test_ocr_payload_page_image_sizes_absent_when_missing_from_results() -> None:
    """If image_size_px is absent from OCR results, pageImageSizes omits that page."""
    ocr_payload = _make_ocr_payload(pages=[8])
    # Strip image_size_px from page 8
    del ocr_payload["results"]["8"]["image_size_px"]
    result = so._ocr_payload_to_pdf_tokens_payload(ocr_payload)
    assert "8" not in result.get("pageImageSizes", {}), (
        "pageImageSizes should not contain page 8 when image_size_px is absent"
    )


def test_pdf_truth_stage_generates_karaoke_pages_dir(
    data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pdf_truth stage must pass a page_images_dir to the OCR runner and create karaoke_pages/."""
    pdf_src = data_root.parent / "source2.pdf"
    pdf_src.write_bytes(b"%PDF-1.4\n%fake\n")
    so.run_stage(
        "source_intake",
        experiment="coord_test",
        youtube_url="https://youtu.be/x",
        pdf_path=str(pdf_src),
        pages="5",
    )
    (data_root / "coord_test" / "input" / "audio.mp3").write_bytes(b"x")

    captured: dict[str, Any] = {}

    def _fake_ocr_runner(
        *,
        pdf_path: Path,
        pages: str,
        top_trim_ratio: float | None,
        bottom_trim_ratio: float | None,
        out_path: Path,
        cleaned_out_path: Path,
    ) -> None:
        # Simulate the OCR runner creating karaoke_pages as it would in production.
        karaoke_pages_dir = out_path.parent / "karaoke_pages"
        karaoke_pages_dir.mkdir(parents=True, exist_ok=True)
        (karaoke_pages_dir / "page_0005.png").write_bytes(b"FAKEPNG")
        captured["karaoke_pages_dir"] = karaoke_pages_dir
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"pageImageSizes": {"5": {"w": 2550, "h": 3300}}, "pages": {}}),
            encoding="utf-8",
        )
        cleaned_out_path.write_text('{"pages":{}}', encoding="utf-8")

    monkeypatch.setattr(so, "run_ocr_pdf_truth", _fake_ocr_runner)
    so.run_stage("pdf_truth", experiment="coord_test")

    karaoke_dir = data_root / "coord_test" / "out" / "karaoke_pages"
    assert karaoke_dir.exists(), "karaoke_pages directory should be created by Stage 3"
    assert (karaoke_dir / "page_0005.png").exists(), "Stage 3 should produce page_0005.png"
