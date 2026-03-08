from __future__ import annotations

import json
from pathlib import Path

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


def test_pdf_truth_stage_invokes_agent4(data_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    called: dict[str, list[str]] = {}

    def _fake_agent4(argv: list[str] | None = None) -> int:
        assert argv is not None
        called["argv"] = argv
        out_path = data_root / "exp2" / "out" / "pdf_tokens.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text('{"pages":{}}', encoding="utf-8")
        return 0

    monkeypatch.setattr(so, "run_agent4_pdf_truth", _fake_agent4)
    out = so.run_stage("pdf_truth", experiment="exp2", top_trim_ratio=0.08, bottom_trim_ratio=0.06)
    assert out.name == "pdf_tokens.json"
    assert out.exists()
    assert "--experiment" in called["argv"]
    assert "--pdf" in called["argv"]


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
