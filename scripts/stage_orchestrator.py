from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.agent4_pdf_truth import main as run_agent4_pdf_truth


STAGES = ("source_intake", "asr_preparation", "pdf_truth", "alignment_config", "quality_gates")


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


@dataclass(frozen=True)
class BuildPaths:
    experiment: str
    root: Path
    input_dir: Path
    out_dir: Path


def build_paths(experiment: str) -> BuildPaths:
    slug = _slugify(experiment)
    root = (Path("data") / slug).resolve()
    return BuildPaths(
        experiment=slug,
        root=root,
        input_dir=(root / "input"),
        out_dir=(root / "out"),
    )


def _resolve_user_pdf_path(raw_pdf: str) -> Path:
    raw = raw_pdf.strip()
    candidate = Path(raw).expanduser()
    attempts: list[Path] = []

    if candidate.is_absolute():
        attempts.append(candidate)
    else:
        attempts.append(Path.cwd() / candidate)
        attempts.append(Path.home() / candidate)
        attempts.append(candidate)
        attempts.append(Path.home() / "Downloads" / candidate.name)

    # If user passed "Downloads/foo.pdf", map to ~/Downloads/foo.pdf.
    parts = candidate.parts
    if parts and parts[0].lower() == "downloads":
        attempts.append(Path.home() / Path(*parts[1:]))

    for p in attempts:
        p = p.expanduser()
        if p.exists() and p.is_file():
            return p.resolve()

    attempted = ", ".join(str(p) for p in attempts)
    raise FileNotFoundError(f"PDF not found for '{raw_pdf}'. Tried: {attempted}")


def _copy_pdf_to_input(paths: BuildPaths, raw_pdf: str) -> Path:
    src = _resolve_user_pdf_path(raw_pdf)
    dst = paths.input_dir / "source.pdf"
    shutil.copy2(src, dst)
    (paths.input_dir / "source_pdf_local.txt").write_text(str(dst) + "\n", encoding="utf-8")
    return dst


def guardrails_for_stage(stage: str, paths: BuildPaths) -> list[str]:
    if stage not in STAGES:
        return [f"unknown stage: {stage}"]
    errors: list[str] = []
    if stage in {"asr_preparation", "pdf_truth", "alignment_config", "quality_gates"}:
        if not paths.input_dir.exists():
            errors.append(f"missing input dir: {paths.input_dir}")
    if stage in {"alignment_config", "quality_gates"}:
        if not paths.out_dir.exists():
            errors.append(f"missing out dir: {paths.out_dir}")
    if stage in {"asr_preparation", "pdf_truth", "alignment_config", "quality_gates"}:
        if not (paths.input_dir / "yt_link.txt").exists():
            errors.append("missing yt_link.txt in input dir")
    if stage in {"asr_preparation", "alignment_config", "quality_gates"}:
        if not (paths.input_dir / "audio.mp3").exists():
            errors.append("missing audio.mp3 in input dir")
    if stage in {"alignment_config", "quality_gates"}:
        if not (paths.out_dir / "0.json").exists():
            errors.append("missing sarvam output 0.json in out dir")
    if stage == "quality_gates":
        if not (paths.out_dir / "pdf_tokens.json").exists():
            errors.append("missing pdf_tokens.json in out dir")
    return errors


def run_stage(
    stage: str,
    *,
    experiment: str,
    youtube_url: str | None = None,
    pdf_path: str | None = None,
    pages: str | None = None,
    ai_model: str = "saaras:v3",
    language_code: str = "sa-IN",
    top_trim_ratio: float | None = None,
    bottom_trim_ratio: float | None = None,
    min_confidence: float = 0.78,
    max_edit_cost: float = 0.32,
) -> Path:
    paths = build_paths(experiment)
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    if stage == "source_intake":
        if not youtube_url:
            raise ValueError("source_intake requires youtube_url")
        if not pdf_path:
            raise ValueError("source_intake requires pdf_path")
        if not pages:
            raise ValueError("source_intake requires pages")
        (paths.input_dir / "yt_link.txt").write_text(youtube_url.strip() + "\n", encoding="utf-8")
        (paths.input_dir / "source_pdf.txt").write_text(pdf_path.strip() + "\n", encoding="utf-8")
        (paths.input_dir / "pages.txt").write_text(pages.strip() + "\n", encoding="utf-8")
        manifest = {
            "stage": stage,
            "experiment": paths.experiment,
            "input_dir": str(paths.input_dir),
            "out_dir": str(paths.out_dir),
            "source_pdf": pdf_path,
            "pages": pages,
        }
        out = paths.out_dir / "source_intake_manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    errors = guardrails_for_stage(stage, paths)
    if errors:
        raise RuntimeError("guardrails failed: " + "; ".join(errors))

    if stage == "asr_preparation":
        manifest = {
            "stage": stage,
            "experiment": paths.experiment,
            "audio": str(paths.input_dir / "audio.mp3"),
            "model": ai_model,
            "language_code": language_code,
            "note": "ASR execution is handled by run-sarvam-stt; this stage validates prerequisites.",
        }
        out = paths.out_dir / "asr_preparation_manifest.json"
        out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    if stage == "pdf_truth":
        local_pdf_path: Path | None = None
        if not pdf_path:
            source_pdf = (paths.input_dir / "source_pdf.txt")
            if source_pdf.exists():
                pdf_path = source_pdf.read_text(encoding="utf-8").strip()
        if not pages:
            source_pages = (paths.input_dir / "pages.txt")
            if source_pages.exists():
                pages = source_pages.read_text(encoding="utf-8").strip()
        if not pdf_path or not pages:
            raise ValueError("pdf_truth requires pdf_path and pages (or source files from source_intake)")
        local_pdf_path = _copy_pdf_to_input(paths, pdf_path)
        args = [
            "--experiment",
            paths.experiment,
            "--pdf",
            str(local_pdf_path),
            "--pages",
            pages,
        ]
        if top_trim_ratio is not None:
            args.extend(["--trim-top-ratio", str(top_trim_ratio)])
        if bottom_trim_ratio is not None:
            args.extend(["--trim-bottom-ratio", str(bottom_trim_ratio)])
        run_agent4_pdf_truth(args)
        return paths.out_dir / "pdf_tokens.json"

    if stage == "alignment_config":
        payload = {
            "stage": stage,
            "experiment": paths.experiment,
            "strategy": "merge-split-aware-dp",
            "thresholds": {"minConfidence": min_confidence, "maxEditCost": max_edit_cost},
            "inputs": {"sarvam": str(paths.out_dir / "0.json"), "pdf_tokens": str(paths.out_dir / "pdf_tokens.json")},
        }
        out = paths.out_dir / "alignment_config.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    if stage == "quality_gates":
        payload = {
            "stage": stage,
            "experiment": paths.experiment,
            "required_artifacts": [
                str(paths.input_dir / "audio.mp3"),
                str(paths.out_dir / "0.json"),
                str(paths.out_dir / "pdf_tokens.json"),
                str(paths.out_dir / "alignment_config.json"),
            ],
            "gates": {
                "minConfidence": min_confidence,
                "maxEditCost": max_edit_cost,
                "routeOnFailure": "human_review",
            },
        }
        out = paths.out_dir / "quality_gates.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out

    raise ValueError(f"unknown stage: {stage}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run-build-stage", description="Run one build stage with guardrails.")
    p.add_argument("--stage", required=True, choices=list(STAGES))
    p.add_argument("--experiment", required=True)
    p.add_argument("--youtube-url", default=None)
    p.add_argument("--pdf", default=None)
    p.add_argument("--pages", default=None)
    p.add_argument("--ai-model", default="saaras:v3")
    p.add_argument("--language-code", default="sa-IN")
    p.add_argument("--trim-top-ratio", type=float, default=None)
    p.add_argument("--trim-bottom-ratio", type=float, default=None)
    p.add_argument("--min-confidence", type=float, default=0.78)
    p.add_argument("--max-edit-cost", type=float, default=0.32)
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    out = run_stage(
        args.stage,
        experiment=args.experiment,
        youtube_url=args.youtube_url,
        pdf_path=args.pdf,
        pages=args.pages,
        ai_model=args.ai_model,
        language_code=args.language_code,
        top_trim_ratio=args.trim_top_ratio,
        bottom_trim_ratio=args.trim_bottom_ratio,
        min_confidence=args.min_confidence,
        max_edit_cost=args.max_edit_cost,
    )
    print(f"stage={args.stage}")
    print(f"output={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
