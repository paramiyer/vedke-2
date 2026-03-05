from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_segmenter(audio: Path, sentence_inputs: Path, run_dir: Path, config_path: Path) -> None:
    cmd = [
        sys.executable,
        "scripts/sentence_segmenter.py",
        "--audio",
        str(audio),
        "--sentence_inputs",
        str(sentence_inputs),
        "--out",
        str(run_dir),
        "--config_json",
        str(config_path),
    ]
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    env.setdefault("XDG_CACHE_HOME", "/tmp")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"segmenter run failed for {run_dir.name}:\n{proc.stderr}\n{proc.stdout}")


def _meets_targets(summary: dict[str, Any]) -> bool:
    return bool(summary.get("meets_best_case_targets", False))


def _should_stop_stall(best_history: list[float]) -> bool:
    if len(best_history) < 4:
        return False
    prev = best_history[-4]
    now = best_history[-1]
    if prev <= 0:
        return False
    improvement = (prev - now) / prev
    return improvement < 0.03


def _copy_best_run(best_run_dir: Path, best_out_dir: Path, best_config: dict[str, Any]) -> None:
    if best_out_dir.exists():
        shutil.rmtree(best_out_dir)
    shutil.copytree(best_run_dir, best_out_dir)
    _write_json(best_out_dir / "best_config.json", best_config)


def _print_row(run_id: int, summary: dict[str, Any]) -> None:
    print(
        f"run_{run_id:03d} "
        f"score={summary['score']:.6f} "
        f"conf_mean={summary['conf_mean']:.4f} "
        f"conf_p10={summary['conf_p10']:.4f} "
        f"zero_sil={summary['zero_silence_frac']:.4f} "
        f"spike_p95={summary['spike_p95']:.1f} "
        f"short_frac={summary['short_frac']:.4f} "
        f"filtered_K={summary['filtered_K']}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tune-segmenter", description="Iterative deterministic tuning for sentence segmenter")
    p.add_argument("--audio", required=True)
    p.add_argument("--sentence_inputs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_runs", type=int, default=12)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out).expanduser().resolve()
    runs_dir = out_dir / "runs"
    best_out_dir = out_dir / "best_run"
    runs_dir.mkdir(parents=True, exist_ok=True)

    audio = Path(args.audio).expanduser().resolve()
    sentence_inputs = Path(args.sentence_inputs).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")
    if not sentence_inputs.exists():
        raise FileNotFoundError(f"sentence_inputs file not found: {sentence_inputs}")

    grid = list(
        itertools.product(
            [300, 450, 600],
            [900, 1500, 2000],
            [250, 350],
            [20, 25, 30],
        )
    )
    grid = grid[: max(1, int(args.max_runs))]

    best_score = float("inf")
    best_run_dir: Path | None = None
    best_config: dict[str, Any] | None = None
    best_history: list[float] = []
    run_results: list[dict[str, Any]] = []

    print("run_id score conf_mean conf_p10 zero_silence_frac spike_p95 short_frac filtered_K")
    for run_id, (min_silence_boundary_ms, min_seg_ms, fill_gap_ms, thr_pct) in enumerate(grid):
        run_dir = runs_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "min_silence_boundary_ms": min_silence_boundary_ms,
            "min_seg_ms": min_seg_ms,
            "fill_gap_ms": fill_gap_ms,
            "thr_pct": thr_pct,
        }
        config_path = run_dir / "config.json"
        _write_json(config_path, config)

        _run_segmenter(audio=audio, sentence_inputs=sentence_inputs, run_dir=run_dir, config_path=config_path)
        summary = _read_json(run_dir / "run_summary.json")
        _print_row(run_id, summary)

        score = float(summary["score"])
        if score < best_score:
            best_score = score
            best_run_dir = run_dir
            best_config = config

        best_history.append(best_score)
        run_results.append({"run_id": run_id, "config": config, "summary": summary})

        if _meets_targets(summary):
            print(f"Early stop: run_{run_id:03d} meets best-case targets.")
            break

        if _should_stop_stall(best_history):
            print("Early stop: improvements stalled (<3% over last 3 runs).")
            break

    if best_run_dir is None or best_config is None:
        raise RuntimeError("No successful tuning run produced outputs")

    _copy_best_run(best_run_dir, best_out_dir, best_config)
    _write_json(
        out_dir / "tuning_summary.json",
        {
            "best_run": best_run_dir.name,
            "best_score": best_score,
            "best_config": best_config,
            "n_runs_completed": len(run_results),
            "runs": run_results,
        },
    )

    print(f"Best run: {best_run_dir.name} score={best_score:.6f}")
    print(f"Best artifacts copied to: {best_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
