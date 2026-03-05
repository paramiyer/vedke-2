from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import wave
from bisect import bisect_left, bisect_right
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import numpy as np
from scipy.signal import butter, filtfilt

try:
    from scripts.lib.text_filters import filter_sentences
except ModuleNotFoundError:
    from lib.text_filters import filter_sentences

matplotlib.use("Agg")
from matplotlib import pyplot as plt


BEST_CASE_TARGETS = {
    "conf_mean": 0.60,
    "conf_p10": 0.40,
    "zero_silence_frac": 0.10,
    "spike_p95": 6000.0,
    "short_frac": 0.15,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_ffmpeg_decode(audio_path: Path, out_wav_path: Path, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-vn",
        "-sn",
        str(out_wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed:\n{proc.stderr.strip()}")


def _load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sampwidth == 2:
        arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr.astype(np.float32), sr


def _save_wav_mono_float32(path: Path, y: np.ndarray, sr: int) -> None:
    clipped = np.clip(y, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def _butter_band(y: np.ndarray, sr: int, hp_hz: float, lp_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    hp = max(1.0, min(hp_hz, nyq * 0.95))
    lp = max(hp + 10.0, min(lp_hz, nyq * 0.99))

    b_hp, a_hp = butter(order, hp / nyq, btype="highpass")
    b_lp, a_lp = butter(order, lp / nyq, btype="lowpass")
    y1 = filtfilt(b_hp, a_hp, y).astype(np.float32)
    y2 = filtfilt(b_lp, a_lp, y1).astype(np.float32)
    return y2


def _compute_rms_db(y: np.ndarray, frame_len: int, hop: int, sr: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))
    starts = np.arange(0, len(y) - frame_len + 1, hop, dtype=np.int64)
    rms = np.empty(len(starts), dtype=np.float32)
    for i, s in enumerate(starts):
        frame = y[s : s + frame_len]
        rms[i] = float(np.sqrt(np.mean(frame * frame) + 1e-12))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    centers = (starts + (frame_len // 2)) / float(sr)
    return rms_db.astype(np.float32), centers.astype(np.float32)


def _rle_runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    if mask.size == 0:
        return []
    runs: list[tuple[int, int, bool]] = []
    start = 0
    curr = bool(mask[0])
    for i in range(1, len(mask)):
        val = bool(mask[i])
        if val != curr:
            runs.append((start, i, curr))
            start = i
            curr = val
    runs.append((start, len(mask), curr))
    return runs


def _apply_morphology(voiced: np.ndarray, frame_ms: float, fill_gap_ms: float, min_voiced_ms: float) -> np.ndarray:
    out = voiced.copy()
    max_gap_frames = int(round(fill_gap_ms / frame_ms))
    min_voiced_frames = int(round(min_voiced_ms / frame_ms))

    runs = _rle_runs(out)
    for start, end, is_voiced in runs:
        if is_voiced:
            continue
        if (end - start) <= max_gap_frames and start > 0 and end < len(out):
            if out[start - 1] and out[end]:
                out[start:end] = True

    runs = _rle_runs(out)
    for start, end, is_voiced in runs:
        if not is_voiced:
            continue
        if (end - start) < min_voiced_frames:
            out[start:end] = False
    return out


def _build_voiced_mask(
    rms_db: np.ndarray,
    frame_ms: float,
    thr_pct: float,
    fill_gap_ms: float,
    min_voiced_ms: float,
) -> tuple[np.ndarray, float]:
    thr_db = float(np.percentile(rms_db, thr_pct))
    voiced = rms_db > thr_db
    voiced = _apply_morphology(voiced, frame_ms, fill_gap_ms, min_voiced_ms)
    return voiced.astype(bool), thr_db


def _find_trim_leading_ms(times_sec: np.ndarray, voiced: np.ndarray, trim_sustain_ms: float, frame_ms: float) -> float:
    need_frames = max(1, int(round(trim_sustain_ms / frame_ms)))
    for i in range(0, len(voiced) - need_frames + 1):
        if np.all(voiced[i : i + need_frames]):
            return float(times_sec[i] * 1000.0)
    return 0.0


def _candidate_boundaries_ms(
    times_sec: np.ndarray,
    voiced: np.ndarray,
    t_trim_ms: float,
    min_silence_boundary_ms: float,
    grid_step_ms: float,
) -> np.ndarray:
    frame_ms = float((times_sec[1] - times_sec[0]) * 1000.0) if len(times_sec) > 1 else 16.0
    silence = ~voiced
    candidates: list[float] = [0.0, float(t_trim_ms)]

    runs = _rle_runs(silence)
    min_silence_frames = max(1, int(round(min_silence_boundary_ms / frame_ms)))
    for start, end, is_sil in runs:
        if not is_sil:
            continue
        if (end - start) >= min_silence_frames:
            center = (start + end - 1) // 2
            if 0 <= center < len(times_sec):
                candidates.append(float(times_sec[center] * 1000.0))

    grid = np.arange(0.0, t_trim_ms + 1e-6, grid_step_ms, dtype=np.float64)
    candidates.extend(grid.tolist())

    uniq = sorted({round(v, 3) for v in candidates if 0.0 <= v <= t_trim_ms})
    if not uniq or uniq[0] != 0.0:
        uniq = [0.0] + uniq
    if uniq[-1] != round(t_trim_ms, 3):
        uniq.append(round(t_trim_ms, 3))
    return np.asarray(uniq, dtype=np.float64)


def _nearest_frame_idx(times_sec: np.ndarray, t_ms: float) -> int:
    if len(times_sec) == 0:
        return 0
    t_sec = t_ms / 1000.0
    idx = int(np.argmin(np.abs(times_sec - t_sec)))
    return max(0, min(idx, len(times_sec) - 1))


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    w = max(1, int(w))
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        a = max(0, i - w + 1)
        out[i] = float(np.mean(x[a : i + 1]))
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(0, dtype=np.float64)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pick_changepoints(ds: np.ndarray, max_points: int = 6, min_sep: int = 5) -> list[int]:
    if len(ds) == 0:
        return []
    thr = float(np.percentile(ds, 95))
    candidates = [i + 1 for i, v in enumerate(ds) if v > thr]
    ranked = sorted(candidates, key=lambda i: float(ds[i - 1]), reverse=True)
    chosen: list[int] = []
    for idx in ranked:
        if all(abs(idx - c) >= min_sep for c in chosen):
            chosen.append(idx)
        if len(chosen) >= max_points:
            break
    return sorted(chosen)


def _drift_start_idx(drift_ms: np.ndarray, drift_thr_ms: float) -> int | None:
    n = len(drift_ms)
    for i in range(n):
        j = min(n, i + 7)
        window = drift_ms[i:j]
        if len(window) >= 5 and int(np.sum(np.abs(window) > drift_thr_ms)) >= 5:
            return i
    return None


def _dp_segment(
    candidate_ms: np.ndarray,
    expected_ms: np.ndarray,
    reward_by_candidate: np.ndarray,
    min_seg_ms: float,
    max_seg_ms: float,
    duration_strength: float,
    boundary_strength: float,
) -> list[int]:
    n = len(candidate_ms)
    k = len(expected_ms)
    inf = float("inf")

    dp = np.full((k + 1, n), inf, dtype=np.float64)
    back = np.full((k + 1, n), -1, dtype=np.int32)
    dp[0, 0] = 0.0

    for i in range(1, k + 1):
        min_j = i
        max_j = n - (k - i) - 1
        if i == k:
            min_j = max_j = n - 1
        for j in range(min_j, max_j + 1):
            t_j = candidate_ms[j]
            lo_t = t_j - max_seg_ms
            hi_t = t_j - min_seg_ms
            lo = bisect_left(candidate_ms, lo_t)
            hi = bisect_right(candidate_ms, hi_t) - 1
            lo = max(lo, i - 1)
            hi = min(hi, j - 1)
            if lo > hi:
                continue

            e_i = max(1e-6, float(expected_ms[i - 1]))
            reward = float(reward_by_candidate[j])
            b_cost = -boundary_strength * reward

            best_val = inf
            best_k = -1
            for prev in range(lo, hi + 1):
                prev_cost = dp[i - 1, prev]
                if not math.isfinite(prev_cost):
                    continue
                dur = float(t_j - candidate_ms[prev])
                d_cost = duration_strength * ((dur - e_i) / e_i) ** 2
                val = prev_cost + d_cost + b_cost
                if val < best_val:
                    best_val = val
                    best_k = prev
            dp[i, j] = best_val
            back[i, j] = best_k

    if not math.isfinite(dp[k, n - 1]):
        raise RuntimeError(
            "DP could not find a valid segmentation to audio end. "
            "Try decreasing --min_seg_ms, or decreasing --min_silence_boundary_ms, "
            "or decreasing --grid_step_ms."
        )

    idxs = [n - 1]
    i, j = k, n - 1
    while i > 0:
        p = int(back[i, j])
        if p < 0:
            raise RuntimeError(
                "DP backtrack failed. Try decreasing --min_seg_ms, or decreasing "
                "--min_silence_boundary_ms, or decreasing --grid_step_ms."
            )
        idxs.append(p)
        i -= 1
        j = p
    return list(reversed(idxs))


def _boundary_metrics(
    boundary_ms: np.ndarray,
    times_sec: np.ndarray,
    rms_db: np.ndarray,
    voiced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_ms = float((times_sec[1] - times_sec[0]) * 1000.0) if len(times_sec) > 1 else 16.0
    silence = ~voiced
    half_span_frames = max(1, int(round(1500.0 / frame_ms)))
    around_frames = max(1, int(round(200.0 / frame_ms)))
    prepost_frames = max(1, int(round(300.0 / frame_ms)))

    silence_width_ms = np.zeros(len(boundary_ms), dtype=np.float64)
    rms_end = np.zeros(len(boundary_ms), dtype=np.float64)
    contrast = np.zeros(len(boundary_ms), dtype=np.float64)

    for i, t in enumerate(boundary_ms):
        f = _nearest_frame_idx(times_sec, float(t))
        lo = max(0, f - half_span_frames)
        hi = min(len(silence) - 1, f + half_span_frames)
        if silence[f]:
            a = f
            b = f
            while a - 1 >= lo and silence[a - 1]:
                a -= 1
            while b + 1 <= hi and silence[b + 1]:
                b += 1
            silence_width_ms[i] = (b - a + 1) * frame_ms
        else:
            silence_width_ms[i] = 0.0

        a1 = max(0, f - around_frames)
        b1 = min(len(rms_db), f + around_frames + 1)
        rms_end[i] = float(np.mean(rms_db[a1:b1])) if b1 > a1 else float(rms_db[f])

        p0 = max(0, f - prepost_frames)
        p1 = f
        q0 = f
        q1 = min(len(rms_db), f + prepost_frames)
        pre = float(np.mean(rms_db[p0:p1])) if p1 > p0 else float(rms_db[f])
        post = float(np.mean(rms_db[q0:q1])) if q1 > q0 else float(rms_db[f])
        contrast[i] = pre - post

    return silence_width_ms, rms_end, contrast


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _compute_quality_metrics(
    conf: np.ndarray,
    silence_width_ms: np.ndarray,
    durs: np.ndarray,
    min_seg_ms: float,
    drift: np.ndarray,
    drift_jump: np.ndarray,
) -> dict[str, float]:
    conf_mean = float(np.mean(conf)) if len(conf) else 0.0
    conf_p10 = float(np.percentile(conf, 10)) if len(conf) else 0.0
    zero_silence_frac = float(np.mean(silence_width_ms == 0.0)) if len(silence_width_ms) else 1.0
    short_frac = float(np.mean(durs <= (min_seg_ms + 150.0))) if len(durs) else 1.0
    drift_p95 = float(np.percentile(np.abs(drift), 95)) if len(drift) else 0.0
    spike_p95 = float(np.percentile(np.abs(drift_jump), 95)) if len(drift_jump) else 0.0

    score = (
        0.35 * zero_silence_frac
        + 0.25 * (1.0 - conf_mean)
        + 0.15 * (1.0 - conf_p10)
        + 0.15 * (spike_p95 / 8000.0)
        + 0.10 * short_frac
    )

    return {
        "score": float(score),
        "conf_mean": conf_mean,
        "conf_p10": conf_p10,
        "zero_silence_frac": zero_silence_frac,
        "short_frac": short_frac,
        "drift_p95": drift_p95,
        "spike_p95": spike_p95,
    }


def _meets_best_case_targets(metrics: dict[str, float]) -> bool:
    return (
        metrics["conf_mean"] >= BEST_CASE_TARGETS["conf_mean"]
        and metrics["conf_p10"] >= BEST_CASE_TARGETS["conf_p10"]
        and metrics["zero_silence_frac"] <= BEST_CASE_TARGETS["zero_silence_frac"]
        and metrics["spike_p95"] <= BEST_CASE_TARGETS["spike_p95"]
        and metrics["short_frac"] <= BEST_CASE_TARGETS["short_frac"]
    )


def _apply_config_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config_json:
        return args
    cfg_path = Path(args.config_json).expanduser().resolve()
    cfg = load_json(cfg_path)
    if not isinstance(cfg, dict):
        raise ValueError("--config_json must point to a JSON object")
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def build_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    review_dir = out_dir / "review_pack"
    review_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(Path(args.sentence_inputs).expanduser().resolve())
    sentences_raw = payload.get("sentences") if isinstance(payload, dict) else None
    if not isinstance(sentences_raw, list) or not sentences_raw:
        raise ValueError("sentence_inputs.json must contain non-empty 'sentences' list")

    filtered_sentences, filtered_out, mapping = filter_sentences([s for s in sentences_raw if isinstance(s, dict)])
    original_k = len(sentences_raw)
    filtered_k = len(filtered_sentences)
    if filtered_k <= 0:
        raise RuntimeError("All sentences were filtered out; cannot segment empty set")

    filter_report = {
        "original_K": original_k,
        "filtered_K": filtered_k,
        "filtered_out_count": len(filtered_out),
        "filtered_out": filtered_out,
        "mapping": mapping,
    }
    _write_json(out_dir / "filter_report.json", filter_report)

    # K follows filtered sentences by design.
    k = filtered_k

    weights = []
    for i, s in enumerate(filtered_sentences):
        w = s.get("weight", s.get("widthSumWords", 1.0))
        try:
            wv = float(w)
        except Exception as exc:
            raise ValueError(f"Invalid weight at filtered idx {i}") from exc
        weights.append(max(1e-6, wv))
    w = np.asarray(weights, dtype=np.float64)

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    decoded_wav = out_dir / "decoded_input_16k_mono.wav"
    _run_ffmpeg_decode(audio_path, decoded_wav, sr=16000)
    y, sr = _load_wav_mono_float32(decoded_wav)
    if sr != 16000:
        raise RuntimeError(f"Decoded sample rate must be 16000, got {sr}")

    y_f = _butter_band(y, sr, hp_hz=float(args.hp_hz), lp_hz=float(args.lp_hz), order=4)

    rms_db, times_sec = _compute_rms_db(y_f, frame_len=int(args.rms_frame), hop=int(args.rms_hop), sr=sr)
    frame_ms = (float(args.rms_hop) / sr) * 1000.0
    voiced, _ = _build_voiced_mask(
        rms_db,
        frame_ms=frame_ms,
        thr_pct=float(args.thr_pct),
        fill_gap_ms=float(args.fill_gap_ms),
        min_voiced_ms=float(args.min_voiced_ms),
    )
    trim_leading_ms = _find_trim_leading_ms(times_sec, voiced, trim_sustain_ms=float(args.trim_sustain_ms), frame_ms=frame_ms)
    trim_samp = int(round(trim_leading_ms * sr / 1000.0))
    trim_samp = max(0, min(trim_samp, max(0, len(y_f) - 1)))
    y_trim = y_f[trim_samp:].copy()

    rms_db_t, times_sec_t = _compute_rms_db(y_trim, frame_len=int(args.rms_frame), hop=int(args.rms_hop), sr=sr)
    voiced_t, _ = _build_voiced_mask(
        rms_db_t,
        frame_ms=frame_ms,
        thr_pct=float(args.thr_pct),
        fill_gap_ms=float(args.fill_gap_ms),
        min_voiced_ms=float(args.min_voiced_ms),
    )
    t_trim_ms = (len(y_trim) / sr) * 1000.0

    candidate_ms = _candidate_boundaries_ms(
        times_sec_t,
        voiced_t,
        t_trim_ms=t_trim_ms,
        min_silence_boundary_ms=float(args.min_silence_boundary_ms),
        grid_step_ms=float(args.grid_step_ms),
    )
    if len(candidate_ms) < (k + 1):
        raise RuntimeError(
            f"Not enough candidate boundaries ({len(candidate_ms)}) for K={k}. "
            "Try decreasing --min_silence_boundary_ms or --grid_step_ms."
        )

    neg_rms = -rms_db_t
    smooth = np.convolve(neg_rms, np.ones(7) / 7.0, mode="same")
    reward_by_candidate = np.zeros(len(candidate_ms), dtype=np.float64)
    for j, t in enumerate(candidate_ms):
        fi = _nearest_frame_idx(times_sec_t, float(t))
        reward_by_candidate[j] = float(smooth[fi]) if len(smooth) else 0.0

    expected_ms = (t_trim_ms * w / max(1e-9, float(np.sum(w)))).astype(np.float64)
    boundary_idx = _dp_segment(
        candidate_ms=candidate_ms,
        expected_ms=expected_ms,
        reward_by_candidate=reward_by_candidate,
        min_seg_ms=float(args.min_seg_ms),
        max_seg_ms=float(args.max_seg_ms),
        duration_strength=float(args.duration_strength),
        boundary_strength=float(args.boundary_strength),
    )
    b_ms = candidate_ms[np.asarray(boundary_idx, dtype=np.int64)]
    if len(b_ms) != (k + 1):
        raise RuntimeError(f"Expected {k+1} boundaries, got {len(b_ms)}")

    starts = b_ms[:-1]
    ends = b_ms[1:]
    durs = ends - starts
    if np.any(durs <= 0):
        raise RuntimeError("Non-monotonic boundaries produced by DP")
    if abs(float(ends[-1]) - float(t_trim_ms)) > 50.0:
        raise RuntimeError("Last segment end is not within ~50ms of trimmed audio end")

    silence_width_ms, rms_end, contrast_db = _boundary_metrics(ends, times_sec_t, rms_db_t, voiced_t)
    z_sil = _zscore(silence_width_ms)
    z_rms = _zscore(-rms_end)
    z_con = _zscore(contrast_db)
    conf_raw = 0.6 * z_sil + 0.25 * z_rms + 0.15 * z_con
    conf = _sigmoid(conf_raw)

    cum_w = np.cumsum(w)
    exp_end = t_trim_ms * (cum_w / max(1e-9, float(np.sum(w))))
    obs_end = ends.copy()
    drift = obs_end - exp_end
    drift_jump = np.empty_like(drift)
    drift_jump[0] = drift[0]
    drift_jump[1:] = drift[1:] - drift[:-1]
    roll_drift = _rolling_mean(drift, int(args.rolling_w))
    roll_jump = _rolling_mean(drift_jump, int(args.rolling_w))
    ds = np.abs(np.diff(roll_jump))
    changepoints = _pick_changepoints(ds, max_points=6, min_sep=5)

    seg_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    for i, s in enumerate(filtered_sentences):
        start_t = float(starts[i])
        end_t = float(ends[i])
        duration = float(durs[i])
        merged = dict(s)
        merged["idx"] = i
        merged["start_ms_trimmed"] = round(start_t, 3)
        merged["end_ms_trimmed"] = round(end_t, 3)
        merged["trim_leading_ms"] = round(float(trim_leading_ms), 3)
        merged["start_ms_original"] = round(start_t + float(trim_leading_ms), 3)
        merged["end_ms_original"] = round(end_t + float(trim_leading_ms), 3)
        merged["duration_ms"] = round(duration, 3)
        merged["silence_width_ms"] = round(float(silence_width_ms[i]), 3)
        merged["rms_db_at_end"] = round(float(rms_end[i]), 4)
        merged["contrast_db"] = round(float(contrast_db[i]), 4)
        merged["confidence"] = round(float(conf[i]), 6)
        merged["mapping_original_idx"] = int(mapping[i]["original_idx"])
        seg_rows.append(merged)

        diag_rows.append(
            {
                "idx": i,
                "original_idx": int(mapping[i]["original_idx"]),
                "sentenceId": s.get("sentenceId", ""),
                "text": s.get("text", ""),
                "start_ms_trimmed": round(start_t, 3),
                "end_ms_trimmed": round(end_t, 3),
                "duration_ms": round(duration, 3),
                "exp_end_ms": round(float(exp_end[i]), 3),
                "obs_end_ms": round(float(obs_end[i]), 3),
                "drift_ms": round(float(drift[i]), 3),
                "drift_jump_ms": round(float(drift_jump[i]), 3),
                "silence_width_ms": round(float(silence_width_ms[i]), 3),
                "rms_db_at_end": round(float(rms_end[i]), 4),
                "contrast_db": round(float(contrast_db[i]), 4),
                "confidence": round(float(conf[i]), 6),
                "weight": round(float(w[i]), 6),
            }
        )

    metrics = _compute_quality_metrics(conf, silence_width_ms, durs, float(args.min_seg_ms), drift, drift_jump)
    meets_targets = _meets_best_case_targets(metrics)

    segments_payload = {
        "target_segments": k,
        "original_K": original_k,
        "filtered_K": filtered_k,
        "filtered_out": filtered_out,
        "mapping": mapping,
        "sentences": seg_rows,
        "params": {
            "pause_bonus_danda": 1.2,
            "pause_bonus_double_danda": 1.5,
            "alpha_svara": 0.15,
            "min_seg_ms": float(args.min_seg_ms),
            "max_seg_ms": float(args.max_seg_ms),
            "boundary_silence_window_ms": 200,
            "duration_prior_strength": float(args.duration_strength),
        },
        "trim_leading_ms": round(float(trim_leading_ms), 3),
    }
    _write_json(out_dir / "segments_sentence.json", segments_payload)

    _write_csv(
        out_dir / "segments_sentence.csv",
        seg_rows,
        fields=[
            "idx",
            "mapping_original_idx",
            "sentenceId",
            "start_ms_trimmed",
            "end_ms_trimmed",
            "duration_ms",
            "weight",
            "confidence",
            "text",
        ],
    )
    _write_csv(
        out_dir / "alignment_diagnostics.csv",
        diag_rows,
        fields=[
            "idx",
            "original_idx",
            "sentenceId",
            "text",
            "start_ms_trimmed",
            "end_ms_trimmed",
            "duration_ms",
            "exp_end_ms",
            "obs_end_ms",
            "drift_ms",
            "drift_jump_ms",
            "silence_width_ms",
            "rms_db_at_end",
            "contrast_db",
            "confidence",
            "weight",
        ],
    )

    x = np.arange(k)
    plt.figure(figsize=(12, 4))
    plt.plot(times_sec_t, rms_db_t, linewidth=0.8)
    for b in ends:
        plt.axvline(x=float(b) / 1000.0, color="red", alpha=0.18, linewidth=0.6)
    plt.title("Trimmed RMS dB with All Boundaries")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS (dB)")
    plt.tight_layout()
    plt.savefig(out_dir / "diag_rms_boundaries.png", dpi=140)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(x, drift, label="drift_ms", linewidth=1.0)
    plt.plot(x, roll_drift, label=f"rolling_mean_{args.rolling_w}", linewidth=1.2)
    plt.axhline(y=float(args.drift_thr_ms), color="orange", linestyle="--")
    plt.axhline(y=-float(args.drift_thr_ms), color="orange", linestyle="--")
    for cp in changepoints:
        plt.axvline(x=cp, color="red", alpha=0.5, linewidth=0.9)
    plt.title("Drift Curve")
    plt.xlabel("Sentence idx")
    plt.ylabel("Drift (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "drift_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.bar(x, drift_jump, alpha=0.5, width=0.9, label="drift_jump_ms")
    plt.plot(x, roll_jump, color="black", linewidth=1.2, label=f"rolling_mean_{args.rolling_w}")
    for cp in changepoints:
        plt.axvline(x=cp, color="red", alpha=0.5, linewidth=0.9)
    plt.title("Drift Slope")
    plt.xlabel("Sentence idx")
    plt.ylabel("Delta Drift (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "drift_slope.png", dpi=140)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(x, obs_end / 1000.0, label="observed_end_s")
    plt.plot(x, exp_end / 1000.0, label="expected_end_s")
    for cp in changepoints:
        plt.axvline(x=cp, color="red", alpha=0.5, linewidth=0.9)
    plt.title("Expected vs Observed End Times")
    plt.xlabel("Sentence idx")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "expected_vs_observed.png", dpi=140)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(x, conf, label="confidence")
    plt.plot(x, (z_sil - z_sil.min()) / (np.ptp(z_sil) + 1e-9), label="silence_z_scaled", alpha=0.7)
    plt.plot(x, (z_rms - z_rms.min()) / (np.ptp(z_rms) + 1e-9), label="-rms_z_scaled", alpha=0.7)
    for cp in changepoints:
        plt.axvline(x=cp, color="red", alpha=0.5, linewidth=0.9)
    plt.title("Boundary Confidence")
    plt.xlabel("Sentence idx")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "boundary_confidence.png", dpi=140)
    plt.close()

    _write_json(
        out_dir / "changepoints.json",
        {
            "changepoints": changepoints,
            "method": "abs(diff(rolling_mean(drift_jump))) > p95, top6 with min separation 5",
        },
    )

    z_jump = _zscore(drift_jump)
    unc = (1.0 - conf) + 0.35 * np.abs(z_jump)
    ranked = sorted(range(k), key=lambda i: float(unc[i]), reverse=True)

    def pick_with_spacing(min_spacing: int) -> list[int]:
        chosen: list[int] = []
        for i in ranked:
            if all(abs(i - c) >= min_spacing for c in chosen):
                chosen.append(i)
            if len(chosen) >= int(args.active_k):
                break
        return chosen

    chosen = pick_with_spacing(3)
    if len(chosen) < int(args.active_k):
        chosen = pick_with_spacing(2)
    chosen = chosen[: int(args.active_k)]

    p95_jump = float(np.percentile(np.abs(drift_jump), 95)) if len(drift_jump) else 0.0
    samples: list[dict[str, Any]] = []
    for idx in chosen:
        s = seg_rows[idx]
        clip_start = max(0.0, float(s["start_ms_trimmed"]) - float(args.active_pad_ms))
        clip_end = min(t_trim_ms, float(s["end_ms_trimmed"]) + float(args.active_pad_ms))
        a = int(round(clip_start * sr / 1000.0))
        b = int(round(clip_end * sr / 1000.0))
        clip = y_trim[a:b]
        clip_name = f"active_sample_{idx:03d}_{int(round(clip_start))}_{int(round(clip_end))}.wav"
        _save_wav_mono_float32(review_dir / clip_name, clip, sr)

        if float(conf[idx]) < 0.35 and float(silence_width_ms[idx]) < 200.0:
            rec = "WEAK_BOUNDARY: likely not a sentence pause. Prefer higher min_silence_boundary_ms or merge/shift later."
        elif abs(float(drift_jump[idx])) > p95_jump:
            rec = "DRIFT_SPIKE: boundary/structure kink. Review padded clip; consider shifting boundary to nearest stronger silence."
        elif float(s["duration_ms"]) <= (float(args.min_seg_ms) + 150.0):
            rec = "TOO_SHORT: likely spurious split; increase min_seg_ms or min_silence_boundary_ms."
        else:
            rec = "REVIEW"

        samples.append(
            {
                "idx": int(idx),
                "original_idx": int(mapping[idx]["original_idx"]),
                "sentenceId": s.get("sentenceId"),
                "text": s.get("text"),
                "start_ms_trimmed": s["start_ms_trimmed"],
                "end_ms_trimmed": s["end_ms_trimmed"],
                "duration_ms": s["duration_ms"],
                "drift_ms": round(float(drift[idx]), 3),
                "drift_jump_ms": round(float(drift_jump[idx]), 3),
                "silence_width_ms": round(float(silence_width_ms[idx]), 3),
                "confidence": round(float(conf[idx]), 6),
                "uncertainty": round(float(unc[idx]), 6),
                "recommendation": rec,
                "clip": f"review_pack/{clip_name}",
            }
        )

    _write_json(
        out_dir / "active_learning_manifest.json",
        {
            "trim_leading_ms": round(float(trim_leading_ms), 3),
            "drift_start_idx": _drift_start_idx(drift, float(args.drift_thr_ms)),
            "changepoints": changepoints,
            "samples": samples,
        },
    )

    run_summary = {
        "score": round(metrics["score"], 8),
        "conf_mean": round(metrics["conf_mean"], 6),
        "conf_p10": round(metrics["conf_p10"], 6),
        "zero_silence_frac": round(metrics["zero_silence_frac"], 6),
        "short_frac": round(metrics["short_frac"], 6),
        "drift_p95": round(metrics["drift_p95"], 3),
        "spike_p95": round(metrics["spike_p95"], 3),
        "original_K": int(original_k),
        "filtered_K": int(filtered_k),
        "meets_best_case_targets": bool(meets_targets),
    }
    _write_json(out_dir / "run_summary.json", run_summary)

    run_manifest = {
        "params": {
            "hp_hz": float(args.hp_hz),
            "lp_hz": float(args.lp_hz),
            "rms_frame": int(args.rms_frame),
            "rms_hop": int(args.rms_hop),
            "thr_pct": float(args.thr_pct),
            "fill_gap_ms": float(args.fill_gap_ms),
            "min_voiced_ms": float(args.min_voiced_ms),
            "trim_sustain_ms": float(args.trim_sustain_ms),
            "min_silence_boundary_ms": float(args.min_silence_boundary_ms),
            "grid_step_ms": float(args.grid_step_ms),
            "min_seg_ms": float(args.min_seg_ms),
            "max_seg_ms": float(args.max_seg_ms),
            "boundary_strength": float(args.boundary_strength),
            "duration_strength": float(args.duration_strength),
            "active_k": int(args.active_k),
            "active_pad_ms": float(args.active_pad_ms),
            "drift_thr_ms": float(args.drift_thr_ms),
            "rolling_w": int(args.rolling_w),
        },
        "summary": run_summary,
        "targets": BEST_CASE_TARGETS,
    }
    _write_json(out_dir / "run_manifest.json", run_manifest)

    print(
        f"K={k} (original={original_k}) score={metrics['score']:.6f} "
        f"conf_mean={metrics['conf_mean']:.4f} conf_p10={metrics['conf_p10']:.4f} "
        f"zero_silence_frac={metrics['zero_silence_frac']:.4f}"
    )
    print(f"Outputs written to: {out_dir}")
    return run_manifest


def _self_test(args: argparse.Namespace) -> None:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    syn_audio = out_dir / "self_test_audio.wav"
    syn_inputs = out_dir / "self_test_sentence_inputs.json"

    sr = 16000
    rng = np.random.default_rng(12345)
    parts: list[np.ndarray] = []
    for i in range(7):
        dur_s = 0.9 + 0.2 * (i % 3)
        t = np.arange(int(sr * dur_s), dtype=np.float32) / sr
        voiced = 0.2 * np.sin(2 * np.pi * (170 + 30 * i) * t) + 0.015 * rng.standard_normal(len(t))
        silence = np.zeros(int(sr * 0.35), dtype=np.float32)
        parts.extend([voiced.astype(np.float32), silence])
    y = np.concatenate(parts).astype(np.float32)
    _save_wav_mono_float32(syn_audio, y, sr)

    texts = [
        "ॐ गणपतये नमः।",
        "Visit www.example.com now",
        "देवाय नमः॥",
        "Email: test@example.com",
        "अग्निमीळे पुरोहितम्।",
        "12345",
        "त्वमेव प्रत्यक्षं तत्त्वमसि॥",
    ]
    payload = {
        "target_segments": len(texts),
        "sentences": [
            {
                "idx": i,
                "sentenceId": f"P0001_L0001_S{i+1:04d}",
                "text": texts[i],
                "weight": float(i + 1),
            }
            for i in range(len(texts))
        ],
        "params": {},
    }
    _write_json(syn_inputs, payload)

    local = argparse.Namespace(**vars(args))
    local.audio = str(syn_audio)
    local.sentence_inputs = str(syn_inputs)
    local.target_segments = None
    build_pipeline(local)

    out_json = load_json(out_dir / "segments_sentence.json")
    run_summary = load_json(out_dir / "run_summary.json")
    if out_json["filtered_K"] >= out_json["original_K"]:
        raise RuntimeError("self_test failed: expected some sentences to be filtered")
    if int(out_json["target_segments"]) != int(out_json["filtered_K"]):
        raise RuntimeError("self_test failed: target_segments must equal filtered_K")

    prev_end = -1e9
    for row in out_json["sentences"]:
        s = float(row["start_ms_trimmed"])
        e = float(row["end_ms_trimmed"])
        if e <= s or s < prev_end:
            raise RuntimeError("self_test failed: non-monotonic timestamps")
        prev_end = e

    required = [
        out_dir / "drift_curve.png",
        out_dir / "drift_slope.png",
        out_dir / "expected_vs_observed.png",
        out_dir / "boundary_confidence.png",
        out_dir / "active_learning_manifest.json",
        out_dir / "filter_report.json",
        out_dir / "run_summary.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"self_test failed: missing outputs {missing}")
    if "score" not in run_summary:
        raise RuntimeError("self_test failed: run_summary missing score")
    print("self_test passed")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sentence-segmenter", description="Sentence-level segmenter + validator pipeline.")
    p.add_argument("--audio", required=False, help="Path to input audio file (mp3/wav).")
    p.add_argument("--sentence_inputs", required=False, help="Path to sentence_inputs.json")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--config_json", default=None, help="Optional JSON file with CLI parameter overrides")
    p.add_argument("--target_segments", type=int, default=None)
    p.add_argument("--hp_hz", type=float, default=80.0)
    p.add_argument("--lp_hz", type=float, default=4500.0)
    p.add_argument("--rms_frame", type=int, default=2048)
    p.add_argument("--rms_hop", type=int, default=256)
    p.add_argument("--thr_pct", type=float, default=25.0)
    p.add_argument("--fill_gap_ms", type=float, default=250.0)
    p.add_argument("--min_voiced_ms", type=float, default=150.0)
    p.add_argument("--trim_sustain_ms", type=float, default=800.0)
    p.add_argument("--min_silence_boundary_ms", type=float, default=300.0)
    p.add_argument("--grid_step_ms", type=float, default=400.0)
    p.add_argument("--min_seg_ms", type=float, default=900.0)
    p.add_argument("--max_seg_ms", type=float, default=20000.0)
    p.add_argument("--boundary_strength", type=float, default=1.2)
    p.add_argument("--duration_strength", type=float, default=1.0)
    p.add_argument("--active_k", type=int, default=10)
    p.add_argument("--active_pad_ms", type=float, default=250.0)
    p.add_argument("--drift_thr_ms", type=float, default=1500.0)
    p.add_argument("--rolling_w", type=int, default=7)
    p.add_argument("--self_test", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args = _apply_config_overrides(args)
    if args.self_test:
        _self_test(args)
        return 0
    if not args.audio or not args.sentence_inputs:
        raise ValueError("--audio and --sentence_inputs are required unless --self_test is set")
    build_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
