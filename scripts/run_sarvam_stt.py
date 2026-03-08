from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests

API_BASE = "https://api.sarvam.ai/speech-to-text/job/v1"


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


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


def _headers_json(api_key: str) -> dict[str, str]:
    return {"api-subscription-key": api_key, "Content-Type": "application/json"}


def _headers_key_only(api_key: str) -> dict[str, str]:
    return {"api-subscription-key": api_key}


def _request_json(method: str, url: str, *, headers: dict[str, str], timeout: float, **kwargs: Any) -> dict[str, Any]:
    resp = requests.request(method=method, url=url, headers=headers, timeout=timeout, **kwargs)
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {url} failed ({resp.status_code}): {resp.text[:500]}")
    try:
        payload = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"{method} {url} returned non-JSON response: {resp.text[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{method} {url} returned unexpected payload type: {type(payload).__name__}")
    return payload


def _extract_put_url(upload_info: dict[str, Any], audio_name: str) -> str:
    upload_urls = upload_info.get("upload_urls")
    if not isinstance(upload_urls, dict):
        raise RuntimeError(f"upload-files missing upload_urls: {json.dumps(upload_info)[:800]}")

    entry = upload_urls.get(audio_name)
    if entry is None:
        raise RuntimeError(f"upload-files missing key '{audio_name}' in upload_urls: {json.dumps(upload_info)[:800]}")

    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        file_url = entry.get("file_url")
        if isinstance(file_url, str):
            return file_url
    raise RuntimeError(f"upload-files invalid upload_urls entry for '{audio_name}': {json.dumps(upload_info)[:800]}")


def _poll_status(api_key: str, job_id: str, poll_interval: float, timeout_s: float) -> dict[str, Any]:
    t0 = time.time()
    while True:
        status = _request_json(
            "GET",
            f"{API_BASE}/{job_id}/status",
            headers=_headers_key_only(api_key),
            timeout=60,
        )
        state = status.get("job_state")
        print(
            f"state: {state} | success: {status.get('successful_files_count')} "
            f"fail: {status.get('failed_files_count')}"
        )
        if state in ("Completed", "Failed"):
            return status
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for job completion after {timeout_s:.1f}s")
        time.sleep(poll_interval)


def _collect_output_files(status: dict[str, Any], input_audio_name: str) -> list[str]:
    if status.get("job_state") != "Completed":
        raise RuntimeError(f"Job did not complete successfully: {status.get('error_message')}")

    output_files: list[str] = []
    for jd in (status.get("job_details") or []):
        if not isinstance(jd, dict):
            continue
        for out in (jd.get("outputs") or []):
            if isinstance(out, dict):
                file_name = out.get("file_name")
                if isinstance(file_name, str) and file_name:
                    output_files.append(file_name)
            elif isinstance(out, str) and out:
                output_files.append(out)

    if not output_files:
        output_files = [input_audio_name]

    return output_files


def _extract_download_urls(dl_info: dict[str, Any]) -> dict[str, str]:
    download_urls = dl_info.get("download_urls")
    if not isinstance(download_urls, dict):
        raise RuntimeError(f"download-files missing download_urls: {json.dumps(dl_info)[:800]}")

    out: dict[str, str] = {}
    for fname, meta in download_urls.items():
        if not isinstance(fname, str):
            continue
        if isinstance(meta, str):
            out[fname] = meta
            continue
        if isinstance(meta, dict) and isinstance(meta.get("file_url"), str):
            out[fname] = str(meta["file_url"])

    if not out:
        raise RuntimeError(f"download-files had no usable URLs: {json.dumps(dl_info)[:800]}")
    return out


def _split_duration_by_lengths(total_ms: int, lengths: list[int]) -> list[int]:
    if total_ms <= 0 or not lengths or sum(lengths) <= 0:
        return [0] * len(lengths)
    total_len = sum(lengths)
    raw = [total_ms * x / total_len for x in lengths]
    base = [int(x) for x in raw]
    remainder = total_ms - sum(base)
    frac_order = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for idx in frac_order[:remainder]:
        base[idx] += 1
    return base


def build_audio_cleaned_tokens_payload(stt_payload: dict[str, Any]) -> dict[str, Any]:
    ts = stt_payload.get("timestamps") if isinstance(stt_payload, dict) else None
    words = ts.get("words", []) if isinstance(ts, dict) else []
    starts = ts.get("start_time_seconds", []) if isinstance(ts, dict) else []
    ends = ts.get("end_time_seconds", []) if isinstance(ts, dict) else []
    if not isinstance(words, list) or len(words) == 0:
        raise RuntimeError("Invalid STT output: 0.json timestamps.words is null/empty")

    tokens: list[dict[str, Any]] = []
    segment_count = min(len(words), len(starts), len(ends))
    token_counter = 0

    for seg_idx in range(segment_count):
        segment_text = str(words[seg_idx]).strip()
        if not segment_text:
            continue
        segment_tokens = segment_text.split()
        if not segment_tokens:
            continue

        try:
            seg_start_ms = int(round(float(starts[seg_idx]) * 1000))
            seg_end_ms = int(round(float(ends[seg_idx]) * 1000))
        except (TypeError, ValueError):
            continue

        seg_dur = max(0, seg_end_ms - seg_start_ms)
        lengths = [len(tok) for tok in segment_tokens]
        alloc = _split_duration_by_lengths(seg_dur, lengths)

        cursor = seg_start_ms
        for tok_idx, (tok, tok_len, tok_ms) in enumerate(zip(segment_tokens, lengths, alloc), start=1):
            token_counter += 1
            start_ms = cursor
            end_ms = cursor + tok_ms
            cursor = end_ms
            tokens.append(
                {
                    "token_id": f"S{seg_idx+1:04d}_T{tok_idx:04d}",
                    "token": tok,
                    "char_count": tok_len,
                    "segment_index": seg_idx,
                    "token_index_in_segment": tok_idx - 1,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "start_s": round(start_ms / 1000.0, 3),
                    "end_s": round(end_ms / 1000.0, 3),
                }
            )

    return {
        "request_id": stt_payload.get("request_id"),
        "source": "0.json",
        "segments_count": segment_count,
        "token_count": len(tokens),
        "fields": ["token_id", "token", "char_count", "segment_index", "token_index_in_segment", "start_ms", "end_ms"],
        "tokens": tokens,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run-sarvam-stt",
        description="Run Sarvam STT batch API for data/<experiment>/input/audio.mp3 and save outputs under data/<experiment>/out.",
    )
    parser.add_argument("--experiment", required=True, help="Experiment name used as folder slug under data/")
    parser.add_argument("--audio", default=None, help="Optional override for input audio path")
    parser.add_argument("--job-id", default=None, help="Existing Sarvam job_id (optional). If omitted, a new job is created.")
    parser.add_argument("--model", default="saaras:v3", help="Sarvam model name")
    parser.add_argument("--mode", default="verbatim", help="Sarvam mode, e.g. verbatim")
    parser.add_argument("--language-code", default="sa-IN", help="Language code, e.g. sa-IN")
    parser.add_argument("--with-timestamps", action="store_true", help="Request timestamps in Sarvam output")
    parser.add_argument("--num-speakers", type=int, default=1, help="Number of speakers (default: 1)")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="Seconds between status polls")
    parser.add_argument("--timeout-s", type=float, default=1800.0, help="Max wait for job completion")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _load_dotenv(Path(".env").resolve())
    api_key = os.getenv("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing SARVAM_API_KEY. Set it in environment or .env")

    slug = _slugify(args.experiment)
    input_dir = (Path("data") / slug / "input").resolve()
    out_dir = (Path("data") / slug / "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.audio:
        audio_path = Path(args.audio).expanduser().resolve()
    else:
        audio_path = (input_dir / "audio.mp3").resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    headers_json = _headers_json(api_key)
    headers_keyonly = _headers_key_only(api_key)

    job_id = args.job_id
    if not job_id:
        init_payload = {
            "job_parameters": {
                "model": args.model,
                "mode": args.mode,
                "language_code": args.language_code,
                "with_timestamps": bool(args.with_timestamps),
                "num_speakers": int(args.num_speakers),
            }
        }
        created = _request_json("POST", API_BASE, headers=headers_json, timeout=60, json=init_payload)
        job_id_value = created.get("job_id")
        if not isinstance(job_id_value, str) or not job_id_value:
            raise RuntimeError(f"Could not read job_id from init response: {json.dumps(created)[:600]}")
        job_id = job_id_value
        print(f"job_id={job_id}")

    upload_meta = _request_json(
        "POST",
        f"{API_BASE}/upload-files",
        headers=headers_json,
        timeout=60,
        json={"job_id": job_id, "files": [audio_path.name]},
    )
    put_url = _extract_put_url(upload_meta, audio_path.name)

    with audio_path.open("rb") as f:
        put = requests.put(
            put_url,
            data=f,
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "audio/mpeg",
            },
            timeout=300,
        )
    if put.status_code >= 400:
        raise RuntimeError(f"Upload PUT failed ({put.status_code}): {put.text[:500]}")
    print("upload=done")

    _request_json("POST", f"{API_BASE}/{job_id}/start", headers=headers_keyonly, timeout=60)
    print("job=start")

    status = _poll_status(api_key, job_id, float(args.poll_interval), float(args.timeout_s))
    (out_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_files = _collect_output_files(status, audio_path.name)
    print(f"output_files={output_files}")

    dl_info = _request_json(
        "POST",
        f"{API_BASE}/download-files",
        headers=headers_json,
        timeout=60,
        json={"job_id": job_id, "files": output_files},
    )
    download_urls = _extract_download_urls(dl_info)

    saved: list[str] = []
    for fname, dl_url in download_urls.items():
        r = requests.get(dl_url, timeout=180)
        if r.status_code >= 400:
            raise RuntimeError(f"Download failed for {fname} ({r.status_code}): {r.text[:500]}")
        out_file = out_dir / fname
        out_file.write_bytes(r.content)
        saved.append(str(out_file))

    primary_json = out_dir / "0.json"
    if primary_json.exists():
        stt_payload = json.loads(primary_json.read_text(encoding="utf-8"))
        cleaned = build_audio_cleaned_tokens_payload(stt_payload)
        cleaned_path = out_dir / "audio_cleaned_tokens.json"
        cleaned_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        saved.append(str(cleaned_path))

    manifest = {
        "experiment": slug,
        "input_dir": str(input_dir),
        "job_id": job_id,
        "audio": str(audio_path),
        "saved_files": saved,
        "out_dir": str(out_dir),
        "request_payload": {
            "model": args.model,
            "mode": args.mode,
            "language_code": args.language_code,
            "with_timestamps": bool(args.with_timestamps),
            "num_speakers": int(args.num_speakers),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"saved={len(saved)}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
