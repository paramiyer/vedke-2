from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _require_tool(name: str, hint: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required tool '{name}'. {hint}")


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


def _download_mp3(url: str, out_mp3: Path, overwrite: bool) -> None:
    out_tmpl = str(out_mp3.with_suffix(".%(ext)s"))
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--no-playlist",
        "--output",
        out_tmpl,
    ]
    if overwrite:
        cmd.append("--force-overwrites")
    else:
        cmd.append("--no-overwrites")
    cmd.append(url)

    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed:\n{result.stderr.strip()}")
    if not out_mp3.exists():
        raise RuntimeError(f"Expected mp3 output not found: {out_mp3}")


def _convert_wav_16k_mono(in_mp3: Path, out_wav: Path, overwrite: bool) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    cmd.extend(
        [
            "-i",
            str(in_mp3),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out_wav),
        ]
    )
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed:\n{result.stderr.strip()}")
    if not out_wav.exists():
        raise RuntimeError(f"Expected wav output not found: {out_wav}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepare-asr-audio",
        description="Download YouTube audio into data/<experiment>/input as mp3 + 16kHz mono wav.",
    )
    parser.add_argument("--experiment", required=True, help="Experiment name used as folder slug under data/")
    parser.add_argument("--url", required=True, help="YouTube URL")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _require_tool("yt-dlp", "Install it (e.g. `brew install yt-dlp`).")
    _require_tool("ffmpeg", "Install it (e.g. `brew install ffmpeg`).")

    slug = _slugify(args.experiment)
    input_dir = Path("data") / slug / "input"
    input_dir = input_dir.expanduser().resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    mp3_path = input_dir / "audio.mp3"
    wav_path = input_dir / "audio.16k.wav"
    yt_link_path = input_dir / "yt_link.txt"

    _download_mp3(url=args.url, out_mp3=mp3_path, overwrite=bool(args.overwrite))
    _convert_wav_16k_mono(in_mp3=mp3_path, out_wav=wav_path, overwrite=bool(args.overwrite))
    yt_link_path.write_text(f"{args.url.strip()}\n", encoding="utf-8")

    print(f"experiment={slug}")
    print(f"input_dir={input_dir}")
    print(f"yt_link={yt_link_path}")
    print(f"mp3={mp3_path}")
    print(f"wav_16k_mono={wav_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
