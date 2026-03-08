from scripts.run_sarvam_stt import (
    _collect_output_files,
    _extract_download_urls,
    _extract_put_url,
    _split_duration_by_lengths,
    build_audio_cleaned_tokens_payload,
)
import pytest


def test_extract_put_url_from_upload_urls_file_url_shape() -> None:
    payload = {
        "job_id": "job_1",
        "upload_urls": {"audio.mp3": {"file_url": "https://example.com/upload"}},
    }
    assert _extract_put_url(payload, "audio.mp3") == "https://example.com/upload"


def test_collect_output_files_from_job_details() -> None:
    status = {
        "job_state": "Completed",
        "job_details": [{"outputs": [{"file_name": "0.json"}]}],
    }
    assert _collect_output_files(status, "audio.mp3") == ["0.json"]


def test_extract_download_urls_from_mapping() -> None:
    payload = {"download_urls": {"0.json": {"file_url": "https://example.com/0.json"}}}
    urls = _extract_download_urls(payload)
    assert urls == {"0.json": "https://example.com/0.json"}


def test_split_duration_by_lengths_exact_sum() -> None:
    alloc = _split_duration_by_lengths(14170, [3, 5, 21, 5, 21])
    assert sum(alloc) == 14170
    assert alloc == [773, 1288, 5411, 1288, 5410]


def test_build_audio_cleaned_tokens_payload() -> None:
    stt = {
        "request_id": "r1",
        "timestamps": {
            "words": ["ओम् भद्रं"],
            "start_time_seconds": [8.39],
            "end_time_seconds": [9.39],
        },
    }
    out = build_audio_cleaned_tokens_payload(stt)
    assert out["request_id"] == "r1"
    assert out["segments_count"] == 1
    assert out["token_count"] == 2
    t0 = out["tokens"][0]
    t1 = out["tokens"][1]
    assert t0["token"] == "ओम्"
    assert t1["token"] == "भद्रं"
    assert t0["char_count"] == 3
    assert t1["char_count"] == 5
    assert t0["start_ms"] == 8390
    assert t1["end_ms"] == 9390


def test_build_audio_cleaned_tokens_payload_raises_on_empty_words() -> None:
    stt = {"timestamps": {"words": [], "start_time_seconds": [], "end_time_seconds": []}}
    with pytest.raises(RuntimeError, match="timestamps.words is null/empty"):
        build_audio_cleaned_tokens_payload(stt)
