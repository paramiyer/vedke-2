# Segmenter Tuning

## Objective
The tuner runs deterministic sentence segmentation repeatedly and chooses the run with the lowest quality score.

## Pre-segmentation Filtering
Before segmentation, sentences are filtered out when likely non-chant/English:
- ASCII-heavy: `ascii_ratio >= 0.25`
- Email/URL pattern detected
- No Devanagari but alphabetic chars present
- Numeric-only marker (after removing danda/punctuation/whitespace)

Artifacts:
- `filter_report.json` includes removed rows, reasons, and filtered-to-original mapping.

## Quality Score (lower is better)
From each run:
- `conf_mean = mean(confidence)`
- `conf_p10 = 10th percentile(confidence)`
- `zero_silence_frac = frac(silence_width_ms == 0)`
- `short_frac = frac(duration_ms <= min_seg_ms + 150)`
- `spike_p95 = p95(abs(drift_jump_ms))`

Score:

`score = 0.35*zero_silence_frac + 0.25*(1-conf_mean) + 0.15*(1-conf_p10) + 0.15*(spike_p95/8000) + 0.10*short_frac`

## Best-case Targets
A run is considered strong when:
- `conf_mean >= 0.60`
- `conf_p10 >= 0.40`
- `zero_silence_frac <= 0.10`
- `spike_p95 <= 6000`
- `short_frac <= 0.15`

## Parameter Sweep
Grid:
- `min_silence_boundary_ms ∈ {300, 450, 600}`
- `min_seg_ms ∈ {900, 1500, 2000}`
- `fill_gap_ms ∈ {250, 350}`
- `thr_pct ∈ {20, 25, 30}`

The tuner evaluates configurations in deterministic order and stops at `--max_runs`.

## Stop Criteria
Early stop if:
1. Current run meets best-case targets, or
2. Best-score improvement over last 3 completed runs is < 3%.

## Outputs
- `out/runs/run_XXX/*` for each attempt
- `out/best_run/*` copied artifacts from best run
- `out/best_run/best_config.json`
- `out/tuning_summary.json` with all run summaries
