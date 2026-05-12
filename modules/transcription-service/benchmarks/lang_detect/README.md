# Language Detection Tuning Harness

Two-stage offline framework for tuning `WhisperLanguageDetector` against
real audio without booting the full stack on every iteration.

## Motivation

Production `WhisperLanguageDetector` (in `src/language_detection.py`) uses
hysteresis to prevent flapping: `confidence_margin=0.2`, `min_dwell_frames=4`,
`min_dwell_ms=10000`. The unit tests in `tests/unit/test_sustained_detector.py`
verify state-machine correctness with very different parameters
(`min_dwell_ms=250`). Nothing currently tests whether the *production
calibration* recovers from a wrong initial Whisper LID guess — the failure
mode that surfaces as "captions appear but no translation, no error".

This harness lets us:

1. Capture real Whisper LID traces from real audio (Stage 1).
2. Replay those traces through the detector with arbitrary params (Stage 2).
3. Sweep parameter grids in seconds, scored against ground-truth labels (Stage 3).
4. Test proposed code variants as opt-in toggles without modifying
   production until the data supports it.

## Layout

```
benchmarks/lang_detect/
├── __init__.py
├── types.py                 # FrameTrace, GroundTruthSegment, DetectorParams, RunResult
├── detector_variants.py     # TunableDetector — wraps WhisperLanguageDetector + proposed fixes
├── scoring.py               # ground-truth comparison → metrics
├── capture.py               # Stage 1: WAV → JSONL trace (needs vllm-mlx :8005)
├── replay.py                # Stage 2: JSONL + params → RunResult (pure Python)
├── sweep.py                 # Stage 3: grid over params × fixtures → ranked TSV
├── sweep_config.yaml        # default starter grid
├── fixtures/
│   ├── ground_truth.yaml    # per-fixture labelled language spans
│   ├── *.jsonl              # captured traces (committed)
└── results/                 # ranked TSV outputs (gitignored)
```

## Workflow

### One-time: install soundfile + scipy + httpx

Already in the workspace; if a fresh worktree:

```bash
uv sync --all-packages --group dev
```

### Stage 1 — Capture (per fixture, ~real-time playback duration)

Needs `vllm-mlx serve` on `:8005` with `large-v3-turbo` loaded
(`just dev-stt` works).

```bash
uv run python -m benchmarks.lang_detect.capture \
    --wav modules/dashboard-service/tests/fixtures/lang_detect_zh_short_48k.wav \
    --out modules/transcription-service/benchmarks/lang_detect/fixtures/zh_short_60s.jsonl
```

Output: a JSONL file with one record per Whisper inference window:

```json
{"t_ms": 0.0, "chunk_dur_s": 3.5, "language": "zh", "confidence": 0.82,
 "text": "...", "no_speech_prob": 0.01, "audio_rms": 0.12}
```

Commit the JSONL — Stage 2 is then fully offline.

### Stage 2 — Replay (one combo, <1 s)

Smoke-test a single parameter set:

```bash
uv run python -m benchmarks.lang_detect.replay \
    --trace modules/transcription-service/benchmarks/lang_detect/fixtures/zh_short_60s.jsonl \
    --ground-truth modules/transcription-service/benchmarks/lang_detect/fixtures/ground_truth.yaml \
    --confidence-margin 0.2 --min-dwell-frames 4 --min-dwell-ms 10000
```

Prints a `RunResult` JSON: `time_to_correct_lang_ms`, `false_switches`,
`correct_at_end`, etc.

### Stage 3 — Sweep (Cartesian grid, seconds total)

```bash
uv run python -m benchmarks.lang_detect.sweep \
    --config modules/transcription-service/benchmarks/lang_detect/sweep_config.yaml \
    --output-dir modules/transcription-service/benchmarks/lang_detect/results
```

Outputs `results/sweep_<timestamp>.tsv` (full ranked rows) and appends to
`results/sweep_index.jsonl` for cross-run regression tracking.

## Proposed detector variants (sweepable)

Set as parameters in `sweep_config.yaml` — all default off so the harness
reproduces production behavior unless a fix is explicitly enabled.

1. **`initial_confidence_threshold`** (float, default `0.0`).
   If Whisper's first-chunk LID confidence is below this, `detect_initial`
   is skipped and the detector waits for a better first frame. Avoids the
   "locked on a 0.5-confidence English guess" trap.

2. **`script_tiebreaker_enabled`** (bool, default `false`).
   When Whisper returns low-confidence LID but the transcript text contains
   ≥ `script_tiebreaker_min_ratio` characters from a non-Latin script
   (Han, kana, Hangul, Arabic, Cyrillic), override the LID hint toward the
   script-implied language with confidence 0.8. Catches the "text says
   你好, LID says en" contradiction we observed in production logs.

If either variant produces consistent wins across fixtures, the
corresponding logic gets promoted into `src/language_detection.py` in
a separate PR — with the harness rerun against the same fixtures to
prove no regressions.

## Adding new fixtures

1. Drop the WAV anywhere (`tests/fixtures/` is conventional).
2. Add an entry to `fixtures/ground_truth.yaml` with the labelled spans.
3. Run Stage 1 to capture the JSONL trace.
4. Add the fixture to `sweep_config.yaml` under `fixtures:`.

For transition tests (the actual production failure scenario): concatenate
a known-English clip onto an existing Chinese clip with a clean cut and
label both spans. Or record a fresh session that crosses language boundaries.

## Scoring methodology

Each detector switch is classified into one of six buckets, so the
ranking can punish the *harmful* kind of switch (breaking a correct
state) more than the *helpful* kind (recovering from a wrong initial
lock). Previous versions of this scoring conflated the two and would
have penalised the script-tiebreaker variant for "false switching" when
in fact it was *correcting* a wrong initial detection.

Per-frame metrics:

- **`time_to_correct_lang_ms`** — first frame `t_ms` at which the
  detector's current language equals the ground-truth language. `null`
  if the detector never reaches truth.
- **`correct_at_end`** — bool: final detector state matches truth at
  the last frame.
- **`frames_with_wrong_lang`** — cumulative wrong-state frames. The
  honest "how much did users suffer" metric.

Switch classification (all of these come from inspecting `(from_lang,
to_lang)` at the time of each switch, with `truth_at_t` from ground
truth):

| Bucket | Definition | Severity |
|---|---|---|
| `correct_initial` | First lock-on landed on the correct language | Best |
| `transitions_caught` | Switch matches a real ground-truth language change | Good |
| `correction_switches` | Wrong → right (recovery from bad initial lock) | Good |
| `wrong_initial` | First lock-on landed on the wrong language | Bad |
| `wrong_recovery_switches` | Wrong → still wrong (different wrong language) | Bad |
| `flap_switches` | Right → wrong (broke a correct state) | **Worst** |
| `missed_transitions` | Real transitions the detector never caught | Bad |

Default sort priority (ascending; `False` ranks better than `True` for
bool columns):

1. `flap_switches` — minimize the worst failure mode first.
2. `wrong_initial` — prefer detectors that get the first guess right.
3. `frames_with_wrong_lang` — minimize downstream impact.
4. `time_to_correct_lang_ms` — and finally, recover fast when needed.

Why this order: a detector that locks correctly on frame 1 is strictly
better than one that locks incorrectly and recovers — even though both
end up in the right state. The old sort would have ranked them
equivalent.
