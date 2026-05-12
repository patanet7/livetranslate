"""Stage 2 — replay a captured JSONL trace through TunableDetector.

Pure Python, no audio, no services. Designed to run thousands of times per
sweep in seconds. Loads a trace + ground truth, returns a RunResult.

CLI:
    uv run python -m benchmarks.lang_detect.replay \\
        --trace benchmarks/lang_detect/fixtures/zh_short_60s.jsonl \\
        --ground-truth benchmarks/lang_detect/fixtures/ground_truth.yaml \\
        --confidence-margin 0.2 --min-dwell-frames 4 --min-dwell-ms 10000
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml  # type: ignore

from .detector_variants import TunableDetector
from .scoring import score_run
from .types import DetectorParams, FixtureTrace, FrameTrace, GroundTruthSegment, RunResult


def load_trace(jsonl_path: Path) -> list[FrameTrace]:
    """Read a captured JSONL trace into typed FrameTrace records."""
    frames: list[FrameTrace] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            frames.append(FrameTrace(
                t_ms=d["t_ms"],
                chunk_dur_s=d["chunk_dur_s"],
                language=d["language"],
                confidence=d["confidence"],
                text=d.get("text", ""),
                no_speech_prob=d.get("no_speech_prob"),
                audio_rms=d.get("audio_rms"),
            ))
    return frames


def load_ground_truth(yaml_path: Path, fixture_id: str) -> list[GroundTruthSegment]:
    """Load labelled segments for ``fixture_id`` from a ground_truth.yaml."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    fixtures = data.get("fixtures", {})
    if fixture_id not in fixtures:
        raise KeyError(f"fixture_id {fixture_id!r} not in {yaml_path}")
    raw_segs = fixtures[fixture_id].get("segments", [])
    return [
        GroundTruthSegment(
            t_ms_start=float(s["t_ms_start"]),
            t_ms_end=float(s["t_ms_end"]),
            language=str(s["language"]),
        )
        for s in raw_segs
    ]


def run_replay(fixture: FixtureTrace, params: DetectorParams) -> RunResult:
    """Feed ``fixture.frames`` through a TunableDetector(params); score the result."""
    detector = TunableDetector(params)
    states: list[tuple[FrameTrace, str | None, bool]] = []
    for frame in fixture.frames:
        lang, switched = detector.ingest(frame)
        states.append((frame, lang, switched))
    return score_run(fixture, states, params)


def _serialise_result(r: RunResult) -> dict:
    """Convert RunResult → plain dict for JSON/TSV output (replaces dataclass-in-dataclass)."""
    d = asdict(r)
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True, help="Path to captured JSONL")
    ap.add_argument("--ground-truth", type=Path, required=True, help="ground_truth.yaml")
    ap.add_argument("--fixture-id", type=str, default=None,
                    help="Defaults to trace filename stem.")
    ap.add_argument("--confidence-margin", type=float, default=0.2)
    ap.add_argument("--min-dwell-frames", type=int, default=4)
    ap.add_argument("--min-dwell-ms", type=float, default=10_000.0)
    ap.add_argument("--initial-confidence-threshold", type=float, default=0.0)
    ap.add_argument("--script-tiebreaker", action="store_true")
    args = ap.parse_args()

    fixture_id = args.fixture_id or args.trace.stem
    frames = load_trace(args.trace)
    gt = load_ground_truth(args.ground_truth, fixture_id)
    fixture = FixtureTrace(
        fixture_id=fixture_id,
        wav_path=args.trace,
        frames=frames,
        ground_truth=gt,
        total_duration_ms=frames[-1].t_ms if frames else 0.0,
    )

    params = DetectorParams(
        confidence_margin=args.confidence_margin,
        min_dwell_frames=args.min_dwell_frames,
        min_dwell_ms=args.min_dwell_ms,
        initial_confidence_threshold=args.initial_confidence_threshold,
        script_tiebreaker_enabled=args.script_tiebreaker,
    )

    result = run_replay(fixture, params)
    print(json.dumps(_serialise_result(result), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
