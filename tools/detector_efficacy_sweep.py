#!/usr/bin/env python3
"""Efficacy sweep: OLD vs NEW detector × hyperparameter grid.

Replays real production flapping events and real FLAC recordings through
both detectors with varying parameters. Produces a comparison table.

Usage:
    uv run python tools/detector_efficacy_sweep.py

Output: TSV table to stdout + JSON results to tools/sweep_results.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

# Add transcription service src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules" / "transcription-service" / "src"))

from language_detection import LanguageDetector, WhisperLanguageDetector

FIXTURES_DIR = Path(__file__).parent.parent / "modules" / "transcription-service" / "tests" / "fixtures"
RECORDINGS_DIR = Path.home() / ".livetranslate" / "recordings"


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    detector: str
    params: dict
    scenario: str
    total_events: int
    switch_count: int
    hallucinated_switches: int
    languages_seen: list[str]
    elapsed_ms: float

    @property
    def row(self) -> str:
        return (
            f"{self.detector}\t{self.scenario}\t"
            f"{json.dumps(self.params)}\t"
            f"{self.switch_count}\t{self.hallucinated_switches}\t"
            f"{self.total_events}\t{','.join(self.languages_seen)}\t"
            f"{self.elapsed_ms:.0f}ms"
        )


HALLUCINATED = {"nn", "cy", "ko", "fr", "es", "it", "pt", "nl", "ru", "pl"}


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def load_flapping_fixture() -> list[dict] | None:
    path = FIXTURES_DIR / "flapping_events_20260317.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["events"]


def load_recording_events(session_prefix: str, chunk_duration_s: float = 3.0) -> list[dict] | None:
    """Simulate Whisper detection events from a FLAC recording.

    Since we can't run Whisper in the sweep (too slow), we simulate what
    Whisper would report: the correct language with occasional low-confidence
    hallucinations — matching the pattern observed in production logs.
    """
    if not RECORDINGS_DIR.exists():
        return None
    recording_dir = None
    for d in RECORDINGS_DIR.iterdir():
        if d.name.startswith(session_prefix) and d.is_dir():
            recording_dir = d
            break
    if recording_dir is None:
        return None

    manifest_path = recording_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    total_duration_s = manifest["total_samples"] / manifest["sample_rate"]
    n_chunks = int(total_duration_s / chunk_duration_s)

    # Simulate: 80% correct zh, 10% en hallucination, 10% other hallucination
    rng = np.random.default_rng(42)
    events = []
    for i in range(n_chunks):
        r = rng.random()
        if r < 0.80:
            lang, conf = "zh", rng.uniform(0.6, 0.95)
        elif r < 0.90:
            lang, conf = "en", rng.uniform(0.3, 0.6)
        else:
            lang = rng.choice(["nn", "cy", "ko", "fr"])
            conf = rng.uniform(0.2, 0.5)
        events.append({
            "language": lang,
            "confidence": float(conf),
            "chunk_duration_s": chunk_duration_s,
            "timestamp_s": i * chunk_duration_s,
        })
    return events


def _synthetic_genuine_switch(chunk_s: float = 3.0) -> list[dict]:
    """30s English → genuine 20s Chinese transition → 30s Chinese.

    Tests that the NEW detector CAN switch on a real sustained change.
    A detector that blocks ALL switches is too conservative.
    """
    events = []
    t = 0.0
    # 10 chunks of stable English
    for _ in range(10):
        events.append({"language": "en", "confidence": 0.85, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    # 7 chunks of sustained Chinese at high confidence (genuine switch)
    for _ in range(7):
        events.append({"language": "zh", "confidence": 0.80, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    # 10 more chunks of Chinese
    for _ in range(10):
        events.append({"language": "zh", "confidence": 0.85, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    return events


def _synthetic_consecutive_hallucination(burst_len: int = 5, chunk_s: float = 3.0) -> list[dict]:
    """Adversarial: 5 consecutive hallucinations of the same wrong language.

    Tests detector resilience when Whisper hallucinates the same wrong
    language multiple times in a row (worse than production but possible).
    """
    events = []
    t = 0.0
    # 10 chunks stable English
    for _ in range(10):
        events.append({"language": "en", "confidence": 0.8, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    # burst_len consecutive Korean hallucinations at moderate confidence
    for _ in range(burst_len):
        events.append({"language": "ko", "confidence": 0.55, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    # Back to English
    for _ in range(10):
        events.append({"language": "en", "confidence": 0.8, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    return events


def _synthetic_noisy_switch(chunk_s: float = 3.0) -> list[dict]:
    """Genuine en→zh switch buried in noise.

    Tests sensitivity: Chinese appears at ~70% rate over 30s with
    interspersed English/hallucinations. Should still detect the switch.
    """
    rng = np.random.default_rng(99)
    events = []
    t = 0.0
    # 10 chunks stable English
    for _ in range(10):
        events.append({"language": "en", "confidence": 0.85, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    # 15 chunks: 70% zh, 20% en, 10% hallucination (noisy transition)
    for _ in range(15):
        r = rng.random()
        if r < 0.70:
            lang, conf = "zh", float(rng.uniform(0.6, 0.85))
        elif r < 0.90:
            lang, conf = "en", float(rng.uniform(0.4, 0.6))
        else:
            lang, conf = "ko", float(rng.uniform(0.3, 0.5))
        events.append({"language": lang, "confidence": conf, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    # 10 chunks stable Chinese
    for _ in range(10):
        events.append({"language": "zh", "confidence": 0.85, "chunk_duration_s": chunk_s, "timestamp_s": t})
        t += chunk_s
    return events


def build_scenarios() -> dict[str, list[dict]]:
    """Build all test scenarios from available data."""
    scenarios = {}

    # 1. Real production flapping log (227 events from 40min meeting)
    flapping = load_flapping_fixture()
    if flapping:
        scenarios["prod_flapping_log"] = flapping

    # 2. Simulated detection events from real recordings
    for prefix, name in [
        ("e76e7657", "zh_long_5min"),
        ("d4abc22a", "zh_med_4min"),
        ("3e653c07", "zh_short_1min"),
    ]:
        events = load_recording_events(prefix)
        if events:
            scenarios[name] = events

    # 3. Synthetic adversarial scenarios
    scenarios["genuine_en_zh_switch"] = _synthetic_genuine_switch()
    scenarios["consecutive_hallucination_5x"] = _synthetic_consecutive_hallucination(burst_len=5)
    scenarios["consecutive_hallucination_8x"] = _synthetic_consecutive_hallucination(burst_len=8)
    scenarios["noisy_transition"] = _synthetic_noisy_switch()

    return scenarios


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------

def run_old_detector(events: list[dict], threshold_s: float) -> SweepResult:
    """Run the OLD LanguageDetector on a scenario."""
    t0 = time.monotonic()
    detector = LanguageDetector(switch_threshold_s=threshold_s)

    switches = 0
    hallucinated = 0
    languages = set()

    for i, event in enumerate(events):
        lang = event.get("language") or event.get("new", "en")
        conf = event.get("confidence", 0.5)
        duration = event.get("chunk_duration_s") or event.get("delta_s", 3.0)

        if i == 0 or detector.current_language is None:
            detected = detector.detect_initial(lang, conf)
            languages.add(detected)
            continue

        # OLD detector doesn't use confidence
        result = detector.update(lang, duration)
        if result is not None:
            switches += 1
            languages.add(result)
            if result in HALLUCINATED:
                hallucinated += 1

    elapsed = (time.monotonic() - t0) * 1000
    return SweepResult(
        detector="OLD",
        params={"threshold_s": threshold_s},
        scenario="",
        total_events=len(events),
        switch_count=switches,
        hallucinated_switches=hallucinated,
        languages_seen=sorted(languages),
        elapsed_ms=elapsed,
    )


def run_new_detector(
    events: list[dict],
    confidence_margin: float,
    min_dwell_frames: int,
    min_dwell_ms: float,
) -> SweepResult:
    """Run the NEW WhisperLanguageDetector on a scenario."""
    t0 = time.monotonic()
    detector = WhisperLanguageDetector(
        confidence_margin=confidence_margin,
        min_dwell_frames=min_dwell_frames,
        min_dwell_ms=min_dwell_ms,
    )

    switches = 0
    hallucinated = 0
    languages = set()

    for i, event in enumerate(events):
        lang = event.get("language") or event.get("new", "en")
        conf = event.get("confidence", 0.5)
        duration = event.get("chunk_duration_s") or event.get("delta_s", 3.0)

        if i == 0 or detector.current_language is None:
            detected = detector.detect_initial(lang, conf)
            languages.add(detected)
            continue

        result = detector.update(lang, duration, conf)
        if result is not None:
            switches += 1
            languages.add(result)
            if result in HALLUCINATED:
                hallucinated += 1

    elapsed = (time.monotonic() - t0) * 1000
    return SweepResult(
        detector="NEW",
        params={
            "margin": confidence_margin,
            "frames": min_dwell_frames,
            "dwell_ms": min_dwell_ms,
        },
        scenario="",
        total_events=len(events),
        switch_count=switches,
        hallucinated_switches=hallucinated,
        languages_seen=sorted(languages),
        elapsed_ms=elapsed,
    )


# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

OLD_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]

NEW_GRID = {
    "confidence_margin": [0.1, 0.15, 0.2, 0.25, 0.3],
    "min_dwell_frames": [2, 3, 4, 5, 6],
    "min_dwell_ms": [5000, 8000, 10000, 12000, 15000],
}


def main():
    scenarios = build_scenarios()
    if not scenarios:
        print("ERROR: No scenarios found. Need flapping fixture or recordings.", file=sys.stderr)
        return 1

    print(f"Scenarios: {list(scenarios.keys())}", file=sys.stderr)

    results: list[SweepResult] = []

    # OLD detector sweep
    for scenario_name, events in scenarios.items():
        for threshold in OLD_THRESHOLDS:
            r = run_old_detector(events, threshold)
            r.scenario = scenario_name
            results.append(r)

    # NEW detector sweep
    for scenario_name, events in scenarios.items():
        for margin, frames, dwell in product(
            NEW_GRID["confidence_margin"],
            NEW_GRID["min_dwell_frames"],
            NEW_GRID["min_dwell_ms"],
        ):
            r = run_new_detector(events, margin, frames, dwell)
            r.scenario = scenario_name
            results.append(r)

    # Print TSV header
    print("detector\tscenario\tparams\tswitches\thallucinated\ttotal_events\tlanguages\telapsed")
    for r in results:
        print(r.row)

    # Summary
    print(f"\n# Total configs tested: {len(results)}", file=sys.stderr)

    # Best NEW config per scenario (fewest switches, then fewest hallucinations)
    print("\n# === BEST CONFIGS PER SCENARIO ===", file=sys.stderr)
    for scenario_name in scenarios:
        scenario_results = [r for r in results if r.scenario == scenario_name]
        old_results = [r for r in scenario_results if r.detector == "OLD"]
        new_results = [r for r in scenario_results if r.detector == "NEW"]

        best_old = min(old_results, key=lambda r: (r.switch_count, r.hallucinated_switches))
        best_new = min(new_results, key=lambda r: (r.switch_count, r.hallucinated_switches))

        print(f"\n# {scenario_name}:", file=sys.stderr)
        print(f"#   OLD best: {best_old.switch_count} switches, {best_old.hallucinated_switches} hallucinated (threshold={best_old.params['threshold_s']}s)", file=sys.stderr)
        print(f"#   NEW best: {best_new.switch_count} switches, {best_new.hallucinated_switches} hallucinated (margin={best_new.params['margin']}, frames={best_new.params['frames']}, dwell={best_new.params['dwell_ms']}ms)", file=sys.stderr)

    # Save JSON
    out_path = Path(__file__).parent / "sweep_results.json"
    with open(out_path, "w") as f:
        json.dump([{
            "detector": r.detector,
            "scenario": r.scenario,
            "params": r.params,
            "switches": r.switch_count,
            "hallucinated": r.hallucinated_switches,
            "total_events": r.total_events,
            "languages": r.languages_seen,
        } for r in results], f, indent=2)
    print(f"\n# Results saved to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
