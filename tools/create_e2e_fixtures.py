#!/usr/bin/env python3
"""Generate 48kHz audio fixtures for Playwright E2E tests.

The meeting WAVs in transcription-service/tests/fixtures/audio/ are 16kHz
(transcription-ready). Browser captures at 48kHz and orchestration downsamples.
Playwright needs 48kHz versions to inject into getUserMedia.

Usage:
    uv run python tools/create_e2e_fixtures.py

Output:
    modules/dashboard-service/tests/fixtures/meeting_en_48k.wav
    modules/dashboard-service/tests/fixtures/meeting_zh_48k.wav
    modules/dashboard-service/tests/fixtures/meeting_es_48k.wav
    modules/dashboard-service/tests/fixtures/meeting_ja_48k.wav
    modules/dashboard-service/tests/fixtures/meeting_mixed_zh_en_48k.wav
"""
from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_DIR = PROJECT_ROOT / "modules" / "transcription-service" / "tests" / "fixtures" / "audio"
OUTPUT_DIR = PROJECT_ROOT / "modules" / "dashboard-service" / "tests" / "fixtures"

# Map of output filename → source filename
FIXTURES = {
    "meeting_en_48k.wav": "meeting_en_long.wav",
    "meeting_zh_48k.wav": "meeting_zh_long.wav",
    "meeting_es_48k.wav": "meeting_es_long.wav",
    "meeting_ja_48k.wav": "meeting_ja_long.wav",
    "meeting_mixed_zh_en_48k.wav": "meeting_mixed_zh_en.wav",
}

# Also copy the JFK fixture (already 48kHz in the transcription fixtures)
JFK_SOURCE = SOURCE_DIR / "jfk.wav"
JFK_OUTPUT = OUTPUT_DIR / "jfk_48k.wav"

TARGET_SR = 48000


def upsample_wav(source: Path, output: Path) -> None:
    """Load a WAV at its native sample rate and resample to 48kHz."""
    audio, sr = librosa.load(str(source), sr=None, mono=True)
    print(f"  {source.name}: {sr}Hz, {len(audio)/sr:.1f}s, {len(audio)} samples")

    if sr == TARGET_SR:
        print(f"  Already {TARGET_SR}Hz — copying as-is")
        resampled = audio
    else:
        resampled = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        print(f"  Resampled {sr}Hz → {TARGET_SR}Hz: {len(resampled)} samples")

    # Write as 16-bit PCM WAV (compact, browser-compatible)
    sf.write(str(output), resampled, TARGET_SR, subtype="PCM_16")
    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"  Written: {output.name} ({size_mb:.1f} MB)")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    errors = []
    for output_name, source_name in FIXTURES.items():
        source = SOURCE_DIR / source_name
        output = OUTPUT_DIR / output_name
        if not source.exists():
            print(f"SKIP: {source} not found")
            errors.append(source_name)
            continue
        print(f"\nProcessing {source_name} → {output_name}")
        upsample_wav(source, output)

    # JFK: resample from 16kHz source
    if JFK_SOURCE.exists():
        print(f"\nProcessing jfk.wav → jfk_48k.wav")
        upsample_wav(JFK_SOURCE, JFK_OUTPUT)
    else:
        print(f"SKIP: {JFK_SOURCE} not found")
        errors.append("jfk.wav")

    if errors:
        print(f"\nWarning: {len(errors)} source files not found: {errors}")
        print("Run from project root. Source audio may need to be generated first.")
        return 1

    successful = len(FIXTURES) - len(errors) + (1 if JFK_SOURCE.exists() else 0)
    print(f"\nDone! {successful} fixture(s) written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
