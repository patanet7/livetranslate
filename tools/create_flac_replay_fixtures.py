#!/usr/bin/env python3
"""Convert FLAC recording chunks to 48kHz WAV fixtures for Playwright E2E tests.

Reads a recording manifest, extracts audio at specified offsets/durations,
resamples to 48kHz (browser sample rate), and writes WAV fixtures.

Usage:
    uv run python tools/create_flac_replay_fixtures.py \\
        --session af5b37c9 \\
        --output modules/dashboard-service/tests/fixtures/

Generates:
    lang_detect_en_full_48k.wav         — Short English session, full
    lang_detect_mixed_start_48k.wav     — Long meeting, first 3 minutes
    lang_detect_zh_section_48k.wav      — Long meeting, ~17:00-20:00 (Chinese section)
    lang_detect_transition_48k.wav      — Long meeting, ~16:00-21:00 (en→zh transition)
    lang_detect_full_meeting_48k.wav    — Long meeting, first 10 minutes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

RECORDINGS_DIR = Path(os.getenv("RECORDING_BASE_PATH", str(Path.home() / ".livetranslate" / "recordings")))
DEFAULT_OUTPUT = Path("modules/dashboard-service/tests/fixtures")

# Fixture definitions: (name, session_prefix, start_s, duration_s)
# Available sessions (all zh, recorded 2026-03-17):
#   d4abc22a — ~4.4min Chinese
#   e76e7657 — ~4.9min Chinese
#   3e653c07 — ~1min Chinese
FIXTURES = [
    ("lang_detect_zh_short", "3e653c07", 0, 60),           # Short zh, full session
    ("lang_detect_zh_section", "d4abc22a", 0, 180),        # 3min zh from first session
    ("lang_detect_zh_full", "e76e7657", 0, 290),           # ~5min zh, longest session
    # TODO: record English and mixed en↔zh sessions, then add:
    # ("lang_detect_en_full", "<en_session>", 0, 120),
    # ("lang_detect_transition", "<mixed_session>", <offset>, 300),
]


def find_recording(session_prefix: str) -> Path | None:
    if not RECORDINGS_DIR.exists():
        return None
    for d in RECORDINGS_DIR.iterdir():
        if d.name.startswith(session_prefix) and d.is_dir():
            return d
    return None


def load_recording_slice(
    recording_dir: Path, start_s: float, duration_s: float
) -> tuple[np.ndarray, int] | None:
    """Load a time slice from a recording directory.

    Returns (audio_float32, sample_rate) or None if not enough audio.
    """
    import soundfile as sf

    manifest_path = recording_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    sample_rate = manifest["sample_rate"]
    chunks = sorted(manifest["chunks"], key=lambda c: c["sequence"])

    # Calculate chunk boundaries
    chunk_duration_s = manifest.get("chunk_duration_s", 30.0)  # typical FLAC chunk length

    # Load all chunks that overlap with our time window
    start_chunk = int(start_s / chunk_duration_s)
    end_s = start_s + duration_s
    end_chunk = int(end_s / chunk_duration_s) + 1

    audio_parts = []
    for chunk_info in chunks[start_chunk:end_chunk]:
        chunk_path = recording_dir / chunk_info["filename"]
        if chunk_path.exists():
            data, sr = sf.read(str(chunk_path))
            if data.ndim == 2:
                data = data.mean(axis=1)
            audio_parts.append(data.astype(np.float32))

    if not audio_parts:
        return None

    full_audio = np.concatenate(audio_parts)

    # Trim to exact time window (relative to start_chunk)
    chunk_start_s = start_chunk * chunk_duration_s
    offset_in_audio = int((start_s - chunk_start_s) * sample_rate)
    samples_needed = int(duration_s * sample_rate)
    sliced = full_audio[offset_in_audio : offset_in_audio + samples_needed]

    if len(sliced) < sample_rate:  # less than 1 second
        return None

    return sliced, sample_rate


def resample_to_48k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 48kHz for browser playback."""
    if orig_sr == 48000:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=48000)
    except ImportError:
        # Nearest-neighbor fallback
        ratio = 48000 / orig_sr
        indices = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
        return audio[indices]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 48000) -> None:
    """Write 16-bit PCM WAV."""
    import soundfile as sf
    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Create FLAC replay fixtures for Playwright")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="List fixtures without creating")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0

    for name, session_prefix, start_s, duration_s in FIXTURES:
        out_path = args.output / f"{name}_48k.wav"
        recording_dir = find_recording(session_prefix)

        if recording_dir is None:
            print(f"  SKIP {name}: recording {session_prefix}* not found")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  WOULD CREATE {out_path} ({duration_s}s from {session_prefix} @ {start_s}s)")
            continue

        result = load_recording_slice(recording_dir, start_s, duration_s)
        if result is None:
            print(f"  SKIP {name}: not enough audio in slice")
            skipped += 1
            continue

        audio, sr = result
        audio_48k = resample_to_48k(audio, sr)
        write_wav(out_path, audio_48k)
        print(f"  CREATED {out_path} ({len(audio_48k) / 48000:.1f}s)")
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
