"""FLAC chunk integrity, manifest continuity, and 4-hour extrapolation tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meeting.recorder import FlacChunkRecorder

from .conftest import generate_audio, generate_audio_chunks

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_recorder(
    tmp_path: Path,
    session_id: str = "test-session",
    sample_rate: int = 48000,
    channels: int = 1,
    chunk_duration_s: float = 30.0,
) -> FlacChunkRecorder:
    rec = FlacChunkRecorder(
        session_id=session_id,
        base_path=tmp_path / "recordings",
        sample_rate=sample_rate,
        channels=channels,
        chunk_duration_s=chunk_duration_s,
    )
    rec.start()
    return rec


def _read_manifest(rec: FlacChunkRecorder) -> dict:
    manifest_path = rec.session_dir / "manifest.json"
    return json.loads(manifest_path.read_text())


# ===========================================================================
# TestFlacChunkIntegrity
# ===========================================================================


class TestFlacChunkIntegrity:
    """Each flushed FLAC chunk is a self-contained valid file."""

    def test_each_chunk_is_valid_flac(self, tmp_path: Path):
        """Write 90s audio → 3 chunks of 30s each, each readable by soundfile."""
        rec = _make_recorder(tmp_path, chunk_duration_s=30.0)
        for chunk in generate_audio_chunks(90.0, chunk_s=5.0):
            rec.write(chunk)
        rec.stop()

        manifest = _read_manifest(rec)
        assert len(manifest["chunks"]) == 3

        for entry in manifest["chunks"]:
            path = rec.session_dir / entry["filename"]
            info = sf.info(str(path))
            assert info.samplerate == 48000
            assert info.frames == entry["samples"]

    def test_manifest_updated_atomically(self, tmp_path: Path):
        """After each flush, manifest is valid JSON with no .tmp residue."""
        rec = _make_recorder(tmp_path, chunk_duration_s=30.0)
        audio = generate_audio(60.0)
        rec.write(audio)
        rec.stop()

        # No temp files left behind
        tmp_files = list(rec.session_dir.glob("*.tmp"))
        assert tmp_files == [], f"Leftover tmp files: {tmp_files}"

        manifest = _read_manifest(rec)
        assert isinstance(manifest["chunks"], list)
        assert len(manifest["chunks"]) == 2

    def test_simulated_crash_last_chunk_consistent(self, tmp_path: Path):
        """Write 5s into a 2s-chunk recorder, skip stop(), verify manifest has exactly 2 chunks."""
        rec = _make_recorder(tmp_path, chunk_duration_s=2.0)
        audio = generate_audio(5.0)
        rec.write(audio)
        # Deliberately do NOT call rec.stop() — simulates crash

        manifest = _read_manifest(rec)
        # 5s / 2s = 2 full chunks flushed, ~1s buffered (lost on crash)
        assert len(manifest["chunks"]) == 2
        for entry in manifest["chunks"]:
            path = rec.session_dir / entry["filename"]
            assert path.exists()
            _data, sr = sf.read(str(path))
            assert sr == 48000

    def test_buffer_overflow_single_large_write(self, tmp_path: Path):
        """Write 5s in one call with 1s chunks → 5 chunks, no sample loss."""
        rec = _make_recorder(tmp_path, chunk_duration_s=1.0)
        audio = generate_audio(5.0)
        rec.write(audio)
        rec.stop()

        manifest = _read_manifest(rec)
        assert len(manifest["chunks"]) == 5
        assert manifest["total_samples"] == len(audio)

    def test_stereo_channels_preserved(self, tmp_path: Path):
        """2-channel FLAC verified via soundfile.info."""
        rec = _make_recorder(tmp_path, channels=2, chunk_duration_s=1.0)
        audio = generate_audio(1.5)  # Mono signal; recorder reshapes to stereo
        rec.write(audio)
        rec.stop()

        manifest = _read_manifest(rec)
        assert len(manifest["chunks"]) >= 1
        path = rec.session_dir / manifest["chunks"][0]["filename"]
        info = sf.info(str(path))
        assert info.channels == 2


# ===========================================================================
# TestManifestContinuity
# ===========================================================================


class TestManifestContinuity:
    """Manifest accurately tracks sample-exact continuity."""

    def test_gapless_sample_count(self, tmp_path: Path):
        """Sum of chunk samples == total_samples == N * sample_rate."""
        duration_s = 10.0
        rec = _make_recorder(tmp_path, chunk_duration_s=2.0)
        audio = generate_audio(duration_s)
        rec.write(audio)
        rec.stop()

        manifest = _read_manifest(rec)
        chunk_sum = sum(c["samples"] for c in manifest["chunks"])
        assert chunk_sum == manifest["total_samples"]
        assert chunk_sum == len(audio)

    def test_chunk_timestamps_monotonically_increasing(self, tmp_path: Path):
        """Each chunk's timestamp_ms is >= the previous."""
        rec = _make_recorder(tmp_path, chunk_duration_s=1.0)
        for chunk in generate_audio_chunks(5.0, chunk_s=1.0):
            rec.write(chunk)
        rec.stop()

        manifest = _read_manifest(rec)
        timestamps = [c["timestamp_ms"] for c in manifest["chunks"]]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_manifest_enables_gapless_reconstruction(self, tmp_path: Path):
        """Read all chunks and concatenate — total length matches original."""
        duration_s = 6.0
        rec = _make_recorder(tmp_path, chunk_duration_s=2.0)
        audio = generate_audio(duration_s)
        rec.write(audio)
        rec.stop()

        manifest = _read_manifest(rec)
        reconstructed_parts = []
        for entry in manifest["chunks"]:
            path = rec.session_dir / entry["filename"]
            data, _ = sf.read(str(path), dtype="float32")
            reconstructed_parts.append(data)

        reconstructed = np.concatenate(reconstructed_parts)
        assert len(reconstructed) == len(audio)


# ===========================================================================
# TestFourHourExtrapolation
# ===========================================================================


@pytest.mark.slow
class TestFourHourExtrapolation:
    """Extrapolate 4-hour recording behaviour from smaller writes."""

    def test_four_hour_chunk_count_and_numbering(self, tmp_path: Path):
        """480 chunk-sized writes → 480 manifest entries, sequences [0..479]."""
        chunk_duration_s = 30.0
        num_chunks = 480  # 4 hours at 30s each
        rec = _make_recorder(tmp_path, chunk_duration_s=chunk_duration_s)

        chunk_audio = generate_audio(chunk_duration_s)
        for _ in range(num_chunks):
            rec.write(chunk_audio.copy())
        rec.stop()

        manifest = _read_manifest(rec)
        assert len(manifest["chunks"]) == num_chunks
        sequences = [c["sequence"] for c in manifest["chunks"]]
        assert sequences == list(range(num_chunks))

    def test_manifest_file_size_bounded(self, tmp_path: Path):
        """480 chunks → manifest under 200 KB."""
        chunk_duration_s = 30.0
        num_chunks = 480
        rec = _make_recorder(tmp_path, chunk_duration_s=chunk_duration_s)

        chunk_audio = generate_audio(chunk_duration_s)
        for _ in range(num_chunks):
            rec.write(chunk_audio.copy())
        rec.stop()

        manifest_path = rec.session_dir / "manifest.json"
        size_kb = manifest_path.stat().st_size / 1024
        assert size_kb < 200, f"Manifest is {size_kb:.1f} KB, expected < 200 KB"

    def test_four_hour_disk_estimate(self, tmp_path: Path):
        """Estimate and log projected 4h disk usage from a single chunk."""
        rec = _make_recorder(tmp_path, channels=2, chunk_duration_s=30.0)
        audio = generate_audio(30.0)
        rec.write(audio)
        rec.stop()

        manifest = _read_manifest(rec)
        chunk_file = rec.session_dir / manifest["chunks"][0]["filename"]
        chunk_bytes = chunk_file.stat().st_size
        projected_mb = (chunk_bytes * 480) / (1024 * 1024)

        # 48kHz stereo FLAC for 4h should be well under 2 GB
        assert projected_mb < 2048, f"Projected 4h = {projected_mb:.0f} MB, expected < 2048 MB"

    def test_recording_path_structure(self, tmp_path: Path):
        """Verify session_dir layout: {base}/{session_id}/chunk_*.flac + manifest.json."""
        session_id = "my-test-session"
        rec = _make_recorder(tmp_path, session_id=session_id, chunk_duration_s=1.0)
        audio = generate_audio(2.5)
        rec.write(audio)
        rec.stop()

        expected_dir = tmp_path / "recordings" / session_id
        assert expected_dir.exists()
        assert (expected_dir / "manifest.json").exists()

        flac_files = list(expected_dir.glob("chunk_*.flac"))
        assert len(flac_files) >= 2
