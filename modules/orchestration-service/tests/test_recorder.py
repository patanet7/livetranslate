"""Tests for FLAC chunk recorder — crash-safe continuous recording."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure the meeting package resolves correctly from the tests directory
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meeting.recorder import FlacChunkRecorder


class TestFlacChunkRecorder:
    @pytest.fixture
    def rec_dir(self, tmp_path):
        return tmp_path / "recordings" / "test-session"

    def test_start_creates_directory(self, rec_dir):
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=48000,
            channels=2,
            chunk_duration_s=5,
        )
        recorder.start()
        assert rec_dir.exists()
        manifest = json.loads((rec_dir / "manifest.json").read_text())
        assert manifest["session_id"] == "test-session"
        assert manifest["sample_rate"] == 48000
        recorder.stop()

    def test_write_chunk_creates_flac(self, rec_dir):
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=48000,
            channels=1,
            chunk_duration_s=1,
        )
        recorder.start()

        # Write 1 second of audio (48000 samples at 48kHz)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        recorder.write(audio)

        recorder.stop()

        flac_files = list(rec_dir.glob("chunk_*.flac"))
        assert len(flac_files) >= 1

        manifest = json.loads((rec_dir / "manifest.json").read_text())
        assert len(manifest["chunks"]) >= 1

    def test_manifest_tracks_samples(self, rec_dir):
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=16000,
            channels=1,
            chunk_duration_s=2,
        )
        recorder.start()

        # Write 3 seconds (should trigger at least 1 full 2-second chunk + partial)
        for _ in range(3):
            audio = np.random.randn(16000).astype(np.float32) * 0.1
            recorder.write(audio)

        recorder.stop()

        manifest = json.loads((rec_dir / "manifest.json").read_text())
        total_samples = sum(c["samples"] for c in manifest["chunks"])
        assert total_samples == 48000  # 3 seconds × 16000

    def test_flac_file_is_readable(self, rec_dir):
        """Written FLAC files must be readable by soundfile."""
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=16000,
            channels=1,
            chunk_duration_s=1,
        )
        recorder.start()
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        recorder.write(audio)
        recorder.stop()

        flac_files = list(rec_dir.glob("chunk_*.flac"))
        assert flac_files
        data, sr = sf.read(str(flac_files[0]))
        assert sr == 16000
        assert len(data) > 0

    def test_stop_is_idempotent(self, rec_dir):
        """Calling stop() twice must not raise."""
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=16000,
            channels=1,
            chunk_duration_s=5,
        )
        recorder.start()
        recorder.stop()
        recorder.stop()  # second call must be a no-op

    def test_write_after_stop_is_ignored(self, rec_dir):
        """Audio written after stop() must not raise and must not create new chunks."""
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=16000,
            channels=1,
            chunk_duration_s=1,
        )
        recorder.start()
        recorder.stop()
        # This must be a silent no-op
        recorder.write(np.zeros(16000, dtype=np.float32))
        manifest = json.loads((rec_dir / "manifest.json").read_text())
        assert manifest["total_samples"] == 0

    def test_chunk_sequence_numbers_are_monotonic(self, rec_dir):
        """Chunk sequence numbers in the manifest must be strictly increasing."""
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=16000,
            channels=1,
            chunk_duration_s=1,
        )
        recorder.start()
        # Write 5 seconds to produce multiple chunks
        for _ in range(5):
            recorder.write(np.random.randn(16000).astype(np.float32) * 0.1)
        recorder.stop()

        manifest = json.loads((rec_dir / "manifest.json").read_text())
        sequences = [c["sequence"] for c in manifest["chunks"]]
        assert sequences == list(range(len(sequences)))
