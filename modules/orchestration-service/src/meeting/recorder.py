"""FLAC chunk recorder for crash-safe continuous meeting recording.

Writes audio in fixed-duration FLAC chunks with a manifest file
for crash recovery and gapless concatenation.

Key design:
- Flush-on-write: each chunk is a complete FLAC file (lose at most 1 chunk on crash)
- Manifest updated per chunk (tracks sequence, sample counts, timestamps)
- Sample-exact continuity: monotonic counter, no gaps or overlaps
- Native quality: 48kHz+ stereo, NOT downsampled
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from livetranslate_common.logging import get_logger

logger = get_logger()


class FlacChunkRecorder:
    def __init__(
        self,
        session_id: str,
        base_path: Path,
        sample_rate: int = 48000,
        channels: int = 2,
        chunk_duration_s: float = 30.0,
    ):
        self.session_id = session_id
        self.session_dir = base_path / session_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(chunk_duration_s * sample_rate)

        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._sequence = 0
        self._total_samples = 0
        self._manifest: dict = {}
        self._running = False

    def start(self) -> None:
        """Create the session directory, write the initial manifest, and begin recording."""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = {
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunks": [],
            "total_samples": 0,
        }
        self._write_manifest()
        self._running = True
        logger.info("recorder_started", session_id=self.session_id, path=str(self.session_dir))

    def write(self, audio: np.ndarray) -> None:
        """Append audio samples. Flushes a complete FLAC chunk whenever the buffer is full."""
        if not self._running:
            return

        self._buffer.append(audio)
        self._buffer_samples += len(audio)

        while self._buffer_samples >= self.chunk_samples:
            self._flush_chunk()

    def stop(self) -> None:
        """Flush any remaining buffered audio and finalise recording."""
        if not self._running:
            return

        if self._buffer_samples > 0:
            self._flush_chunk()

        self._running = False
        logger.info(
            "recorder_stopped",
            session_id=self.session_id,
            total_samples=self._total_samples,
            chunks=self._sequence,
        )

    def _flush_chunk(self) -> None:
        """Combine buffered arrays into one FLAC chunk and update the manifest atomically."""
        if not self._buffer:
            return

        combined = np.concatenate(list(self._buffer))
        self._buffer.clear()
        self._buffer_samples = 0

        # If the combined audio exceeds one chunk, keep the overflow for the next flush.
        if len(combined) > self.chunk_samples:
            overflow = combined[self.chunk_samples:]
            combined = combined[: self.chunk_samples]
            self._buffer.append(overflow)
            self._buffer_samples = len(overflow)

        timestamp_ms = int(time.time() * 1000)
        filename = f"chunk_{self._sequence:06d}_{timestamp_ms}.flac"
        filepath = self.session_dir / filename

        # soundfile needs (samples, channels) for multi-channel; (samples,) for mono.
        audio_to_write = combined
        if self.channels > 1 and combined.ndim == 1:
            # Re-shape mono buffer into multi-channel by repeating (best-effort)
            audio_to_write = np.stack([combined] * self.channels, axis=1)

        try:
            sf.write(str(filepath), audio_to_write, self.sample_rate, format="FLAC")
            written_samples = len(combined)
        except OSError as exc:
            logger.error("chunk_write_failed", filename=filename, error=str(exc))
            written_samples = 0

        chunk_info = {
            "sequence": self._sequence,
            "filename": filename,
            "samples": written_samples,
            "timestamp_ms": timestamp_ms,
        }
        self._manifest["chunks"].append(chunk_info)
        self._total_samples += written_samples
        self._manifest["total_samples"] = self._total_samples
        self._write_manifest()

        self._sequence += 1
        if written_samples > 0:
            logger.debug("chunk_flushed", filename=filename, samples=written_samples)

    def _write_manifest(self) -> None:
        """Write manifest.json atomically (write to tmp then rename)."""
        manifest_path = self.session_dir / "manifest.json"
        tmp_path = self.session_dir / "manifest.json.tmp"
        tmp_path.write_text(json.dumps(self._manifest, indent=2))
        tmp_path.replace(manifest_path)
