"""Fixture recorder for capturing live sessions as replayable test data.

Enable with LIVETRANSLATE_RECORD_FIXTURES=1 environment variable.
Output directory: FIXTURE_RECORDING_PATH (default: /tmp/livetranslate/fixture-recordings/)

Produces:
  - <session_id>.wav  — Raw 48kHz audio as received from the browser
  - <session_id>.json — Timestamped segment and translation events (sidecar)

The WAV + JSON pair can be used to replay sessions through Playwright or
backend-only translation playback tests without needing live services.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from livetranslate_common.logging import get_logger

logger = get_logger()

RECORD_FIXTURES = os.getenv("LIVETRANSLATE_RECORD_FIXTURES", "").strip() in ("1", "true", "yes")
RECORDING_PATH = Path(os.getenv("FIXTURE_RECORDING_PATH", str(Path.home() / ".livetranslate" / "fixture-recordings")))


class FixtureRecorder:
    """Records a single WebSocket session's audio and events for test replay."""

    def __init__(self, session_id: str, sample_rate: int = 48000):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self._start_time = time.monotonic()
        self._audio_frames: list[np.ndarray] = []
        self._events: list[dict] = []
        self._output_dir = RECORDING_PATH
        self._stopped = False

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start_time) * 1000)

    def write_audio(self, audio: np.ndarray) -> None:
        """Record a raw audio frame (before downsampling)."""
        if self._stopped:
            return
        self._audio_frames.append(audio.copy())

    def log_event(self, event_type: str, data: dict) -> None:
        """Record a segment or translation event with timestamp."""
        if self._stopped:
            return
        self._events.append({
            "t_ms": self._elapsed_ms(),
            "type": event_type,
            "data": data,
        })

    def stop(self) -> Path | None:
        """Finalize recording: write WAV + JSON sidecar. Returns output directory."""
        if self._stopped:
            return None
        self._stopped = True

        if not self._audio_frames and not self._events:
            logger.debug("fixture_recorder_empty", session_id=self.session_id)
            return None

        self._output_dir.mkdir(parents=True, exist_ok=True)

        wav_path = self._output_dir / f"{self.session_id}.wav"
        json_path = self._output_dir / f"{self.session_id}.json"

        # Write audio
        if self._audio_frames:
            combined = np.concatenate(self._audio_frames)
            sf.write(str(wav_path), combined, self.sample_rate, subtype="PCM_16")
            duration_s = len(combined) / self.sample_rate
            logger.info(
                "fixture_audio_written",
                session_id=self.session_id,
                path=str(wav_path),
                duration_s=round(duration_s, 1),
                frames=len(self._audio_frames),
            )

        # Write event sidecar
        sidecar = {
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "duration_ms": self._elapsed_ms(),
            "events": self._events,
        }
        json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
        logger.info(
            "fixture_sidecar_written",
            session_id=self.session_id,
            path=str(json_path),
            event_count=len(self._events),
        )

        return self._output_dir
