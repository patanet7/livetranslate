"""Meeting pipeline coordinator.

Ties together: audio source → recording fork → downsampling →
transcription WebSocket → translation → DB persistence → frontend broadcast.

Handles both ephemeral (stream-through) and active meeting modes.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import numpy as np
from livetranslate_common.logging import get_logger

from meeting.downsampler import downsample_to_16k, normalize_audio_shape
from meeting.recorder import FlacChunkRecorder
from meeting.session_manager import MeetingSessionManager

logger = get_logger()


class MeetingPipeline:
    """Coordinates the meeting audio pipeline for a single session.

    In ephemeral mode: audio passes through to transcription only.
    In meeting mode: audio is also recorded to FLAC and persisted to DB.
    """

    def __init__(
        self,
        session_manager: MeetingSessionManager,
        recording_base_path: Path,
        source_type: str = "loopback",
        sample_rate: int = 48000,
        channels: int = 1,
        auto_record: bool = True,
    ):
        self.session_manager = session_manager
        self.recording_base_path = recording_base_path
        self.source_type = source_type
        self.sample_rate = sample_rate
        self.channels = channels
        self.auto_record = auto_record

        self._session_id: uuid.UUID | None = None
        self._recorder: FlacChunkRecorder | None = None
        self._is_meeting: bool = False
        self._running: bool = False
        # Monotonic timestamp of the last DB heartbeat update (throttled to 30 s)
        self._last_heartbeat_at: float = 0.0

    @property
    def session_id(self) -> uuid.UUID | None:
        return self._session_id

    @property
    def is_meeting(self) -> bool:
        return self._is_meeting

    async def start(self) -> uuid.UUID:
        """Create an ephemeral session and begin accepting audio."""
        session = await self.session_manager.create_session(
            source_type=self.source_type,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        self._session_id = session.id
        self._running = True

        if self.auto_record:
            self._recorder = FlacChunkRecorder(
                session_id=str(self._session_id),
                base_path=self.recording_base_path,
                sample_rate=self.sample_rate,
                channels=self.channels,
            )
            self._recorder.start()
            self._is_meeting = True

        logger.info(
            "pipeline_started",
            session_id=str(session.id),
            mode="meeting" if self.auto_record else "ephemeral",
        )
        return session.id

    async def promote_to_meeting(self) -> None:
        """Promote the current ephemeral session to a full recording meeting."""
        if self._session_id is None:
            raise RuntimeError("Cannot promote: no active session")

        await self.session_manager.promote_to_meeting(self._session_id)

        self._recorder = FlacChunkRecorder(
            session_id=str(self._session_id),
            base_path=self.recording_base_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        self._recorder.start()
        self._is_meeting = True

        logger.info("pipeline_promoted", session_id=str(self._session_id))

    async def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process an incoming audio chunk.

        Steps:
        1. If meeting mode: record at native quality via FLAC recorder.
        2. Downsample to 16kHz mono for forwarding to the transcription service.
        3. Throttled heartbeat DB update (at most once per 30 s).

        Returns:
            Downsampled 16kHz mono float32 array ready for transcription.
        """
        if not self._running:
            return np.array([], dtype=np.float32)

        normalized_audio = normalize_audio_shape(audio, channels=self.channels)

        # Fork 1 — Record native quality (meeting mode only)
        if self._is_meeting and self._recorder is not None:
            self._recorder.write(normalized_audio)

        # Fork 2 — Downsample for transcription
        downsampled = downsample_to_16k(
            normalized_audio,
            source_rate=self.sample_rate,
            channels=self.channels,
        )

        # Throttled heartbeat: update DB at most once every 30 seconds
        if self._session_id is not None and self._is_meeting:
            now = time.monotonic()
            if now - self._last_heartbeat_at > 30.0:
                await self.session_manager.update_heartbeat(self._session_id)
                self._last_heartbeat_at = now

        return downsampled

    async def end(self) -> None:
        """End the session, flush recordings, and mark it completed in the DB."""
        if self._recorder is not None:
            self._recorder.stop()
            self._recorder = None

        if self._session_id is not None:
            if self._is_meeting:
                await self.session_manager.end_meeting(self._session_id)
            else:
                await self.session_manager.discard_session(self._session_id)

        self._running = False
        self._is_meeting = False
        logger.info("pipeline_ended", session_id=str(self._session_id))
