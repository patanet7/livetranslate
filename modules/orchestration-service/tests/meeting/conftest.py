"""Shared fixtures for meeting persistence tests.

All DB fixtures delegate to the root conftest (testcontainers PostgreSQL + Alembic).
Audio generation uses voice-like signals matching the real pipeline (48 kHz mono float32).
"""
from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure src is on path
_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from meeting.pipeline import MeetingPipeline
from meeting.recorder import FlacChunkRecorder
from meeting.session_manager import MeetingSessionManager

# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------


def generate_audio(duration_s: float, sample_rate: int = 48000) -> np.ndarray:
    """Generate voice-like mono float32 audio at the given sample rate."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    signal = 0.3 * np.sin(2 * np.pi * 120 * t)
    signal += 0.2 * np.sin(2 * np.pi * 240 * t)
    signal += 0.1 * np.sin(2 * np.pi * 360 * t)
    signal += 0.1 * np.sin(2 * np.pi * 800 * t)
    signal += 0.05 * np.sin(2 * np.pi * 1200 * t)
    signal += 0.01 * np.random.default_rng(42).standard_normal(len(signal))
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


def generate_audio_chunks(
    total_s: float,
    chunk_s: float = 1.0,
    sample_rate: int = 48000,
) -> Iterator[np.ndarray]:
    """Yield fixed-duration chunks from a single continuous audio signal."""
    full = generate_audio(total_s, sample_rate)
    chunk_samples = int(chunk_s * sample_rate)
    for start in range(0, len(full), chunk_samples):
        chunk = full[start : start + chunk_samples]
        if len(chunk) > 0:
            yield chunk


# ---------------------------------------------------------------------------
# Meeting infrastructure fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def meeting_session_manager(db_session: AsyncSession, tmp_path: Path) -> MeetingSessionManager:
    """Real MeetingSessionManager backed by testcontainer PostgreSQL."""
    return MeetingSessionManager(
        db=db_session,
        recording_base_path=tmp_path / "recordings",
        heartbeat_timeout_s=120,
    )


@pytest.fixture
def meeting_pipeline(
    meeting_session_manager: MeetingSessionManager, tmp_path: Path
) -> MeetingPipeline:
    """Real MeetingPipeline (48 kHz mono) — not yet started."""
    return MeetingPipeline(
        session_manager=meeting_session_manager,
        recording_base_path=tmp_path / "recordings",
        source_type="loopback",
        sample_rate=48000,
        channels=1,
    )


@pytest_asyncio.fixture(loop_scope="function")
async def promoted_pipeline(meeting_pipeline: MeetingPipeline) -> AsyncIterator[MeetingPipeline]:
    """Started + promoted pipeline. Yields the pipeline and calls end() on teardown."""
    await meeting_pipeline.start()
    await meeting_pipeline.promote_to_meeting()
    yield meeting_pipeline
    await meeting_pipeline.end()


@pytest.fixture
def flac_recorder(tmp_path: Path) -> FlacChunkRecorder:
    """Standalone FlacChunkRecorder for FLAC-only tests (no DB)."""
    recorder = FlacChunkRecorder(
        session_id="test-session",
        base_path=tmp_path / "recordings",
        sample_rate=48000,
        channels=1,
        chunk_duration_s=30.0,
    )
    recorder.start()
    return recorder
