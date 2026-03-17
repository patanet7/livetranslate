"""Long session resilience tests — memory boundedness, heartbeat, and storage estimates."""
from __future__ import annotations

import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import Meeting

from meeting.pipeline import MeetingPipeline
from meeting.recorder import FlacChunkRecorder
from meeting.session_manager import MeetingSessionManager

from .conftest import generate_audio, generate_audio_chunks

# ===========================================================================
# TestMemoryBoundedness
# ===========================================================================


class TestMemoryBoundedness:
    """Recorder buffer stays bounded during long sessions."""

    def test_recorder_buffer_stays_bounded(self, tmp_path: Path):
        """Write 600s of 1s chunks at 48 kHz; buffer stays < 2 x chunk_samples."""
        rec = FlacChunkRecorder(
            session_id="bounded-test",
            base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
            chunk_duration_s=30.0,
        )
        rec.start()

        chunk_audio = generate_audio(1.0)
        for i in range(600):
            rec.write(chunk_audio.copy())
            # After each batch of 30 writes, check buffer isn't growing unbounded
            if (i + 1) % 30 == 0:
                assert rec._buffer_samples < 2 * rec.chunk_samples, (
                    f"Buffer at {rec._buffer_samples} samples after {i + 1}s, "
                    f"expected < {2 * rec.chunk_samples}"
                )
        rec.stop()

    def test_segment_id_to_chunk_mapping_concept(self):
        """Verify that a bounded dict (LRU-style eviction) stays bounded.

        This tests the concept used in websocket_audio.py where we track
        segment_id → chunk_id mappings. The dict should not grow unboundedly.
        """
        mapping: dict[int, uuid.UUID] = {}
        max_size = 200

        for i in range(1000):
            mapping[i] = uuid.uuid4()
            # Evict oldest entries when over max size
            if len(mapping) > max_size:
                oldest_keys = sorted(mapping.keys())[: len(mapping) - max_size]
                for k in oldest_keys:
                    del mapping[k]

        assert len(mapping) <= max_size


# ===========================================================================
# TestHeartbeatKeepsSessionAlive
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestHeartbeatKeepsSessionAlive:
    """Heartbeat prevents orphan detection; stale sessions are detected."""

    async def test_heartbeat_prevents_orphan_detection(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        await meeting_session_manager.promote_to_meeting(session.id)

        # Update heartbeat — session should be fresh
        await meeting_session_manager.update_heartbeat(session.id)

        orphans = await meeting_session_manager.detect_orphans()
        assert len(orphans) == 0

    async def test_orphan_detection_only_kills_stale(
        self, db_session: AsyncSession, tmp_path: Path
    ):
        """Two sessions: one stale (200s ago), one fresh. Only stale is detected."""
        mgr = MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

        fresh = await mgr.create_session("loopback")
        await mgr.promote_to_meeting(fresh.id)
        await mgr.update_heartbeat(fresh.id)

        stale = await mgr.create_session("loopback")
        await mgr.promote_to_meeting(stale.id)
        # Manually backdate last_activity_at
        from sqlalchemy import update as sa_update
        await db_session.execute(
            sa_update(Meeting)
            .where(Meeting.id == stale.id)
            .values(last_activity_at=datetime.now(UTC) - timedelta(seconds=200))
        )
        await db_session.commit()

        orphans = await mgr.detect_orphans()
        orphan_ids = {o.id for o in orphans}
        assert stale.id in orphan_ids
        assert fresh.id not in orphan_ids

    async def test_pipeline_heartbeat_throttle(
        self, meeting_session_manager: MeetingSessionManager, tmp_path: Path
    ):
        """Two rapid process_audio() calls; _last_heartbeat_at updates only once."""
        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        audio = generate_audio(0.1)
        await pipeline.process_audio(audio)
        first_hb = pipeline._last_heartbeat_at

        await pipeline.process_audio(audio)
        second_hb = pipeline._last_heartbeat_at

        # Both calls happen within the 30s window, so heartbeat shouldn't update twice
        assert first_hb == second_hb

        await pipeline.end()


# ===========================================================================
# TestCleanEndAfterLongDuration
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestCleanEndAfterLongDuration:
    """Pipeline end() after many chunks sets correct DB state."""

    async def test_end_after_many_chunks(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession, tmp_path: Path,
    ):
        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # Write 60s of audio in 1s chunks
        for chunk in generate_audio_chunks(60.0, chunk_s=1.0):
            await pipeline.process_audio(chunk)

        await pipeline.end()

        session = await db_session.get(Meeting, pipeline.session_id)
        assert session.status == "completed"
        assert session.ended_at is not None

    @pytest.mark.slow
    async def test_db_query_performance_with_many_chunks(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession,
    ):
        """Insert 1000 chunks, time recover_untranslated(), assert < 2s."""
        session = await meeting_session_manager.create_session("loopback")
        await meeting_session_manager.promote_to_meeting(session.id)

        for i in range(1000):
            await meeting_session_manager.add_transcript(
                session_id=session.id,
                text=f"Chunk {i}: some transcript text here",
                timestamp_ms=i * 1000,
                language="en",
                confidence=0.9,
                is_final=True,
            )

        start = time.monotonic()
        untranslated = await meeting_session_manager.recover_untranslated()
        elapsed = time.monotonic() - start

        assert len(untranslated) == 1000
        assert elapsed < 2.0, f"recover_untranslated took {elapsed:.2f}s, expected < 2s"


# ===========================================================================
# TestStorageEstimates
# ===========================================================================


@pytest.mark.slow
class TestStorageEstimates:
    """Measure and project FLAC storage requirements."""

    def test_flac_compression_ratio(self, tmp_path: Path):
        """Write known audio, measure FLAC size vs raw size."""
        audio = generate_audio(30.0, sample_rate=48000)
        raw_bytes = len(audio) * 4  # float32 = 4 bytes

        rec = FlacChunkRecorder(
            session_id="compress-test",
            base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
            chunk_duration_s=30.0,
        )
        rec.start()
        rec.write(audio)
        rec.stop()

        import json
        manifest = json.loads((rec.session_dir / "manifest.json").read_text())
        flac_path = rec.session_dir / manifest["chunks"][0]["filename"]
        flac_bytes = flac_path.stat().st_size

        ratio = flac_bytes / raw_bytes
        # FLAC typically achieves 30-70% compression on audio
        assert ratio < 1.0, f"FLAC larger than raw? ratio={ratio:.2f}"

    def test_projected_four_hour_storage(self, tmp_path: Path):
        """From compression ratio, project 4h disk usage. Assert < 2 GB."""
        audio = generate_audio(30.0, sample_rate=48000)

        rec = FlacChunkRecorder(
            session_id="storage-test",
            base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=2,  # stereo
            chunk_duration_s=30.0,
        )
        rec.start()
        rec.write(audio)
        rec.stop()

        import json
        manifest = json.loads((rec.session_dir / "manifest.json").read_text())
        flac_path = rec.session_dir / manifest["chunks"][0]["filename"]
        chunk_bytes = flac_path.stat().st_size

        # 4 hours = 480 chunks at 30s each
        projected_bytes = chunk_bytes * 480
        projected_mb = projected_bytes / (1024 * 1024)

        assert projected_mb < 2048, f"Projected 4h = {projected_mb:.0f} MB, expected < 2 GB"
