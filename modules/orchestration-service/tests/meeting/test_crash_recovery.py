"""Crash recovery tests — startup recovery, untranslated detection, recording survival."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import soundfile as sf
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
# TestStartupRecovery
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestStartupRecovery:
    """recover_on_startup() marks only active sessions as interrupted."""

    async def test_recover_on_startup_marks_active_interrupted(
        self, db_session: AsyncSession, tmp_path: Path,
    ):
        mgr = MeetingSessionManager(
            db=db_session, recording_base_path=tmp_path / "recordings",
        )

        ephemeral = await mgr.create_session("loopback")  # status=ephemeral
        active = await mgr.create_session("loopback")
        await mgr.promote_to_meeting(active.id)  # status=active
        completed = await mgr.create_session("loopback")
        await mgr.promote_to_meeting(completed.id)
        await mgr.end_meeting(completed.id)  # status=completed

        orphans = await mgr.recover_on_startup()
        orphan_ids = {o.id for o in orphans}

        # Only the active session should be interrupted
        assert active.id in orphan_ids
        assert ephemeral.id not in orphan_ids
        assert completed.id not in orphan_ids

        # Verify status changed in DB
        row = await db_session.get(Meeting, active.id)
        assert row.status == "interrupted"

    async def test_recover_on_startup_returns_all_orphans(
        self, db_session: AsyncSession, tmp_path: Path,
    ):
        mgr = MeetingSessionManager(
            db=db_session, recording_base_path=tmp_path / "recordings",
        )

        active_ids = []
        for _ in range(5):
            s = await mgr.create_session("loopback")
            await mgr.promote_to_meeting(s.id)
            active_ids.append(s.id)

        orphans = await mgr.recover_on_startup()
        assert len(orphans) == 5
        assert {o.id for o in orphans} == set(active_ids)


# ===========================================================================
# TestUntranslatedRecovery
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestUntranslatedRecovery:
    """recover_untranslated() finds final chunks without translations."""

    async def test_recover_untranslated_finds_finals_without_translation(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession,
    ):
        session = await meeting_session_manager.create_session("loopback")
        ts = int(time.time() * 1000)

        c1 = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Chunk 1", timestamp_ms=ts,
            language="en", confidence=0.9, is_final=True,
        )
        c2 = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Chunk 2", timestamp_ms=ts + 1000,
            language="en", confidence=0.9, is_final=True,
        )
        c3 = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Chunk 3", timestamp_ms=ts + 2000,
            language="en", confidence=0.9, is_final=True,
        )

        # Only translate c1
        await meeting_session_manager.save_translation(
            chunk_id=c1.id, translated_text="Trozo 1",
            source_language="en", target_language="es", model_used="test",
        )

        untranslated = await meeting_session_manager.recover_untranslated()
        untranslated_ids = {c.id for c in untranslated}
        assert c2.id in untranslated_ids
        assert c3.id in untranslated_ids
        assert c1.id not in untranslated_ids

    async def test_recover_untranslated_excludes_non_final(
        self, meeting_session_manager: MeetingSessionManager,
    ):
        session = await meeting_session_manager.create_session("loopback")
        await meeting_session_manager.add_transcript(
            session_id=session.id, text="Non-final chunk", timestamp_ms=0,
            language="en", confidence=0.9, is_final=False,
        )

        untranslated = await meeting_session_manager.recover_untranslated()
        assert len(untranslated) == 0

    async def test_recover_untranslated_excludes_translated(
        self, meeting_session_manager: MeetingSessionManager,
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Translated chunk", timestamp_ms=0,
            language="en", confidence=0.9, is_final=True,
        )
        await meeting_session_manager.save_translation(
            chunk_id=chunk.id, translated_text="Trozo traducido",
            source_language="en", target_language="es", model_used="test",
        )

        untranslated = await meeting_session_manager.recover_untranslated()
        assert all(c.id != chunk.id for c in untranslated)


# ===========================================================================
# TestRecordingSurvival
# ===========================================================================


class TestRecordingSurvival:
    """FLAC files survive simulated crashes."""

    def test_flac_files_survive_crash(self, tmp_path: Path):
        """Write 3 chunks, skip stop(), verify flushed files exist and are readable."""
        rec = FlacChunkRecorder(
            session_id="crash-test",
            base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
            chunk_duration_s=1.0,
        )
        rec.start()

        for chunk in generate_audio_chunks(3.5, chunk_s=1.0):
            rec.write(chunk)
        # Deliberately skip stop() — simulates crash

        manifest = json.loads((rec.session_dir / "manifest.json").read_text())
        assert len(manifest["chunks"]) == 3  # 3 full chunks flushed

        for entry in manifest["chunks"]:
            path = rec.session_dir / entry["filename"]
            assert path.exists()
            data, sr = sf.read(str(path))
            assert sr == 48000
            assert len(data) > 0

    def test_partial_tmp_file_doesnt_corrupt_manifest(self, tmp_path: Path):
        """Write chunks, create fake .tmp file, verify manifest.json is still valid."""
        rec = FlacChunkRecorder(
            session_id="tmp-test",
            base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
            chunk_duration_s=1.0,
        )
        rec.start()
        audio = generate_audio(2.5)
        rec.write(audio)
        rec.stop()

        # Create a fake leftover .tmp file (simulates interrupted atomic write)
        fake_tmp = rec.session_dir / "manifest.json.tmp"
        fake_tmp.write_text('{"corrupted": true}')

        # The real manifest should still be valid
        manifest = json.loads((rec.session_dir / "manifest.json").read_text())
        assert "chunks" in manifest
        assert len(manifest["chunks"]) >= 2


# ===========================================================================
# TestFullCrashRecoveryFlow
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestFullCrashRecoveryFlow:
    """End-to-end crash recovery: create → record → transcribe → crash → recover."""

    async def test_end_to_end_crash_recovery(
        self, db_session: AsyncSession, tmp_path: Path,
    ):
        # Phase 1: Create a session and generate data
        mgr = MeetingSessionManager(
            db=db_session, recording_base_path=tmp_path / "recordings",
        )
        pipeline = MeetingPipeline(
            session_manager=mgr,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )

        await pipeline.start()
        await pipeline.promote_to_meeting()
        session_id = pipeline.session_id

        # Write audio
        for chunk in generate_audio_chunks(5.0, chunk_s=1.0):
            await pipeline.process_audio(chunk)

        # Add transcripts
        ts = int(time.time() * 1000)
        c1 = await mgr.add_transcript(
            session_id=session_id, text="First chunk", timestamp_ms=ts,
            language="en", confidence=0.9, is_final=True,
        )
        c2 = await mgr.add_transcript(
            session_id=session_id, text="Second chunk", timestamp_ms=ts + 1000,
            language="en", confidence=0.85, is_final=True,
        )
        c3 = await mgr.add_transcript(
            session_id=session_id, text="Third chunk", timestamp_ms=ts + 2000,
            language="en", confidence=0.92, is_final=True,
        )

        # Translate only the first chunk (simulate partial progress)
        await mgr.save_translation(
            chunk_id=c1.id, translated_text="Primer trozo",
            source_language="en", target_language="es", model_used="test",
        )

        # Phase 2: Simulate crash — do NOT call pipeline.end()
        # (just abandon the pipeline object)

        # Phase 3: New manager on "restart"
        mgr2 = MeetingSessionManager(
            db=db_session, recording_base_path=tmp_path / "recordings",
        )

        # recover_on_startup should find the active session
        orphans = await mgr2.recover_on_startup()
        assert len(orphans) >= 1
        orphan_ids = {o.id for o in orphans}
        assert session_id in orphan_ids

        # Verify session is now interrupted
        session = await db_session.get(Meeting, session_id)
        assert session.status == "interrupted"

        # recover_untranslated should find c2 and c3
        untranslated = await mgr2.recover_untranslated()
        untranslated_ids = {c.id for c in untranslated}
        assert c2.id in untranslated_ids
        assert c3.id in untranslated_ids
        assert c1.id not in untranslated_ids  # already translated

        # FLAC files should still be readable
        recording_dir = tmp_path / "recordings" / str(session_id)
        if recording_dir.exists():
            manifest_path = recording_dir / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                for entry in manifest["chunks"]:
                    path = recording_dir / entry["filename"]
                    assert path.exists()
                    _data, sr = sf.read(str(path))
                    assert sr == 48000
