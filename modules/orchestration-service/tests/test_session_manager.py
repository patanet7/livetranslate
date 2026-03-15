"""Tests for MeetingSessionManager — lifecycle, promotion, heartbeat."""
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

# Ensure the meeting package resolves from the tests directory
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import Meeting, MeetingTranslation
from meeting.session_manager import MeetingSessionManager


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingSessionManager:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

    async def test_create_ephemeral(self, manager):
        session = await manager.create_session(source_type="loopback")
        assert session.status == "ephemeral"
        assert session.source == "loopback"
        assert session.source_type == "loopback"  # property alias
        assert session.recording_path is None

    async def test_promote_to_meeting(self, manager):
        session = await manager.create_session(source_type="loopback")
        promoted = await manager.promote_to_meeting(session.id)
        assert promoted.status == "active"
        assert promoted.recording_path is not None

    async def test_end_meeting(self, manager):
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)
        ended = await manager.end_meeting(session.id)
        assert ended.status == "completed"
        assert ended.ended_at is not None

    async def test_detect_orphans(self, manager, db_session: AsyncSession):
        """Sessions with no activity for > timeout seconds should be returned as orphans."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # Manually backdate last_activity_at to 200 seconds ago
        result = await db_session.get(Meeting, session.id)
        result.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        await db_session.commit()

        orphans = await manager.detect_orphans()
        assert len(orphans) >= 1
        assert any(o.id == session.id for o in orphans)

    async def test_mark_interrupted(self, manager, db_session: AsyncSession):
        """mark_interrupted should set status=interrupted and ended_at."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)
        await manager.mark_interrupted(session.id)

        result = await db_session.get(Meeting, session.id)
        assert result.status == "interrupted"
        assert result.ended_at is not None

    async def test_recover_on_startup(self, manager, db_session: AsyncSession):
        """recover_on_startup should mark all 'active' sessions as interrupted."""
        s1 = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(s1.id)
        s2 = await manager.create_session(source_type="gmeet")
        await manager.promote_to_meeting(s2.id)

        orphans = await manager.recover_on_startup()
        assert len(orphans) == 2

        for sid in [s1.id, s2.id]:
            row = await db_session.get(Meeting, sid)
            assert row.status == "interrupted"

    async def test_recover_untranslated(self, manager):
        """Chunks without translations should be returned for re-submission."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        chunk = await manager.add_transcript(
            session_id=session.id,
            text="Hello world",
            timestamp_ms=1000,
            language="en",
            confidence=0.95,
            is_final=True,
        )

        untranslated = await manager.recover_untranslated()
        assert len(untranslated) >= 1
        assert any(t.id == chunk.id for t in untranslated)

    async def test_recover_untranslated_excludes_translated(self, manager, db_session: AsyncSession):
        """Chunks that already have a translation must NOT be returned."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        chunk = await manager.add_transcript(
            session_id=session.id,
            text="Hello world",
            timestamp_ms=1000,
            language="en",
            confidence=0.95,
            is_final=True,
        )

        translation = MeetingTranslation(
            chunk_id=chunk.id,
            target_language="es",
            translated_text="Hola mundo",
            model_used="qwen3.5:7b",
        )
        db_session.add(translation)
        await db_session.commit()

        untranslated = await manager.recover_untranslated()
        assert not any(t.id == chunk.id for t in untranslated)

    async def test_recover_untranslated_excludes_non_final(self, manager):
        """Non-final chunks must be excluded from untranslated recovery."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        chunk = await manager.add_transcript(
            session_id=session.id,
            text="Partial...",
            timestamp_ms=500,
            language="en",
            confidence=0.5,
            is_final=False,  # Not final — should be excluded
        )

        untranslated = await manager.recover_untranslated()
        assert not any(t.id == chunk.id for t in untranslated)

    async def test_update_heartbeat(self, manager, db_session: AsyncSession):
        """update_heartbeat should refresh last_activity_at."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # Backdate the heartbeat
        row = await db_session.get(Meeting, session.id)
        row.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        await db_session.commit()

        await manager.update_heartbeat(session.id)

        # After heartbeat update, the session should not be an orphan
        orphans = await manager.detect_orphans()
        assert not any(o.id == session.id for o in orphans)
