"""Tests for heartbeat background monitor — behavioral test against a real DB."""
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import Meeting
from meeting.heartbeat import run_heartbeat_monitor
from meeting.session_manager import MeetingSessionManager


@pytest.mark.asyncio
@pytest.mark.integration
class TestHeartbeatMonitor:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

    async def test_stale_session_marked_interrupted(self, manager, db_session: AsyncSession):
        """An active session whose last_activity_at is beyond the timeout must be interrupted."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # Backdate last_activity_at to 200 seconds ago (> 120 s timeout)
        db_obj = await db_session.get(Meeting, session.id)
        db_obj.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        await db_session.commit()

        # Run one iteration of the heartbeat monitor then cancel
        task = asyncio.create_task(
            run_heartbeat_monitor(manager, check_interval_s=3600)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await db_session.refresh(db_obj)
        assert db_obj.status == "interrupted"
        assert db_obj.ended_at is not None

    async def test_active_session_not_interrupted(self, manager, db_session: AsyncSession):
        """An active session with a recent heartbeat must NOT be interrupted."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)
        # last_activity_at is now — well within the 120 s window

        task = asyncio.create_task(
            run_heartbeat_monitor(manager, check_interval_s=3600)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        db_obj = await db_session.get(Meeting, session.id)
        assert db_obj.status == "active"

    async def test_ephemeral_session_not_interrupted(self, manager, db_session: AsyncSession):
        """Ephemeral sessions must never be detected as orphans (only active ones are)."""
        session = await manager.create_session(source_type="loopback")
        # Do NOT promote — leave as ephemeral

        # Backdate so it would qualify if status were checked incorrectly
        db_obj = await db_session.get(Meeting, session.id)
        db_obj.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        await db_session.commit()

        task = asyncio.create_task(
            run_heartbeat_monitor(manager, check_interval_s=3600)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await db_session.refresh(db_obj)
        assert db_obj.status == "ephemeral"

    async def test_monitor_handles_exception_and_continues(self, manager, db_session: AsyncSession, monkeypatch):
        """A transient exception in detect_orphans must not crash the monitor loop."""
        call_count = 0
        original_detect = manager.detect_orphans

        async def flaky_detect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient DB error")
            return await original_detect()

        monkeypatch.setattr(manager, "detect_orphans", flaky_detect)

        task = asyncio.create_task(
            run_heartbeat_monitor(manager, check_interval_s=0)
        )
        # Give it enough time to run at least two iterations
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The monitor must have survived the first exception and attempted a second call
        assert call_count >= 2
