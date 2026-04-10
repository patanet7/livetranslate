"""Operational visibility tests for meeting translation backlog."""
from __future__ import annotations

import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import Meeting, MeetingSentence
from meeting.session_manager import MeetingSessionManager
from meeting.translation_recovery import (
    get_translation_recovery_metrics,
    recover_pending_translations,
)
from routers import meetings as meetings_router
from services.meeting_store import MeetingStore

from types import SimpleNamespace

from livetranslate_common.models import TranslationResponse


class _FakeTranslationService:
    """Minimal fake for observability tests."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            context_window_size=3,
            max_context_tokens=512,
            cross_direction_max_tokens=128,
            model="fake-recovery-model",
        )

    async def translate(self, request, skip_context: bool = False, context_store=None):
        return TranslationResponse(
            translated_text=f"{request.target_language}:{request.text}",
            source_language=request.source_language,
            target_language=request.target_language,
            model_used=self.config.model,
            latency_ms=1.5,
        )


@pytest.mark.asyncio
@pytest.mark.integration
class TestTranslationObservability:
    async def test_store_reports_backlog_counts(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        manager = MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
        )
        loopback = await manager.create_session(source_type="loopback", target_languages=["es"])
        await manager.promote_to_meeting(loopback.id)
        await manager.add_transcript(
            session_id=loopback.id,
            text="Chunk pending translation",
            timestamp_ms=1000,
            language="en",
            confidence=0.9,
            is_final=True,
        )

        fireflies = Meeting(
            id=uuid.uuid4(),
            source="fireflies",
            status="completed",
            source_languages=["en"],
            target_languages=["fr"],
            started_at=datetime.now(UTC),
            last_activity_at=datetime.now(UTC),
        )
        db_session.add(fireflies)
        await db_session.flush()
        db_session.add(
            MeetingSentence(
                meeting_id=fireflies.id,
                text="Sentence pending translation",
                speaker_name="Alice",
                start_time=0.0,
                end_time=1.0,
                chunk_ids=[],
            )
        )
        await db_session.commit()

        store = MeetingStore(os.environ["DATABASE_URL"])
        await store.initialize()
        try:
            loopback_backlog = await store.get_meeting_translation_backlog(str(loopback.id))
            fleet_backlog = await store.list_translation_backlog(limit=10, offset=0, only_pending=True)
        finally:
            await store.close()

        assert loopback_backlog is not None
        assert loopback_backlog["pending_chunk_translation_count"] == 1
        assert loopback_backlog["pending_sentence_translation_count"] == 0
        assert fleet_backlog["summary"]["pending_translation_count"] == 2
        assert fleet_backlog["summary"]["pending_chunk_translation_count"] == 1
        assert fleet_backlog["summary"]["pending_sentence_translation_count"] == 1
        assert fleet_backlog["total"] == 2

    async def test_recovery_counters_increment(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        manager = MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
        )
        session = await manager.create_session(source_type="loopback", target_languages=["es"])
        await manager.promote_to_meeting(session.id)
        await manager.add_transcript(
            session_id=session.id,
            text="Counter test",
            timestamp_ms=2000,
            language="en",
            confidence=0.8,
            is_final=True,
        )

        before = await get_translation_recovery_metrics()
        stats = await recover_pending_translations(manager, _FakeTranslationService())
        after = await get_translation_recovery_metrics()

        assert stats == {"pending": 1, "recovered": 1, "failed": 0}
        assert after["runs"] == before["runs"] + 1
        assert after["total_pending_seen"] == before["total_pending_seen"] + 1
        assert after["total_recovered"] == before["total_recovered"] + 1
        assert after["last_run_pending"] == 1
        assert after["last_run_failed"] == 0

    async def test_admin_recover_endpoint_returns_before_and_after(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        manager = MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
        )
        session = await manager.create_session(source_type="loopback", target_languages=["es"])
        await manager.promote_to_meeting(session.id)
        await manager.add_transcript(
            session_id=session.id,
            text="Endpoint test",
            timestamp_ms=3000,
            language="en",
            confidence=0.95,
            is_final=True,
        )

        store = MeetingStore(os.environ["DATABASE_URL"])
        await store.initialize()

        class _FakeDbManager:
            async def get_session_direct(self):
                return db_session

        async def _fake_get_store():
            return store

        async def _fake_recover(db, recording_base_path, translation_service, *, meeting_ids=None, limit=200):
            return await recover_pending_translations(
                manager,
                _FakeTranslationService(),
                meeting_ids=meeting_ids,
                limit=limit,
            )

        monkeypatch.setattr(meetings_router, "_get_store", _fake_get_store)
        monkeypatch.setattr(meetings_router, "get_database_manager", lambda: _FakeDbManager())
        monkeypatch.setattr(meetings_router, "get_translation_service_client", lambda: _FakeTranslationService())
        monkeypatch.setattr(
            meetings_router,
            "recover_pending_translations_with_fresh_session",
            _fake_recover,
        )

        try:
            payload = await meetings_router.recover_meeting_translations(str(session.id), limit=10)
        finally:
            await store.close()

        assert payload["before"]["pending_translation_count"] == 1
        assert payload["recovery"] == {"pending": 1, "recovered": 1, "failed": 0}
        assert payload["after"]["pending_translation_count"] == 0
