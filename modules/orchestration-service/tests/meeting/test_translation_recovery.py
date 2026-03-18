"""Recovery tests for untranslated meeting backlog.

Validates the shared recovery path for:
- loopback/gmeet finalized chunks
- Fireflies sentence-backed transcripts persisted outside the live websocket flow
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import Meeting, MeetingSentence, MeetingTranslation
from meeting.session_manager import MeetingSessionManager
from meeting.translation_recovery import recover_pending_translations
from routers.fireflies import _store_transcript_to_db
from services.meeting_store import MeetingStore
from livetranslate_common.models import TranslationResponse


class _FakeTranslationService:
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
class TestMeetingTranslationRecovery:
    async def test_recovers_chunk_and_sentence_backlog(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        manager = MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
        )

        loopback = await manager.create_session(
            source_type="loopback",
            target_languages=["es"],
        )
        await manager.promote_to_meeting(loopback.id)
        chunk = await manager.add_transcript(
            session_id=loopback.id,
            text="Hello from loopback",
            timestamp_ms=1000,
            language="en",
            confidence=0.99,
            is_final=True,
        )

        fireflies_meeting = Meeting(
            id=uuid.uuid4(),
            source="fireflies",
            status="completed",
            target_languages=["fr"],
            source_languages=["en"],
            started_at=datetime.now(UTC),
            last_activity_at=datetime.now(UTC),
        )
        db_session.add(fireflies_meeting)
        await db_session.flush()

        sentence = MeetingSentence(
            meeting_id=fireflies_meeting.id,
            text="Hello from fireflies",
            speaker_name="Alice",
            start_time=0.0,
            end_time=1.0,
            chunk_ids=[],
        )
        db_session.add(sentence)
        await db_session.commit()

        stats = await recover_pending_translations(manager, _FakeTranslationService())

        translations = (
            await db_session.execute(select(MeetingTranslation).order_by(MeetingTranslation.created_at.asc()))
        ).scalars().all()

        assert stats == {"pending": 2, "recovered": 2, "failed": 0}
        assert len(translations) == 2
        assert any(t.chunk_id == chunk.id and t.target_language == "es" for t in translations)
        assert any(t.sentence_id == sentence.id and t.target_language == "fr" for t in translations)

    async def test_fireflies_synced_transcript_can_use_shared_recovery_pipeline(
        self,
        db_session: AsyncSession,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        manager = MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
        )
        store = MeetingStore(os.environ["DATABASE_URL"])
        await store.initialize()
        monkeypatch.setenv("DEFAULT_TARGET_LANGUAGE", "de")

        try:
            meeting_id = await _store_transcript_to_db(
                {
                    "id": "ff-sync-001",
                    "title": "Fireflies Sync",
                    "sentences": [
                        {
                            "text": "Thanks everyone for joining",
                            "speaker_name": "Sam",
                            "start_time": 0.0,
                            "end_time": 1.2,
                        }
                    ],
                },
                store,
            )
        finally:
            await store.close()

        meeting = await db_session.get(Meeting, uuid.UUID(meeting_id))
        assert meeting is not None
        assert meeting.target_languages == ["de"]

        stats = await recover_pending_translations(
            manager,
            _FakeTranslationService(),
            meeting_ids=[uuid.UUID(meeting_id)],
        )

        translations = (
            await db_session.execute(
                select(MeetingTranslation).join(MeetingSentence).where(MeetingSentence.meeting_id == uuid.UUID(meeting_id))
            )
        ).scalars().all()

        assert stats == {"pending": 1, "recovered": 1, "failed": 0}
        assert len(translations) == 1
        assert translations[0].sentence_id is not None
        assert translations[0].target_language == "de"
        assert translations[0].translated_text == "de:Thanks everyone for joining"
