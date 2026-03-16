"""Tests for translation persistence — save_translation on MeetingSessionManager.

Integration tests use real Postgres via testcontainers (no mocks).
Tests the full path: create session → add chunk → save translation → verify FK link.
"""
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import MeetingChunk, MeetingTranslation
from meeting.session_manager import MeetingSessionManager


@pytest.mark.asyncio
@pytest.mark.integration
class TestSaveTranslation:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
        )

    async def _create_meeting_with_chunk(
        self, manager: MeetingSessionManager
    ) -> tuple[uuid.UUID, MeetingChunk]:
        """Helper: create a meeting session and add a transcript chunk."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)
        chunk = await manager.add_transcript(
            session_id=session.id,
            text="你好世界",
            timestamp_ms=1000,
            language="zh",
            confidence=0.95,
            is_final=True,
        )
        return session.id, chunk

    async def test_save_translation_creates_row(self, manager, db_session: AsyncSession):
        """save_translation should create a MeetingTranslation linked to the chunk."""
        meeting_id, chunk = await self._create_meeting_with_chunk(manager)

        translation = await manager.save_translation(
            chunk_id=chunk.id,
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3-4b",
        )

        assert translation.id is not None
        assert translation.chunk_id == chunk.id
        assert translation.translated_text == "Hello world"
        assert translation.source_language == "zh"
        assert translation.target_language == "en"
        assert translation.model_used == "qwen3-4b"

    async def test_save_translation_links_to_chunk_via_fk(self, manager, db_session: AsyncSession):
        """The saved translation should be queryable via the chunk's relationship."""
        meeting_id, chunk = await self._create_meeting_with_chunk(manager)

        await manager.save_translation(
            chunk_id=chunk.id,
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3-4b",
        )

        # Query via FK join
        result = await db_session.execute(
            select(MeetingTranslation).where(MeetingTranslation.chunk_id == chunk.id)
        )
        translations = list(result.scalars().all())
        assert len(translations) == 1
        assert translations[0].translated_text == "Hello world"

    async def test_save_translation_removes_from_untranslated(self, manager):
        """After saving a translation, recover_untranslated should exclude this chunk."""
        meeting_id, chunk = await self._create_meeting_with_chunk(manager)

        # Before: chunk is untranslated
        untranslated = await manager.recover_untranslated()
        assert any(c.id == chunk.id for c in untranslated)

        # Save translation
        await manager.save_translation(
            chunk_id=chunk.id,
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3-4b",
        )

        # After: chunk should not be in untranslated
        untranslated = await manager.recover_untranslated()
        assert not any(c.id == chunk.id for c in untranslated)

    async def test_save_translation_with_timing(self, manager):
        """save_translation should accept optional translation_time_ms."""
        meeting_id, chunk = await self._create_meeting_with_chunk(manager)

        translation = await manager.save_translation(
            chunk_id=chunk.id,
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3-4b",
            translation_time_ms=245.3,
        )

        assert translation.translation_time_ms == pytest.approx(245.3, abs=0.1)

    async def test_save_translation_without_chunk_id(self, manager, db_session: AsyncSession):
        """save_translation with chunk_id=None should still persist (orphaned translation)."""
        translation = await manager.save_translation(
            chunk_id=None,
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3-4b",
        )

        assert translation.id is not None
        assert translation.chunk_id is None
        assert translation.translated_text == "Hello world"
