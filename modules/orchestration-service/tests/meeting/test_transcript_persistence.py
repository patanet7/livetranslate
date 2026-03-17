"""Transcript + translation persistence tests — verify DB round-trips and FK linkage."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import MeetingChunk, MeetingTranslation

from meeting.session_manager import MeetingSessionManager

# ===========================================================================
# TestTranscriptPersistence
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestTranscriptPersistence:
    """Verify add_transcript() persists chunks with all fields intact."""

    async def test_add_transcript_persists_chunk(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id,
            text="Hello world",
            timestamp_ms=int(time.time() * 1000),
            language="en",
            confidence=0.95,
            is_final=True,
            speaker_id="speaker_0",
        )

        result = await db_session.execute(
            select(MeetingChunk).where(MeetingChunk.id == chunk.id)
        )
        row = result.scalar_one()
        assert row.text == "Hello world"
        assert row.source_language == "en"
        assert row.confidence == pytest.approx(0.95)
        assert row.is_final is True
        assert row.speaker_id == "speaker_0"
        assert row.meeting_id == session.id

    async def test_source_language_stored_correctly(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        languages = ["en", "zh", "ja"]
        for lang in languages:
            await meeting_session_manager.add_transcript(
                session_id=session.id,
                text=f"Text in {lang}",
                timestamp_ms=int(time.time() * 1000),
                language=lang,
                confidence=0.9,
                is_final=True,
            )

        result = await db_session.execute(
            select(MeetingChunk).where(MeetingChunk.meeting_id == session.id)
        )
        chunks = list(result.scalars().all())
        assert {c.source_language for c in chunks} == {"en", "zh", "ja"}

    async def test_final_vs_non_final_flag(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        ts = int(time.time() * 1000)

        final = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Final", timestamp_ms=ts,
            language="en", confidence=0.9, is_final=True,
        )
        non_final = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Non-final", timestamp_ms=ts + 1,
            language="en", confidence=0.8, is_final=False,
        )

        f_row = await db_session.get(MeetingChunk, final.id)
        nf_row = await db_session.get(MeetingChunk, non_final.id)
        assert f_row.is_final is True
        assert nf_row.is_final is False

    async def test_speaker_id_persisted(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Speaker test", timestamp_ms=0,
            language="en", confidence=0.9, is_final=True, speaker_id="spk_42",
        )
        row = await db_session.get(MeetingChunk, chunk.id)
        assert row.speaker_id == "spk_42"


# ===========================================================================
# TestTranslationPersistence
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestTranslationPersistence:
    """Verify save_translation() persists and links correctly to chunks."""

    async def test_save_translation_linked_to_chunk(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="你好世界", timestamp_ms=0,
            language="zh", confidence=0.9, is_final=True,
        )
        translation = await meeting_session_manager.save_translation(
            chunk_id=chunk.id,
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3.5:7b",
            translation_time_ms=150.0,
        )

        row = await db_session.get(MeetingTranslation, translation.id)
        assert row.chunk_id == chunk.id
        assert row.translated_text == "Hello world"

    async def test_translation_fields_round_trip(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="test", timestamp_ms=0,
            language="en", confidence=0.9, is_final=True,
        )
        t = await meeting_session_manager.save_translation(
            chunk_id=chunk.id,
            translated_text="prueba",
            source_language="en",
            target_language="es",
            model_used="qwen3.5:7b",
            translation_time_ms=200.5,
        )

        row = await db_session.get(MeetingTranslation, t.id)
        assert row.translated_text == "prueba"
        assert row.target_language == "es"
        assert row.source_language == "en"
        assert row.model_used == "qwen3.5:7b"
        assert row.translation_time_ms == pytest.approx(200.5)

    async def test_chunk_translation_relationship(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="关系测试", timestamp_ms=0,
            language="zh", confidence=0.9, is_final=True,
        )
        await meeting_session_manager.save_translation(
            chunk_id=chunk.id,
            translated_text="Relationship test",
            source_language="zh",
            target_language="en",
            model_used="test-model",
        )

        result = await db_session.execute(
            select(MeetingChunk)
            .options(selectinload(MeetingChunk.direct_translations))
            .where(MeetingChunk.id == chunk.id)
        )
        loaded = result.scalar_one()
        assert len(loaded.direct_translations) == 1
        assert loaded.direct_translations[0].translated_text == "Relationship test"

    async def test_multiple_translations_per_chunk(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Multi-lang source", timestamp_ms=0,
            language="en", confidence=0.9, is_final=True,
        )
        await meeting_session_manager.save_translation(
            chunk_id=chunk.id, translated_text="Fuente multi-idioma",
            source_language="en", target_language="es", model_used="test",
        )
        await meeting_session_manager.save_translation(
            chunk_id=chunk.id, translated_text="Mehrsprachige Quelle",
            source_language="en", target_language="de", model_used="test",
        )

        result = await db_session.execute(
            select(MeetingChunk)
            .options(selectinload(MeetingChunk.direct_translations))
            .where(MeetingChunk.id == chunk.id)
        )
        loaded = result.scalar_one()
        langs = {t.target_language for t in loaded.direct_translations}
        assert langs == {"es", "de"}


# ===========================================================================
# TestOriginalVsTranslated
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestOriginalVsTranslated:
    """Verify original transcript and translation coexist with correct linkage."""

    async def test_original_and_translated_both_persisted(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="你好世界", timestamp_ms=0,
            language="zh", confidence=0.92, is_final=True,
        )
        await meeting_session_manager.save_translation(
            chunk_id=chunk.id, translated_text="Hello world",
            source_language="zh", target_language="en", model_used="test",
        )

        # Reload chunk with translations
        result = await db_session.execute(
            select(MeetingChunk)
            .options(selectinload(MeetingChunk.direct_translations))
            .where(MeetingChunk.id == chunk.id)
        )
        loaded = result.scalar_one()
        trans = loaded.direct_translations[0]

        assert loaded.text != trans.translated_text
        assert loaded.source_language != trans.target_language
        assert trans.chunk_id == loaded.id

    async def test_bilingual_meeting_chunks_separated(
        self, meeting_session_manager: MeetingSessionManager, db_session: AsyncSession
    ):
        session = await meeting_session_manager.create_session("loopback")
        ts = int(time.time() * 1000)

        # Alternate en/zh chunks
        en_chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="Hello everyone", timestamp_ms=ts,
            language="en", confidence=0.9, is_final=True,
        )
        zh_chunk = await meeting_session_manager.add_transcript(
            session_id=session.id, text="大家好", timestamp_ms=ts + 1000,
            language="zh", confidence=0.88, is_final=True,
        )

        # Translate each to the other language
        await meeting_session_manager.save_translation(
            chunk_id=en_chunk.id, translated_text="大家好 (translated)",
            source_language="en", target_language="zh", model_used="test",
        )
        await meeting_session_manager.save_translation(
            chunk_id=zh_chunk.id, translated_text="Hello everyone (translated)",
            source_language="zh", target_language="en", model_used="test",
        )

        # Verify linkage
        result = await db_session.execute(
            select(MeetingChunk)
            .options(selectinload(MeetingChunk.direct_translations))
            .where(MeetingChunk.meeting_id == session.id)
            .order_by(MeetingChunk.timestamp_ms)
        )
        chunks = list(result.scalars().all())
        assert len(chunks) == 2

        assert chunks[0].source_language == "en"
        assert chunks[0].direct_translations[0].target_language == "zh"

        assert chunks[1].source_language == "zh"
        assert chunks[1].direct_translations[0].target_language == "en"
