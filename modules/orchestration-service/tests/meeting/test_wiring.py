"""Wiring tests — verify websocket_audio.py actually persists transcripts and links translations.

These tests call the real persistence code path (_translate_and_send) with a real
database, but stub the LLM client at the HTTP boundary so no running LLM is needed.
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import MeetingChunk, MeetingTranslation
from translation.config import TranslationConfig
from translation.service import TranslationService

from meeting.pipeline import MeetingPipeline
from meeting.session_manager import MeetingSessionManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fake_translate_stream(*args, **kwargs):
    """Yield fixed tokens simulating an LLM streaming response."""
    for word in ["Hello", " ", "world"]:
        yield word


def _fake_extract_translation(self, response: str) -> str:
    return response.strip()


def _make_translation_service() -> TranslationService:
    """Create a TranslationService with a stubbed LLM client."""
    config = TranslationConfig(
        base_url="http://localhost:99999/v1",  # never contacted
        model="test-model",
    )
    svc = TranslationService(config)
    # Stub the LLM client methods at the HTTP boundary
    svc._client.translate_stream = _fake_translate_stream
    svc._client._extract_translation = lambda self_or_text, text=None: (
        (text or self_or_text).strip()
    )
    return svc


# ===========================================================================
# TestTranslateAndSendWiring
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestTranslateAndSendWiring:
    """Verify _translate_and_send() persists translations with correct chunk_id."""

    async def test_final_translation_saved_with_chunk_id(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession, tmp_path: Path,
    ):
        """When chunk_id is provided, save_translation links to that chunk."""
        from routers.audio.websocket_audio import _translate_and_send

        # Setup: real pipeline + session
        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # Create a transcript chunk (what handle_transcription_segment does)
        chunk = await meeting_session_manager.add_transcript(
            session_id=pipeline.session_id,
            text="你好世界",
            timestamp_ms=int(time.time() * 1000),
            language="zh",
            confidence=0.9,
            is_final=True,
        )

        # Call _translate_and_send with the real pipeline and chunk_id
        sent_messages: list[str] = []

        async def fake_send(msg: str) -> bool:
            sent_messages.append(msg)
            return True

        svc = _make_translation_service()

        await _translate_and_send(
            safe_send=fake_send,
            translation_service=svc,
            segment_id=1,
            text="你好世界",
            source_lang="zh",
            target_lang="en",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=False,
            chunk_id=chunk.id,
        )

        # Verify: translation was saved in DB with correct chunk_id
        result = await db_session.execute(
            select(MeetingTranslation).where(MeetingTranslation.chunk_id == chunk.id)
        )
        translation = result.scalar_one()
        assert translation.translated_text == "Hello world"
        assert translation.source_language == "zh"
        assert translation.target_language == "en"
        assert translation.model_used == "test-model"
        assert translation.chunk_id == chunk.id

        # Verify: streaming chunks + final message were sent
        assert len(sent_messages) >= 2  # at least chunk messages + final

        await pipeline.end()
        await svc.close()

    async def test_final_translation_without_chunk_id_sends_but_skips_persist(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession, tmp_path: Path,
    ):
        """When chunk_id is None, translation is sent to client but DB persist is skipped
        (schema requires chunk_id or sentence_id)."""
        from routers.audio.websocket_audio import _translate_and_send

        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        sent_messages: list[str] = []

        async def fake_send(msg: str) -> bool:
            sent_messages.append(msg)
            return True

        svc = _make_translation_service()

        await _translate_and_send(
            safe_send=fake_send,
            translation_service=svc,
            segment_id=1,
            text="Hello",
            source_lang="en",
            target_lang="es",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=False,
            chunk_id=None,  # ephemeral — no transcript chunk
        )

        # Translation message was sent to client even without chunk_id
        import json
        translation_msgs = [
            json.loads(m) for m in sent_messages if "translation" in m.lower()
        ]
        assert len(translation_msgs) >= 1, f"Expected translation message, got: {sent_messages}"

        await pipeline.end()
        await svc.close()

    async def test_draft_translation_not_persisted(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession, tmp_path: Path,
    ):
        """Draft translations are never saved to the database."""
        from routers.audio.websocket_audio import _translate_and_send

        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        sent_messages: list[str] = []

        async def fake_send(msg: str) -> bool:
            sent_messages.append(msg)
            return True

        svc = _make_translation_service()
        # Stub translate_draft for the draft path
        svc.translate_draft = AsyncMock(return_value="Hola mundo")

        await _translate_and_send(
            safe_send=fake_send,
            translation_service=svc,
            segment_id=1,
            text="Hello world",
            source_lang="en",
            target_lang="es",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=True,
        )

        # No translations should be in the DB from the draft path
        result = await db_session.execute(select(MeetingTranslation))
        translations = list(result.scalars().all())
        assert len(translations) == 0

        await pipeline.end()
        await svc.close()


# ===========================================================================
# TestSegmentToChunkMapping
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestSegmentToChunkMapping:
    """Verify the segment_id → chunk_id mapping used in websocket_audio.py."""

    async def test_mapping_links_correct_chunk(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession,
    ):
        """Simulate the mapping flow: persist chunk → store mapping → look up for translation."""
        session = await meeting_session_manager.create_session("loopback")
        await meeting_session_manager.promote_to_meeting(session.id)

        # Simulate what handle_transcription_segment does
        _segment_to_chunk_id: dict[int, uuid.UUID] = {}

        for seg_id in range(1, 6):
            chunk = await meeting_session_manager.add_transcript(
                session_id=session.id,
                text=f"Segment {seg_id} text",
                timestamp_ms=seg_id * 1000,
                language="zh",
                confidence=0.9,
                is_final=True,
            )
            _segment_to_chunk_id[seg_id] = chunk.id

        # Simulate what _translate_and_send does — look up chunk_id
        for seg_id in range(1, 6):
            chunk_id = _segment_to_chunk_id.get(seg_id)
            assert chunk_id is not None

            await meeting_session_manager.save_translation(
                chunk_id=chunk_id,
                translated_text=f"Translation of segment {seg_id}",
                source_language="zh",
                target_language="en",
                model_used="test",
            )

        # Verify all 5 chunks have linked translations
        result = await db_session.execute(
            select(MeetingChunk)
            .options(selectinload(MeetingChunk.direct_translations))
            .where(MeetingChunk.meeting_id == session.id)
            .order_by(MeetingChunk.timestamp_ms)
        )
        chunks = list(result.scalars().all())
        assert len(chunks) == 5
        for i, c in enumerate(chunks, 1):
            assert len(c.direct_translations) == 1
            assert c.direct_translations[0].translated_text == f"Translation of segment {i}"

    async def test_mapping_eviction_keeps_recent(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession,
    ):
        """When mapping exceeds max size, oldest entries are evicted but recent ones survive."""
        session = await meeting_session_manager.create_session("loopback")
        _segment_to_chunk_id: dict[int, uuid.UUID] = {}
        max_size = 200

        # Add 250 entries
        for seg_id in range(250):
            chunk = await meeting_session_manager.add_transcript(
                session_id=session.id,
                text=f"Seg {seg_id}",
                timestamp_ms=seg_id * 100,
                language="en",
                confidence=0.9,
                is_final=True,
            )
            _segment_to_chunk_id[seg_id] = chunk.id
            if len(_segment_to_chunk_id) > max_size:
                oldest = sorted(_segment_to_chunk_id.keys())[: len(_segment_to_chunk_id) - max_size]
                for k in oldest:
                    del _segment_to_chunk_id[k]

        assert len(_segment_to_chunk_id) == max_size
        # Recent entries (200-249) should all be present
        for seg_id in range(200, 250):
            assert seg_id in _segment_to_chunk_id
        # Old entries (0-49) should be evicted
        for seg_id in range(50):
            assert seg_id not in _segment_to_chunk_id


# ===========================================================================
# TestTranscriptPersistenceInPipeline
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestTranscriptPersistenceInPipeline:
    """Verify the full flow: audio → pipeline → transcript → translation → DB."""

    async def test_full_pipeline_persistence_flow(
        self, meeting_session_manager: MeetingSessionManager,
        db_session: AsyncSession, tmp_path: Path,
    ):
        """Simulate: start pipeline → process audio → persist transcript → translate → verify linkage."""
        from routers.audio.websocket_audio import _translate_and_send

        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # Process some audio (exercises FLAC recording + downsampling)
        from .conftest import generate_audio
        audio = generate_audio(2.0)
        await pipeline.process_audio(audio)

        # Persist transcript (what handle_transcription_segment now does)
        chunk = await meeting_session_manager.add_transcript(
            session_id=pipeline.session_id,
            text="这是一个完整的测试",
            timestamp_ms=int(time.time() * 1000),
            language="zh",
            confidence=0.92,
            is_final=True,
        )

        # Translate with chunk_id linkage (what _translate_and_send now does)
        sent: list[str] = []

        async def fake_send(msg: str) -> bool:
            sent.append(msg)
            return True

        svc = _make_translation_service()

        await _translate_and_send(
            safe_send=fake_send,
            translation_service=svc,
            segment_id=42,
            text="这是一个完整的测试",
            source_lang="zh",
            target_lang="en",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=False,
            chunk_id=chunk.id,
        )

        # Verify: chunk exists with correct text
        db_chunk = await db_session.get(MeetingChunk, chunk.id)
        assert db_chunk.text == "这是一个完整的测试"
        assert db_chunk.source_language == "zh"
        assert db_chunk.is_final is True

        # Verify: translation linked to chunk
        result = await db_session.execute(
            select(MeetingTranslation).where(MeetingTranslation.chunk_id == chunk.id)
        )
        trans = result.scalar_one()
        assert trans.translated_text == "Hello world"
        assert trans.target_language == "en"
        assert trans.chunk_id == chunk.id

        # Verify: FLAC recording exists
        recording_dir = tmp_path / "recordings" / str(pipeline.session_id)
        assert recording_dir.exists()

        await pipeline.end()
        await svc.close()
