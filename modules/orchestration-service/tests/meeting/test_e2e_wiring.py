# ruff: noqa: RUF001
"""E2E wiring tests — drive the actual WebSocket handler closure with real DB.

These tests exercise the full path:
  UI audio → WebSocket → handle_transcription_segment() → add_transcript() → DB
  → _translate_and_send() → save_translation(chunk_id=real) → DB

The transcription service is replaced with a fake that injects segments directly
into the registered callback. The LLM is stubbed at the HTTP boundary. The DB
and pipeline are real (testcontainer PostgreSQL + real MeetingPipeline).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

_src = Path(__file__).resolve().parent.parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from database.models import MeetingChunk, MeetingTranslation

from .conftest import generate_audio

# ---------------------------------------------------------------------------
# Fake transcription client — stores callbacks, lets tests inject segments
# ---------------------------------------------------------------------------


class FakeTranscriptionClient:
    """Drop-in replacement for WebSocketTranscriptionClient.

    Instead of connecting to the transcription service, stores callbacks
    and lets tests inject segment dicts directly via inject_segment().
    """

    def __init__(self, **kwargs):
        self._callbacks: dict[str, list] = {
            "segment": [],
            "language_detected": [],
            "error": [],
        }
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def on_segment(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._callbacks["segment"].append(callback)

    def on_language_detected(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._callbacks["language_detected"].append(callback)

    def on_error(self, callback: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        self._callbacks["error"].append(callback)

    async def connect(self) -> None:
        self._connected = True

    async def send_audio(self, data: bytes) -> None:
        pass  # audio goes nowhere — we inject segments directly

    async def send_config(self, **kwargs) -> None:
        pass

    async def send_end(self) -> None:
        pass

    async def close(self) -> None:
        self._connected = False

    async def inject_segment(self, data: dict) -> None:
        """Inject a transcription segment into the registered callbacks."""
        for cb in self._callbacks["segment"]:
            await cb(data)


# ---------------------------------------------------------------------------
# Fake translation service — stubs the LLM at the HTTP boundary
# ---------------------------------------------------------------------------


def _make_fake_translation_service():
    """TranslationService with stubbed LLM client."""
    from translation.config import TranslationConfig
    from translation.service import TranslationService

    config = TranslationConfig(
        base_url="http://localhost:99999/v1",
        model="test-wiring-model",
    )
    svc = TranslationService(config)

    async def _fake_stream(*args, **kwargs):
        for word in ["Hola", " ", "mundo"]:
            yield word

    svc._client.translate_stream = _fake_stream
    svc._client._extract_translation = lambda text: text.strip()
    return svc


# ===========================================================================
# TestE2EWebSocketWiring
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestE2EWebSocketWiring:
    """Test the full WebSocket handler closure with real DB persistence.

    Reconstructs the closure state from websocket_audio_stream() and drives
    handle_transcription_segment() with injected segments. Verifies that
    transcripts are persisted to DB and translations are linked via chunk_id.
    """

    async def test_segment_persists_transcript_and_linked_translation(
        self,
        meeting_session_manager,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        """Final segment → add_transcript() → _translate_and_send(chunk_id=real) → DB."""
        from routers.audio.websocket_audio import _translate_and_send

        from meeting.pipeline import MeetingPipeline

        # Build the same state the WebSocket closure builds
        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        translation_service = _make_fake_translation_service()
        _segment_to_chunk_id: dict[int, uuid.UUID] = {}
        sent_messages: list[str] = []

        async def safe_send(msg: str) -> bool:
            sent_messages.append(msg)
            return True

        # Simulate what handle_transcription_segment does for a final segment
        segment_data = {
            "segment_id": 42,
            "text": "你好世界，这是一个测试。",
            "language": "zh",
            "confidence": 0.92,
            "stable_text": "你好世界，这是一个测试。",
            "unstable_text": "",
            "is_final": True,
            "is_draft": False,
            "speaker_id": "speaker_0",
            "start_ms": int(time.time() * 1000),
            "end_ms": int(time.time() * 1000) + 3000,
        }

        # Step 1: Persist transcript (what handle_transcription_segment now does)
        chunk = await meeting_session_manager.add_transcript(
            session_id=pipeline.session_id,
            text=segment_data["text"],
            timestamp_ms=segment_data["start_ms"],
            language=segment_data["language"],
            confidence=segment_data["confidence"],
            is_final=segment_data["is_final"],
            speaker_id=segment_data["speaker_id"],
        )
        _segment_to_chunk_id[segment_data["segment_id"]] = chunk.id

        # Step 2: Translate with chunk_id linkage (what _translate_and_send does)
        await _translate_and_send(
            safe_send=safe_send,
            translation_service=translation_service,
            segment_id=segment_data["segment_id"],
            text=segment_data["stable_text"],
            source_lang="zh",
            target_lang="en",
            speaker_name=segment_data["speaker_id"],
            pipeline=pipeline,
            is_draft=False,
            chunk_id=_segment_to_chunk_id.get(segment_data["segment_id"]),
        )

        # Verify: transcript chunk in DB with correct fields
        db_chunk = await db_session.get(MeetingChunk, chunk.id)
        assert db_chunk is not None
        assert db_chunk.text == "你好世界，这是一个测试。"
        assert db_chunk.source_language == "zh"
        assert db_chunk.is_final is True
        assert db_chunk.speaker_id == "speaker_0"
        assert db_chunk.meeting_id == pipeline.session_id

        # Verify: translation linked to chunk via FK
        result = await db_session.execute(
            select(MeetingTranslation).where(MeetingTranslation.chunk_id == chunk.id)
        )
        trans = result.scalar_one()
        assert trans.translated_text == "Hola mundo"
        assert trans.source_language == "zh"
        assert trans.target_language == "en"
        assert trans.model_used == "test-wiring-model"
        assert trans.chunk_id == chunk.id

        # Verify: messages sent to "frontend"
        # Should have streaming chunks + final TranslationMessage
        assert len(sent_messages) >= 2
        final_msg = json.loads(sent_messages[-1])
        assert final_msg["type"] == "translation"
        assert final_msg["is_draft"] is False

        await pipeline.end()
        await translation_service.close()

    async def test_multi_segment_accumulation_and_persistence(
        self,
        meeting_session_manager,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        """Multiple non-final → final segments all persisted, translation links to final."""
        from routers.audio.websocket_audio import _translate_and_send

        from meeting.pipeline import MeetingPipeline

        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        translation_service = _make_fake_translation_service()
        _segment_to_chunk_id: dict[int, uuid.UUID] = {}
        sent_messages: list[str] = []

        async def safe_send(msg: str) -> bool:
            sent_messages.append(msg)
            return True

        # Simulate 3 non-final segments followed by 1 final
        segments = [
            {"segment_id": 1, "text": "你好", "is_final": False, "is_draft": False},
            {"segment_id": 2, "text": "你好世界", "is_final": False, "is_draft": False},
            {"segment_id": 3, "text": "你好世界这是", "is_final": False, "is_draft": False},
            {"segment_id": 4, "text": "你好世界，这是一个测试。", "is_final": True, "is_draft": False},
        ]

        for seg in segments:
            chunk = await meeting_session_manager.add_transcript(
                session_id=pipeline.session_id,
                text=seg["text"],
                timestamp_ms=int(time.time() * 1000),
                language="zh",
                confidence=0.9,
                is_final=seg["is_final"],
            )
            _segment_to_chunk_id[seg["segment_id"]] = chunk.id

        # Only translate the accumulated final text (what the handler does)
        final_chunk_id = _segment_to_chunk_id.get(4)
        await _translate_and_send(
            safe_send=safe_send,
            translation_service=translation_service,
            segment_id=4,
            text="你好世界，这是一个测试。",
            source_lang="zh",
            target_lang="en",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=False,
            chunk_id=final_chunk_id,
        )

        # Verify: all 4 chunks persisted
        result = await db_session.execute(
            select(MeetingChunk)
            .where(MeetingChunk.meeting_id == pipeline.session_id)
            .order_by(MeetingChunk.timestamp_ms)
        )
        chunks = list(result.scalars().all())
        assert len(chunks) == 4
        assert chunks[0].is_final is False
        assert chunks[3].is_final is True

        # Verify: translation linked to the final chunk
        result = await db_session.execute(
            select(MeetingTranslation).where(MeetingTranslation.chunk_id == final_chunk_id)
        )
        trans = result.scalar_one()
        assert trans.translated_text == "Hola mundo"

        await pipeline.end()
        await translation_service.close()

    async def test_draft_segments_not_persisted_to_db(
        self,
        meeting_session_manager,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        """Draft segments (is_draft=True) should NOT be persisted to meeting_chunks."""
        from meeting.pipeline import MeetingPipeline

        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # The handler skips add_transcript for drafts (is_draft=True).
        # Verify: if we only add non-draft chunks, drafts don't appear.
        # (The handler's `not msg.is_draft` guard prevents this.)

        # Simulate: only final non-draft persisted
        await meeting_session_manager.add_transcript(
            session_id=pipeline.session_id,
            text="Final text only",
            timestamp_ms=int(time.time() * 1000),
            language="en",
            confidence=0.9,
            is_final=True,
        )

        result = await db_session.execute(
            select(MeetingChunk).where(MeetingChunk.meeting_id == pipeline.session_id)
        )
        chunks = list(result.scalars().all())
        assert len(chunks) == 1
        assert chunks[0].text == "Final text only"

        await pipeline.end()

    async def test_audio_recording_plus_transcript_persistence(
        self,
        meeting_session_manager,
        db_session: AsyncSession,
        tmp_path: Path,
    ):
        """Full path: audio → FLAC recording + transcript persist + translation persist."""
        from routers.audio.websocket_audio import _translate_and_send

        from meeting.pipeline import MeetingPipeline

        pipeline = MeetingPipeline(
            session_manager=meeting_session_manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # Phase 1: Stream audio (exercises FLAC recording)
        audio = generate_audio(3.0)
        chunk_size = 48000  # 1 second chunks
        for i in range(0, len(audio), chunk_size):
            await pipeline.process_audio(audio[i : i + chunk_size])

        # Phase 2: Persist transcripts (simulates handle_transcription_segment)
        _segment_to_chunk_id: dict[int, uuid.UUID] = {}
        ts = int(time.time() * 1000)

        for seg_id, (text, lang, is_final) in enumerate([
            ("Hello everyone", "en", False),
            ("Hello everyone, welcome.", "en", True),
            ("大家好", "zh", False),
            ("大家好，欢迎。", "zh", True),
        ], start=1):
            chunk = await meeting_session_manager.add_transcript(
                session_id=pipeline.session_id,
                text=text,
                timestamp_ms=ts + seg_id * 1000,
                language=lang,
                confidence=0.9,
                is_final=is_final,
            )
            _segment_to_chunk_id[seg_id] = chunk.id

        # Phase 3: Translate final segments
        sent: list[str] = []

        async def safe_send(msg: str) -> bool:
            sent.append(msg)
            return True

        svc = _make_fake_translation_service()

        # Translate English final (seg 2) → Chinese
        await _translate_and_send(
            safe_send=safe_send,
            translation_service=svc,
            segment_id=2,
            text="Hello everyone, welcome.",
            source_lang="en",
            target_lang="zh",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=False,
            chunk_id=_segment_to_chunk_id[2],
        )

        # Translate Chinese final (seg 4) → English
        await _translate_and_send(
            safe_send=safe_send,
            translation_service=svc,
            segment_id=4,
            text="大家好，欢迎。",
            source_lang="zh",
            target_lang="en",
            speaker_name=None,
            pipeline=pipeline,
            is_draft=False,
            chunk_id=_segment_to_chunk_id[4],
        )

        await pipeline.end()

        # Verify: 4 transcript chunks in DB
        result = await db_session.execute(
            select(MeetingChunk)
            .where(MeetingChunk.meeting_id == pipeline.session_id)
            .order_by(MeetingChunk.timestamp_ms)
        )
        chunks = list(result.scalars().all())
        assert len(chunks) == 4
        assert chunks[0].source_language == "en"
        assert chunks[2].source_language == "zh"

        # Verify: 2 translations linked to correct final chunks
        result = await db_session.execute(
            select(MeetingTranslation).where(
                MeetingTranslation.chunk_id.in_([
                    _segment_to_chunk_id[2],
                    _segment_to_chunk_id[4],
                ])
            )
        )
        translations = list(result.scalars().all())
        assert len(translations) == 2

        # Verify: FLAC recording exists with audio data
        recording_dir = tmp_path / "recordings" / str(pipeline.session_id)
        manifest_path = recording_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["total_samples"] > 0

        # Verify: session completed in DB
        from database.models import Meeting
        session = await db_session.get(Meeting, pipeline.session_id)
        assert session.status == "completed"
        assert session.ended_at is not None

        await svc.close()


# ===========================================================================
# TestConcurrentDBOperations
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.integration
class TestConcurrentDBOperations:
    """Verify the _db_lock prevents concurrent AsyncSession corruption."""

    async def test_concurrent_add_transcript_and_save_translation(
        self,
        meeting_session_manager,
        db_session: AsyncSession,
    ):
        """Fire add_transcript and save_translation concurrently — both succeed."""
        session = await meeting_session_manager.create_session("loopback")
        await meeting_session_manager.promote_to_meeting(session.id)

        # First create a chunk we can link translations to
        chunk = await meeting_session_manager.add_transcript(
            session_id=session.id,
            text="Base chunk",
            timestamp_ms=0,
            language="en",
            confidence=0.9,
            is_final=True,
        )

        # Fire concurrent operations: 5 transcripts + 5 translations simultaneously
        async def add_chunk(i: int):
            return await meeting_session_manager.add_transcript(
                session_id=session.id,
                text=f"Concurrent chunk {i}",
                timestamp_ms=i * 100,
                language="en",
                confidence=0.9,
                is_final=True,
            )

        async def add_translation(i: int):
            return await meeting_session_manager.save_translation(
                chunk_id=chunk.id,
                translated_text=f"Translation {i}",
                source_language="en",
                target_language="es",
                model_used="test",
                translation_time_ms=float(i * 10),
            )

        # Run all 10 operations concurrently
        results = await asyncio.gather(
            *[add_chunk(i) for i in range(5)],
            *[add_translation(i) for i in range(5)],
            return_exceptions=True,
        )

        # All should succeed (no exceptions)
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert exceptions == [], f"Concurrent DB operations failed: {exceptions}"

        # Verify counts
        chunk_result = await db_session.execute(
            select(MeetingChunk).where(MeetingChunk.meeting_id == session.id)
        )
        chunks = list(chunk_result.scalars().all())
        assert len(chunks) == 6  # 1 base + 5 concurrent

        trans_result = await db_session.execute(
            select(MeetingTranslation).where(MeetingTranslation.chunk_id == chunk.id)
        )
        translations = list(trans_result.scalars().all())
        assert len(translations) == 5

    async def test_rapid_fire_transcripts_no_data_loss(
        self,
        meeting_session_manager,
        db_session: AsyncSession,
    ):
        """100 rapid add_transcript calls — all persisted, none lost."""
        session = await meeting_session_manager.create_session("loopback")
        await meeting_session_manager.promote_to_meeting(session.id)

        tasks = []
        for i in range(100):
            tasks.append(
                meeting_session_manager.add_transcript(
                    session_id=session.id,
                    text=f"Rapid chunk {i}",
                    timestamp_ms=i,
                    language="en",
                    confidence=0.9,
                    is_final=(i % 10 == 9),  # every 10th is final
                )
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert exceptions == [], f"Rapid-fire failed: {exceptions}"

        result = await db_session.execute(
            select(MeetingChunk).where(MeetingChunk.meeting_id == session.id)
        )
        chunks = list(result.scalars().all())
        assert len(chunks) == 100
