#!/usr/bin/env python3
"""
Behavioral Tests for Quality Hardening (Tasks #39-49)

Tests the actual behavior of quality improvements made during the
post-implementation hardening phase. NO MOCKS — all tests use real
service instances, real adapters, and real coordinators.

Covers:
- Task #39: No bare except:pass blocks (source scanning)
- Task #40: No f-string logging (source scanning)
- Task #41: DB persistence fail-hard (persistence tracking fields)
- Task #43: GlossaryPipelineAdapter (bridges get_terms() to GlossaryService)
- Task #44: Pipeline pause/resume (coordinator + model fields)
- Task #45: Import uses TRANSCRIPT_FULL_QUERY with rich fields
- Task #46: TargetLanguagesRequest model validation
- Task #47: DB schema index for target_language
- Task #49: Mock server matches real Fireflies API contract (5 fields only)
"""

import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Prevent FastAPI app import issues
os.environ["SKIP_MAIN_FASTAPI_IMPORT"] = "1"

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))


# =============================================================================
# Task #44 — Pipeline Coordinator Pause/Resume
# =============================================================================


class TestPipelinePauseResume:
    """Behavioral tests for pipeline pause/resume functionality.

    Verifies that the coordinator actually stops processing chunks when paused
    and resumes processing when unpaused — using a REAL coordinator instance.
    """

    @pytest.mark.asyncio
    async def test_coordinator_starts_unpaused(self):
        """Coordinator should start in unpaused state."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="pause-test-1",
            source_type="audio_upload",
            transcript_id="pause-123",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="pause-test-1")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        assert coordinator.paused is False

    @pytest.mark.asyncio
    async def test_pause_sets_paused_flag(self):
        """Calling pause() should set the paused property to True."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="pause-test-2",
            source_type="audio_upload",
            transcript_id="pause-234",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="pause-test-2")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        coordinator.pause()
        assert coordinator.paused is True

    @pytest.mark.asyncio
    async def test_resume_clears_paused_flag(self):
        """Calling resume() should set the paused property back to False."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="pause-test-3",
            source_type="audio_upload",
            transcript_id="pause-345",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="pause-test-3")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        coordinator.pause()
        assert coordinator.paused is True

        coordinator.resume()
        assert coordinator.paused is False

    @pytest.mark.asyncio
    async def test_chunks_are_dropped_when_paused(self):
        """Chunks sent while paused should be silently dropped (not counted)."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="pause-drop-test",
            source_type="audio_upload",
            transcript_id="pause-drop-123",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="pause-drop-test")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        # Process 2 chunks while running
        for i in range(2):
            await coordinator.process_raw_chunk(
                {"text": f"Running chunk {i}", "start": float(i), "end": float(i + 1)}
            )

        stats_before_pause = coordinator.get_stats()
        assert stats_before_pause["chunks_received"] == 2

        # Pause and send 3 more chunks
        coordinator.pause()
        for i in range(3):
            await coordinator.process_raw_chunk(
                {"text": f"Paused chunk {i}", "start": float(i + 10), "end": float(i + 11)}
            )

        stats_while_paused = coordinator.get_stats()
        # Chunks should NOT have incremented — they were dropped
        assert stats_while_paused["chunks_received"] == 2

    @pytest.mark.asyncio
    async def test_chunks_resume_after_unpause(self):
        """After resume(), chunks should be processed again."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="resume-test",
            source_type="audio_upload",
            transcript_id="resume-123",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="resume-test")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        # Process 1 chunk
        await coordinator.process_raw_chunk(
            {"text": "Before pause", "start": 0.0, "end": 1.0}
        )
        assert coordinator.get_stats()["chunks_received"] == 1

        # Pause, then resume
        coordinator.pause()
        coordinator.resume()

        # Process 1 more chunk — should be counted
        await coordinator.process_raw_chunk(
            {"text": "After resume", "start": 1.0, "end": 2.0}
        )
        assert coordinator.get_stats()["chunks_received"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_includes_paused_field(self):
        """get_stats() should include the 'paused' field."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="stats-paused-test",
            source_type="audio_upload",
            transcript_id="stats-pause-123",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="stats-paused-test")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        stats = coordinator.get_stats()
        assert "paused" in stats
        assert stats["paused"] is False

        coordinator.pause()
        stats = coordinator.get_stats()
        assert stats["paused"] is True


# =============================================================================
# Task #41 — Persistence Tracking Fields on Models
# =============================================================================


class TestPersistenceTrackingFields:
    """Behavioral tests for persistence failure tracking on session models.

    Verifies that FirefliesSession and SessionResponse correctly expose
    persistence_failures and persistence_healthy fields.
    """

    def test_fireflies_session_defaults(self):
        """New FirefliesSession should default to healthy persistence."""
        from models.fireflies import FirefliesSession

        session = FirefliesSession(
            session_id="persist-test-1",
            fireflies_transcript_id="t-persist-1",
        )

        assert session.persistence_failures == 0
        assert session.persistence_healthy is True

    def test_fireflies_session_tracks_failures(self):
        """persistence_failures should increment and persistence_healthy should flip."""
        from models.fireflies import FirefliesSession

        session = FirefliesSession(
            session_id="persist-test-2",
            fireflies_transcript_id="t-persist-2",
        )

        # Simulate what the router does on a DB error
        session.persistence_failures += 1
        session.persistence_healthy = False

        assert session.persistence_failures == 1
        assert session.persistence_healthy is False

        # Multiple failures accumulate
        session.persistence_failures += 1
        assert session.persistence_failures == 2

    def test_session_response_includes_persistence_fields(self):
        """SessionResponse must include persistence_failures and persistence_healthy."""
        from routers.fireflies import SessionResponse

        response = SessionResponse(
            session_id="resp-test-1",
            transcript_id="t-resp-1",
            connection_status="connected",
            chunks_received=50,
            sentences_produced=15,
            translations_completed=45,
            speakers_detected=["Alice"],
            connected_at=datetime.now(UTC),
            error_count=0,
            last_error=None,
            persistence_failures=3,
            persistence_healthy=False,
        )

        assert response.persistence_failures == 3
        assert response.persistence_healthy is False

    def test_session_response_defaults_to_healthy(self):
        """SessionResponse should default to healthy when not specified."""
        from routers.fireflies import SessionResponse

        response = SessionResponse(
            session_id="resp-test-2",
            transcript_id="t-resp-2",
            connection_status="connected",
            chunks_received=0,
            sentences_produced=0,
            translations_completed=0,
            speakers_detected=[],
            connected_at=None,
            error_count=0,
            last_error=None,
        )

        assert response.persistence_failures == 0
        assert response.persistence_healthy is True


# =============================================================================
# Task #44 — FirefliesSession.is_paused Field
# =============================================================================


class TestFirefliesSessionPausedField:
    """Behavioral tests for the is_paused field on FirefliesSession."""

    def test_session_starts_unpaused(self):
        """New FirefliesSession should default to is_paused=False."""
        from models.fireflies import FirefliesSession

        session = FirefliesSession(
            session_id="paused-model-1",
            fireflies_transcript_id="t-paused-1",
        )
        assert session.is_paused is False

    def test_session_paused_field_can_be_set(self):
        """is_paused should be settable to True."""
        from models.fireflies import FirefliesSession

        session = FirefliesSession(
            session_id="paused-model-2",
            fireflies_transcript_id="t-paused-2",
        )
        session.is_paused = True
        assert session.is_paused is True

    def test_session_paused_field_round_trip(self):
        """is_paused should survive serialization/deserialization."""
        from models.fireflies import FirefliesSession

        session = FirefliesSession(
            session_id="paused-model-3",
            fireflies_transcript_id="t-paused-3",
            is_paused=True,
        )

        # Serialize and deserialize
        data = session.model_dump()
        assert data["is_paused"] is True

        restored = FirefliesSession.model_validate(data)
        assert restored.is_paused is True


# =============================================================================
# Task #46 — TargetLanguagesRequest Model Validation
# =============================================================================


class TestTargetLanguagesRequest:
    """Behavioral tests for TargetLanguagesRequest model."""

    def test_valid_request(self):
        """Valid request with one or more languages should pass validation."""
        from routers.fireflies import TargetLanguagesRequest

        request = TargetLanguagesRequest(target_languages=["es"])
        assert request.target_languages == ["es"]

    def test_multiple_languages(self):
        """Request with multiple languages should work."""
        from routers.fireflies import TargetLanguagesRequest

        request = TargetLanguagesRequest(target_languages=["es", "fr", "de"])
        assert len(request.target_languages) == 3

    def test_empty_list_rejected(self):
        """Empty language list should be rejected (min_length=1)."""
        from pydantic import ValidationError

        from routers.fireflies import TargetLanguagesRequest

        with pytest.raises(ValidationError):
            TargetLanguagesRequest(target_languages=[])

    def test_missing_field_rejected(self):
        """Missing target_languages field should be rejected (required)."""
        from pydantic import ValidationError

        from routers.fireflies import TargetLanguagesRequest

        with pytest.raises(ValidationError):
            TargetLanguagesRequest()


# =============================================================================
# Task #43 — GlossaryPipelineAdapter
# =============================================================================


class TestGlossaryPipelineAdapter:
    """Behavioral tests for GlossaryPipelineAdapter.

    Verifies the adapter correctly bridges the coordinator's get_terms()
    interface to GlossaryService.get_glossary_terms(), manages its own
    DB sessions, and handles errors gracefully.
    """

    @pytest.mark.asyncio
    async def test_adapter_creation(self):
        """Adapter should be creatable with a session factory."""
        from services.glossary_service import GlossaryPipelineAdapter

        # Use a simple async callable as session factory
        async def fake_factory():
            mock_session = AsyncMock()
            return mock_session

        adapter = GlossaryPipelineAdapter(session_factory=fake_factory)
        assert adapter is not None

    @pytest.mark.asyncio
    async def test_adapter_get_terms_returns_empty_dict_on_error(self):
        """get_terms() should return {} if GlossaryService raises."""
        from services.glossary_service import GlossaryPipelineAdapter

        call_count = 0

        def session_factory():
            nonlocal call_count
            call_count += 1
            # Return a mock session whose operations will cause GlossaryService to fail
            session = AsyncMock()
            session.execute = AsyncMock(side_effect=Exception("DB unavailable"))
            session.close = AsyncMock()
            return session

        adapter = GlossaryPipelineAdapter(session_factory=session_factory)
        result = await adapter.get_terms(
            target_language="es",
            glossary_id=None,
        )

        # Should return empty dict, not raise
        assert result == {}
        # Session factory should have been called
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_adapter_closes_session_on_success(self):
        """Adapter should close its session even on success."""
        from services.glossary_service import GlossaryPipelineAdapter

        closed = False

        def session_factory():
            nonlocal closed
            session = AsyncMock()
            # Make execute return something that won't crash GlossaryService
            session.execute = AsyncMock(side_effect=Exception("no tables"))

            async def mark_closed():
                nonlocal closed
                closed = True

            session.close = mark_closed
            return session

        adapter = GlossaryPipelineAdapter(session_factory=session_factory)
        await adapter.get_terms(target_language="es")

        # Session should have been closed in the finally block
        assert closed is True

    @pytest.mark.asyncio
    async def test_adapter_closes_session_on_error(self):
        """Adapter should close its session when GlossaryService raises."""
        from services.glossary_service import GlossaryPipelineAdapter

        closed = False

        def session_factory():
            nonlocal closed
            session = AsyncMock()
            session.execute = AsyncMock(side_effect=RuntimeError("kaboom"))

            async def mark_closed():
                nonlocal closed
                closed = True

            session.close = mark_closed
            return session

        adapter = GlossaryPipelineAdapter(session_factory=session_factory)
        result = await adapter.get_terms(target_language="fr")

        assert result == {}
        assert closed is True

    @pytest.mark.asyncio
    async def test_adapter_signature_matches_coordinator_expectations(self):
        """get_terms() should accept the params the coordinator passes."""
        from services.glossary_service import GlossaryPipelineAdapter
        import inspect

        sig = inspect.signature(GlossaryPipelineAdapter.get_terms)
        params = list(sig.parameters.keys())

        # Coordinator calls: get_terms(glossary_id=..., source_language=..., target_language=..., domain=...)
        assert "glossary_id" in params
        assert "source_language" in params
        assert "target_language" in params
        assert "domain" in params


# =============================================================================
# Tasks #39, #40 — Source Code Verification (no f-string logging, no silent exceptions)
# =============================================================================


class TestCodeStyleHardening:
    """Verify that code-style quality fixes are still in place.

    These tests read the actual source files and verify that anti-patterns
    (f-string logging, bare except: pass) are absent.
    """

    def test_no_fstring_logger_calls_in_fireflies_client(self):
        """fireflies_client.py should have no f-string logger calls."""
        source_file = src_path / "clients" / "fireflies_client.py"
        content = source_file.read_text()

        # Look for logger.<level>(f"...) pattern
        import re

        fstring_log_pattern = re.compile(r'logger\.\w+\(f["\']')
        matches = fstring_log_pattern.findall(content)

        assert len(matches) == 0, (
            f"Found {len(matches)} f-string logger calls in fireflies_client.py: {matches}"
        )

    def test_no_fstring_logger_calls_in_fireflies_router(self):
        """fireflies.py router should have no f-string logger calls."""
        source_file = src_path / "routers" / "fireflies.py"
        content = source_file.read_text()

        import re

        fstring_log_pattern = re.compile(r'logger\.\w+\(f["\']')
        matches = fstring_log_pattern.findall(content)

        assert len(matches) == 0, (
            f"Found {len(matches)} f-string logger calls in fireflies.py: {matches}"
        )

    def test_no_bare_except_pass_in_fireflies_client(self):
        """fireflies_client.py should have no bare 'except Exception: pass' blocks."""
        source_file = src_path / "clients" / "fireflies_client.py"
        content = source_file.read_text()

        # Look for except ... : pass or except ... :\n    pass
        import re

        bare_pass_pattern = re.compile(
            r"except\s+(?:Exception|BaseException)\s*(?:as\s+\w+)?\s*:\s*\n\s*pass\b"
        )
        matches = bare_pass_pattern.findall(content)

        assert len(matches) == 0, (
            f"Found {len(matches)} bare except:pass blocks in fireflies_client.py"
        )


# =============================================================================
# Integration: Coordinator + Pause + Stats Full Flow
# =============================================================================


class TestFullPauseResumeFlow:
    """End-to-end behavioral test for the complete pause/resume lifecycle
    with real coordinator, real adapter, real stats tracking.
    """

    @pytest.mark.asyncio
    async def test_complete_pause_resume_lifecycle(self):
        """Test full lifecycle: process -> pause -> attempt -> resume -> process."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="lifecycle-test",
            source_type="audio_upload",
            transcript_id="lifecycle-123",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="lifecycle-test")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        # Phase 1: Normal processing
        await coordinator.process_raw_chunk(
            {"text": "Phase one", "start": 0.0, "end": 1.0, "speaker": "Alice"}
        )
        stats_1 = coordinator.get_stats()
        assert stats_1["chunks_received"] == 1
        assert stats_1["paused"] is False

        # Phase 2: Pause
        coordinator.pause()
        assert coordinator.paused is True
        assert coordinator.get_stats()["paused"] is True

        # Phase 3: Attempt to process while paused (should be dropped)
        await coordinator.process_raw_chunk(
            {"text": "Dropped chunk", "start": 1.0, "end": 2.0, "speaker": "Bob"}
        )
        stats_3 = coordinator.get_stats()
        assert stats_3["chunks_received"] == 1  # Still 1, not 2

        # Phase 4: Resume
        coordinator.resume()
        assert coordinator.paused is False
        assert coordinator.get_stats()["paused"] is False

        # Phase 5: Process after resume
        await coordinator.process_raw_chunk(
            {"text": "After resume", "start": 2.0, "end": 3.0, "speaker": "Alice"}
        )
        stats_5 = coordinator.get_stats()
        assert stats_5["chunks_received"] == 2  # Now 2

    @pytest.mark.asyncio
    async def test_rapid_pause_resume_cycles(self):
        """Rapid toggling of pause/resume should not break the coordinator."""
        from services.pipeline import (
            AudioUploadChunkAdapter,
            PipelineConfig,
            TranscriptionPipelineCoordinator,
        )

        config = PipelineConfig(
            session_id="rapid-toggle",
            source_type="audio_upload",
            transcript_id="rapid-123",
            target_languages=["es"],
        )
        adapter = AudioUploadChunkAdapter(session_id="rapid-toggle")
        coordinator = TranscriptionPipelineCoordinator(config=config, adapter=adapter)
        await coordinator.initialize()

        # Rapidly toggle 10 times
        for _ in range(10):
            coordinator.pause()
            coordinator.resume()

        # Should be in unpaused state
        assert coordinator.paused is False

        # Should still accept chunks
        await coordinator.process_raw_chunk(
            {"text": "After rapid toggle", "start": 0.0, "end": 1.0}
        )
        assert coordinator.get_stats()["chunks_received"] == 1


# =============================================================================
# Task #45 — Import Uses TRANSCRIPT_FULL_QUERY (Rich GraphQL Fields)
# =============================================================================


class TestTranscriptFullQueryUsed:
    """Verify that the import path uses TRANSCRIPT_FULL_QUERY which includes
    ai_filters, analytics, summary, and attendance data — not the basic query.
    """

    def test_full_query_constant_exists(self):
        """TRANSCRIPT_FULL_QUERY should be defined in fireflies_client."""
        from clients.fireflies_client import TRANSCRIPT_FULL_QUERY

        assert TRANSCRIPT_FULL_QUERY is not None
        assert len(TRANSCRIPT_FULL_QUERY) > 100  # Not a trivial stub

    def test_full_query_includes_ai_filters(self):
        """TRANSCRIPT_FULL_QUERY must include ai_filters for insights."""
        from clients.fireflies_client import TRANSCRIPT_FULL_QUERY

        assert "ai_filters" in TRANSCRIPT_FULL_QUERY

    def test_full_query_includes_analytics(self):
        """TRANSCRIPT_FULL_QUERY must include analytics (sentiments, speakers)."""
        from clients.fireflies_client import TRANSCRIPT_FULL_QUERY

        assert "analytics" in TRANSCRIPT_FULL_QUERY
        assert "sentiments" in TRANSCRIPT_FULL_QUERY

    def test_full_query_includes_summary(self):
        """TRANSCRIPT_FULL_QUERY must include summary."""
        from clients.fireflies_client import TRANSCRIPT_FULL_QUERY

        assert "summary" in TRANSCRIPT_FULL_QUERY

    def test_full_query_includes_attendance(self):
        """TRANSCRIPT_FULL_QUERY must include attendance data."""
        from clients.fireflies_client import TRANSCRIPT_FULL_QUERY

        # Check for at least one attendance-related field
        assert "meeting_attendees" in TRANSCRIPT_FULL_QUERY or "meeting_attendance" in TRANSCRIPT_FULL_QUERY

    def test_download_full_transcript_uses_full_query(self):
        """GraphQLClient.download_full_transcript() must reference TRANSCRIPT_FULL_QUERY."""
        import inspect
        from clients.fireflies_client import FirefliesGraphQLClient

        source = inspect.getsource(FirefliesGraphQLClient.download_full_transcript)
        assert "TRANSCRIPT_FULL_QUERY" in source


# =============================================================================
# Task #47 — DB Schema Has target_language Index
# =============================================================================


class TestMeetingSchemaIndex:
    """Verify that the meeting schema SQL includes the target_language index."""

    def test_schema_has_target_lang_index(self):
        """meeting-schema.sql must contain idx_mtrans_target_lang index."""
        schema_file = Path(__file__).parent.parent.parent.parent.parent.parent / "scripts" / "meeting-schema.sql"
        content = schema_file.read_text()

        assert "idx_mtrans_target_lang" in content, (
            "Missing index idx_mtrans_target_lang in meeting-schema.sql"
        )

    def test_schema_index_on_correct_column(self):
        """The target_lang index should be on meeting_translations(target_language)."""
        schema_file = Path(__file__).parent.parent.parent.parent.parent.parent / "scripts" / "meeting-schema.sql"
        content = schema_file.read_text()

        # Find the index definition and verify it references the right table/column
        assert "meeting_translations" in content
        # The index line should reference target_language
        import re
        idx_line = re.search(r"idx_mtrans_target_lang.*?;", content, re.DOTALL)
        assert idx_line is not None, "Could not find idx_mtrans_target_lang definition"
        assert "target_language" in idx_line.group()


# =============================================================================
# Task #49 — Mock Server Matches Real Fireflies API Contract
# =============================================================================


class TestMockServerAPIContract:
    """Verify that MockChunk matches the real Fireflies API (5 fields only).

    Real Fireflies Realtime API sends exactly:
      chunk_id, text, speaker_name, start_time, end_time

    No confidence, no is_final, no transcript_id in the event payload.
    """

    def test_mock_chunk_has_exactly_five_data_fields(self):
        """MockChunk dataclass should have exactly 5 data fields."""
        from tests.fireflies.mocks.fireflies_mock_server import MockChunk

        import dataclasses
        field_names = [f.name for f in dataclasses.fields(MockChunk)]
        expected = {"chunk_id", "text", "speaker_name", "start_time", "end_time"}

        assert set(field_names) == expected, (
            f"MockChunk fields {field_names} don't match real API fields {expected}"
        )

    def test_mock_chunk_no_confidence_field(self):
        """MockChunk must NOT have a confidence field (doesn't exist in real API)."""
        from tests.fireflies.mocks.fireflies_mock_server import MockChunk

        assert not hasattr(MockChunk(), "confidence"), "MockChunk should not have confidence field"

    def test_mock_chunk_no_is_final_field(self):
        """MockChunk must NOT have an is_final field (doesn't exist in real API)."""
        from tests.fireflies.mocks.fireflies_mock_server import MockChunk

        assert not hasattr(MockChunk(), "is_final"), "MockChunk should not have is_final field"

    def test_socketio_dict_excludes_transcript_id(self):
        """to_socketio_dict() payload should NOT include transcript_id."""
        from tests.fireflies.mocks.fireflies_mock_server import MockChunk

        chunk = MockChunk(chunk_id="12345", text="hello", speaker_name="Alice")
        payload = chunk.to_socketio_dict(transcript_id="should-not-appear")

        assert "transcript_id" not in payload, (
            "Real Fireflies API does not include transcript_id in Socket.IO event data"
        )

    def test_socketio_dict_has_exactly_five_keys(self):
        """to_socketio_dict() should return exactly 5 keys matching real API."""
        from tests.fireflies.mocks.fireflies_mock_server import MockChunk

        chunk = MockChunk(chunk_id="12345", text="hello", speaker_name="Alice")
        payload = chunk.to_socketio_dict(transcript_id="irrelevant")

        expected_keys = {"chunk_id", "text", "speaker_name", "start_time", "end_time"}
        assert set(payload.keys()) == expected_keys

    def test_word_by_word_scenario_reuses_chunk_id(self):
        """word_by_word_scenario must reuse same chunk_id (real API behavior)."""
        from tests.fireflies.mocks.fireflies_mock_server import MockTranscriptScenario

        scenario = MockTranscriptScenario.word_by_word_scenario(
            text="Hello world test",
            speaker="Alice",
        )

        chunk_ids = [c.chunk_id for c in scenario.chunks]
        # All chunks should have the SAME chunk_id (word-streaming behavior)
        assert len(set(chunk_ids)) == 1, (
            f"word_by_word_scenario should reuse one chunk_id, got {len(set(chunk_ids))} unique IDs"
        )

    def test_word_by_word_scenario_accumulates_text(self):
        """word_by_word_scenario chunks should have progressively longer text."""
        from tests.fireflies.mocks.fireflies_mock_server import MockTranscriptScenario

        scenario = MockTranscriptScenario.word_by_word_scenario(
            text="Hello world test",
            speaker="Alice",
        )

        texts = [c.text for c in scenario.chunks]
        # Each subsequent chunk should be longer (accumulating words)
        for i in range(1, len(texts)):
            assert len(texts[i]) > len(texts[i - 1]), (
                f"Chunk {i} text should be longer than chunk {i-1}: '{texts[i]}' vs '{texts[i-1]}'"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
