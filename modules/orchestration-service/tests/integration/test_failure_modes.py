"""Phase 3: Failure mode tests for orchestration service.

Tests verify the system degrades gracefully when downstream services
(transcription, translation, filesystem) are unreachable or failing.
Each test targets ONE specific failure scenario.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from routers.audio import websocket_audio as ws_mod
from routers.audio.websocket_audio import router


def create_test_app() -> FastAPI:
    """Build a minimal FastAPI app with just the audio WebSocket router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/audio")
    return app


async def _always_none_pipeline(sample_rate: int, channels: int):
    """Stub that simulates no DB available — prevents session leaks in tests."""
    return None


@pytest.fixture(autouse=True)
def disable_pipeline_for_failure_tests(monkeypatch):
    """Disable MeetingPipeline in failure mode tests to avoid DB session leaks.

    The failure mode tests focus on transcription/translation failures, not
    pipeline behaviour. Creating real pipelines here causes SAWarnings because
    the TestClient's anyio portal races with the session cleanup.
    """
    monkeypatch.setattr(ws_mod, "_try_create_pipeline", _always_none_pipeline)


# ---------------------------------------------------------------------------
# 3.1 Transcription Service Unreachable
# ---------------------------------------------------------------------------


class TestTranscriptionServiceUnreachable:
    """When the transcription service is not running, start_session should
    return a structured error and service_status with transcription=down,
    not crash or hang."""

    @pytest.fixture(autouse=True)
    def point_at_unreachable_host(self, monkeypatch):
        """Route transcription client to a port nothing listens on."""
        monkeypatch.setenv("TRANSCRIPTION_HOST", "127.0.0.1")
        monkeypatch.setenv("TRANSCRIPTION_PORT", "1")
        # Prevent any database or translation setup from interfering
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

    def test_start_session_returns_error_not_crash(self):
        """start_session with unreachable transcription sends error + service_status."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"
            assert "session_id" in connected

            ws.send_json({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            })

            # Server sends exactly 2 messages: error + service_status
            msg1 = ws.receive_json(mode="text")
            msg2 = ws.receive_json(mode="text")
            messages = [msg1, msg2]
            types = [m["type"] for m in messages]

            assert "error" in types, (
                f"Expected an 'error' message when transcription is unreachable, got: {types}"
            )
            assert "service_status" in types, (
                f"Expected a 'service_status' message, got: {types}"
            )

            status_msg = next(m for m in messages if m["type"] == "service_status")
            assert status_msg["transcription"] == "down"
            assert status_msg["translation"] == "down"

    def test_websocket_stays_open_after_transcription_failure(self):
        """The WebSocket connection itself remains alive after a failed start_session."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            })

            # Server sends exactly 2 messages: error + service_status
            msg1 = ws.receive_json(mode="text")
            msg2 = ws.receive_json(mode="text")
            types = {msg1["type"], msg2["type"]}
            assert "error" in types or "service_status" in types

            # Connection should still be open — send end_session gracefully
            ws.send_json({"type": "end_session"})


# ---------------------------------------------------------------------------
# 3.2 Translation (Ollama) Unreachable -- Graceful Degradation
# ---------------------------------------------------------------------------


class TestTranslationUnreachableGracefulDegradation:
    """When Ollama (LLM) is unreachable, translation should fail fast
    without corrupting the TranslationService's internal state."""

    @pytest.fixture(autouse=True)
    def unreachable_llm(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")

    @pytest.mark.asyncio
    async def test_translate_raises_on_unreachable_llm(self):
        """TranslationService.translate() raises, does not hang forever."""
        from translation.config import TranslationConfig
        from translation.service import TranslationService
        from livetranslate_common.models import TranslationRequest

        config = TranslationConfig(
            base_url="http://127.0.0.1:1/v1",
            timeout_s=2,
        )
        service = TranslationService(config)

        request = TranslationRequest(
            text="Hello world",
            source_language="en",
            target_language="es",
        )

        with pytest.raises(Exception):
            await service.translate(request)

        await service.close()

    @pytest.mark.asyncio
    async def test_service_reusable_after_failure(self):
        """TranslationService internal state is not corrupted after a connection failure."""
        from translation.config import TranslationConfig
        from translation.service import TranslationService
        from livetranslate_common.models import TranslationRequest

        config = TranslationConfig(
            base_url="http://127.0.0.1:1/v1",
            timeout_s=2,
        )
        service = TranslationService(config)

        request = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="es",
        )

        # First call fails
        with pytest.raises(Exception):
            await service.translate(request)

        # Context store and queue should still be intact
        assert service.context_store is not None
        assert not service._queue.full()

        # Second call should also fail (not hang or panic)
        with pytest.raises(Exception):
            await service.translate(request)

        await service.close()

    @pytest.mark.asyncio
    async def test_enqueue_translation_rejects_when_llm_down(self):
        """Queued translations eventually propagate the error via the future."""
        from translation.config import TranslationConfig
        from translation.service import TranslationService
        from livetranslate_common.models import TranslationRequest

        config = TranslationConfig(
            base_url="http://127.0.0.1:1/v1",
            timeout_s=2,
            max_queue_depth=5,
        )
        service = TranslationService(config)

        request = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="es",
        )

        with pytest.raises(Exception):
            await service.enqueue_translation(request)

        await service.close()


# ---------------------------------------------------------------------------
# 3.3 Service Status Reported Accurately at start_session
# ---------------------------------------------------------------------------


class TestServiceStatusOnStartSession:
    """Verify that start_session sends a ServiceStatusMessage reflecting
    actual downstream availability."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)

    def test_service_status_translation_down_when_no_llm_config(self, monkeypatch):
        """When LLM_BASE_URL is empty, translation should be reported as 'down'."""
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        # Need a real transcription service for this to be 'up', but we
        # are only testing the translation part. Point transcription at
        # unreachable host to keep the test fast.
        monkeypatch.setenv("TRANSCRIPTION_HOST", "127.0.0.1")
        monkeypatch.setenv("TRANSCRIPTION_PORT", "1")

        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            ws.send_json({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            })

            # Server sends exactly 2 messages: error + service_status
            msg1 = ws.receive_json(mode="text")
            msg2 = ws.receive_json(mode="text")
            messages = [msg1, msg2]

            status_messages = [m for m in messages if m["type"] == "service_status"]
            assert len(status_messages) >= 1, (
                f"Expected at least one service_status message, got types: "
                f"{[m['type'] for m in messages]}"
            )
            assert status_messages[0]["translation"] == "down"


# ---------------------------------------------------------------------------
# 3.4 Audio Forwarding When Transcription Client Not Connected
# ---------------------------------------------------------------------------


class TestAudioForwardingWithoutTranscriptionClient:
    """Binary audio frames sent before start_session (or after transcription
    failure) should be silently dropped, not crash the connection."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("TRANSCRIPTION_HOST", "127.0.0.1")
        monkeypatch.setenv("TRANSCRIPTION_PORT", "1")

    def test_audio_before_start_session_silently_dropped(self):
        """Sending binary audio before start_session does not crash."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            connected = ws.receive_json()
            assert connected["type"] == "connected"

            # Send audio BEFORE start_session
            audio = np.random.randn(4800).astype(np.float32)
            ws.send_bytes(audio.tobytes())

            # Connection should still be alive -- end gracefully
            ws.send_json({"type": "end_session"})

    def test_audio_after_failed_start_session_silently_dropped(self):
        """Sending binary audio after transcription connect failure does not crash."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            # start_session will fail because transcription is unreachable
            ws.send_json({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            })

            # Receive exactly 2 messages: error + service_status
            ws.receive_json(mode="text")
            ws.receive_json(mode="text")

            # Now send audio -- should be silently dropped
            audio = np.random.randn(4800).astype(np.float32)
            ws.send_bytes(audio.tobytes())

            # Connection should still be alive
            ws.send_json({"type": "end_session"})


# ---------------------------------------------------------------------------
# 3.6 FLAC Recorder Disk-Full Resilience
# ---------------------------------------------------------------------------


class TestFlacRecorderDiskFull:
    """FlacChunkRecorder.write() must not re-raise OSError from soundfile.
    A disk-full scenario should be absorbed so the audio pipeline continues."""

    def test_recorder_absorbs_disk_full_oserror(self, monkeypatch):
        """Simulated disk-full on sf.write does not propagate to caller."""
        import soundfile as sf

        from meeting.recorder import FlacChunkRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = FlacChunkRecorder(
                session_id="test-disk-full",
                base_path=Path(tmpdir),
                sample_rate=48000,
                channels=1,
                chunk_duration_s=0.1,  # tiny chunk so flush triggers fast
            )
            recorder.start()

            def failing_write(*args, **kwargs):
                raise OSError("No space left on device")

            monkeypatch.setattr(sf, "write", failing_write)

            # Generate enough audio to trigger a flush (chunk_duration_s=0.1 => 4800 samples)
            audio = np.random.randn(4800).astype(np.float32)

            # This SHOULD NOT raise -- the recorder should absorb the error.
            # If the current implementation does NOT absorb it, this test
            # correctly fails, signalling that the recorder needs hardening.
            try:
                recorder.write(audio)
                absorbed = True
            except OSError:
                absorbed = False

            recorder._running = False  # force stop to avoid secondary flush

            if not absorbed:
                pytest.fail(
                    "FlacChunkRecorder.write() propagated OSError to the caller. "
                    "The recorder's _flush_chunk() should wrap sf.write() in a "
                    "try/except so disk-full errors do not crash the audio pipeline."
                )

    def test_recorder_continues_after_transient_write_failure(self, monkeypatch):
        """After a transient sf.write failure, the recorder can write subsequent chunks."""
        import soundfile as sf

        from meeting.recorder import FlacChunkRecorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = FlacChunkRecorder(
                session_id="test-transient",
                base_path=Path(tmpdir),
                sample_rate=48000,
                channels=1,
                chunk_duration_s=0.1,
            )
            recorder.start()

            original_write = sf.write
            call_count = 0

            def intermittent_write(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("No space left on device")
                return original_write(*args, **kwargs)

            monkeypatch.setattr(sf, "write", intermittent_write)

            audio_chunk = np.random.randn(4800).astype(np.float32)

            # First chunk triggers a failing sf.write
            try:
                recorder.write(audio_chunk)
            except OSError:
                # If not absorbed, the recorder needs hardening
                pytest.skip(
                    "Recorder does not yet absorb OSError -- "
                    "test_recorder_absorbs_disk_full_oserror should catch this first."
                )

            # Second chunk should succeed (disk recovered)
            recorder.write(audio_chunk)
            recorder.stop()

            # At least one FLAC chunk should have been written
            flac_files = list(Path(tmpdir).rglob("*.flac"))
            assert len(flac_files) >= 1, "Expected at least one FLAC chunk after recovery"


# ---------------------------------------------------------------------------
# 3.7 Unknown/Malformed WebSocket Messages
# ---------------------------------------------------------------------------


class TestMalformedWebSocketMessages:
    """The WebSocket handler should ignore unknown or malformed messages
    without crashing the connection."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

    def test_unknown_message_type_ignored(self):
        """Sending a JSON message with an unknown type does not crash."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            ws.send_json({"type": "this_type_does_not_exist", "data": 42})

            # Connection should still be alive
            ws.send_json({"type": "end_session"})

    def test_invalid_json_ignored(self):
        """Sending non-JSON text does not crash the WebSocket handler."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            ws.send_text("this is not json {{{")

            # Connection should still be alive
            ws.send_json({"type": "end_session"})

    def test_empty_binary_frame_ignored(self):
        """Sending an empty binary frame does not crash."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            ws.send_bytes(b"")

            # Connection should still be alive
            ws.send_json({"type": "end_session"})


# ---------------------------------------------------------------------------
# 3.8 Promote to Meeting Without Database
# ---------------------------------------------------------------------------


class TestPromoteToMeetingWithoutDatabase:
    """promote_to_meeting when no database is configured should return
    a structured error, not crash."""

    @pytest.fixture(autouse=True)
    def no_database(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.setenv("TRANSCRIPTION_HOST", "127.0.0.1")
        monkeypatch.setenv("TRANSCRIPTION_PORT", "1")

    def test_promote_returns_error_when_no_database(self):
        """promote_to_meeting without a pipeline returns a structured error."""
        app = create_test_app()
        client = TestClient(app)

        with client.websocket_connect("/api/audio/stream") as ws:
            ws.receive_json()  # connected

            # start_session (will fail on transcription; pipeline=None due to patched factory)
            ws.send_json({
                "type": "start_session",
                "sample_rate": 48000,
                "channels": 1,
            })
            # Receive exactly 2 messages: error + service_status
            ws.receive_json(mode="text")
            ws.receive_json(mode="text")

            # Now try promote_to_meeting
            ws.send_json({"type": "promote_to_meeting"})

            msg = ws.receive_json(mode="text")
            assert msg["type"] == "error"
            assert "pipeline" in msg["message"].lower() or "database" in msg["message"].lower()
