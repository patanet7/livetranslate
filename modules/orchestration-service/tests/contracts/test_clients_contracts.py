import json
import os
import sys
from typing import Any

import pytest

# Ensure orchestration src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from clients.audio_service_client import AudioServiceClient
from clients.translation_service_client import (
    LanguageDetectionResponse,
    TranslationRequest,
    TranslationServiceClient,
)


class MockResponse:
    def __init__(self, status: int = 200, payload: dict[str, Any] | None = None):
        self.status = status
        self._payload = payload or {}
        self._text = json.dumps(self._payload)

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockRequest:
    def __init__(self, response):
        self._response = response

    def __await__(self):
        async def _():
            return self._response

        return _().__await__()

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class MockSession:
    def __init__(self):
        self.closed = False
        self.calls = []

    async def close(self):
        self.closed = True

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/api/realtime/start"):
            return MockRequest(MockResponse(payload={"session_id": "session_123"}))
        if url.endswith("/api/realtime/audio"):
            return MockRequest(MockResponse(payload={"status": "chunk_added"}))
        if url.endswith("/api/realtime/stop"):
            return MockRequest(MockResponse(payload={"status": "streaming_stopped"}))
        if url.endswith("/api/analyze"):
            return MockRequest(
                MockResponse(payload={"status": "success", "metrics": {"duration_seconds": 1.0}})
            )
        if url.endswith("/api/process-pipeline"):
            return MockRequest(
                MockResponse(payload={"status": "success", "transcription": {"text": "hello"}})
            )
        if url.endswith("/api/translate"):
            return MockRequest(
                MockResponse(payload={"translated_text": "hola", "target_language": "es"})
            )
        if url.endswith("/api/detect"):
            return MockRequest(MockResponse(payload={"language": "en", "confidence": 0.9}))
        raise AssertionError(f"Unexpected POST URL {url}")

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/api/realtime/status/session_123"):
            return MockRequest(
                MockResponse(payload={"session_id": "session_123", "streaming_active": True})
            )
        if url.endswith("/api/stream-results/session_123"):
            return MockRequest(MockResponse(payload={"results": ["segment"], "count": 1}))
        if url.endswith("/api/processing-stats"):
            return MockRequest(MockResponse(payload={"active_sessions": 1}))
        if url.endswith("/api/languages"):
            return MockRequest(
                MockResponse(payload={"languages": [{"code": "en", "name": "English"}]})
            )
        if url.endswith("/api/performance"):
            return MockRequest(MockResponse(payload={"active_sessions": 0}))
        if url.endswith("/api/status"):
            return MockRequest(MockResponse(payload={"status": "ok"}))
        raise AssertionError(f"Unexpected GET URL {url}")


@pytest.mark.asyncio
async def test_audio_client_realtime_contracts():
    client = AudioServiceClient(base_url="http://mock-whisper")
    mock_session = MockSession()
    client.session = mock_session

    session_id = await client.start_realtime_session({"session_id": "session_123"})
    assert session_id == "session_123"

    chunk_result = await client.send_realtime_audio("session_123", b"audio-bytes")
    assert chunk_result["status"] == "chunk_added"

    stop_result = await client.stop_realtime_session("session_123")
    assert stop_result["status"] == "streaming_stopped"

    status_payload = await client.get_session_status("session_123")
    assert status_payload["streaming_active"] is True

    analysis = await client.analyze_audio(b"audio-bytes")
    assert analysis["status"] == "success"

    pipeline = await client.process_audio_batch({"audio_file": b"audio-bytes"}, "req-1")
    assert pipeline["status"] == "success"

    expected_post_urls = {
        "http://mock-whisper/api/realtime/start",
        "http://mock-whisper/api/realtime/audio",
        "http://mock-whisper/api/realtime/stop",
        "http://mock-whisper/api/analyze",
        "http://mock-whisper/api/process-pipeline",
    }
    seen_posts = {call[1] for call in mock_session.calls if call[0] == "POST"}
    assert expected_post_urls.issubset(seen_posts)


@pytest.mark.asyncio
async def test_translation_client_contracts():
    client = TranslationServiceClient(base_url="http://mock-translation")
    mock_session = MockSession()
    client.session = mock_session

    languages = await client.get_supported_languages()
    assert languages[0]["code"] == "en"

    detection = await client.detect_language("hello world")
    assert isinstance(detection, LanguageDetectionResponse)
    assert detection.language == "en"

    request = TranslationRequest(text="hello", target_language="es")
    response = await client.translate(request)
    assert response.translated_text == "hola"

    performance_calls = {call[1] for call in mock_session.calls}
    assert "http://mock-translation/api/languages" in performance_calls
    assert "http://mock-translation/api/detect" in performance_calls
    assert "http://mock-translation/api/translate" in performance_calls
