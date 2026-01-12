from datetime import datetime, timezone

import numpy as np
import pytest

from audio.audio_coordinator import ServiceClientPool
from audio.models import (
    AudioChunkMetadata,
    AudioFormat,
    ProcessingStatus,
    SourceType,
)
from clients.audio_service_client import (
    AudioServiceClient,
    TranscriptionRequest,
    TranscriptionResponse,
)
from clients.translation_service_client import (
    TranslationRequest as ClientTranslationRequest,
    TranslationResponse as ClientTranslationResponse,
)
from internal_services import audio as audio_module
from internal_services.audio import (
    get_unified_audio_service,
    UnifiedAudioError,
)


def _reset_singleton():
    audio_module._AUDIO_SINGLETON = None  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_audio_facade(monkeypatch):
    _reset_singleton()
    monkeypatch.setattr(audio_module, "AUDIO_MODULE_AVAILABLE", False, raising=False)
    yield
    _reset_singleton()


@pytest.mark.asyncio
async def test_audio_health_degraded_when_module_missing():
    service = get_unified_audio_service()
    health = await service.health()

    assert health["module_available"] is False
    assert health["status"] == "degraded"


@pytest.mark.asyncio
async def test_audio_client_uses_embedded(monkeypatch):
    audio_module.AUDIO_MODULE_AVAILABLE = True
    service = get_unified_audio_service()

    class DummyTranscriptionRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyTranscriptionResult:
        def __init__(self):
            self.text = "test transcript"
            self.language = "en"
            self.segments = [{"start": 0.0, "end": 1.0, "text": "test"}]
            self.speakers = [{"speaker": "A"}]
            self.confidence_score = 0.87
            self.processing_time = 0.05
            self.session_id = "session"
            self.timestamp = datetime.now(timezone.utc).isoformat()

    class DummyWhisperService:
        async def transcribe(self, request):
            assert isinstance(request, DummyTranscriptionRequest)
            return DummyTranscriptionResult()

    monkeypatch.setattr(
        audio_module, "_TranscriptionRequest", DummyTranscriptionRequest, raising=False
    )
    monkeypatch.setattr(
        audio_module, "_TranscriptionResult", DummyTranscriptionResult, raising=False
    )

    async def ensure_service():
        service._service = DummyWhisperService()
        return service._service

    monkeypatch.setattr(service, "_ensure_service", ensure_service)

    client = AudioServiceClient(base_url="embedded")
    request = TranscriptionRequest(language="en", model="whisper-base")
    response = await client.transcribe_stream(b"abc", request)

    assert response.text == "test transcript"
    assert response.language == "en"
    assert response.segments
    stats = await service.get_statistics()
    assert stats["successful_requests"] == 1


@pytest.mark.asyncio
async def test_audio_client_raises_when_no_backend(monkeypatch):
    service = get_unified_audio_service()

    async def ensure_none():
        return None

    monkeypatch.setattr(service, "_ensure_service", ensure_none)

    client = AudioServiceClient(base_url="embedded")

    with pytest.raises(UnifiedAudioError):
        await client.transcribe_stream(b"abc", TranscriptionRequest(language="en"))


@pytest.mark.asyncio
async def test_service_client_pool_uses_embedded_audio_client():
    audio_metadata = AudioChunkMetadata(
        chunk_id="chunk-1",
        session_id="session-1",
        file_path="chunk.wav",
        file_name="chunk.wav",
        file_size=1024,
        file_format=AudioFormat.WAV,
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        chunk_sequence=0,
        chunk_start_time=0.0,
        chunk_end_time=1.0,
        overlap_duration=0.0,
        processing_status=ProcessingStatus.PENDING,
        source_type=SourceType.BOT_AUDIO,
    )

    class StubAudioClient:
        async def transcribe_stream(
            self, audio_bytes: bytes, request: TranscriptionRequest
        ):
            assert isinstance(request, TranscriptionRequest)
            assert len(audio_bytes) > 0
            return TranscriptionResponse(
                text="hello there",
                language="en",
                segments=[{"start": 0.0, "end": 1.0, "text": "hello there"}],
                speakers=[{"speaker": "A"}],
                processing_time=0.12,
                confidence=0.93,
            )

    pool = ServiceClientPool(
        {"whisper_service": "embedded"},
        audio_client=StubAudioClient(),
    )

    audio_chunk = np.zeros(1600, dtype=np.float32)
    result = await pool.send_to_whisper_service(
        "session-1", audio_metadata, audio_chunk
    )

    assert result is not None
    assert result["text"] == "hello there"
    assert result["language"] == "en"
    assert result["confidence"] > 0.9


@pytest.mark.asyncio
async def test_service_client_pool_uses_embedded_translation_client():
    class StubTranslationClient:
        async def translate(self, request: ClientTranslationRequest):
            assert isinstance(request, ClientTranslationRequest)
            return ClientTranslationResponse(
                translated_text="hola",
                source_language=request.source_language or "auto",
                target_language=request.target_language,
                confidence=0.95,
                processing_time=0.2,
                model_used="embedded-model",
                backend_used="embedded",
            )

    pool = ServiceClientPool(
        {"translation_service": "embedded"},
        translation_client=StubTranslationClient(),  # type: ignore[arg-type]
    )

    transcript = {"text": "hello", "language": "en"}
    result = await pool.send_to_translation_service("session-1", transcript, "es")

    assert result is not None
    assert result["translated_text"] == "hola"
    assert result["confidence"] == 0.95
