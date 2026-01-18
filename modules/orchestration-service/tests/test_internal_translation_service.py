from datetime import UTC, datetime

import pytest
from clients.translation_service_client import (
    TranslationRequest,
    TranslationServiceClient,
)
from internal_services import translation as translation_module
from internal_services.translation import (
    UnifiedTranslationError,
    get_unified_translation_service,
)


def _reset_singleton():
    """Ensure every test starts with a fresh facade instance."""
    translation_module._TRANSLATION_SINGLETON = None  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def reset_facade_state(monkeypatch):
    _reset_singleton()
    monkeypatch.setattr(translation_module, "TRANSLATION_MODULE_AVAILABLE", False, raising=False)
    yield
    _reset_singleton()


@pytest.mark.asyncio
async def test_health_reports_degraded_when_module_unavailable():
    service = get_unified_translation_service()
    health = await service.health()

    assert health["module_available"] is False
    assert health["status"] == "degraded"
    assert health["fallback_mode"] is True


@pytest.mark.asyncio
async def test_client_uses_embedded_service_when_available(monkeypatch):
    translation_module.TRANSLATION_MODULE_AVAILABLE = True
    service = get_unified_translation_service()

    # Replace translation-service dataclasses with lightweight test doubles
    class DummyTranslationRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DummyTranslationResult:
        def __init__(self):
            self.translated_text = "Hola mundo"
            self.source_language = "en"
            self.target_language = "es"
            self.confidence_score = 0.92
            self.processing_time = 0.12
            self.backend_used = "dummy"
            self.session_id = "test-session"
            self.timestamp = datetime.now(UTC).isoformat()

    class DummyService:
        async def translate(self, request):
            assert isinstance(request, DummyTranslationRequest)
            return DummyTranslationResult()

    monkeypatch.setattr(
        translation_module,
        "_TranslationRequest",
        DummyTranslationRequest,
        raising=False,
    )
    monkeypatch.setattr(
        translation_module, "_TranslationResult", DummyTranslationResult, raising=False
    )

    async def fake_initializer():
        service._service = DummyService()
        return service._service

    monkeypatch.setattr(service, "_ensure_service", fake_initializer)

    client = TranslationServiceClient(base_url="embedded")
    request = TranslationRequest(text="Hello world", source_language="en", target_language="es")

    response = await client.translate(request)

    assert response.translated_text == "Hola mundo"
    assert response.target_language == "es"
    assert response.backend_used == "dummy"

    stats = await service.get_statistics()
    assert stats["total_translations"] == 1
    assert stats["successful_translations"] == 1


@pytest.mark.asyncio
async def test_client_raises_when_no_backend(monkeypatch):
    service = get_unified_translation_service()

    async def no_service():
        return None

    monkeypatch.setattr(service, "_ensure_service", no_service)

    client = TranslationServiceClient(base_url="embedded")

    with pytest.raises(UnifiedTranslationError):
        await client.translate(
            TranslationRequest(text="Hello", source_language="en", target_language="es")
        )
