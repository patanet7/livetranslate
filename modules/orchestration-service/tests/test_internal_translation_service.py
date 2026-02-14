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
    """Verify TranslationServiceClient delegates to embedded backend when enabled.

    The client disables embedded mode by default (``_prefer_embedded = False``).
    This test re-enables it and injects a dummy service matching the interface
    that ``_translate_embedded`` expects: keyword-arg ``translate()`` returning
    a dict, plus ``is_available()`` returning True.
    """
    translation_module.TRANSLATION_MODULE_AVAILABLE = True

    class DummyEmbeddedService:
        """Matches the interface expected by TranslationServiceClient._translate_embedded."""

        def is_available(self):
            return True

        async def translate(
            self,
            *,
            text,
            source_language,
            target_language,
            session_id=None,
            quality=None,
            model=None,
        ):
            return {
                "translated_text": "Hola mundo",
                "source_language": source_language or "en",
                "target_language": target_language,
                "confidence": 0.92,
                "processing_time": 0.12,
                "model_used": model or "default",
                "backend_used": "dummy",
                "session_id": session_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }

    client = TranslationServiceClient(base_url="embedded")
    # Re-enable embedded mode and inject the dummy service
    client._prefer_embedded = True
    client._embedded_service = DummyEmbeddedService()

    request = TranslationRequest(text="Hello world", source_language="en", target_language="es")

    response = await client.translate(request)

    assert response.translated_text == "Hola mundo"
    assert response.target_language == "es"
    assert response.backend_used == "dummy"


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
