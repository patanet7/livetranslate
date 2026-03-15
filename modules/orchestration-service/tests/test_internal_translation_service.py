"""
NOTE: This test file previously tested TranslationServiceClient and
internal_services.translation.UnifiedTranslationService. Both have been removed:

- clients/translation_service_client.py — deleted (superseded by translation.service.TranslationService)
- internal_services/translation.py — deleted (the embedded translation-service facade
  is no longer needed since TranslationService calls Ollama directly)

TODO: Add tests for translation.service.TranslationService when ready.
See modules/orchestration-service/src/translation/ for the new implementation.
"""

import pytest


def test_translation_service_module_placeholder():
    """Placeholder test — real tests for translation.service.TranslationService pending."""
    from translation.service import TranslationService
    from translation.config import TranslationConfig

    # Verify the new service can be instantiated
    config = TranslationConfig(
        llm_base_url="http://localhost:11434/v1",
        model="test-model",
    )
    service = TranslationService(config)
    assert service.config.model == "test-model"
