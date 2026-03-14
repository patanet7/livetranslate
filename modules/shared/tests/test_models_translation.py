"""Behavioral tests for TranslationContext, TranslationRequest, TranslationResponse."""

from __future__ import annotations

import pytest

from livetranslate_common.models.translation import (
    TranslationContext,
    TranslationRequest,
    TranslationResponse,
)


class TestTranslationContext:
    def test_context_creation(self) -> None:
        ctx = TranslationContext(text="Good morning", translation="Bonjour")
        assert ctx.text == "Good morning"
        assert ctx.translation == "Bonjour"


class TestTranslationRequest:
    def test_request_with_context(self) -> None:
        ctx = TranslationContext(text="Hello", translation="Hola")
        req = TranslationRequest(
            text="Goodbye",
            source_language="en",
            target_language="es",
            context=[ctx],
        )
        assert req.text == "Goodbye"
        assert req.source_language == "en"
        assert req.target_language == "es"
        assert len(req.context) == 1
        assert req.context[0].translation == "Hola"

    def test_request_no_context(self) -> None:
        req = TranslationRequest(
            text="Test sentence.",
            source_language="en",
            target_language="de",
        )
        assert req.context == []
        assert req.context_window_size == 5
        assert req.max_context_tokens == 500
        assert req.glossary_terms == {}
        assert req.speaker_name is None

    def test_request_with_glossary(self) -> None:
        req = TranslationRequest(
            text="The API endpoint returned a 200 OK.",
            source_language="en",
            target_language="ja",
            glossary_terms={"API": "API", "endpoint": "エンドポイント"},
            speaker_name="Alice",
        )
        assert req.glossary_terms["endpoint"] == "エンドポイント"
        assert req.speaker_name == "Alice"

    def test_request_json_roundtrip(self) -> None:
        ctx_items = [TranslationContext(text=f"s{i}", translation=f"t{i}") for i in range(3)]
        original = TranslationRequest(
            text="Round trip test.",
            source_language="en",
            target_language="fr",
            context=ctx_items,
            context_window_size=10,
            max_context_tokens=1000,
            glossary_terms={"LiveTranslate": "LiveTranslate"},
            speaker_name="Bob",
        )
        json_str = original.model_dump_json()
        restored = TranslationRequest.model_validate_json(json_str)
        assert restored.text == original.text
        assert restored.target_language == original.target_language
        assert len(restored.context) == 3
        assert restored.speaker_name == "Bob"
        assert restored.glossary_terms == {"LiveTranslate": "LiveTranslate"}


class TestTranslationResponse:
    def test_response_creation(self) -> None:
        resp = TranslationResponse(
            translated_text="Bonjour le monde",
            source_language="en",
            target_language="fr",
            model_used="qwen3.5:7b",
            latency_ms=87.4,
        )
        assert resp.translated_text == "Bonjour le monde"
        assert resp.source_language == "en"
        assert resp.target_language == "fr"
        assert resp.model_used == "qwen3.5:7b"
        assert resp.latency_ms == pytest.approx(87.4)
        assert resp.quality_score is None

    def test_response_with_quality_score(self) -> None:
        resp = TranslationResponse(
            translated_text="Guten Morgen",
            source_language="en",
            target_language="de",
            model_used="qwen3.5:7b",
            latency_ms=112.0,
            quality_score=0.92,
        )
        assert resp.quality_score == pytest.approx(0.92)
