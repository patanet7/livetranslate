"""
Contract Tests for Translation Service Integration

These tests validate that:
1. RollingWindowTranslator correctly uses TranslationRequest contracts
2. TranslationResponse contracts are properly handled
3. The entire flow uses actual service contracts, not fake models

This ensures the Fireflies integration isn't a "surprise" to the translation service.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(orchestration_root))

# Import ACTUAL contracts from production code
from clients.translation_service_client import (
    TranslationRequest,
    TranslationResponse,
)
from models.fireflies import (
    TranslationUnit,
)

# =============================================================================
# Contract Validation Helpers
# =============================================================================


def validate_translation_request(request: TranslationRequest) -> None:
    """Validate a TranslationRequest matches expected contract."""
    # Required fields
    assert isinstance(request.text, str), "text must be a string"
    assert len(request.text) > 0, "text cannot be empty"
    assert isinstance(request.target_language, str), "target_language must be a string"

    # Optional fields with defaults
    assert request.source_language is None or isinstance(request.source_language, str)
    assert isinstance(request.model, str), "model must have a default"
    assert request.quality in ("fast", "balanced", "quality"), f"Invalid quality: {request.quality}"


def validate_translation_response(response: TranslationResponse) -> None:
    """Validate a TranslationResponse matches expected contract."""
    assert isinstance(response.translated_text, str), "translated_text must be a string"
    assert isinstance(response.target_language, str), "target_language must be a string"
    assert isinstance(response.confidence, float), "confidence must be a float"
    assert 0.0 <= response.confidence <= 1.0, "confidence must be between 0 and 1"
    assert isinstance(response.processing_time, float), "processing_time must be a float"


# =============================================================================
# Contract-Based Mock Client
# =============================================================================


class ContractValidatingTranslationClient:
    """
    Mock translation client that validates contracts.

    This mock ensures that:
    1. All requests match the TranslationRequest contract
    2. All responses match the TranslationResponse contract
    3. Contract violations are detected immediately
    """

    def __init__(self, translations: dict | None = None):
        self.translations = translations or {}
        self.requests_received = []
        self.call_count = 0

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate with contract validation."""
        # VALIDATE INPUT CONTRACT
        validate_translation_request(request)
        self.requests_received.append(request)
        self.call_count += 1

        # Generate response
        translated_text = self.translations.get(
            request.text, f"[{request.target_language}] {request.text}"
        )

        # BUILD RESPONSE USING ACTUAL CONTRACT
        response = TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language or "auto",
            target_language=request.target_language,
            confidence=0.95,
            processing_time=0.05,
            model_used=request.model,
            backend_used="contract_test",
            session_id=request.session_id,
            timestamp=datetime.now(UTC).isoformat(),
        )

        # VALIDATE OUTPUT CONTRACT
        validate_translation_response(response)

        return response

    async def health_check(self):
        return {"status": "healthy", "mode": "contract_test"}

    async def close(self):
        pass


# =============================================================================
# TranslationRequest Contract Tests
# =============================================================================


class TestTranslationRequestContract:
    """Test that TranslationRequest contract is correctly used."""

    def test_request_requires_text(self):
        """TranslationRequest must have text field."""
        # GIVEN: Required fields
        # WHEN: Creating a request
        request = TranslationRequest(
            text="Hello world",
            target_language="es",
        )

        # THEN: Required fields are set
        assert request.text == "Hello world"
        assert request.target_language == "es"

    def test_request_has_defaults(self):
        """TranslationRequest has sensible defaults."""
        request = TranslationRequest(
            text="Hello",
            target_language="es",
        )

        # THEN: Defaults are applied
        assert request.source_language is None  # auto-detect
        assert request.model == "default"
        assert request.quality == "balanced"
        assert request.session_id is None

    def test_request_validates_quality(self):
        """TranslationRequest accepts valid quality values."""
        for quality in ["fast", "balanced", "quality"]:
            request = TranslationRequest(
                text="Hello",
                target_language="es",
                quality=quality,
            )
            assert request.quality == quality

    def test_request_serialization(self):
        """TranslationRequest can be serialized to dict."""
        request = TranslationRequest(
            text="Hello world",
            target_language="es",
            source_language="en",
            quality="balanced",
        )

        # THEN: Can serialize
        data = request.model_dump()
        assert data["text"] == "Hello world"
        assert data["target_language"] == "es"
        assert data["source_language"] == "en"


# =============================================================================
# TranslationResponse Contract Tests
# =============================================================================


class TestTranslationResponseContract:
    """Test that TranslationResponse contract is correctly used."""

    def test_response_has_required_fields(self):
        """TranslationResponse has required translated_text."""
        response = TranslationResponse(
            translated_text="Hola mundo",
            target_language="es",
        )

        assert response.translated_text == "Hola mundo"
        assert response.target_language == "es"

    def test_response_has_defaults(self):
        """TranslationResponse has sensible defaults."""
        response = TranslationResponse(
            translated_text="Hola",
            target_language="es",
        )

        # THEN: Defaults are applied
        assert response.source_language == "auto"
        assert response.confidence == 0.95
        assert response.processing_time == 0.0
        assert response.model_used == "default"

    def test_response_accepts_all_fields(self):
        """TranslationResponse accepts all optional fields."""
        response = TranslationResponse(
            translated_text="Hola mundo",
            source_language="en",
            target_language="es",
            confidence=0.98,
            processing_time=0.123,
            model_used="advanced",
            backend_used="local_llm",
            session_id="session-123",
            timestamp="2026-01-10T12:00:00Z",
        )

        assert response.confidence == 0.98
        assert response.processing_time == 0.123
        assert response.backend_used == "local_llm"


# =============================================================================
# RollingWindowTranslator Contract Integration Tests
# =============================================================================


class TestRollingWindowTranslatorContracts:
    """Test that RollingWindowTranslator correctly uses service contracts."""

    @pytest.fixture
    def contract_client(self):
        """Create contract-validating client."""
        return ContractValidatingTranslationClient(
            translations={
                "Hello, how are you?": "Hola, ¿cómo estás?",
                "Good morning everyone.": "Buenos días a todos.",
                "Let's start the meeting.": "Comencemos la reunión.",
            }
        )

    @pytest.fixture
    def translation_unit(self):
        """Create a test translation unit."""
        return TranslationUnit(
            text="Hello, how are you?",
            speaker_name="Alice",
            start_time=0.0,
            end_time=2.0,
            session_id="session-001",
            transcript_id="transcript-001",
            chunk_ids=["chunk-001"],
            boundary_type="punctuation",
        )

    @pytest.mark.asyncio
    async def test_translator_sends_valid_request(self, contract_client, translation_unit):
        """RollingWindowTranslator sends valid TranslationRequest."""
        from services.rolling_window_translator import RollingWindowTranslator

        translator = RollingWindowTranslator(
            translation_client=contract_client,
            window_size=3,
        )

        # WHEN: Translating
        await translator.translate(
            unit=translation_unit,
            target_language="es",
            source_language="en",
        )

        # THEN: Request was valid (contract validation passed)
        assert contract_client.call_count == 1
        request = contract_client.requests_received[0]

        # Verify request contract fields
        assert request.text == "Hello, how are you?"
        assert request.target_language == "es"
        assert request.source_language == "en"
        assert request.quality == "balanced"  # Default from translator

    @pytest.mark.asyncio
    async def test_translator_handles_response_correctly(self, contract_client, translation_unit):
        """RollingWindowTranslator correctly processes TranslationResponse."""
        from services.rolling_window_translator import RollingWindowTranslator

        translator = RollingWindowTranslator(
            translation_client=contract_client,
        )

        # WHEN: Translating
        result = await translator.translate(
            unit=translation_unit,
            target_language="es",
        )

        # THEN: Result is properly mapped from response
        assert result.original == "Hello, how are you?"
        assert result.translated == "Hola, ¿cómo estás?"
        assert result.target_language == "es"
        assert result.confidence >= 0.0  # From response
        assert result.speaker_name == "Alice"  # From unit

    @pytest.mark.asyncio
    async def test_multiple_translations_use_valid_contracts(self, contract_client):
        """Multiple translations all use valid contracts."""
        from services.rolling_window_translator import RollingWindowTranslator

        translator = RollingWindowTranslator(
            translation_client=contract_client,
            window_size=3,
        )

        sentences = [
            ("Hello, how are you?", "Alice"),
            ("Good morning everyone.", "Bob"),
            ("Let's start the meeting.", "Alice"),
        ]

        # WHEN: Translating multiple sentences
        for i, (text, speaker) in enumerate(sentences):
            unit = TranslationUnit(
                text=text,
                speaker_name=speaker,
                start_time=float(i * 2),
                end_time=float(i * 2 + 2),
                session_id="session-001",
                transcript_id="transcript-001",
                chunk_ids=[f"chunk-{i:03d}"],
                boundary_type="punctuation",
            )
            await translator.translate(unit, target_language="es")

        # THEN: All requests were valid
        assert contract_client.call_count == 3

        # Each request had valid contract
        for request in contract_client.requests_received:
            validate_translation_request(request)

    @pytest.mark.asyncio
    async def test_context_window_doesnt_break_contracts(self, contract_client):
        """Context window feature still uses valid contracts."""
        from services.rolling_window_translator import RollingWindowTranslator

        translator = RollingWindowTranslator(
            translation_client=contract_client,
            window_size=2,
            include_cross_speaker_context=True,
        )

        # Build context by translating multiple sentences
        units = [
            TranslationUnit(
                text="First sentence here.",
                speaker_name="Alice",
                start_time=0.0,
                end_time=2.0,
                session_id="test",
                transcript_id="test",
                chunk_ids=["c1"],
                boundary_type="punctuation",
            ),
            TranslationUnit(
                text="Second sentence here.",
                speaker_name="Alice",
                start_time=2.0,
                end_time=4.0,
                session_id="test",
                transcript_id="test",
                chunk_ids=["c2"],
                boundary_type="punctuation",
            ),
            TranslationUnit(
                text="Third sentence with context.",
                speaker_name="Alice",
                start_time=4.0,
                end_time=6.0,
                session_id="test",
                transcript_id="test",
                chunk_ids=["c3"],
                boundary_type="punctuation",
            ),
        ]

        # WHEN: Translating with context
        for unit in units:
            await translator.translate(unit, target_language="es")

        # THEN: All requests still valid despite context building
        for request in contract_client.requests_received:
            validate_translation_request(request)
            # Text field should contain the unit text, not context
            assert "sentence" in request.text.lower()


# =============================================================================
# End-to-End Contract Flow Tests
# =============================================================================


class TestEndToEndContractFlow:
    """Test complete flow maintains contract validity."""

    @pytest.mark.asyncio
    async def test_fireflies_to_translation_contract_flow(self):
        """Test Fireflies chunk -> TranslationUnit -> TranslationRequest flow."""
        from models.fireflies import FirefliesChunk, TranslationUnit
        from services.rolling_window_translator import RollingWindowTranslator

        # GIVEN: A Fireflies chunk (what we receive from Fireflies API)
        chunk = FirefliesChunk(
            transcript_id="meeting-001",
            chunk_id="chunk-001",
            text="Hello everyone, let's discuss the API changes.",
            speaker_name="Product Manager",
            start_time=0.0,
            end_time=3.5,
        )

        # WHEN: Converting to TranslationUnit
        unit = TranslationUnit(
            text=chunk.text,
            speaker_name=chunk.speaker_name,
            start_time=chunk.start_time,
            end_time=chunk.end_time,
            session_id="session-001",
            transcript_id=chunk.transcript_id,
            chunk_ids=[chunk.chunk_id],
            boundary_type="punctuation",
        )

        # AND: Creating translator with contract client
        client = ContractValidatingTranslationClient()
        translator = RollingWindowTranslator(translation_client=client)

        # AND: Translating
        result = await translator.translate(unit, target_language="es")

        # THEN: The TranslationRequest was valid
        assert client.call_count == 1
        request = client.requests_received[0]
        validate_translation_request(request)

        # AND: The request text matches original
        assert request.text == chunk.text

        # AND: The result has correct metadata
        assert result.speaker_name == chunk.speaker_name
        assert result.original == chunk.text


class TestContractErrorHandling:
    """Test that contract violations are detected."""

    @pytest.mark.asyncio
    async def test_empty_text_fails_contract(self):
        """Empty text should fail contract validation."""
        with pytest.raises(AssertionError, match="cannot be empty"):
            request = TranslationRequest(
                text="",
                target_language="es",
            )
            validate_translation_request(request)

    def test_invalid_confidence_fails_contract(self):
        """Confidence outside 0-1 should fail."""
        response = TranslationResponse(
            translated_text="Hola",
            target_language="es",
            confidence=1.5,  # Invalid!
        )

        with pytest.raises(AssertionError, match="between 0 and 1"):
            validate_translation_response(response)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
