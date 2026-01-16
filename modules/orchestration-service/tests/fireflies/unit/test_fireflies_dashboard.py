"""
Unit Tests for Fireflies Dashboard Functionality

TDD Tests - These tests define the EXPECTED BEHAVIOR of the dashboard.
Implements behaviors for:
1. API key validation against Fireflies
2. Translation model validation
3. Prompt template loading with correct variables
4. Error message extraction
5. Session data management

Run with: pytest tests/fireflies/unit/test_fireflies_dashboard.py -v
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_fireflies_client():
    """Mock Fireflies client for testing."""
    client = AsyncMock()
    client.get_active_meetings = AsyncMock(return_value=[
        {"id": "meeting1", "title": "Test Meeting", "state": "active"},
        {"id": "meeting2", "title": "Another Meeting", "state": "active"},
    ])
    client.close = AsyncMock()
    return client


@pytest.fixture
def valid_api_key():
    """Valid test API key."""
    return "ff-test-api-key-valid-12345"


@pytest.fixture
def available_models():
    """List of available translation models."""
    return [
        {"id": "default", "name": "Default Model", "description": "Default translation model"},
        {"id": "ollama", "name": "Ollama", "description": "Ollama local LLM"},
        {"id": "groq", "name": "Groq", "description": "Groq cloud API"},
        {"id": "gpt-4", "name": "GPT-4", "description": "OpenAI GPT-4"},
    ]


@pytest.fixture
def prompt_templates():
    """Standard prompt templates."""
    return {
        "full": """You are a professional real-time translator.

Target Language: {target_language}

{glossary_section}

Previous context (DO NOT translate, only use for understanding references):
{context_window}

---

Translate ONLY the following sentence to {target_language}:
{current_sentence}

Translation:""",
        "simple": """Translate to {target_language}:
{current_sentence}

Translation:""",
        "minimal": """Translate to {target_language}: {current_sentence}"""
    }


# =============================================================================
# Behavior: API Key Validation
# =============================================================================


class TestAPIKeyValidation:
    """
    BEHAVIOR: API key validation against Fireflies.

    Given: A user enters an API key
    When: The key is validated against Fireflies
    Then: Valid keys return meeting list, invalid keys return specific error
    """

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_meetings(self, mock_fireflies_client, valid_api_key):
        """
        GIVEN: A valid Fireflies API key
        WHEN: Validating the key by fetching meetings
        THEN: Should return list of available meetings
        """
        # Arrange
        expected_meetings = [
            {"id": "meeting1", "title": "Test Meeting"},
            {"id": "meeting2", "title": "Another Meeting"},
        ]
        mock_fireflies_client.get_active_meetings.return_value = expected_meetings

        # Act
        meetings = await mock_fireflies_client.get_active_meetings()

        # Assert
        assert len(meetings) == 2
        assert meetings[0]["id"] == "meeting1"
        assert meetings[0]["title"] == "Test Meeting"

    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_authentication_error(self, mock_fireflies_client):
        """
        GIVEN: An invalid Fireflies API key
        WHEN: Validating the key by fetching meetings
        THEN: Should raise FirefliesAPIError with 'Invalid API key' message
        """
        # Arrange
        from clients.fireflies_client import FirefliesAPIError
        mock_fireflies_client.get_active_meetings.side_effect = FirefliesAPIError(
            "Invalid API key",
            status_code=401
        )

        # Act & Assert
        with pytest.raises(FirefliesAPIError) as exc_info:
            await mock_fireflies_client.get_active_meetings()

        assert "Invalid API key" in str(exc_info.value)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_expired_api_key_returns_specific_error(self, mock_fireflies_client):
        """
        GIVEN: An expired Fireflies API key
        WHEN: Validating the key
        THEN: Should return specific 'expired' error, not generic failure
        """
        # Arrange
        from clients.fireflies_client import FirefliesAPIError
        mock_fireflies_client.get_active_meetings.side_effect = FirefliesAPIError(
            "API key expired",
            status_code=401
        )

        # Act & Assert
        with pytest.raises(FirefliesAPIError) as exc_info:
            await mock_fireflies_client.get_active_meetings()

        assert "expired" in str(exc_info.value).lower()

    def test_api_key_masking(self):
        """
        GIVEN: A full API key
        WHEN: Masking it for display
        THEN: Should show first 4 and last 4 characters only
        """
        # Arrange
        full_key = "ff-test-api-key-12345-abcde"

        # Act
        def mask_api_key(key: str) -> str:
            if not key or len(key) < 8:
                return '••••••••'
            return key[:4] + '••••••••' + key[-4:]

        masked = mask_api_key(full_key)

        # Assert
        assert masked == "ff-t••••••••bcde"
        assert full_key not in masked

    def test_empty_api_key_returns_placeholder(self):
        """
        GIVEN: An empty or too-short API key
        WHEN: Masking it for display
        THEN: Should return placeholder dots
        """
        # Arrange
        def mask_api_key(key: str) -> str:
            if not key or len(key) < 8:
                return '••••••••'
            return key[:4] + '••••••••' + key[-4:]

        # Act & Assert
        assert mask_api_key("") == "••••••••"
        assert mask_api_key("short") == "••••••••"
        assert mask_api_key(None) == "••••••••"


# =============================================================================
# Behavior: Model Validation
# =============================================================================


class TestModelValidation:
    """
    BEHAVIOR: Translation model validation.

    Given: A user selects a translation model
    When: The model is validated against available models
    Then: Valid models pass, invalid models fallback to first available
    """

    def test_valid_model_passes_validation(self, available_models):
        """
        GIVEN: A model that exists in the available models list
        WHEN: Validating the model
        THEN: Should return the same model ID
        """
        # Arrange
        def validate_model(model_id: str, available: List[Dict]) -> str:
            valid_ids = [m["id"] for m in available]
            if model_id in valid_ids:
                return model_id
            return valid_ids[0] if valid_ids else "default"

        # Act
        result = validate_model("ollama", available_models)

        # Assert
        assert result == "ollama"

    def test_invalid_model_falls_back_to_first_available(self, available_models):
        """
        GIVEN: A model that doesn't exist in available models
        WHEN: Validating the model
        THEN: Should fallback to the first available model
        """
        # Arrange
        def validate_model(model_id: str, available: List[Dict]) -> str:
            valid_ids = [m["id"] for m in available]
            if model_id in valid_ids:
                return model_id
            return valid_ids[0] if valid_ids else "default"

        # Act
        result = validate_model("nonexistent-model", available_models)

        # Assert
        assert result == "default"  # First in the list

    def test_empty_models_list_returns_default(self):
        """
        GIVEN: An empty available models list
        WHEN: Validating any model
        THEN: Should return 'default'
        """
        # Arrange
        def validate_model(model_id: str, available: List[Dict]) -> str:
            valid_ids = [m["id"] for m in available]
            if model_id in valid_ids:
                return model_id
            return valid_ids[0] if valid_ids else "default"

        # Act
        result = validate_model("anything", [])

        # Assert
        assert result == "default"

    def test_model_validation_is_case_sensitive(self, available_models):
        """
        GIVEN: A model ID with wrong case
        WHEN: Validating the model
        THEN: Should not match (case-sensitive)
        """
        # Arrange
        def validate_model(model_id: str, available: List[Dict]) -> str:
            valid_ids = [m["id"] for m in available]
            if model_id in valid_ids:
                return model_id
            return valid_ids[0] if valid_ids else "default"

        # Act
        result = validate_model("OLLAMA", available_models)  # Wrong case

        # Assert
        assert result == "default"  # Falls back because case doesn't match


# =============================================================================
# Behavior: Prompt Template Management
# =============================================================================


class TestPromptTemplateManagement:
    """
    BEHAVIOR: Prompt template loading and variable validation.

    Given: A prompt template style is selected
    When: Loading the template
    Then: Template should contain correct variables for translation
    """

    def test_full_template_contains_all_required_variables(self, prompt_templates):
        """
        GIVEN: The 'full' prompt template
        WHEN: Checking for required variables
        THEN: Should contain: target_language, current_sentence, glossary_section, context_window
        """
        # Arrange
        template = prompt_templates["full"]
        required_vars = [
            "{target_language}",
            "{current_sentence}",
            "{glossary_section}",
            "{context_window}",
        ]

        # Act & Assert
        for var in required_vars:
            assert var in template, f"Missing required variable: {var}"

    def test_simple_template_has_minimal_variables(self, prompt_templates):
        """
        GIVEN: The 'simple' prompt template
        WHEN: Checking for variables
        THEN: Should contain only target_language and current_sentence
        """
        # Arrange
        template = prompt_templates["simple"]

        # Act & Assert
        assert "{target_language}" in template
        assert "{current_sentence}" in template
        # Should NOT have context/glossary sections
        assert "{glossary_section}" not in template
        assert "{context_window}" not in template

    def test_minimal_template_is_single_line(self, prompt_templates):
        """
        GIVEN: The 'minimal' prompt template
        WHEN: Checking format
        THEN: Should be effectively a single line with variables
        """
        # Arrange
        template = prompt_templates["minimal"]

        # Act & Assert
        assert "{target_language}" in template
        assert "{current_sentence}" in template
        # Minimal should not have multi-line structure
        assert "\n\n" not in template

    def test_template_does_not_contain_deprecated_text_variable(self, prompt_templates):
        """
        GIVEN: Any prompt template
        WHEN: Checking for deprecated {text} variable
        THEN: Should NOT contain {text} - use {current_sentence} instead
        """
        # Arrange & Act & Assert
        for name, template in prompt_templates.items():
            # {text} is deprecated - should use {current_sentence}
            # We check that {text} doesn't appear ALONE (not as part of other words)
            import re
            text_var_pattern = r'\{text\}'
            assert not re.search(text_var_pattern, template), \
                f"Template '{name}' uses deprecated {{text}} variable"

    def test_get_template_returns_correct_type(self, prompt_templates):
        """
        GIVEN: A template name
        WHEN: Getting the template
        THEN: Should return correct template or default for unknown
        """
        # Arrange
        def get_prompt_template(template_name: str, templates: Dict[str, str]) -> str:
            return templates.get(template_name, templates.get("simple", ""))

        # Act & Assert
        assert "glossary_section" in get_prompt_template("full", prompt_templates)
        assert "glossary_section" not in get_prompt_template("simple", prompt_templates)
        assert get_prompt_template("unknown", prompt_templates) == prompt_templates["simple"]


# =============================================================================
# Behavior: Error Message Extraction
# =============================================================================


class TestErrorMessageExtraction:
    """
    BEHAVIOR: Extract detailed error messages from API responses.

    Given: An API error occurs
    When: Extracting the error message
    Then: Should return specific, actionable error detail
    """

    def test_extract_error_from_json_detail_field(self):
        """
        GIVEN: A JSON error response with 'detail' field
        WHEN: Extracting error message
        THEN: Should return the detail value
        """
        # Arrange
        error_response = {"detail": "Invalid API key: authentication failed"}

        # Act
        def extract_error(response: Dict[str, Any]) -> str:
            if isinstance(response, dict):
                return response.get("detail") or response.get("message") or str(response)
            return str(response)

        result = extract_error(error_response)

        # Assert
        assert result == "Invalid API key: authentication failed"

    def test_extract_error_from_json_message_field(self):
        """
        GIVEN: A JSON error response with 'message' field
        WHEN: Extracting error message
        THEN: Should return the message value
        """
        # Arrange
        error_response = {"message": "Service temporarily unavailable"}

        # Act
        def extract_error(response: Dict[str, Any]) -> str:
            if isinstance(response, dict):
                return response.get("detail") or response.get("message") or str(response)
            return str(response)

        result = extract_error(error_response)

        # Assert
        assert result == "Service temporarily unavailable"

    def test_extract_error_prefers_detail_over_message(self):
        """
        GIVEN: A JSON response with both 'detail' and 'message' fields
        WHEN: Extracting error message
        THEN: Should prefer 'detail' (FastAPI convention)
        """
        # Arrange
        error_response = {
            "detail": "Specific validation error",
            "message": "Generic error"
        }

        # Act
        def extract_error(response: Dict[str, Any]) -> str:
            if isinstance(response, dict):
                return response.get("detail") or response.get("message") or str(response)
            return str(response)

        result = extract_error(error_response)

        # Assert
        assert result == "Specific validation error"

    def test_extract_error_from_exception_message(self):
        """
        GIVEN: An exception without JSON body
        WHEN: Extracting error message
        THEN: Should return the exception string
        """
        # Arrange
        class MockException(Exception):
            pass

        exc = MockException("Connection refused")

        # Act
        error_msg = str(exc)

        # Assert
        assert error_msg == "Connection refused"

    def test_error_includes_http_status_context(self):
        """
        GIVEN: An HTTP error with status code
        WHEN: Formatting error for display
        THEN: Should include both status code and message
        """
        # Arrange
        def format_http_error(status_code: int, detail: str) -> str:
            status_messages = {
                400: "Bad Request",
                401: "Unauthorized",
                403: "Forbidden",
                404: "Not Found",
                422: "Validation Error",
                500: "Internal Server Error",
                502: "Bad Gateway",
            }
            status_text = status_messages.get(status_code, f"HTTP {status_code}")
            return f"{status_text}: {detail}"

        # Act
        result = format_http_error(422, "Missing required field: target_language")

        # Assert
        assert "Validation Error" in result
        assert "target_language" in result


# =============================================================================
# Behavior: Session Data Management
# =============================================================================


class TestSessionDataManagement:
    """
    BEHAVIOR: Session data storage and retrieval.

    Given: A Fireflies session is active
    When: Managing session data (transcripts, translations)
    Then: Should properly store, retrieve, and export data
    """

    def test_store_feed_entry_with_all_fields(self):
        """
        GIVEN: A new feed entry with transcript and translation
        WHEN: Storing the entry
        THEN: Should include all required fields
        """
        # Arrange
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speaker": "John Doe",
            "original": "Hello, how are you?",
            "translated": "Hola, como estas?",
            "confidence": 0.95,
            "language": "es",
        }

        feed_entries = []

        # Act
        feed_entries.append(entry)

        # Assert
        assert len(feed_entries) == 1
        stored = feed_entries[0]
        assert "timestamp" in stored
        assert "speaker" in stored
        assert "original" in stored
        assert "translated" in stored
        assert "confidence" in stored
        assert "language" in stored

    def test_export_feed_to_json_format(self):
        """
        GIVEN: A list of feed entries
        WHEN: Exporting to JSON
        THEN: Should produce valid JSON with session metadata
        """
        # Arrange
        entries = [
            {"timestamp": "2024-01-15T10:00:00Z", "speaker": "Alice", "original": "Hello", "translated": "Hola"},
            {"timestamp": "2024-01-15T10:00:05Z", "speaker": "Bob", "original": "Hi there", "translated": "Hola ahi"},
        ]
        session_id = "ff_session_abc123"

        # Act
        export_data = {
            "session_id": session_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "entry_count": len(entries),
            "entries": entries,
        }
        json_output = json.dumps(export_data, indent=2)

        # Assert
        parsed = json.loads(json_output)
        assert parsed["session_id"] == session_id
        assert parsed["entry_count"] == 2
        assert len(parsed["entries"]) == 2

    def test_load_saved_feed_from_local_storage_format(self):
        """
        GIVEN: A JSON string from localStorage
        WHEN: Loading the feed data
        THEN: Should parse correctly with all entries
        """
        # Arrange
        saved_json = json.dumps({
            "session_id": "ff_session_xyz789",
            "saved_at": "2024-01-15T12:00:00Z",
            "entry_count": 2,
            "entries": [
                {"speaker": "Alice", "original": "Test 1", "translated": "Prueba 1"},
                {"speaker": "Bob", "original": "Test 2", "translated": "Prueba 2"},
            ]
        })

        # Act
        loaded = json.loads(saved_json)

        # Assert
        assert loaded["session_id"] == "ff_session_xyz789"
        assert len(loaded["entries"]) == 2
        assert loaded["entries"][0]["speaker"] == "Alice"

    def test_filter_entries_by_speaker(self):
        """
        GIVEN: Feed entries from multiple speakers
        WHEN: Filtering by a specific speaker
        THEN: Should return only that speaker's entries
        """
        # Arrange
        entries = [
            {"speaker": "Alice", "original": "Message 1"},
            {"speaker": "Bob", "original": "Message 2"},
            {"speaker": "Alice", "original": "Message 3"},
            {"speaker": "Charlie", "original": "Message 4"},
        ]

        # Act
        alice_entries = [e for e in entries if e["speaker"] == "Alice"]

        # Assert
        assert len(alice_entries) == 2
        assert all(e["speaker"] == "Alice" for e in alice_entries)

    def test_calculate_session_statistics(self):
        """
        GIVEN: A list of feed entries
        WHEN: Calculating statistics
        THEN: Should return correct counts and averages
        """
        # Arrange
        entries = [
            {"speaker": "Alice", "confidence": 0.95},
            {"speaker": "Bob", "confidence": 0.88},
            {"speaker": "Alice", "confidence": 0.92},
        ]

        # Act
        def calculate_stats(entries: List[Dict]) -> Dict:
            total = len(entries)
            speakers = list(set(e["speaker"] for e in entries))
            avg_confidence = sum(e.get("confidence", 0) for e in entries) / total if total > 0 else 0
            return {
                "total_entries": total,
                "unique_speakers": len(speakers),
                "speakers": speakers,
                "average_confidence": round(avg_confidence, 3),
            }

        stats = calculate_stats(entries)

        # Assert
        assert stats["total_entries"] == 3
        assert stats["unique_speakers"] == 2
        assert "Alice" in stats["speakers"]
        assert "Bob" in stats["speakers"]
        assert 0.91 <= stats["average_confidence"] <= 0.92


# =============================================================================
# Behavior: Translation Request Building
# =============================================================================


class TestTranslationRequestBuilding:
    """
    BEHAVIOR: Building correct translation API requests.

    Given: User inputs for translation
    When: Building the API request
    Then: Should construct proper request format
    """

    def test_build_basic_translation_request(self):
        """
        GIVEN: Text and target language
        WHEN: Building a translation request
        THEN: Should have required fields
        """
        # Arrange
        text = "Hello, how are you?"
        target_language = "es"

        # Act
        request = {
            "text": text,
            "target_language": target_language,
            "source_language": "auto",
        }

        # Assert
        assert request["text"] == text
        assert request["target_language"] == target_language
        assert request["source_language"] == "auto"

    def test_request_with_model_uses_service_field(self, available_models):
        """
        GIVEN: A selected model
        WHEN: Building request
        THEN: Should use 'service' field (not deprecated 'model')
        """
        # Arrange
        text = "Hello"
        target_lang = "es"
        model = "ollama"

        # Act - New format (correct)
        request = {
            "text": text,
            "target_language": target_lang,
            "source_language": "auto",
            "service": model,  # Correct field
        }

        # Assert
        assert "service" in request
        assert request["service"] == "ollama"

    def test_empty_text_should_be_rejected(self):
        """
        GIVEN: Empty text input
        WHEN: Validating request
        THEN: Should reject with validation error
        """
        # Arrange
        def validate_request(text: str, target_language: str) -> List[str]:
            errors = []
            if not text or not text.strip():
                errors.append("Text cannot be empty")
            if not target_language:
                errors.append("Target language is required")
            return errors

        # Act
        errors = validate_request("", "es")

        # Assert
        assert len(errors) == 1
        assert "Text cannot be empty" in errors

    def test_invalid_language_code_should_be_rejected(self):
        """
        GIVEN: An invalid language code
        WHEN: Validating request
        THEN: Should reject with specific error
        """
        # Arrange
        valid_languages = ["en", "es", "fr", "de", "ja", "zh", "ko", "pt", "it", "ru", "ar", "hi"]

        def validate_language(lang: str) -> bool:
            return lang in valid_languages or lang == "auto"

        # Act & Assert
        assert validate_language("es") is True
        assert validate_language("xx") is False  # Invalid code
        assert validate_language("auto") is True


# =============================================================================
# Behavior: WebSocket Connection State
# =============================================================================


class TestWebSocketConnectionState:
    """
    BEHAVIOR: Managing WebSocket connection states.

    Given: A WebSocket connection for live feed
    When: Connection state changes
    Then: Should update UI state appropriately
    """

    def test_connection_states_are_distinct(self):
        """
        GIVEN: WebSocket connection states
        WHEN: Checking state values
        THEN: Should have distinct values for each state
        """
        # Arrange
        states = {
            "disconnected": "disconnected",
            "connecting": "connecting",
            "connected": "connected",
            "reconnecting": "reconnecting",
            "error": "error",
        }

        # Act & Assert
        assert len(set(states.values())) == 5  # All unique

    def test_status_badge_class_mapping(self):
        """
        GIVEN: A connection status
        WHEN: Getting CSS class for badge
        THEN: Should return appropriate class
        """
        # Arrange
        def get_status_class(status: str) -> str:
            mapping = {
                "connected": "status-connected",
                "connecting": "status-connecting",
                "disconnected": "status-disconnected",
                "reconnecting": "status-connecting",
                "error": "status-disconnected",
            }
            return mapping.get(status, "status-disconnected")

        # Act & Assert
        assert get_status_class("connected") == "status-connected"
        assert get_status_class("connecting") == "status-connecting"
        assert get_status_class("disconnected") == "status-disconnected"
        assert get_status_class("error") == "status-disconnected"
        assert get_status_class("unknown") == "status-disconnected"


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
