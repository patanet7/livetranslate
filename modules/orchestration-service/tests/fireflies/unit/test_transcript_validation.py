"""
Unit Tests for Transcript Validation

TDD Tests - Define behavior for data validation before storage.

Behaviors:
1. Validate transcript text is not empty
2. Validate timestamps are valid (end >= start)
3. Validate session_id is present
4. Validate text length limits
5. Return specific validation errors

Run with: pytest tests/fireflies/unit/test_transcript_validation.py -v
"""

import pytest
from datetime import datetime, timezone
from typing import List
from pathlib import Path
import sys

# Add src path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def valid_transcript():
    """A valid transcript for testing."""
    return {
        "text": "Hello, this is a valid transcript.",
        "start_time": 10.5,
        "end_time": 12.3,
        "session_id": "ff_session_abc123",
        "speaker": "John Doe",
    }


@pytest.fixture
def transcript_validator():
    """
    Transcript validator function.

    This defines the expected behavior - implementation will follow in src/
    """
    def validate(
        text: str,
        start_time: float,
        end_time: float,
        session_id: str,
        max_length: int = 10000,
    ) -> List[str]:
        """
        Validate transcript data before storage.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Rule 1: Text cannot be empty
        if not text or not text.strip():
            errors.append("Empty transcript text")

        # Rule 2: Timestamps must be valid
        if end_time < start_time:
            errors.append(f"Invalid timestamps: end ({end_time}) < start ({start_time})")

        # Rule 3: Session ID is required
        if not session_id:
            errors.append("Missing session_id")

        # Rule 4: Text length limit
        if text and len(text) > max_length:
            errors.append(f"Transcript too long (>{max_length} chars)")

        # Rule 5: Start time cannot be negative
        if start_time < 0:
            errors.append(f"Invalid start_time: {start_time} (cannot be negative)")

        return errors

    return validate


# =============================================================================
# Behavior: Empty Text Validation
# =============================================================================


class TestEmptyTextValidation:
    """
    BEHAVIOR: Validate transcript text is not empty.

    Given: A transcript with text field
    When: Text is empty or whitespace
    Then: Should return validation error
    """

    def test_empty_string_is_invalid(self, transcript_validator):
        """
        GIVEN: Empty string for text
        WHEN: Validating
        THEN: Should return 'Empty transcript text' error
        """
        errors = transcript_validator(
            text="",
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
        )

        assert "Empty transcript text" in errors

    def test_whitespace_only_is_invalid(self, transcript_validator):
        """
        GIVEN: Whitespace-only string for text
        WHEN: Validating
        THEN: Should return 'Empty transcript text' error
        """
        errors = transcript_validator(
            text="   \t\n  ",
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
        )

        assert "Empty transcript text" in errors

    def test_none_text_is_invalid(self, transcript_validator):
        """
        GIVEN: None for text
        WHEN: Validating
        THEN: Should return 'Empty transcript text' error
        """
        errors = transcript_validator(
            text=None,
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
        )

        assert "Empty transcript text" in errors

    def test_valid_text_passes(self, transcript_validator):
        """
        GIVEN: Valid text content
        WHEN: Validating
        THEN: Should not have 'Empty transcript text' error
        """
        errors = transcript_validator(
            text="Hello, this is valid.",
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
        )

        assert "Empty transcript text" not in errors


# =============================================================================
# Behavior: Timestamp Validation
# =============================================================================


class TestTimestampValidation:
    """
    BEHAVIOR: Validate timestamps are valid.

    Given: Start and end timestamps
    When: End time is less than start time
    Then: Should return validation error
    """

    def test_end_before_start_is_invalid(self, transcript_validator):
        """
        GIVEN: end_time < start_time
        WHEN: Validating
        THEN: Should return timestamp error
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=10.0,
            end_time=5.0,  # Invalid: end < start
            session_id="session_123",
        )

        assert any("Invalid timestamps" in e for e in errors)

    def test_equal_timestamps_is_valid(self, transcript_validator):
        """
        GIVEN: end_time == start_time (instant)
        WHEN: Validating
        THEN: Should be valid (no timestamp error)
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=10.0,
            end_time=10.0,  # Equal is ok (single point)
            session_id="session_123",
        )

        assert not any("Invalid timestamps" in e for e in errors)

    def test_normal_timestamps_are_valid(self, transcript_validator):
        """
        GIVEN: end_time > start_time
        WHEN: Validating
        THEN: Should be valid (no timestamp error)
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=10.0,
            end_time=12.5,
            session_id="session_123",
        )

        assert not any("Invalid timestamps" in e for e in errors)

    def test_negative_start_time_is_invalid(self, transcript_validator):
        """
        GIVEN: Negative start_time
        WHEN: Validating
        THEN: Should return error
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=-5.0,  # Invalid: negative
            end_time=10.0,
            session_id="session_123",
        )

        assert any("negative" in e.lower() for e in errors)


# =============================================================================
# Behavior: Session ID Validation
# =============================================================================


class TestSessionIdValidation:
    """
    BEHAVIOR: Validate session_id is present.

    Given: A transcript record
    When: Session ID is missing
    Then: Should return validation error
    """

    def test_empty_session_id_is_invalid(self, transcript_validator):
        """
        GIVEN: Empty session_id
        WHEN: Validating
        THEN: Should return 'Missing session_id' error
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=0.0,
            end_time=1.0,
            session_id="",
        )

        assert "Missing session_id" in errors

    def test_none_session_id_is_invalid(self, transcript_validator):
        """
        GIVEN: None session_id
        WHEN: Validating
        THEN: Should return 'Missing session_id' error
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=0.0,
            end_time=1.0,
            session_id=None,
        )

        assert "Missing session_id" in errors

    def test_valid_session_id_passes(self, transcript_validator):
        """
        GIVEN: Valid session_id
        WHEN: Validating
        THEN: Should not have session_id error
        """
        errors = transcript_validator(
            text="Valid text",
            start_time=0.0,
            end_time=1.0,
            session_id="ff_session_abc123",
        )

        assert "Missing session_id" not in errors


# =============================================================================
# Behavior: Text Length Validation
# =============================================================================


class TestTextLengthValidation:
    """
    BEHAVIOR: Validate text length limits.

    Given: A transcript with text
    When: Text exceeds maximum length
    Then: Should return validation error
    """

    def test_extremely_long_text_is_invalid(self, transcript_validator):
        """
        GIVEN: Text longer than 10000 characters
        WHEN: Validating
        THEN: Should return length error
        """
        long_text = "A" * 15000  # > 10000 chars

        errors = transcript_validator(
            text=long_text,
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
        )

        assert any("too long" in e.lower() for e in errors)

    def test_normal_text_passes_length_check(self, transcript_validator):
        """
        GIVEN: Text within length limit
        WHEN: Validating
        THEN: Should not have length error
        """
        normal_text = "This is a normal length transcript."

        errors = transcript_validator(
            text=normal_text,
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
        )

        assert not any("too long" in e.lower() for e in errors)

    def test_exactly_at_limit_is_valid(self, transcript_validator):
        """
        GIVEN: Text exactly at the limit
        WHEN: Validating
        THEN: Should be valid
        """
        text_at_limit = "A" * 10000  # Exactly at limit

        errors = transcript_validator(
            text=text_at_limit,
            start_time=0.0,
            end_time=1.0,
            session_id="session_123",
            max_length=10000,
        )

        assert not any("too long" in e.lower() for e in errors)


# =============================================================================
# Behavior: Multiple Validation Errors
# =============================================================================


class TestMultipleValidationErrors:
    """
    BEHAVIOR: Return all validation errors at once.

    Given: Multiple validation issues
    When: Validating
    Then: Should return all errors, not just the first
    """

    def test_returns_all_errors(self, transcript_validator):
        """
        GIVEN: Multiple validation problems
        WHEN: Validating
        THEN: Should return all errors
        """
        errors = transcript_validator(
            text="",  # Empty
            start_time=10.0,
            end_time=5.0,  # Invalid timestamps
            session_id="",  # Missing
        )

        assert len(errors) >= 3
        assert "Empty transcript text" in errors
        assert any("Invalid timestamps" in e for e in errors)
        assert "Missing session_id" in errors

    def test_valid_input_returns_empty_list(self, transcript_validator, valid_transcript):
        """
        GIVEN: Valid transcript data
        WHEN: Validating
        THEN: Should return empty error list
        """
        errors = transcript_validator(
            text=valid_transcript["text"],
            start_time=valid_transcript["start_time"],
            end_time=valid_transcript["end_time"],
            session_id=valid_transcript["session_id"],
        )

        assert len(errors) == 0


# =============================================================================
# Behavior: Translation Validation
# =============================================================================


class TestTranslationValidation:
    """
    BEHAVIOR: Validate translation data.

    Given: A translation result
    When: Validating before storage
    Then: Should check all required fields
    """

    def test_translation_requires_transcript_id(self):
        """
        GIVEN: A translation without transcript_id
        WHEN: Validating
        THEN: Should return error
        """
        def validate_translation(
            translated_text: str,
            transcript_id: str,
            target_language: str,
        ) -> List[str]:
            errors = []
            if not translated_text or not translated_text.strip():
                errors.append("Empty translation text")
            if not transcript_id:
                errors.append("Missing transcript_id (translation must link to transcript)")
            if not target_language:
                errors.append("Missing target_language")
            return errors

        errors = validate_translation(
            translated_text="Hola mundo",
            transcript_id="",  # Missing
            target_language="es",
        )

        assert any("transcript_id" in err for err in errors)

    def test_translation_cannot_be_empty(self):
        """
        GIVEN: Empty translation text
        WHEN: Validating
        THEN: Should return error
        """
        def validate_translation(
            translated_text: str,
            transcript_id: str,
            target_language: str,
        ) -> List[str]:
            errors = []
            if not translated_text or not translated_text.strip():
                errors.append("Empty translation text")
            if not transcript_id:
                errors.append("Missing transcript_id")
            if not target_language:
                errors.append("Missing target_language")
            return errors

        errors = validate_translation(
            translated_text="",
            transcript_id="abc123",
            target_language="es",
        )

        assert "Empty translation text" in errors


# =============================================================================
# Behavior: Validation Helper Function
# =============================================================================


class TestValidationHelpers:
    """
    BEHAVIOR: Helper functions for common validation patterns.

    Given: Various validation scenarios
    When: Using helper functions
    Then: Should provide consistent validation
    """

    def test_is_valid_language_code(self):
        """
        GIVEN: A language code
        WHEN: Checking if valid
        THEN: Should recognize standard codes
        """
        valid_codes = ["en", "es", "fr", "de", "ja", "zh", "ko", "pt", "it", "ru", "ar", "hi", "auto"]

        def is_valid_language_code(code: str) -> bool:
            return code in valid_codes

        assert is_valid_language_code("es") is True
        assert is_valid_language_code("xx") is False
        assert is_valid_language_code("auto") is True

    def test_sanitize_text_for_storage(self):
        """
        GIVEN: Text with potential issues
        WHEN: Sanitizing for storage
        THEN: Should clean appropriately
        """
        def sanitize_text(text: str) -> str:
            if not text:
                return ""
            # Remove null bytes and control characters (except newline, tab)
            cleaned = "".join(c for c in text if c.isprintable() or c in "\n\t")
            # Normalize whitespace
            return " ".join(cleaned.split())

        # Test null byte removal
        text_with_null = "Hello\x00World"
        assert "\x00" not in sanitize_text(text_with_null)

        # Test whitespace normalization
        text_with_spaces = "  Hello    World  "
        assert sanitize_text(text_with_spaces) == "Hello World"


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
