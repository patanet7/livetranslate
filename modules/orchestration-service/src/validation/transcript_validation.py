"""
Transcript and Translation Validation

Validates data before storage to ensure data integrity.

Usage:
    from validation import validate_transcript, validate_translation

    # Validate a transcript
    result = validate_transcript(
        text="Hello world",
        start_time=0.0,
        end_time=1.5,
        session_id="ff_session_abc123"
    )
    if not result.is_valid:
        print(f"Validation errors: {result.errors}")

    # Validate a translation
    result = validate_translation(
        translated_text="Hola mundo",
        transcript_id="abc123",
        target_language="es"
    )
"""

from dataclasses import dataclass, field
from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    Attributes:
        is_valid: True if all validations passed
        errors: List of validation error messages
        warnings: List of validation warnings (data is valid but may have issues)
        sanitized_data: Optional dict of sanitized data values
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitized_data: dict[str, Any] | None = None

    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)


# =============================================================================
# Constants
# =============================================================================

# Valid language codes for transcripts/translations
VALID_LANGUAGE_CODES = [
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "zh",
    "ko",
    "ru",
    "ar",
    "hi",
    "nl",
    "sv",
    "pl",
    "tr",
    "vi",
    "th",
    "id",
    "cs",
    "el",
    "ro",
    "hu",
    "da",
    "fi",
    "no",
    "sk",
    "uk",
    "he",
    "bg",
    "auto",  # Special code for auto-detection
]

# Maximum text length (characters)
MAX_TEXT_LENGTH = 10000

# Maximum reasonable duration for a single transcript segment (seconds)
MAX_SEGMENT_DURATION = 300  # 5 minutes


# =============================================================================
# Transcript Validator
# =============================================================================


class TranscriptValidator:
    """
    Validates transcript data before storage.

    Rules:
    1. Text cannot be empty or whitespace-only
    2. Timestamps must be valid (end >= start, start >= 0)
    3. Session ID must be present
    4. Text length must be within limits
    5. Duration must be reasonable
    """

    def __init__(
        self,
        max_text_length: int = MAX_TEXT_LENGTH,
        max_segment_duration: float = MAX_SEGMENT_DURATION,
    ):
        self.max_text_length = max_text_length
        self.max_segment_duration = max_segment_duration

    def validate(
        self,
        text: str,
        start_time: float,
        end_time: float,
        session_id: str,
        speaker: str | None = None,
        sanitize: bool = True,
    ) -> ValidationResult:
        """
        Validate transcript data.

        Args:
            text: Transcript text content
            start_time: Start time in seconds
            end_time: End time in seconds
            session_id: Session this transcript belongs to
            speaker: Optional speaker name
            sanitize: Whether to return sanitized data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized: dict[str, Any] = {}

        # Rule 1: Text cannot be empty
        if not text or not text.strip():
            result.add_error("Empty transcript text")
        else:
            sanitized["text"] = self._sanitize_text(text) if sanitize else text

        # Rule 2a: Start time cannot be negative
        if start_time < 0:
            result.add_error(f"Invalid start_time: {start_time} (cannot be negative)")
        else:
            sanitized["start_time"] = start_time

        # Rule 2b: End time must be >= start time
        if end_time < start_time:
            result.add_error(f"Invalid timestamps: end ({end_time}) < start ({start_time})")
        else:
            sanitized["end_time"] = end_time

        # Rule 3: Session ID is required
        if not session_id:
            result.add_error("Missing session_id")
        else:
            sanitized["session_id"] = session_id

        # Rule 4: Text length limit
        if text and len(text) > self.max_text_length:
            result.add_error(f"Transcript too long (>{self.max_text_length} chars)")

        # Rule 5: Duration check (warning, not error)
        duration = end_time - start_time
        if duration > self.max_segment_duration:
            result.add_warning(
                f"Unusually long segment duration: {duration:.1f}s "
                f"(max recommended: {self.max_segment_duration}s)"
            )

        # Sanitize speaker name
        if speaker:
            sanitized["speaker"] = self._sanitize_text(speaker) if sanitize else speaker

        # Add sanitized data if validation passed
        if result.is_valid and sanitize:
            result.sanitized_data = sanitized

        return result

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize text for storage.

        - Removes null bytes and control characters (except newline, tab)
        - Normalizes whitespace
        - Strips leading/trailing whitespace
        """
        if not text:
            return ""

        # Remove null bytes and non-printable control characters
        cleaned = "".join(c for c in text if c.isprintable() or c in "\n\t")

        # Normalize whitespace (multiple spaces/tabs to single space)
        cleaned = " ".join(cleaned.split())

        return cleaned.strip()


class TranslationValidator:
    """
    Validates translation data before storage.

    Rules:
    1. Translated text cannot be empty
    2. Transcript ID is required (translations link to transcripts)
    3. Target language must be valid
    4. Text length must be within limits
    """

    def __init__(
        self,
        max_text_length: int = MAX_TEXT_LENGTH,
        valid_languages: list[str] | None = None,
    ):
        self.max_text_length = max_text_length
        self.valid_languages = valid_languages or VALID_LANGUAGE_CODES

    def validate(
        self,
        translated_text: str,
        transcript_id: str,
        target_language: str,
        source_language: str | None = None,
        confidence: float | None = None,
        sanitize: bool = True,
    ) -> ValidationResult:
        """
        Validate translation data.

        Args:
            translated_text: The translated text
            transcript_id: ID of the source transcript
            target_language: Target language code
            source_language: Optional source language code
            confidence: Optional confidence score (0.0-1.0)
            sanitize: Whether to return sanitized data

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)
        sanitized: dict[str, Any] = {}

        # Rule 1: Translation text cannot be empty
        if not translated_text or not translated_text.strip():
            result.add_error("Empty translation text")
        else:
            sanitized["translated_text"] = (
                self._sanitize_text(translated_text) if sanitize else translated_text
            )

        # Rule 2: Transcript ID is required
        if not transcript_id:
            result.add_error("Missing transcript_id (translation must link to transcript)")
        else:
            sanitized["transcript_id"] = transcript_id

        # Rule 3: Target language must be valid
        if not target_language:
            result.add_error("Missing target_language")
        elif target_language.lower() not in self.valid_languages:
            result.add_error(
                f"Invalid target_language: {target_language}. "
                f"Valid codes: {', '.join(self.valid_languages[:10])}..."
            )
        else:
            sanitized["target_language"] = target_language.lower()

        # Rule 4: Text length limit
        if translated_text and len(translated_text) > self.max_text_length:
            result.add_error(f"Translation too long (>{self.max_text_length} chars)")

        # Validate source language if provided
        if source_language:
            if source_language.lower() not in self.valid_languages:
                result.add_warning(f"Unknown source_language: {source_language}")
            sanitized["source_language"] = source_language.lower()

        # Validate confidence if provided
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                result.add_warning(f"Invalid confidence score: {confidence} (should be 0.0-1.0)")
            sanitized["confidence"] = max(0.0, min(1.0, confidence))

        # Add sanitized data if validation passed
        if result.is_valid and sanitize:
            result.sanitized_data = sanitized

        return result

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for storage (same as TranscriptValidator)."""
        if not text:
            return ""
        cleaned = "".join(c for c in text if c.isprintable() or c in "\n\t")
        return " ".join(cleaned.split()).strip()


# =============================================================================
# Convenience Functions
# =============================================================================

# Default validator instances
_transcript_validator = TranscriptValidator()
_translation_validator = TranslationValidator()


def validate_transcript(
    text: str,
    start_time: float,
    end_time: float,
    session_id: str,
    speaker: str | None = None,
    sanitize: bool = True,
) -> ValidationResult:
    """
    Validate transcript data using default validator.

    See TranscriptValidator.validate() for details.
    """
    return _transcript_validator.validate(
        text=text,
        start_time=start_time,
        end_time=end_time,
        session_id=session_id,
        speaker=speaker,
        sanitize=sanitize,
    )


def validate_translation(
    translated_text: str,
    transcript_id: str,
    target_language: str,
    source_language: str | None = None,
    confidence: float | None = None,
    sanitize: bool = True,
) -> ValidationResult:
    """
    Validate translation data using default validator.

    See TranslationValidator.validate() for details.
    """
    return _translation_validator.validate(
        translated_text=translated_text,
        transcript_id=transcript_id,
        target_language=target_language,
        source_language=source_language,
        confidence=confidence,
        sanitize=sanitize,
    )


def is_valid_language_code(code: str) -> bool:
    """Check if a language code is valid."""
    return bool(code) and code.lower() in VALID_LANGUAGE_CODES
