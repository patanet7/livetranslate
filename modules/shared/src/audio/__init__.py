"""
Audio Processing Module for LiveTranslate

This module provides comprehensive audio validation, corruption detection,
quality assessment, and format conversion capabilities.

Main Components:
- AudioValidator: Main class for audio processing and validation
- ValidationResult: Comprehensive validation results
- CorruptionAnalysis: Detailed corruption detection results
- AudioMetadata: Audio file metadata structure
- AudioFormat: Supported audio format enumeration
- QualityLevel: Audio quality level enumeration

Convenience Functions:
- validate_audio(): Quick audio validation
- check_audio_corruption(): Quick corruption detection
- convert_audio_format(): Quick format conversion

Usage Example:
    from livetranslate.shared.audio import AudioValidator, AudioFormat

    validator = AudioValidator()
    result = validator.validate_audio_format(audio_data)

    if result.is_valid:
        print(f"Audio quality: {result.quality_level.value}")
    else:
        print(f"Validation errors: {result.errors}")
"""

from .audio_validator import (
    AudioCorruptionError,
    # Enumerations
    AudioFormat,
    AudioFormatError,
    # Data structures
    AudioMetadata,
    AudioQualityError,
    # Exceptions
    AudioValidationError,
    # Main classes
    AudioValidator,
    CorruptionAnalysis,
    QualityLevel,
    ValidationResult,
    check_audio_corruption,
    convert_audio_format,
    # Convenience functions
    validate_audio,
)

__all__ = [
    "AudioCorruptionError",
    # Enumerations
    "AudioFormat",
    "AudioFormatError",
    # Data structures
    "AudioMetadata",
    "AudioQualityError",
    # Exceptions
    "AudioValidationError",
    # Main classes
    "AudioValidator",
    "CorruptionAnalysis",
    "QualityLevel",
    "ValidationResult",
    "check_audio_corruption",
    "convert_audio_format",
    # Convenience functions
    "validate_audio",
]

# Version information
__version__ = "1.0.0"
__author__ = "LiveTranslate Team"
__description__ = "Comprehensive audio validation and processing library"
