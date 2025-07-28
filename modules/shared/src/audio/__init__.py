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
    # Main classes
    AudioValidator,
    
    # Data structures
    AudioMetadata,
    ValidationResult,
    CorruptionAnalysis,
    
    # Enumerations
    AudioFormat,
    QualityLevel,
    
    # Exceptions
    AudioValidationError,
    AudioFormatError,
    AudioCorruptionError,
    AudioQualityError,
    
    # Convenience functions
    validate_audio,
    check_audio_corruption,
    convert_audio_format,
)

__all__ = [
    # Main classes
    'AudioValidator',
    
    # Data structures
    'AudioMetadata',
    'ValidationResult',
    'CorruptionAnalysis',
    
    # Enumerations
    'AudioFormat',
    'QualityLevel',
    
    # Exceptions
    'AudioValidationError',
    'AudioFormatError',
    'AudioCorruptionError',
    'AudioQualityError',
    
    # Convenience functions
    'validate_audio',
    'check_audio_corruption',
    'convert_audio_format',
]

# Version information
__version__ = '1.0.0'
__author__ = 'LiveTranslate Team'
__description__ = 'Comprehensive audio validation and processing library'