# Audio Validation Library

A comprehensive audio format validation, corruption detection, and quality assessment library for the LiveTranslate system.

## Overview

The AudioValidator library provides enterprise-grade audio processing capabilities including:

- **Format Validation**: Support for WAV, MP3, WebM, OGG, MP4, FLAC, M4A formats
- **Corruption Detection**: Advanced signal analysis to detect various types of audio corruption
- **Quality Assessment**: Comprehensive quality scoring with detailed metrics
- **Format Conversion**: High-quality audio format conversion with preservation options
- **Metadata Extraction**: Detailed audio metadata including sample rate, channels, duration, etc.

## Features

### 🎯 Core Validation
- Multi-format audio validation
- Sample rate verification
- Channel count validation
- Bit depth assessment
- Duration analysis

### 🔍 Corruption Detection
- Digital clipping detection
- Dropout identification
- Pop and click detection
- Frequency anomaly detection
- Spectral inconsistency analysis
- Excessive silence detection

### 📊 Quality Assessment
- Signal-to-noise ratio (SNR) calculation
- Dynamic range analysis
- Total harmonic distortion (THD) measurement
- Spectral centroid stability analysis
- Quality level classification (Excellent/Good/Fair/Poor/Unacceptable)

### 🔄 Format Conversion
- High-quality resampling (Kaiser window)
- Format conversion with quality preservation
- Metadata preservation during conversion
- Multiple output format support

### 📈 Performance Metrics
- Processing time tracking
- Memory usage optimization
- Comprehensive error reporting
- Detailed recommendation system

## Quick Start

### Installation

```bash
# Install required dependencies
pip install librosa soundfile scipy numpy matplotlib

# Install the shared module (from project root)
pip install -e modules/shared
```

### Basic Usage

```python
from livetranslate.shared.audio import AudioValidator, AudioFormat

# Initialize validator
validator = AudioValidator(default_sample_rate=16000)

# Validate audio file
result = validator.validate_audio_format('audio.wav', AudioFormat.WAV)

if result.is_valid:
    print(f"Audio quality: {result.quality_level.value}")
    print(f"Quality score: {result.quality_score:.3f}")
else:
    print(f"Validation errors: {result.errors}")
    print(f"Recommendations: {result.recommendations}")
```

### Convenience Functions

```python
from livetranslate.shared.audio import validate_audio, check_audio_corruption, convert_audio_format

# Quick validation
result = validate_audio(audio_data, expected_sample_rate=16000)

# Check for corruption
corruption = check_audio_corruption(audio_data)
if corruption.is_corrupted:
    print(f"Corruption detected: {corruption.corruption_type}")
    print(f"Severity: {corruption.corruption_severity:.3f}")

# Convert format
converted_bytes, metadata = convert_audio_format(
    audio_data, 
    AudioFormat.FLAC, 
    target_sample_rate=44100
)
```

## API Reference

### AudioValidator Class

#### Constructor
```python
AudioValidator(
    default_sample_rate: int = 16000,
    quality_threshold: float = 0.7,
    enable_performance_metrics: bool = True
)
```

#### Core Methods

##### validate_audio_format()
```python
validate_audio_format(
    audio_data: Union[bytes, np.ndarray, str], 
    expected_format: Optional[AudioFormat] = None
) -> ValidationResult
```
Comprehensive audio validation including format, quality, and corruption detection.

**Parameters:**
- `audio_data`: Audio data as bytes, numpy array, or file path
- `expected_format`: Expected audio format for validation

**Returns:** `ValidationResult` with detailed analysis

##### detect_audio_corruption()
```python
detect_audio_corruption(
    audio_data: Union[np.ndarray, bytes, str],
    sample_rate: Optional[int] = None
) -> CorruptionAnalysis
```
Advanced corruption detection using signal analysis.

**Returns:** `CorruptionAnalysis` with corruption details

##### validate_sample_rate()
```python
validate_sample_rate(
    audio_data: Union[np.ndarray, bytes, str],
    expected_rate: int = 16000
) -> Tuple[bool, int, Dict[str, Any]]
```
Validate audio sample rate against expected value.

**Returns:** Tuple of `(is_valid, actual_rate, analysis_details)`

##### validate_audio_quality()
```python
validate_audio_quality(
    audio_data: Union[np.ndarray, bytes, str],
    sample_rate: Optional[int] = None
) -> Tuple[float, QualityLevel, Dict[str, Any]]
```
Comprehensive audio quality assessment.

**Returns:** Tuple of `(quality_score, quality_level, quality_metrics)`

##### standardize_audio_format()
```python
standardize_audio_format(
    audio_data: Union[np.ndarray, bytes, str],
    target_format: AudioFormat,
    target_sample_rate: Optional[int] = None,
    preserve_quality: bool = True
) -> Tuple[bytes, AudioMetadata]
```
Convert audio to standardized format with quality preservation.

**Returns:** Tuple of `(converted_audio_bytes, metadata)`

##### get_audio_metadata()
```python
get_audio_metadata(audio_data: Union[bytes, str]) -> AudioMetadata
```
Extract comprehensive audio metadata.

**Returns:** `AudioMetadata` with detailed information

### Data Structures

#### ValidationResult
```python
@dataclass
class ValidationResult:
    is_valid: bool
    format_valid: bool
    corruption_detected: bool
    quality_score: float
    quality_level: QualityLevel
    sample_rate_valid: bool
    metadata: AudioMetadata
    errors: List[str]
    warnings: List[str]
    processing_time: float
    recommendations: List[str]
```

#### CorruptionAnalysis
```python
@dataclass
class CorruptionAnalysis:
    is_corrupted: bool
    corruption_type: Optional[str]
    corruption_severity: float
    affected_regions: List[Tuple[float, float]]
    confidence: float
    details: Dict[str, Any]
```

#### AudioMetadata
```python
@dataclass
class AudioMetadata:
    format: str
    sample_rate: int
    channels: int
    duration: float
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    codec: Optional[str] = None
    file_size: Optional[int] = None
```

### Enumerations

#### AudioFormat
- `WAV`: Uncompressed PCM audio
- `MP3`: MPEG Layer 3 compressed audio
- `WEBM`: WebM container format
- `OGG`: Ogg Vorbis compressed audio
- `MP4`: MPEG-4 container format
- `FLAC`: Free Lossless Audio Codec
- `M4A`: MPEG-4 Audio

#### QualityLevel
- `EXCELLENT`: High quality audio (score ≥ 0.9)
- `GOOD`: Good quality audio (score ≥ 0.75)
- `FAIR`: Fair quality audio (score ≥ 0.6)
- `POOR`: Poor quality audio (score ≥ 0.4)
- `UNACCEPTABLE`: Unacceptable quality audio (score < 0.4)

## Advanced Usage

### Custom Quality Parameters

```python
validator = AudioValidator()

# Customize quality assessment parameters
validator.quality_params = {
    'min_snr': 15.0,          # Higher SNR requirement
    'max_thd': 0.05,          # Lower THD tolerance
    'min_dynamic_range': 25.0, # Higher dynamic range requirement
    'max_silence_ratio': 0.2   # Lower silence tolerance
}
```

### Batch Processing

```python
import glob
from concurrent.futures import ThreadPoolExecutor

def validate_audio_file(file_path):
    validator = AudioValidator()
    return validator.validate_audio_format(file_path)

# Process multiple files
audio_files = glob.glob("audio/*.wav")
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(validate_audio_file, audio_files))
```

### Custom Corruption Detection

```python
# Detect specific corruption types
corruption = validator.detect_audio_corruption(audio_data)

if corruption.is_corrupted:
    print(f"Primary corruption: {corruption.corruption_type}")
    print(f"Severity: {corruption.corruption_severity:.3f}")
    
    # Check specific corruption indicators
    details = corruption.details
    if 'clipping_ratio' in details:
        print(f"Clipping ratio: {details['clipping_ratio']:.3f}")
    
    # Get affected time regions
    for start_time, end_time in corruption.affected_regions:
        print(f"Affected region: {start_time:.2f}s - {end_time:.2f}s")
```

## Quality Metrics Explained

### Signal-to-Noise Ratio (SNR)
Measures the ratio of signal power to noise power. Higher values indicate cleaner audio.
- **Excellent**: > 25 dB
- **Good**: 15-25 dB
- **Fair**: 10-15 dB
- **Poor**: < 10 dB

### Dynamic Range
Measures the difference between the loudest and quietest parts of the audio.
- **Excellent**: > 30 dB
- **Good**: 20-30 dB
- **Fair**: 15-20 dB
- **Poor**: < 15 dB

### Total Harmonic Distortion (THD)
Measures the amount of harmonic distortion in the signal. Lower values are better.
- **Excellent**: < 0.01 (1%)
- **Good**: 0.01-0.05 (1-5%)
- **Fair**: 0.05-0.1 (5-10%)
- **Poor**: > 0.1 (>10%)

## Error Handling

The library provides comprehensive error handling with specific exception types:

```python
from livetranslate.shared.audio import (
    AudioValidationError, 
    AudioFormatError, 
    AudioCorruptionError, 
    AudioQualityError
)

try:
    result = validator.validate_audio_format(audio_data)
except AudioFormatError as e:
    print(f"Format error: {e}")
except AudioCorruptionError as e:
    print(f"Corruption detected: {e}")
except AudioQualityError as e:
    print(f"Quality issue: {e}")
except AudioValidationError as e:
    print(f"General validation error: {e}")
```

## Performance Considerations

### Memory Usage
- Large audio files are processed in chunks to minimize memory usage
- Temporary files are automatically cleaned up
- Memory-efficient algorithms are used for spectral analysis

### Processing Speed
- Optimized NumPy operations for fast computation
- Optional performance metrics collection
- Parallel processing support for batch operations

### Accuracy vs Speed Trade-offs
- High-quality resampling can be disabled for faster processing
- Corruption detection depth can be adjusted
- Quality assessment can be simplified for real-time applications

## Integration Examples

### With Whisper Service

```python
from livetranslate.shared.audio import AudioValidator, AudioFormat

def preprocess_audio_for_whisper(audio_data):
    validator = AudioValidator(default_sample_rate=16000)
    
    # Validate input audio
    result = validator.validate_audio_format(audio_data)
    
    if not result.is_valid:
        raise ValueError(f"Invalid audio: {result.errors}")
    
    # Convert to required format if needed
    if result.metadata.sample_rate != 16000:
        converted_bytes, metadata = validator.standardize_audio_format(
            audio_data, 
            AudioFormat.WAV, 
            target_sample_rate=16000
        )
        return converted_bytes
    
    return audio_data
```

### With Translation Service

```python
def validate_audio_quality_for_translation(audio_data):
    validator = AudioValidator(quality_threshold=0.6)
    
    result = validator.validate_audio_format(audio_data)
    
    if result.corruption_detected:
        print("Warning: Audio corruption detected, translation quality may be affected")
    
    if result.quality_score < 0.6:
        print("Warning: Poor audio quality, consider noise reduction")
    
    return result.is_valid
```

## Testing

The library includes comprehensive test suites:

### Structure Validation
```bash
cd modules/shared/src/audio
python validate_structure.py
```

### Functional Testing (requires dependencies)
```bash
cd modules/shared/src/audio
python test_audio_validator.py
```

### Integration Testing
```bash
# From project root
python -m pytest modules/shared/tests/test_audio_integration.py
```

## Contributing

When contributing to the audio validation library:

1. Ensure all tests pass
2. Add tests for new functionality
3. Update documentation for API changes
4. Follow the existing code style
5. Add type hints for new functions

## License

This audio validation library is part of the LiveTranslate project and follows the same license terms.

## Support

For issues, questions, or contributions related to the audio validation library, please refer to the main LiveTranslate project documentation and issue tracker.