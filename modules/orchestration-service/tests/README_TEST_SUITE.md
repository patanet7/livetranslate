# Comprehensive Audio Flow Test Suite

This test suite provides comprehensive validation of the complete audio processing pipeline from frontend through orchestration to whisper and translation services.

## Overview

The test suite validates:
- **Complete Pipeline**: Frontend → Orchestration → Whisper → Translation → Response
- **Format Compatibility**: All supported formats (WAV, MP3, WebM, OGG, MP4, FLAC)
- **Error Scenarios**: Service failures, invalid audio, corruption
- **Performance**: Latency, throughput, memory usage, concurrent processing
- **Quality**: Audio quality validation and enhancement testing

## Test Structure

```
tests/
├── conftest.py                    # Global test configuration and fixtures
├── pytest.ini                    # Pytest configuration
├── README_TEST_SUITE.md          # This documentation
├── run_comprehensive_audio_tests.py  # Main test runner
├── fixtures/
│   └── audio_test_data.py        # Comprehensive audio test data generation
├── integration/
│   ├── test_complete_audio_flow.py    # Main end-to-end tests
│   ├── test_audio_coordinator_integration.py
│   └── test_chunk_manager_integration.py
├── unit/
│   ├── test_audio_models.py
│   ├── test_audio_processor.py
│   └── test_speaker_correlator.py
└── performance/
    └── test_audio_performance.py
```

## Test Categories

### 1. Complete Pipeline Tests
**File**: `integration/test_complete_audio_flow.py`

Tests the complete audio processing flow:
- Audio upload and processing
- Service communication and dependency injection
- Transcription and translation pipeline
- Response validation and structure

**Key Test Cases**:
- `test_complete_pipeline_wav_format()` - Full WAV pipeline test
- `test_format_compatibility_all_formats()` - All format support
- `test_concurrent_session_processing()` - Multiple sessions
- `test_configuration_synchronization()` - Config management

### 2. Format Compatibility Tests
**Supported Formats**: WAV, MP3, WebM, OGG, MP4, FLAC

Tests format-specific processing:
- Format conversion accuracy
- Sample rate handling (8kHz to 48kHz)
- Bit depth support (16-bit, 24-bit, 32-bit)
- Codec compatibility

### 3. Error Scenario Tests
**Error Types Tested**:
- Empty audio files
- Corrupted audio data (header, data, truncated)
- Invalid audio formats
- Oversized files
- Service unavailability
- Network timeouts
- Rate limiting

### 4. Performance Tests
**Metrics Measured**:
- Processing latency (target: < 2x real-time)
- Memory usage (with leak detection)
- Throughput (audio minutes per hour)
- Concurrent session handling
- Resource utilization

### 5. Audio Quality Tests
**Quality Scenarios**:
- High quality (SNR > 25dB)
- Medium quality (SNR 10-25dB)
- Low quality (SNR < 10dB)
- Multi-speaker audio
- Noisy environments

## Running Tests

### Quick Start
```bash
# Run all tests (except performance)
python tests/run_comprehensive_audio_tests.py --quick

# Run specific category
python tests/run_comprehensive_audio_tests.py --categories integration

# Run with verbose output
python tests/run_comprehensive_audio_tests.py --verbose
```

### Using Pytest Directly
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_complete_audio_flow.py -v

# Run tests with markers
pytest -m "integration and not slow" -v

# Run performance tests
pytest -m "performance" -v

# Run format compatibility tests
pytest -m "format" -v
```

### Test Categories
```bash
# Unit tests only
python tests/run_comprehensive_audio_tests.py --categories unit

# Integration tests
python tests/run_comprehensive_audio_tests.py --categories integration

# Performance tests
python tests/run_comprehensive_audio_tests.py --categories performance

# Error handling tests
python tests/run_comprehensive_audio_tests.py --categories error_handling

# Format compatibility tests
python tests/run_comprehensive_audio_tests.py --categories format_compatibility

# Multiple categories
python tests/run_comprehensive_audio_tests.py --categories integration performance
```

## Test Configuration

### Environment Variables
```bash
# Test timeout (default: 30 seconds)
export TEST_TIMEOUT=60

# Maximum concurrent tests (default: 10)
export MAX_CONCURRENT_TESTS=5

# Audio cache size (default: 100MB)
export TEST_AUDIO_CACHE_SIZE=200
```

### Pytest Markers
Use markers to filter tests:
```bash
# Audio pipeline tests
pytest -m "audio_pipeline"

# Service integration tests
pytest -m "service_integration"

# Format-specific tests
pytest -m "format_wav" -m "format_mp3"

# Quality-specific tests
pytest -m "high_quality"

# Speaker tests
pytest -m "single_speaker" -m "multi_speaker"
```

## Test Data Generation

The test suite automatically generates comprehensive audio test data:

### Audio Signal Types
- **Speech-like**: Realistic voice characteristics with harmonics and formants
- **Multi-speaker**: Overlapping and sequential speakers
- **Music**: Chord progressions and rhythmic patterns
- **Pure tones**: Sine waves for baseline testing
- **Noise**: White, pink, brown, and environmental noise

### Audio Effects
- **Reverb**: Room acoustics simulation
- **Echo**: Delay effects
- **Distortion**: Signal clipping and compression
- **Filtering**: Low-pass and high-pass filtering

### Quality Variations
- **SNR Levels**: 3dB to 25dB signal-to-noise ratio
- **Bit Depths**: 16-bit, 24-bit, 32-bit
- **Sample Rates**: 8kHz to 48kHz
- **Corruption**: Header, data, and structure corruption

## Output and Reporting

### Test Reports
```bash
# Generate comprehensive report
python tests/run_comprehensive_audio_tests.py --output test_report.json

# HTML reports (auto-generated)
# - test_report_integration.html
# - test_report_performance.html
# - test_report_error_handling.html
```

### Performance Metrics
The test suite tracks:
- **Processing Time**: End-to-end latency
- **Real-time Factor**: Processing time / audio duration
- **Memory Usage**: Peak and average memory consumption
- **Throughput**: Audio processed per unit time
- **Success Rate**: Percentage of successful requests

### Example Report Structure
```json
{
  "test_run_info": {
    "start_time": "2024-01-01T12:00:00Z",
    "total_duration": 45.2,
    "categories_tested": ["integration", "performance"],
    "overall_success": true
  },
  "system_info": {
    "platform": "Windows-10",
    "python_version": "3.11.0",
    "cpu_count": 8,
    "memory_total_gb": 16.0
  },
  "test_results": {
    "integration": {
      "exit_code": 0,
      "duration": 25.1,
      "summary_line": "15 passed, 0 failed"
    }
  }
}
```

## Performance Benchmarks

### Target Performance Metrics
- **Processing Latency**: < 2x real-time for standard audio
- **Memory Usage**: < 500MB growth during test suite
- **Concurrent Sessions**: Support 10+ simultaneous sessions
- **Throughput**: > 60 minutes audio processed per hour
- **Error Rate**: < 1% for valid inputs

### Regression Detection
The test suite detects performance regressions:
- **Latency Increase**: > 20% slower than baseline
- **Memory Growth**: > 100MB increase from baseline
- **Throughput Decrease**: > 15% reduction in throughput

## Troubleshooting

### Common Issues

1. **Service Connection Failures**
   ```bash
   # Check service health
   curl http://localhost:5001/health  # Whisper service
   curl http://localhost:5003/health  # Translation service
   ```

2. **Test Timeouts**
   ```bash
   # Increase timeout
   export TEST_TIMEOUT=120
   python tests/run_comprehensive_audio_tests.py
   ```

3. **Memory Issues**
   ```bash
   # Reduce concurrent tests
   export MAX_CONCURRENT_TESTS=3
   ```

4. **Audio Generation Errors**
   ```bash
   # Clear test cache
   rm -rf /tmp/audio_test_cache/
   ```

### Debug Mode
```bash
# Enable debug logging
export PYTEST_CURRENT_TEST=1
pytest tests/integration/test_complete_audio_flow.py::TestCompleteAudioFlow::test_complete_pipeline_wav_format -v -s --log-cli-level=DEBUG
```

### Mock vs Real Services
Tests use mocked services by default. To test against real services:
```bash
# Set environment variables for real services
export USE_REAL_SERVICES=true
export WHISPER_SERVICE_URL=http://localhost:5001
export TRANSLATION_SERVICE_URL=http://localhost:5003

python tests/run_comprehensive_audio_tests.py
```

## Contributing

### Adding New Tests

1. **Add test to appropriate category**:
   - Unit tests: `tests/unit/`
   - Integration tests: `tests/integration/`
   - Performance tests: `tests/performance/`

2. **Use appropriate markers**:
   ```python
   @pytest.mark.integration
   @pytest.mark.audio_pipeline
   async def test_new_feature():
       pass
   ```

3. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`

4. **Add to test runner** if creating new category:
   - Update `run_comprehensive_audio_tests.py`
   - Add category to `pytest.ini`

### Test Data Guidelines

1. **Use AudioTestDataManager** for consistent test data
2. **Cache generated audio** to improve test performance  
3. **Test edge cases** with various audio characteristics
4. **Document expected behavior** in test docstrings

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Audio Flow Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run audio tests
        run: python tests/run_comprehensive_audio_tests.py --quick
      - name: Upload test results
        uses: actions/upload-artifact@v2
        with:
          name: test-results
          path: test_*.xml
```

## Monitoring and Alerting

### Performance Monitoring
- Track test execution time trends
- Monitor memory usage patterns
- Alert on regression thresholds
- Generate performance dashboards

### Quality Gates
- All integration tests must pass
- Performance tests must meet SLA
- Error handling tests must validate graceful degradation
- Format compatibility must be 100%

This comprehensive test suite ensures the audio processing pipeline is robust, performant, and reliable across all supported scenarios.