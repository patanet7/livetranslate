# Comprehensive Audio Testing Suite

This comprehensive testing suite provides complete validation for all audio processing components in the LiveTranslate orchestration service, including I/O validation and component interaction testing as requested.

## Overview

The testing suite is organized into four main categories:

- **Unit Tests** (`unit/`) - Test individual components in isolation
- **Integration Tests** (`integration/`) - Test component interactions and I/O operations  
- **Performance Tests** (`performance/`) - Test performance characteristics and scalability
- **End-to-End Tests** (`e2e/`) - Test complete workflows (future implementation)

## Quick Start

### Run All Tests

```bash
# Run comprehensive test suite
python run_audio_tests.py

# Run with verbose output
python run_audio_tests.py --verbose

# Run specific test suites
python run_audio_tests.py --suites unit integration

# Generate detailed report
python run_audio_tests.py --report test_report.json
```

### Run Individual Test Suites

```bash
# Unit tests only
pytest unit/ -v

# Integration tests only  
pytest integration/ -v

# Performance tests only
pytest performance/ -v

# With coverage reporting
pytest unit/ --cov=src.audio --cov-report=html
```

## Test Suite Details

### Unit Tests (`unit/`)

**Purpose**: Test individual components in isolation with comprehensive validation.

**Files**:
- `test_audio_models.py` - Test all Pydantic models with validation
- `test_audio_processor.py` - Test complete audio processing pipeline
- `test_chunk_manager.py` - Test audio chunking logic (future)
- `test_database_adapter.py` - Test database operations (future)
- `test_config_manager.py` - Test configuration management (future)

**Key Features**:
- Comprehensive data model validation
- Audio processing pipeline testing (VAD, noise reduction, voice enhancement, compression)
- Parameter validation and edge cases
- Error handling verification
- Mock-based isolation testing

**Example**:
```bash
# Run all unit tests
pytest unit/ -v

# Run specific test file
pytest unit/test_audio_models.py -v

# Run with coverage
pytest unit/ --cov=src.audio --cov-report=term-missing
```

### Integration Tests (`integration/`)

**Purpose**: Test component interactions, database operations, and real I/O operations.

**Files**:
- `test_audio_coordinator_integration.py` - Test AudioCoordinator with real services
- `test_chunk_manager_integration.py` - Test ChunkManager with file and database I/O
- `test_service_communication.py` - Test service-to-service communication (future)
- `test_database_integration.py` - Test database operations end-to-end (future)

**Key Features**:
- Real database operations with PostgreSQL
- File I/O operations with audio files
- Service communication testing
- WebSocket integration testing
- Session lifecycle management
- Error recovery and resilience testing
- Resource cleanup verification

**Example**:
```bash
# Run all integration tests
pytest integration/ -v

# Run with real database (requires PostgreSQL)
pytest integration/ -v --db-url="postgresql://postgres:password@localhost:5432/test_db"

# Run specific integration test
pytest integration/test_audio_coordinator_integration.py::TestAudioCoordinatorIntegration::test_session_lifecycle_management -v
```

### Performance Tests (`performance/`)

**Purpose**: Test performance characteristics, throughput, latency, and scalability.

**Files**:
- `test_audio_performance.py` - Comprehensive performance testing for all components
- `test_load_testing.py` - High-load scenario testing (future)
- `test_memory_profiling.py` - Memory usage profiling (future)

**Key Features**:
- Throughput measurement (processed audio duration vs processing time)
- Latency measurement (per-operation timing)
- Memory usage monitoring
- Concurrent processing testing
- Scalability testing with multiple sessions
- Stress testing with large audio files
- Performance regression detection

**Performance Thresholds**:
- Audio processing: >10x real-time throughput
- Latency: <100ms per operation
- Memory: <200MB growth under normal load
- Success rate: >95% under concurrent load

**Example**:
```bash
# Run all performance tests
pytest performance/ -v

# Run with extended duration
pytest performance/ -v --duration=300

# Performance baseline establishment
pytest performance/test_audio_performance.py::TestPerformanceRegression::test_performance_baseline -v
```

## Test Configuration and Fixtures

### Configuration Files

Tests use configuration fixtures defined in `conftest.py`:

**Sample Configurations**:
```python
# Default test configuration
test_audio_processing_config = {
    "preset_name": "test_preset",
    "enabled_stages": ["vad", "voice_filter", "noise_reduction", "voice_enhancement", "compression", "limiter"],
    "vad": {"enabled": True, "aggressiveness": 2},
    "noise_reduction": {"enabled": True, "strength": 0.7}
}

# Performance test configuration  
performance_test_config = {
    "concurrent_sessions": [1, 5, 10],
    "audio_durations": [1.0, 5.0, 10.0],
    "max_processing_time_ms": {"vad": 10, "noise_reduction": 50},
    "max_memory_usage_mb": 100
}
```

### Test Data Generation

The `AudioTestFixtures` class provides comprehensive test audio generation:

```python
# Generate various audio types
fixtures.generate_voice_like_audio(duration=3.0)     # Realistic voice with harmonics
fixtures.generate_noise(duration=2.0, amplitude=0.5) # White noise
fixtures.generate_silence(duration=1.0)              # Pure silence
fixtures.generate_clipped_audio(duration=2.0)        # Clipped audio for testing
fixtures.generate_noisy_voice(duration=3.0, snr_db=10) # Voice with specific SNR
```

### Mock Services

Integration tests use comprehensive mocking:

```python
# Mock database adapter
mock_database_adapter = AsyncMock(spec=AudioDatabaseAdapter)
mock_database_adapter.store_audio_chunk.return_value = "test_chunk_id"

# Mock service responses
mock_whisper_response = {
    "text": "Test transcription",
    "speaker_id": "speaker_0", 
    "confidence": 0.9
}
```

## Running Tests in CI/CD

### GitHub Actions Integration

```yaml
name: Audio Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run comprehensive audio tests
      run: |
        cd modules/orchestration-service/tests/audio
        python run_audio_tests.py --suites unit integration --report ci_report.json
    
    - name: Upload test results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: modules/orchestration-service/tests/audio/ci_report.json
```

### Docker Testing Environment

```dockerfile
# Test environment Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libportaudio2 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy test code
COPY tests/ /app/tests/
COPY src/ /app/src/

WORKDIR /app/tests/audio

# Run tests
CMD ["python", "run_audio_tests.py", "--suites", "unit", "integration", "performance"]
```

## Test Quality Assurance

### Coverage Requirements

- **Unit Tests**: ≥85% code coverage
- **Integration Tests**: ≥75% code coverage  
- **Performance Tests**: ≥60% code coverage
- **Overall**: ≥80% combined coverage

### Quality Metrics

All tests must pass the following quality checks:

1. **Correctness**: All assertions pass
2. **Performance**: Meet defined performance thresholds
3. **Reliability**: ≥95% success rate under normal conditions
4. **Memory**: No memory leaks or excessive growth
5. **Error Handling**: Graceful failure handling

### Test Data Validation

Tests include comprehensive I/O validation:

```python
def assert_audio_quality(audio_data: np.ndarray):
    """Validate audio data quality."""
    assert not np.isnan(audio_data).any(), "Audio contains NaN values"
    assert not np.isinf(audio_data).any(), "Audio contains infinite values"
    assert 0.0 <= np.max(np.abs(audio_data)) <= 1.0, "Audio levels out of range"

def assert_chunk_metadata_valid(metadata: AudioChunkMetadata):
    """Validate chunk metadata."""
    assert metadata.chunk_id is not None
    assert metadata.duration_seconds > 0
    assert 0.0 <= metadata.audio_quality_score <= 1.0
```

## Debugging and Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Database Connection**: Check PostgreSQL is running for integration tests
3. **Audio Files**: Verify `soundfile` can read test audio files
4. **Performance Failures**: Check system resources and thresholds

### Debug Mode

```bash
# Run tests with debug output
pytest unit/ -v -s --tb=long

# Run single test with full output
pytest unit/test_audio_models.py::TestAudioChunkMetadata::test_valid_chunk_metadata_creation -v -s

# Run with debugger
pytest --pdb unit/test_audio_models.py
```

### Test Output

```bash
🚀 Starting Comprehensive Audio Testing Suite
   Suites: unit, integration, performance
   Base path: /path/to/tests/audio
   Timestamp: 2024-01-15T14:30:00

🔍 Validating test environment...
   ✅ All required packages available
   ✅ unit: 2 test files
   ✅ integration: 2 test files  
   ✅ performance: 1 test files

🧪 Running Unit Tests...
   Path: unit/
   Timeout: 300s

✅ Unit Tests
   Tests: 45✅ 0❌ 0💥 2⏭️
   Duration: 12.3s
   Coverage: 87.2%

🧪 Running Integration Tests...
   Path: integration/
   Timeout: 600s

✅ Integration Tests
   Tests: 23✅ 0❌ 0💥 1⏭️
   Duration: 45.7s
   Coverage: 78.9%

🏁 COMPREHENSIVE AUDIO TESTING SUMMARY
   Status: ✅ PASSED
   Total Tests: 71
   Passed: 68 ✅
   Failed: 0 ❌
   Errors: 0 💥
   Skipped: 3 ⏭️
   Average Coverage: 83.1%
   Total Duration: 58.0s
   Success Rate: 95.8%
   🎉 Excellent test results!
```

## Test Extensions

### Adding New Tests

1. **Unit Test**: Add to appropriate `unit/test_*.py` file
2. **Integration Test**: Add to `integration/test_*_integration.py`
3. **Performance Test**: Add to `performance/test_*_performance.py`

### Custom Test Markers

```python
# Mark tests with custom markers
@pytest.mark.slow
def test_large_audio_processing():
    """Test with large audio files."""
    pass

@pytest.mark.gpu_required
def test_gpu_acceleration():
    """Test requiring GPU acceleration."""
    pass
```

Run marked tests:
```bash
# Run only fast tests
pytest -m "not slow"

# Run GPU tests only
pytest -m "gpu_required"
```

This comprehensive testing suite ensures robust validation of all audio processing components with complete I/O validation and component interaction testing as specifically requested.
