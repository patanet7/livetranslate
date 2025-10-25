# Whisper Service Test Infrastructure

This directory contains the comprehensive test infrastructure for the whisper-service module.

## Directory Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── fixtures/                      # Test fixtures and resources
│   └── audio/                     # Generated audio fixtures
│       ├── hello_world.wav        # 3s speech-like audio
│       ├── silence.wav            # 2s silence
│       ├── noisy.wav              # 3s noisy speech (SNR=10dB)
│       ├── short_speech.wav       # 1s short speech
│       ├── long_speech.wav        # 5s long speech
│       └── white_noise.wav        # 2s white noise
├── unit/                          # Unit tests (isolated components)
│   ├── __init__.py
│   └── test_*.py
├── integration/                   # Integration tests (end-to-end)
│   ├── __init__.py
│   └── test_*.py
├── test_fixtures.py               # Tests for fixture infrastructure
└── README.md                      # This file
```

## Audio Fixtures

All audio fixtures are **real audio files** (not mocks):
- **Format**: WAV (PCM)
- **Sample Rate**: 16kHz (Whisper's native rate)
- **Channels**: Mono
- **Dtype**: float32 (-1.0 to 1.0)

Fixtures are automatically generated on first test run and cached for subsequent runs.

### Available Fixtures

| Fixture | Duration | Description |
|---------|----------|-------------|
| `hello_world_audio` | 3.0s | Multi-formant speech-like audio |
| `silence_audio` | 2.0s | Pure silence (zeros) |
| `noisy_audio` | 3.0s | Speech with 10dB SNR noise |
| `short_speech_audio` | 1.0s | Short speech clip |
| `long_speech_audio` | 5.0s | Longer speech clip |
| `white_noise_audio` | 2.0s | White noise |
| `all_audio_fixtures` | - | Dict of all fixtures |

### Usage Example

```python
def test_my_function(hello_world_audio):
    """Test using audio fixture."""
    audio, sr = hello_world_audio

    assert sr == 16000
    assert audio.dtype == np.float32
    assert len(audio) == sr * 3  # 3 seconds

    # Use audio in your test
    result = my_transcription_function(audio, sr)
    assert result is not None
```

## Pytest Markers

The test infrastructure includes custom markers for organizing and filtering tests:

### Available Markers

| Marker | Description | Usage |
|--------|-------------|-------|
| `@pytest.mark.integration` | Integration/end-to-end tests | `-m integration` |
| `@pytest.mark.slow` | Slow running tests | `-m "not slow"` to skip |
| `@pytest.mark.openvino` | Requires OpenVINO | Auto-skip if not installed |
| `@pytest.mark.gpu` | Requires GPU (CUDA) | Auto-skip if not available |

### Examples

```bash
# Run only unit tests (fast)
pytest tests/unit/ -v

# Run integration tests, skip slow ones
pytest tests/integration/ -v -m "integration and not slow"

# Run only GPU tests
pytest -v -m gpu

# Run everything except slow tests
pytest -v -m "not slow"

# Run with strict marker checking
pytest -v --strict-markers
```

### Using Markers in Tests

```python
import pytest

@pytest.mark.slow
def test_expensive_operation():
    """This test takes a long time."""
    pass

@pytest.mark.gpu
def test_cuda_acceleration():
    """This test requires GPU."""
    pass

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline():
    """Integration test that is also slow."""
    pass
```

## Hardware Detection

The test infrastructure automatically detects available hardware:

### Fixtures

- `has_openvino`: Boolean, True if OpenVINO is available
- `has_gpu`: Boolean, True if CUDA GPU is available
- `device_type`: String, best available device ("openvino", "cuda", or "cpu")

### Usage

```python
def test_device_detection(device_type):
    """Test uses best available device."""
    if device_type == "openvino":
        # Test OpenVINO path
        pass
    elif device_type == "cuda":
        # Test GPU path
        pass
    else:
        # Test CPU path
        pass
```

## Configuration Fixtures

### `default_whisper_config`

Provides default Whisper configuration for tests:

```python
def test_with_config(default_whisper_config):
    """Test using default configuration."""
    config = default_whisper_config

    assert config["model_name"] == "base"
    assert config["sample_rate"] == 16000
    assert config["device"] == "cpu"

    # Modify for your test
    config["temperature"] = 0.5
```

## Mock Fixtures

### `mock_whisper_model`

Provides a mock Whisper model for testing without loading real models:

```python
def test_with_mock(mock_whisper_model):
    """Test using mock model."""
    result = mock_whisper_model.transcribe(np.zeros(16000))

    assert result["text"] == "Hello world"
    assert "segments" in result
```

## Temporary Directories

### `temp_audio_dir`

Creates a temporary directory for test file operations:

```python
def test_file_operations(temp_audio_dir, hello_world_audio):
    """Test writing/reading audio files."""
    import soundfile as sf

    audio, sr = hello_world_audio

    # Write to temp directory
    output_file = temp_audio_dir / "output.wav"
    sf.write(output_file, audio, sr)

    assert output_file.exists()
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_audio_sample.py -v

# Run specific test function
pytest tests/unit/test_audio_sample.py::test_audio_duration -v

# Show print statements
pytest -v -s

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf
```

### Coverage

```bash
# Run with coverage
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Parallel Execution

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest -n 4
```

## Writing New Tests

### Unit Test Template

```python
"""
Test module for [component name].
"""

import numpy as np
import pytest


def test_basic_functionality(hello_world_audio):
    """Test basic functionality."""
    audio, sr = hello_world_audio

    # Your test code here
    assert True


@pytest.mark.slow
def test_expensive_operation(long_speech_audio):
    """Test that takes a long time."""
    audio, sr = long_speech_audio

    # Expensive operation here
    assert True
```

### Integration Test Template

```python
"""
Integration test for [feature name].
"""

import pytest


@pytest.mark.integration
def test_end_to_end(hello_world_audio, default_whisper_config):
    """Test complete pipeline."""
    audio, sr = hello_world_audio
    config = default_whisper_config

    # Test full pipeline
    assert True


@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_pipeline(hello_world_audio):
    """Test GPU-accelerated pipeline."""
    # This test will be skipped if GPU not available
    assert True
```

## Test Organization Guidelines

### Unit Tests (`tests/unit/`)

- Test **individual components** in isolation
- Should be **fast** (< 1 second each)
- Use **mocks** for external dependencies
- No real model loading
- No network calls

### Integration Tests (`tests/integration/`)

- Test **components working together**
- Can be **slower** (mark with `@pytest.mark.slow` if > 5 seconds)
- Can use **real models** (but prefer smaller ones)
- Test **actual hardware acceleration**
- Test **end-to-end pipelines**

## Continuous Integration

The test infrastructure is designed for CI/CD:

```yaml
# Example GitHub Actions workflow
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run fast tests
        run: poetry run pytest -v -m "not slow"

      - name: Run integration tests
        run: poetry run pytest -v -m "integration and not slow"
```

## Troubleshooting

### Fixtures not generated

```bash
# Manually trigger fixture generation
pytest tests/test_fixtures.py -v -s
```

### Import errors

```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Missing dependencies

```bash
# Install test dependencies
poetry install --with dev

# Or with pip
pip install pytest pytest-cov pytest-mock soundfile numpy
```

## Best Practices

1. **Use fixtures** for common test data
2. **Mark slow tests** with `@pytest.mark.slow`
3. **Mark hardware-specific tests** with `@pytest.mark.gpu` or `@pytest.mark.openvino`
4. **Keep unit tests fast** (< 1 second)
5. **Use descriptive test names** that explain what is being tested
6. **One assertion per test** when possible (or closely related assertions)
7. **Clean up resources** in tests (use `temp_audio_dir` for files)
8. **Document complex tests** with docstrings

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers](https://docs.pytest.org/en/stable/mark.html)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
