# LiveTranslate Test Suite

Comprehensive test organization for the LiveTranslate project.

## Test Structure

```
tests/
├── e2e/                    # End-to-end full-stack tests
│   └── test_loopback_fullstack.py
├── integration/            # Service integration tests
│   ├── test_loopback_translation.py
│   └── TEST_RESULTS.md
└── system/                 # System-level tests
    ├── test_error_handling.py
    ├── test_model_selection.py
    ├── test_multipart.py
    └── test_real_audio.py
```

## Test Types

### End-to-End Tests (`e2e/`)
Complete pipeline tests simulating real user workflows:
- **Loopback Full-Stack**: Audio capture → Orchestration → Whisper → Translation → Display
- Tests the complete orchestration API (`/api/audio/upload`)
- Validates real-time streaming with configurable chunk sizes

**Run E2E tests:**
```bash
cd tests/e2e
python test_loopback_fullstack.py
```

### Integration Tests (`integration/`)
Service-to-service integration testing:
- **Loopback Translation**: Direct Whisper + Translation service tests
- Tests service APIs without orchestration layer
- Validates Chinese→English translation flow

**Run integration tests:**
```bash
cd tests/integration
python test_loopback_translation.py
```

### System Tests (`system/`)
System-level validation and edge case testing:
- **Error Handling**: Service resilience and recovery
- **Model Selection**: Model loading and fallback mechanisms
- **Multipart**: File upload and multipart request handling
- **Real Audio**: Production audio file processing

**Run system tests:**
```bash
cd tests/system
pytest -v
```

## Module-Specific Tests

Each service module has its own test suite:

- **Orchestration Service**: `modules/orchestration-service/tests/`
- **Whisper Service**: `modules/whisper-service/tests/`
- **Translation Service**: `modules/translation-service/tests/`
- **Bot Container**: `modules/bot-container/tests/`

See module-specific README files for detailed test documentation.

## Running Tests

### All Tests
```bash
# Root-level tests
pytest tests/ -v

# All module tests
pytest modules/*/tests/ -v
```

### Service-Specific
```bash
# Orchestration
cd modules/orchestration-service
pytest tests/ -v

# Whisper
cd modules/whisper-service
python tests/run_tests.py --all

# Translation
cd modules/translation-service
pytest tests/ -v

# Bot
cd modules/bot-container
pytest tests/ -v
```

### With Coverage
```bash
pytest tests/ -v --cov=modules --cov-report=html
```

### Performance Tests
```bash
pytest tests/ -v -m performance
```

## Test Requirements

### Prerequisites
- **Audio Loopback**: BlackHole or similar virtual audio device (macOS)
- **Services Running**: Orchestration (port 3000), Whisper (port 5001), Translation (port 5003)
- **Python**: 3.10+ with test dependencies installed

### Install Test Dependencies
```bash
# Root level
pip install -r requirements-test.txt

# Per module
pip install -r modules/orchestration-service/requirements-test.txt
pip install -r modules/whisper-service/tests/requirements-test.txt
pip install -r modules/translation-service/requirements-test.txt
```

## Test Conventions

### File Naming
- `test_*.py` - Unit and integration tests
- `test_*_integration.py` - Explicit integration tests
- `test_*_e2e.py` - End-to-end tests (optional suffix)

### Markers
```python
@pytest.mark.integration  # Integration test
@pytest.mark.e2e          # End-to-end test
@pytest.mark.slow         # Long-running test
@pytest.mark.performance  # Performance benchmark
@pytest.mark.requires_db  # Requires database
```

### Test Output
All test results should be saved to `tests/output/` or module-specific `tests/output/`:
```bash
# Example
pytest tests/ -v > tests/output/$(date +%Y%m%d_%H%M%S)_test_results.log
```

## Chinese→English Translation Tests

### Loopback Audio Testing
The primary use case for testing is **Chinese→English real-time translation**:

1. **Full-Stack Test** (`e2e/test_loopback_fullstack.py`):
   - Complete orchestration pipeline
   - Real-time audio chunks (2-5 seconds)
   - WebSocket or REST API

2. **Direct Service Test** (`integration/test_loopback_translation.py`):
   - Direct Whisper + Translation calls
   - No orchestration layer
   - Faster iteration for development

### Running CN→EN Tests
```bash
# Start services
cd modules/orchestration-service && python src/main.py &
cd modules/whisper-service && python src/main.py &
cd modules/translation-service && python src/api_server_fastapi.py &

# Run loopback test
cd tests/e2e
python test_loopback_fullstack.py

# Play Chinese audio on your system to test
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Tests
  run: |
    pytest tests/ -v --tb=short
    pytest modules/*/tests/ -v --tb=short
```

### Pre-commit Hooks
```bash
# Run quick tests before commit
pytest tests/unit/ -v --tb=short
```

## Troubleshooting

### Common Issues

**Services Not Ready:**
```bash
# Check service health
curl http://localhost:3000/api/health  # Orchestration
curl http://localhost:5001/health      # Whisper
curl http://localhost:5003/api/health  # Translation
```

**Audio Loopback Not Found:**
```bash
# Install BlackHole (macOS)
brew install blackhole-2ch

# Set system audio output to BlackHole
# Set test script to use BlackHole input
```

**Import Errors:**
```bash
# Ensure PYTHONPATH includes modules
export PYTHONPATH="${PYTHONPATH}:$(pwd)/modules"
```

## Contributing

When adding new tests:
1. Place in appropriate directory (`e2e/`, `integration/`, `system/`, or module-specific)
2. Use descriptive names (`test_<feature>_<scenario>.py`)
3. Add pytest markers for categorization
4. Update this README if adding new test categories
5. Save test output to `tests/output/` with timestamps
