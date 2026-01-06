# Test Execution Guide - Post Import Fixes

## Quick Start

All import and dependency issues have been resolved. Tests can now run without errors.

---

## Orchestration Service Tests

### Location
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/orchestration-service
```

### Basic Test Execution
```bash
# Run all tests
poetry run pytest tests/ -v

# Run bot lifecycle tests
poetry run pytest tests/test_bot_lifecycle.py -v

# Run with specific markers
poetry run pytest -m "unit" -v                    # Unit tests only
poetry run pytest -m "integration" -v             # Integration tests only
poetry run pytest -m "not e2e" -v                 # Exclude e2e tests
poetry run pytest -m "not slow" -v                # Exclude slow tests

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

# Run specific test file
poetry run pytest tests/unit/test_pipeline_fixes.py -v
```

### Available Markers
- `e2e` - End-to-end tests
- `integration` - Integration tests
- `unit` - Unit tests
- `slow` - Slow running tests
- `audio_pipeline` - Audio pipeline tests
- `performance` - Performance tests

---

## Whisper Service Tests

### Location
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/whisper-service
```

### Basic Test Execution
```bash
# Run all tests
poetry run pytest tests/ -v

# Run integration tests
poetry run pytest tests/integration/ -v

# Run milestone tests
poetry run pytest tests/integration/milestone2/test_real_code_switching.py -v

# Run with markers
poetry run pytest -m "integration" -v             # Integration tests
poetry run pytest -m "not slow" -v                # Exclude slow tests
poetry run pytest -m "not gpu" -v                 # Exclude GPU tests

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html -v
```

### Available Markers
- `integration` - Integration tests
- `slow` - Slow running tests
- `openvino` - Requires OpenVINO
- `gpu` - Requires GPU
- `e2e` - End-to-end tests
- `unit` - Unit tests
- `stress` - Stress tests
- `accuracy` - Accuracy regression tests
- `property` - Property-based tests
- `benchmark` - Performance benchmarks

---

## Translation Service Tests

### Location
```bash
cd /Users/thomaspatane/Documents/GitHub/livetranslate/modules/translation-service
```

### Basic Test Execution
```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

# Run specific test patterns
poetry run pytest tests/unit/ -v
poetry run pytest tests/integration/ -v
```

---

## Common Test Patterns

### Filter by Test Name
```bash
# Run tests matching pattern
poetry run pytest -k "test_bot" -v
poetry run pytest -k "pipeline" -v
poetry run pytest -k "audio" -v
```

### Verbose Output
```bash
# Extra verbose with print statements
poetry run pytest -vv -s tests/

# Show local variables on failure
poetry run pytest -vv -l tests/
```

### Stop on First Failure
```bash
poetry run pytest -x tests/
```

### Run Last Failed Tests
```bash
poetry run pytest --lf tests/
```

### Parallel Execution (if pytest-xdist installed)
```bash
poetry run pytest -n auto tests/
```

---

## Test Output Logging

### Save Test Results
```bash
# Orchestration
cd modules/orchestration-service
poetry run pytest tests/ -v > tests/output/$(date +%Y%m%d_%H%M%S)_test_results.log 2>&1

# Whisper
cd modules/whisper-service
poetry run pytest tests/ -v > tests/output/$(date +%Y%m%d_%H%M%S)_test_results.log 2>&1

# Translation
cd modules/translation-service
poetry run pytest tests/ -v > tests/output/$(date +%Y%m%d_%H%M%S)_test_results.log 2>&1
```

---

## Import Verification

### Quick Import Checks

**Orchestration Service**:
```bash
cd modules/orchestration-service
poetry run python -c "from src.dependencies import get_bot_manager, get_event_publisher; print('✅ Dependencies OK')"
poetry run python -c "from src.routers.bot.bot_lifecycle import router; print('✅ Bot routers OK')"
```

**Whisper Service**:
```bash
cd modules/whisper-service
poetry run python -c "from tests.test_utils import calculate_wer_detailed; print('✅ Test utils OK')"
poetry run python -c "from src.api_server import app; print('✅ API server OK')"
```

**Translation Service**:
```bash
cd modules/translation-service
poetry run python -c "from src.translation_service import TranslationService; print('✅ Translation service OK')"
```

---

## Troubleshooting

### Issue: ModuleNotFoundError
**Solution**: Ensure you're in the correct directory and using poetry run:
```bash
cd modules/[service-name]
poetry run pytest tests/ -v
```

### Issue: Import errors in tests
**Solution**: Check that pythonpath is configured in pyproject.toml:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

### Issue: Marker warnings
**Solution**: Use --strict-markers flag (already in pyproject.toml):
```bash
poetry run pytest --strict-markers tests/
```

### Issue: Coverage not working
**Solution**: Ensure pytest-cov is installed:
```bash
poetry show pytest-cov
# If missing: poetry add -D pytest-cov
```

---

## Continuous Integration

### GitHub Actions Example
```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install Poetry
        run: pip install poetry

      - name: Test Orchestration Service
        run: |
          cd modules/orchestration-service
          poetry install
          poetry run pytest tests/ -v --cov=src

      - name: Test Whisper Service
        run: |
          cd modules/whisper-service
          poetry install
          poetry run pytest tests/ -v --cov=src -m "not slow"

      - name: Test Translation Service
        run: |
          cd modules/translation-service
          poetry install
          poetry run pytest tests/ -v --cov=src
```

---

## Test Categories

### Unit Tests (Fast, No External Dependencies)
```bash
poetry run pytest -m "unit" tests/
```

### Integration Tests (Service Integration)
```bash
poetry run pytest -m "integration" tests/
```

### End-to-End Tests (Full System)
```bash
poetry run pytest -m "e2e" tests/
```

### Smoke Tests (Quick Validation)
```bash
poetry run pytest tests/smoke/ -v
```

---

## Performance Testing

### Orchestration Service
```bash
cd modules/orchestration-service
poetry run pytest tests/integration/test_pipeline_production_readiness.py -v
```

### Whisper Service
```bash
cd modules/whisper-service
poetry run pytest tests/benchmarks/ -v
poetry run pytest -m "benchmark" tests/
```

---

## Expected Results

### All Imports Working ✅
```
✅ src.dependencies imports successful
✅ src.routers.bot.* imports successful
✅ tests.test_utils imports successful
✅ All absolute imports with src. prefix working
```

### All Markers Recognized ✅
```
✅ @pytest.mark.e2e
✅ @pytest.mark.integration
✅ @pytest.mark.unit
✅ @pytest.mark.slow
✅ Custom markers from each service
```

### All Packages Available ✅
```
✅ pytest-cov installed in all services
✅ timecode installed in orchestration-service
✅ psycopg2-binary installed in orchestration-service
```

---

## Quick Reference

### Most Common Commands

```bash
# Orchestration - Bot Tests
cd modules/orchestration-service && poetry run pytest tests/test_bot_lifecycle.py -v

# Whisper - Integration Tests
cd modules/whisper-service && poetry run pytest tests/integration/ -v -m "not slow"

# Translation - All Tests with Coverage
cd modules/translation-service && poetry run pytest tests/ --cov=src --cov-report=html -v

# Run all services (from root)
for service in orchestration-service whisper-service translation-service; do
  echo "Testing $service..."
  cd modules/$service
  poetry run pytest tests/ -v -m "not slow"
  cd ../..
done
```

---

## Success Criteria

✅ All imports resolve without ModuleNotFoundError
✅ All pytest markers recognized without warnings
✅ All tests can execute with coverage reporting
✅ Test utilities (test_utils.py) importable from tests
✅ Absolute imports (src.*) working across all modules

**Status**: All criteria met as of 2026-01-05

---

## Additional Resources

- [DEPENDENCY_FIXES_COMPLETE.md](./DEPENDENCY_FIXES_COMPLETE.md) - Complete fix documentation
- [COMPLETE_TEST_ERROR_ANALYSIS.md](./COMPLETE_TEST_ERROR_ANALYSIS.md) - Original error analysis
- [modules/orchestration-service/fix_imports.py](./modules/orchestration-service/fix_imports.py) - Import fix automation

**All tests ready to run!** 🚀
