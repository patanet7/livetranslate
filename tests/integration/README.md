# LiveTranslate Integration Tests
## TDD Test Suite for System Upgrade

**Status**: 🔴 RED PHASE (Tests written BEFORE implementation)
**Phase**: 0 - Test Infrastructure
**Created**: 2025-10-20

---

## Overview

This directory contains **comprehensive integration tests** for the LiveTranslate system upgrade. Following **Test-Driven Development (TDD)**, all tests were written BEFORE implementation to:

1. **Define requirements clearly**
2. **Ensure comprehensive coverage**
3. **Validate features as they're built**
4. **Prevent regressions**

---

## Test Files

### SimulStreaming Innovations (7 files)

1. **`test_alignatt_streaming.py`** - AlignAtt attention-guided streaming
   - Frame threshold constraints
   - Attention masking
   - Incremental decoding
   - Target: -30-50% latency reduction

2. **`test_beam_search.py`** - Beam search decoding
   - Beam width variations (1, 3, 5, 10)
   - Quality improvements
   - Memory constraints
   - Target: +20-30% quality improvement

3. **`test_in_domain_prompts.py`** - Domain terminology injection
   - Medical, legal, technical, financial domains
   - Custom terminology
   - Scrolling context
   - Target: -40-60% domain errors

4. **`test_computationally_aware_chunking.py`** - Dynamic chunking
   - RTF (Real-Time Factor) calculation
   - Adaptive chunk sizing
   - Buffer overflow prevention
   - Target: -60% jitter reduction

5. **`test_context_carryover.py`** - Context management
   - 30-second window processing
   - Context pruning
   - Coherence improvement
   - Target: +25-40% long-form quality

6. **`test_silero_vad.py`** - Voice Activity Detection
   - Silence detection
   - Speech probability
   - Computational savings
   - Target: -30-50% computation

7. **`test_cif_word_boundaries.py`** - Word boundary detection
   - Incomplete word detection
   - Partial word truncation
   - Re-translation reduction
   - Target: -50% re-translations

### Vexa Innovations (1 file)

8. **`test_websocket_optimization.py`** - Sub-second WebSocket
   - Binary protocol (MessagePack)
   - Event-driven updates
   - Connection pooling
   - Target: <100ms network latency

### New Features (1 file)

9. **`test_chat_history.py`** - Conversation persistence
   - Session storage
   - Message retrieval
   - Date range queries
   - Full-text search
   - User isolation

### Regression Tests (1 file)

10. **`test_feature_preservation.py`** - Existing features
    - Google Meet bot
    - Virtual webcam
    - Speaker attribution
    - Time correlation
    - NPU acceleration
    - Config sync
    - Database integration

---

## Running Tests

### Install Dependencies

```bash
cd tests/integration
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# From project root
pytest tests/integration/ -v

# With coverage
pytest tests/integration/ -v --cov=modules --cov-report=html

# Parallel execution (faster)
pytest tests/integration/ -v -n auto
```

### Run Specific Test Files

```bash
# Single file
pytest tests/integration/test_alignatt_streaming.py -v

# Multiple files
pytest tests/integration/test_beam_search.py tests/integration/test_in_domain_prompts.py -v
```

### Run by Marker

```bash
# Only integration tests
pytest tests/integration/ -v -m integration

# Only slow tests
pytest tests/integration/ -v -m slow

# Exclude slow tests
pytest tests/integration/ -v -m "not slow"

# Feature preservation only
pytest tests/integration/ -v -m feature_preservation

# Tests requiring GPU
pytest tests/integration/ -v -m requires_gpu

# Tests requiring database
pytest tests/integration/ -v -m requires_db
```

### Run with Specific Options

```bash
# Stop on first failure
pytest tests/integration/ -v -x

# Show local variables on failure
pytest tests/integration/ -v -l

# Verbose output with print statements
pytest tests/integration/ -v -s

# Run failed tests from last run
pytest tests/integration/ -v --lf

# Generate HTML report
pytest tests/integration/ -v --html=report.html
```

---

## Test Markers

Tests are marked with the following markers:

- `@pytest.mark.integration` - Integration test
- `@pytest.mark.slow` - Test takes significant time
- `@pytest.mark.requires_gpu` - Requires GPU
- `@pytest.mark.requires_npu` - Requires Intel NPU
- `@pytest.mark.requires_db` - Requires database
- `@pytest.mark.requires_redis` - Requires Redis
- `@pytest.mark.feature_preservation` - Regression test

---

## Expected Test Results

### Current Status (Phase 0 - TDD Red Phase)

Most tests are **EXPECTED TO FAIL** because features are not implemented yet.

| Test File | Expected Status | Reason |
|-----------|----------------|--------|
| `test_alignatt_streaming.py` | 🔴 FAIL | AlignAttDecoder not implemented |
| `test_beam_search.py` | 🔴 FAIL | BeamSearchDecoder not implemented |
| `test_in_domain_prompts.py` | 🔴 FAIL | DomainPromptManager not implemented |
| `test_computationally_aware_chunking.py` | 🔴 FAIL | ComputationallyAwareChunker not implemented |
| `test_context_carryover.py` | 🔴 FAIL | ContextManager not implemented |
| `test_silero_vad.py` | 🔴 FAIL | SileroVAD not implemented |
| `test_cif_word_boundaries.py` | 🔴 FAIL | WordBoundaryDetector not implemented |
| `test_websocket_optimization.py` | 🔴 FAIL | OptimizedWebSocketManager not implemented |
| `test_chat_history.py` | 🔴 FAIL | Chat models not implemented |
| `test_feature_preservation.py` | 🟢 PASS (mostly) | Testing existing features |

**This is EXPECTED and CORRECT** - we're following TDD!

### Implementation Progress Tracking

As features are implemented, tests will transition:
- 🔴 RED (failing) → 🟢 GREEN (passing)

Track progress in `UPGRADE_PLAN.md`.

---

## Test Configuration

### Database Setup

Tests requiring database use PostgreSQL:

```bash
# Set environment variable
export TEST_DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/livetranslate_test"

# Or use fake in-memory database
export USE_FAKE_REDIS=true
```

### Redis Setup

Tests requiring Redis:

```bash
# Set environment variable
export TEST_REDIS_URL="redis://localhost:6379/1"

# Or use fake in-memory Redis (default)
export USE_FAKE_REDIS=true
```

---

## Fixtures

### Database Fixtures
- `postgres_fixture` - PostgreSQL session with schema
- `db_session` - Alias for postgres_fixture

### Service Fixtures
- `whisper_service_fixture` - Mocked Whisper service
- `translation_service_fixture` - Mocked Translation service
- `orchestration_service_fixture` - Orchestration with test dependencies

### Utility Fixtures
- `generate_test_audio` - Generate synthetic audio
- `generate_test_audio_chunks` - Generate chunked audio for streaming
- `calculate_wer` - Calculate Word Error Rate
- `assert_latency` - Assert latency requirements
- `baseline_metrics` - Store/retrieve baseline metrics

See `conftest.py` for full fixture documentation.

---

## Writing New Tests

### Test Structure

```python
"""
TDD Test Suite for [Feature Name]
Tests written BEFORE implementation

Status: 🔴 Expected to FAIL (not implemented yet)
"""
import pytest

class Test[FeatureName]:
    """Test [feature description]"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_[specific_behavior](self, fixture1, fixture2):
        """Test that [specific behavior] works correctly"""
        # EXPECTED TO FAIL - not implemented yet

        try:
            from module.path import FeatureClass
        except ImportError:
            pytest.skip("FeatureClass not implemented yet")

        # Arrange
        feature = FeatureClass()

        # Act
        result = await feature.do_something()

        # Assert
        assert result == expected_value
```

### Test Naming Convention

- File: `test_[feature_name].py`
- Class: `Test[FeatureName]`
- Method: `test_[specific_behavior]`

### Use Descriptive Assertions

```python
# Good
assert latency < 150, f"Expected <150ms, got {latency}ms"

# Bad
assert latency < 150
```

---

## Continuous Integration

Tests will run automatically on:
- ✅ Pull requests
- ✅ Main branch commits
- ✅ Release tags

(Note: CI/CD configuration to be added in production deployment)

---

## Test Coverage Goals

- **Unit Tests**: >80% coverage per module
- **Integration Tests**: >70% coverage of integration points
- **Feature Tests**: 100% coverage of new innovations
- **Regression Tests**: 100% coverage of existing features

---

## Troubleshooting

### Tests Skip with ImportError

**Cause**: Feature not implemented yet (expected during TDD)

**Solution**: Implement the feature, tests will automatically start running

### Database Connection Errors

**Cause**: PostgreSQL not running or wrong connection string

**Solution**:
```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string
echo $TEST_DATABASE_URL

# Use fake database for development
export USE_FAKE_DB=true  # (when implemented)
```

### GPU/NPU Tests Failing

**Cause**: Hardware not available

**Solution**: Tests marked with `@pytest.mark.requires_gpu` will skip if no GPU
```bash
# Skip GPU tests explicitly
pytest tests/integration/ -v -m "not requires_gpu"
```

---

## Contributing

When adding new features:

1. **Write tests FIRST** (TDD)
2. **Mark with appropriate markers**
3. **Add to this README**
4. **Verify tests fail** (red phase)
5. **Implement feature**
6. **Verify tests pass** (green phase)
7. **Refactor if needed** (refactor phase)

---

## Resources

- **Project Plan**: `UPGRADE_PLAN.md`
- **Pytest Docs**: https://docs.pytest.org/
- **Fixtures**: `conftest.py`
- **Coverage Report**: `htmlcov/index.html` (after running with --cov)

---

**Last Updated**: 2025-10-20
**Phase**: 0 - TDD Test Infrastructure Complete ✅
**Next**: Phase 1 - Implement Chat History (make tests green!)
