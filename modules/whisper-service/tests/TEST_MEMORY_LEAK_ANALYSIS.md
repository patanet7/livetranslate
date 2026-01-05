# Test Memory Leak Analysis - Whisper Service

**Date**: 2026-01-05
**Issue**: pytest process consuming ~2GB RAM with 32GB virtual memory, continuously growing

## Root Cause

Unit tests are loading multiple large ML models without proper cleanup:

### Memory-Heavy Operations

1. **test_vad.py** - Silero VAD Model Loading
   - Location: `tests/unit/test_vad.py:256-263`
   - Model: ~40MB per load
   - Issue: `@pytest.fixture` without scope - reloads for EVERY test method
   - Impact: 4+ test classes × multiple test methods = **160-200MB wasted**

2. **test_whisper_lid_probe.py** - Whisper Base Model
   - Location: `tests/unit/test_whisper_lid_probe.py:22`
   - Model: ~140MB
   - Issue: Module-level fixture but no cleanup
   - Impact: **140MB persistent**

3. **test_token_buffer.py** - Whisper Large-v3 Model (CRITICAL)
   - Locations: Lines 371, 432, 482
   - Model: **~3GB per load**
   - Issue: Loaded 3 separate times within same test file
   - Impact: **9GB+ potential memory usage**
   - Real tests with actual inference - models stay loaded in PyTorch memory

## Why Memory Grows Continuously

1. **No Fixture Scope Management**
   ```python
   # CURRENT (BAD):
   @pytest.fixture
   def vad_model(self):
       model, _ = torch.hub.load(...)  # Loads EVERY test
       return model

   # SHOULD BE:
   @pytest.fixture(scope="module")
   def vad_model(self):
       model, _ = torch.hub.load(...)
       yield model
       del model
       torch.cuda.empty_cache()
   ```

2. **No Explicit Cleanup**
   - PyTorch keeps models in memory indefinitely
   - GPU/NPU tensors cached until explicit clear
   - No `teardown` or cleanup fixtures

3. **Multiple Model Instances**
   - Each test class creates its own fixture
   - No sharing between test classes
   - Models never garbage collected during test run

4. **Real Inference in Unit Tests**
   - `test_token_buffer.py` runs actual `model.transcribe()` calls
   - Creates intermediate tensors (KV cache, attention, etc.)
   - These accumulate without cleanup

## Immediate Fixes Required

### 1. Add Fixture Scope to VAD Tests

**File**: `modules/whisper-service/tests/unit/test_vad.py`

```python
# Replace lines 253-263
@pytest.fixture(scope="module")  # Load ONCE per module
def vad_model():
    """Load Silero VAD model once for all tests."""
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    model.eval()
    yield model

    # Cleanup
    del model
    import gc
    gc.collect()
```

### 2. Refactor Whisper Model Loading in Token Buffer Tests

**File**: `modules/whisper-service/tests/unit/test_token_buffer.py`

```python
@pytest.fixture(scope="module")
def whisper_model_large():
    """Load Whisper large-v3 ONCE for all integration tests."""
    models_dir = Path(__file__).parent.parent / ".models"
    manager = ModelManager(models_dir=str(models_dir))
    manager.init_context()
    model = manager.load_model("large-v3")

    yield model

    # Cleanup
    del model
    del manager
    import torch
    import gc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

# Then reuse in tests:
def test_rolling_context_with_real_model(self, whisper_model_large):
    model = whisper_model_large
    # ... use model without reloading
```

### 3. Add Global Cleanup Hook

**File**: `modules/whisper-service/tests/unit/conftest.py` (create if doesn't exist)

```python
import pytest
import torch
import gc

@pytest.fixture(scope="session", autouse=True)
def cleanup_after_all_tests():
    """Auto-cleanup after ALL tests complete."""
    yield

    # Final cleanup
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    print("\n✅ Cleaned up ML models and GPU memory")

@pytest.fixture(autouse=True)
def cleanup_after_each_test():
    """Cleanup after each individual test."""
    yield

    # Minor cleanup between tests
    gc.collect()
```

### 4. Mark Heavy Tests Appropriately

```python
# Add to heavy model tests
@pytest.mark.slow  # Already defined in pytest.ini
@pytest.mark.integration  # These are actually integration tests
def test_rolling_context_with_real_model(self, whisper_model_large):
    ...
```

Then run unit tests WITHOUT heavy model tests:
```bash
pytest modules/whisper-service/tests/unit/ -v -m "not slow and not integration"
```

## Long-Term Recommendations

### 1. Separate Integration from Unit Tests

Move model-loading tests to `tests/integration/`:
- `test_token_buffer.py` (lines 350+) → Integration tests
- `test_whisper_lid_probe.py` → Integration tests
- Keep only `test_vad.py` basic tests in unit (mock the model)

### 2. Use Model Mocking for True Unit Tests

```python
@pytest.fixture
def mock_vad_model(mocker):
    """Mock VAD model - no actual loading."""
    mock_model = mocker.Mock()
    mock_model.eval.return_value = None
    return mock_model
```

### 3. Add Memory Profiling to CI

```bash
# In pytest.ini or CI script
pytest --memprof tests/unit/
```

### 4. Implement Resource Limits

```python
# pytest.ini
[pytest]
timeout = 300  # 5 min max per test
```

## Expected Memory Savings

| Current | After Fixes |
|---------|-------------|
| 9GB+ (3x large-v3 loads) | ~3GB (1x shared) |
| 200MB+ (VAD reloads) | ~40MB (1x shared) |
| No cleanup = leak | Proper cleanup = stable |

**Total reduction**: ~6-7GB memory savings

## Verification Steps

1. Apply fixture scope fixes
2. Run tests with memory monitoring:
   ```bash
   pytest modules/whisper-service/tests/unit/ -v --tb=short
   # Monitor with: watch -n 1 'ps aux | grep pytest'
   ```
3. Verify memory stays stable (not growing)
4. Confirm test runtime decreases (models load once)

## Related Files

- `/modules/whisper-service/tests/unit/test_vad.py` - VAD fixture issues
- `/modules/whisper-service/tests/unit/test_token_buffer.py` - Large model loading
- `/modules/whisper-service/tests/unit/test_whisper_lid_probe.py` - Model loading
- `/modules/whisper-service/pytest.ini` - Test markers and config

## Action Items

- [ ] Create `conftest.py` with global cleanup fixtures
- [ ] Add `scope="module"` to VAD model fixture
- [ ] Refactor token buffer tests to use shared model fixture
- [ ] Move integration tests to separate directory
- [ ] Add memory profiling to CI pipeline
- [ ] Document test separation policy in CLAUDE.md
