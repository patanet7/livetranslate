# Whisper Service Comprehensive Test Suite

This test suite provides comprehensive coverage of the whisper-service code-switching system, based on ML engineer review recommendations.

## Test Structure

```
tests/
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_kv_cache_masking.py  # Priority 1: KV cache edge cases
│   ├── test_sustained_detector.py # Priority 2: Hysteresis & false positives
│   └── test_vad.py               # Priority 5: VAD robustness
├── stress/                        # Stress tests (long-running)
│   └── test_long_session.py      # Priority 3: 60-min sessions, memory leaks
├── accuracy/                      # Accuracy regression tests
│   ├── test_code_switching_accuracy.py  # Priority 4: 70%+ accuracy baseline
│   └── baselines/                # Accuracy baselines (JSON)
├── property/                      # Property-based tests
│   └── test_invariants.py        # Priority 5: System invariants
├── benchmarks/                    # Performance benchmarks
│   ├── test_latency.py           # Priority 6: Latency tracking
│   └── results/                  # Benchmark results (JSON)
└── milestone2/                    # Integration tests
    └── test_real_code_switching.py  # Full system integration
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Fast, isolated tests for individual components.**

#### `test_kv_cache_masking.py` - KV Cache Edge Cases
- **Purpose**: Test critical KV cache mask bug fix in `model.py`
- **Priority**: 1 (Critical)
- **Tests**:
  - Empty query tensor (n_ctx=0) handling
  - Mask slicing at context boundaries (440, 445, 448, 450 tokens)
  - Offset calculation correctness
  - Dynamic mask creation when clipping occurs
  - Rapid KV cache accumulation
- **Run**: `pytest tests/unit/test_kv_cache_masking.py -v`

#### `test_sustained_detector.py` - Hysteresis Logic
- **Purpose**: Test sustained language detection prevents false positives
- **Priority**: 2 (Critical)
- **Tests**:
  - Hysteresis prevents EN→ZH→EN→ZH flapping
  - Sustained switch after 6 frames at 250ms
  - False positive prevention
  - Confidence margin thresholds
  - Adaptive detection parameters
- **Run**: `pytest tests/unit/test_sustained_detector.py -v`

#### `test_vad.py` - VAD Robustness
- **Purpose**: Test VAD handles arbitrary audio without crashes
- **Priority**: 5 (Property-based)
- **Tests**:
  - Speech/silence detection accuracy
  - START/END event handling
  - Arbitrary chunk sizes (FixedVADIterator)
  - Robustness to extreme audio conditions
  - Property: Never crashes on arbitrary audio
- **Run**: `pytest tests/unit/test_vad.py -v`

### 2. Stress Tests (`tests/stress/`)

**Long-running tests for production readiness.**

#### `test_long_session.py` - 60-Minute Stress Test
- **Purpose**: Test memory leaks and stability over extended periods
- **Priority**: 3 (Critical for production)
- **Tests**:
  - 60-minute continuous streaming
  - Memory stays under 500MB
  - KV cache doesn't overflow
  - Session history bounded growth
  - No performance degradation over time
- **Run**: `pytest tests/stress/test_long_session.py -v -m stress`
- **Duration**: ~60 minutes (can reduce for CI)
- **Requirements**: `psutil`, `scipy`

### 3. Accuracy Tests (`tests/accuracy/`)

**Accuracy regression tracking with baseline storage.**

#### `test_code_switching_accuracy.py` - Accuracy Baseline
- **Purpose**: Maintain 70%+ accuracy on code-switching benchmark
- **Priority**: 4 (Quality assurance)
- **Tests**:
  - English accuracy (JFK): 75%+ (25% WER)
  - Chinese accuracy: 70%+ (30% CER)
  - Code-switching accuracy: 70-85% overall
  - Baseline tracking and regression detection
- **Run**: `pytest tests/accuracy/test_code_switching_accuracy.py -v -m accuracy`
- **Baseline Storage**: `tests/accuracy/baselines/code_switching_baselines.json`
- **Note**: First run creates baseline, subsequent runs compare

### 4. Property-Based Tests (`tests/property/`)

**Property-based tests using hypothesis framework.**

#### `test_invariants.py` - System Invariants
- **Purpose**: Test critical invariants hold under arbitrary input
- **Priority**: 5 (Robustness)
- **Tests**:
  - LID probabilities sum to 1.0 (±epsilon)
  - VAD never crashes on arbitrary audio
  - Session state machine invariants
  - Numerical stability invariants
  - Boundary conditions
- **Run**: `pytest tests/property/test_invariants.py -v -m property`
- **Requirements**: `hypothesis` (install: `pip install hypothesis`)
- **Note**: Generates hundreds of test cases automatically

### 5. Performance Benchmarks (`tests/benchmarks/`)

**Performance tracking with regression detection.**

#### `test_latency.py` - Latency Benchmarks
- **Purpose**: Track latency and detect performance regressions
- **Priority**: 6 (Performance monitoring)
- **Tests**:
  - End-to-end latency: P95 < 100ms for 500ms chunks
  - LID probe latency: < 1ms on GPU
  - VAD latency: < 1ms
  - Throughput: >= 1.0x real-time
  - Track p50, p95, p99 percentiles
- **Run**: `pytest tests/benchmarks/test_latency.py -v -m benchmark`
- **Benchmark Storage**: `tests/benchmarks/results/latency_benchmarks.json`

### 6. Integration Tests (`tests/milestone2/`)

**Full system integration tests with real audio.**

#### `test_real_code_switching.py` - System Integration
- **Purpose**: Test complete system with real Whisper models
- **Tests**:
  - Mixed English/Chinese transcription
  - Separate language files with session restart
  - English-only (no false switches)
  - Full accuracy validation
- **Run**: `pytest tests/milestone2/test_real_code_switching.py -v`

## Running Tests

### Quick Start

```bash
# Run all unit tests (fast)
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_kv_cache_masking.py -v

# Run with coverage
pytest tests/unit/ --cov=src --cov-report=html

# Run all tests with detailed logging
pytest tests/ -v --log-cli-level=INFO
```

### By Priority

```bash
# Priority 1: KV Cache (critical bug fix validation)
pytest tests/unit/test_kv_cache_masking.py -v

# Priority 2: Sustained Detector (hysteresis validation)
pytest tests/unit/test_sustained_detector.py -v

# Priority 3: Stress Tests (production readiness)
pytest tests/stress/test_long_session.py -v -m stress

# Priority 4: Accuracy (quality assurance)
pytest tests/accuracy/ -v -m accuracy

# Priority 5: Property Tests (robustness)
pytest tests/property/ -v -m property

# Priority 6: Performance (monitoring)
pytest tests/benchmarks/ -v -m benchmark
```

### By Test Type

```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Stress tests (slow - 60 minutes)
pytest tests/stress/ -v -m stress

# Accuracy tests (requires models)
pytest tests/accuracy/ -v -m accuracy

# Property-based tests (requires hypothesis)
pytest tests/property/ -v -m property

# Performance benchmarks
pytest tests/benchmarks/ -v -m benchmark

# Integration tests (requires models + audio fixtures)
pytest tests/milestone2/ -v
```

### Continuous Integration

```bash
# Fast CI suite (< 5 minutes)
pytest tests/unit/ tests/property/ -v -m "not slow"

# Full CI suite (< 30 minutes)
pytest tests/unit/ tests/accuracy/ tests/property/ tests/benchmarks/ -v

# Nightly stress tests (60+ minutes)
pytest tests/stress/ -v -m stress
```

## Test Markers

Tests are marked with pytest markers for selective execution:

```python
@pytest.mark.slow           # Long-running tests
@pytest.mark.stress         # Stress tests (60+ min)
@pytest.mark.accuracy       # Accuracy regression tests
@pytest.mark.property       # Property-based tests
@pytest.mark.benchmark      # Performance benchmarks
```

Use markers to filter:
```bash
# Run only fast tests
pytest -v -m "not slow and not stress"

# Run only stress tests
pytest -v -m stress

# Run everything except stress
pytest -v -m "not stress"
```

## Requirements

### Core Requirements
```bash
pytest>=7.0.0
numpy>=1.20.0
torch>=2.0.0
soundfile>=0.12.0
librosa>=0.10.0
```

### Additional Requirements
```bash
# For stress tests
psutil>=5.9.0
scipy>=1.9.0

# For property-based tests
hypothesis>=6.0.0

# For coverage
pytest-cov>=4.0.0
```

Install all:
```bash
cd modules/whisper-service
pip install -r requirements.txt -r requirements-test.txt
```

## Test Data

### Audio Fixtures

Located in `tests/fixtures/audio/`:
- `jfk.wav` - English speech (JFK)
- `OSR_cn_000_0072_8k.wav` - Chinese speech
- `test_clean_mixed_en_zh.wav` - Mixed English/Chinese
- `silence.wav` - Pure silence
- `white_noise.wav` - White noise

### Ground Truth

Ground truth transcriptions are defined in test files:
- English: EXPECTED_JFK_TEXT in test_real_code_switching.py
- Chinese: EXPECTED_MIXED_SEGMENTS in test_real_code_switching.py
- Mixed: GROUND_TRUTH in test_code_switching_accuracy.py

## Baseline Management

### Accuracy Baselines

Stored in `tests/accuracy/baselines/code_switching_baselines.json`:
```json
{
  "jfk_english": {
    "best": {
      "timestamp": "2025-11-03T...",
      "metrics": {
        "accuracy": 95.5,
        "wer": 4.5,
        "cer": 2.1
      }
    },
    "history": [...]
  }
}
```

### Performance Baselines

Stored in `tests/benchmarks/results/latency_benchmarks.json`:
```json
{
  "end_to_end_500ms": {
    "best": {
      "timestamp": "2025-11-03T...",
      "stats": {
        "p50_ms": 45.2,
        "p95_ms": 87.3,
        "p99_ms": 120.5
      }
    }
  }
}
```

### Baseline Commands

```bash
# View current baselines
cat tests/accuracy/baselines/code_switching_baselines.json
cat tests/benchmarks/results/latency_benchmarks.json

# Reset baselines (re-run tests)
rm tests/accuracy/baselines/*.json
rm tests/benchmarks/results/*.json
pytest tests/accuracy/ tests/benchmarks/ -v

# Generate accuracy report
pytest tests/accuracy/test_code_switching_accuracy.py::TestAccuracyReportGeneration -v
```

## Coverage Goals

### Target Coverage

- **Unit Tests**: 90%+ coverage of core components
- **Integration Tests**: 80%+ coverage of session_manager.py
- **Stress Tests**: Memory leak detection
- **Accuracy Tests**: 70-85% transcription accuracy
- **Property Tests**: 0 crashes on arbitrary input

### Generate Coverage Report

```bash
# HTML coverage report
pytest tests/unit/ --cov=src --cov-report=html
open htmlcov/index.html

# Terminal coverage report
pytest tests/unit/ --cov=src --cov-report=term-missing

# XML coverage (for CI)
pytest tests/unit/ --cov=src --cov-report=xml
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: |
          pip install -r requirements-test.txt
          pytest tests/unit/ tests/property/ -v --cov=src --cov-report=xml

  accuracy-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run accuracy tests
        run: |
          pytest tests/accuracy/ -v -m accuracy

  nightly-stress:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v3
      - name: Run stress tests
        run: |
          pytest tests/stress/ -v -m stress
```

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Solution: Download model to ~/.whisper/models/large-v3-turbo.pt
   ```

2. **hypothesis not installed**
   ```bash
   pip install hypothesis
   ```

3. **Audio fixtures missing**
   ```
   Check: tests/fixtures/audio/ should contain test audio files
   ```

4. **Memory errors in stress tests**
   ```bash
   # Reduce test duration
   pytest tests/stress/test_long_session.py -v --duration=10  # 10 minutes instead of 60
   ```

5. **Slow tests**
   ```bash
   # Skip slow tests
   pytest -v -m "not slow"
   ```

## Contributing

### Adding New Tests

1. **Unit Test**: Add to appropriate file in `tests/unit/`
2. **Stress Test**: Add to `tests/stress/test_long_session.py`
3. **Accuracy Test**: Add to `tests/accuracy/test_code_switching_accuracy.py`
4. **Property Test**: Add to `tests/property/test_invariants.py`
5. **Benchmark**: Add to `tests/benchmarks/test_latency.py`

### Test Naming Conventions

- Unit tests: `test_<component>_<behavior>`
- Stress tests: `test_<scenario>_<duration>`
- Accuracy tests: `test_<language>_accuracy_baseline`
- Property tests: `test_<property>_invariant`
- Benchmarks: `test_<metric>_<configuration>`

### Test Documentation

Each test should have:
- Clear docstring explaining purpose
- Reference to related code/issues
- Expected behavior
- Pass/fail criteria

Example:
```python
def test_kv_cache_at_boundary_448(self, attention_module, causal_mask):
    """
    Test mask slicing exactly at n_text_ctx=448.

    Scenario: KV cache exactly at context limit
    Reference: model.py lines 167-201
    Expected: No crashes, correct output shape
    """
    # Test implementation
```

## References

- **ML Engineer Review**: See project documentation
- **FEEDBACK.md**: lines 171-184 (code-switching requirements)
- **Session Manager**: `src/session_restart/session_manager.py`
- **KV Cache Fix**: `src/simul_whisper/whisper/model.py` lines 132-208
- **VAD Implementation**: `src/vad_detector.py`
- **Sustained Detector**: `src/language_id/sustained_detector.py`

## Summary

This comprehensive test suite provides:
- ✅ **Unit tests** for all critical components
- ✅ **Stress tests** for production readiness
- ✅ **Accuracy tracking** with regression detection
- ✅ **Property-based tests** for robustness
- ✅ **Performance benchmarks** with baselines
- ✅ **Integration tests** with real audio

**Total Coverage**: 7 test files, 100+ test cases, addressing all 6 ML engineer priorities.
