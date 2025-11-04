# Migration Guide - DX Optimization Updates

> **For existing code using whisper-service**
> This guide helps you migrate to the new configuration and logging system.

## Overview of Changes

The DX optimization introduces:

1. **Centralized Configuration** (`src/service_config.py`) - All magic numbers extracted
2. **Type Hints** (`src/type_definitions.py`) - Better IDE support and type safety
3. **Structured Logging** (`src/logging_utils.py`) - Performance-focused logging
4. **Helper Functions** (`src/vad_helpers.py`) - Simplified complex logic
5. **Development Tools** (`scripts/`) - Validation and benchmarking

## Breaking Changes

### None!

All changes are **backward compatible**. Existing code will continue to work without modifications.

## Recommended Migrations

### 1. Update Configuration Usage

#### Before (still works):
```python
transcriber = SessionRestartTranscriber(
    model_path="/path/to/model.pt",
    target_languages=['en', 'zh'],
    online_chunk_size=1.2,
    vad_threshold=0.5,
    sampling_rate=16000,
    lid_hop_ms=100,
    confidence_margin=0.2,
    min_dwell_frames=6,
    min_dwell_ms=250.0
)
```

#### After (recommended):
```python
from service_config import SessionConfig

# Load from environment (preferred)
config = SessionConfig.from_env(model_path="/path/to/model.pt")
config.configure_logging()

transcriber = SessionRestartTranscriber(
    model_path=config.whisper.model_path,
    models_dir=config.whisper.models_dir,
    target_languages=config.whisper.target_languages,
    online_chunk_size=config.whisper.online_chunk_size,
    vad_threshold=config.vad.threshold,
    sampling_rate=config.whisper.sampling_rate,
    lid_hop_ms=config.lid.lid_hop_ms,
    confidence_margin=config.lid.confidence_margin,
    min_dwell_frames=config.lid.min_dwell_frames,
    min_dwell_ms=config.lid.min_dwell_ms
)
```

**Benefits:**
- Configuration validated at startup
- Easy to override via environment variables
- Single source of truth
- Better error messages

### 2. Add Type Hints

#### Before:
```python
def process_audio(audio_chunk):
    result = transcriber.process(audio_chunk)
    return result
```

#### After:
```python
from type_definitions import ProcessResult
import numpy as np

def process_audio(audio_chunk: np.ndarray) -> ProcessResult:
    result = transcriber.process(audio_chunk)
    return result
```

**Benefits:**
- IDE autocomplete works better
- Type errors caught early
- Better documentation

### 3. Improve Logging

#### Before:
```python
import logging
logger = logging.getLogger(__name__)

# High-frequency logging at INFO level (clutters logs)
logger.info(f"Processing chunk {i}")
logger.info(f"Audio RMS: {rms}, Max: {max_amp}")
```

#### After:
```python
from logging_utils import get_component_logger, log_audio_stats, PerformanceLogger

logger = get_component_logger('my_component')
perf = PerformanceLogger('my_component')

# High-frequency at DEBUG, events at INFO
logger.debug(f"Processing chunk {i}")
log_audio_stats(audio_chunk, logger, level=logging.DEBUG)

# Measure performance
with perf.measure('process_chunk'):
    result = process_chunk(audio)

# Log summary periodically
perf.log_summary()
```

**Benefits:**
- Production logs stay clean (INFO level)
- Debug mode shows details (DEBUG level)
- Performance metrics tracked automatically

### 4. Simplify VAD Logic

#### Before:
```python
vad_result = vad.check_speech(audio_chunk)

should_process = False
should_buffer = False

if vad_result is not None:
    has_end = 'end' in vad_result
    has_start = 'start' in vad_result

    if has_start:
        self.vad_status = 'voice'
        if not has_end:
            should_process = False
        should_buffer = True

    if has_end:
        should_process = True
        if not has_start:
            self.vad_status = 'nonvoice'
            should_buffer = False
else:
    if self.vad_status == 'voice':
        should_process = False
        should_buffer = True
```

#### After:
```python
from vad_helpers import get_vad_action_plan, VADStatus

vad_result = vad.check_speech(audio_chunk)

should_buffer, should_process, self.vad_status = get_vad_action_plan(
    vad_result,
    self.vad_status
)
```

**Benefits:**
- Much clearer intent
- Easier to test
- Fewer bugs
- Better documented

### 5. Add Assertions

#### Before:
```python
def process(self, audio_chunk):
    # Process without validation
    ...
```

#### After:
```python
from vad_helpers import assert_valid_audio_chunk

def process(self, audio_chunk: np.ndarray):
    # Validate inputs
    assert_valid_audio_chunk(audio_chunk)

    # Add invariant checks
    assert self.current_session is not None, "Session must be initialized"
    ...
```

**Benefits:**
- Catch bugs early
- Better error messages
- Documented assumptions

## Step-by-Step Migration

### Step 1: Setup Environment

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Set your configuration
nano .env  # or your favorite editor

# 3. Validate
python scripts/validate_config.py
```

### Step 2: Update Imports

Add to your code:

```python
# New imports
from service_config import SessionConfig, VADConfig, LIDConfig, WhisperConfig
from type_definitions import ProcessResult, SessionSegment, VADResult
from logging_utils import get_component_logger, PerformanceLogger
from vad_helpers import get_vad_action_plan, VADStatus
```

### Step 3: Update Initialization

```python
# Load configuration
config = SessionConfig.from_env(model_path="/path/to/model.pt")
config.configure_logging()

# Use configuration
transcriber = SessionRestartTranscriber(
    model_path=config.whisper.model_path,
    # ... use config.* for all parameters
)
```

### Step 4: Update Logging

```python
# Replace direct logger creation
logger = get_component_logger('my_component')

# Add performance tracking
perf = PerformanceLogger('my_component')

# Use structured logging helpers
from logging_utils import log_audio_stats, log_vad_event, log_language_switch
```

### Step 5: Simplify Complex Logic

```python
# Replace complex VAD handling
should_buffer, should_process, new_status = get_vad_action_plan(
    vad_result, current_status
)
```

### Step 6: Add Type Hints

```python
# Add return type hints
def process(self, audio: np.ndarray) -> ProcessResult:
    ...
```

### Step 7: Test

```bash
# Run tests to verify
pytest tests/ -v

# Run benchmarks to check performance
python scripts/benchmark.py --component all
```

## Migration Checklist

- [ ] Copy `.env.example` to `.env`
- [ ] Configure environment variables
- [ ] Validate configuration: `python scripts/validate_config.py`
- [ ] Update imports to include new modules
- [ ] Replace hardcoded values with `config.*`
- [ ] Add type hints to functions
- [ ] Update logging to use `logging_utils`
- [ ] Simplify VAD logic with `vad_helpers`
- [ ] Add input validation with assertions
- [ ] Run tests: `pytest tests/ -v`
- [ ] Run benchmarks: `python scripts/benchmark.py`
- [ ] Update documentation

## FAQ

### Q: Do I need to migrate immediately?

**A:** No. All changes are backward compatible. Migrate at your own pace.

### Q: Will my existing code break?

**A:** No. Existing code will continue to work without changes.

### Q: What if I don't use environment variables?

**A:** You can still pass parameters directly. The config system is optional but recommended.

### Q: How do I override specific config values?

**A:** Either:
1. Set environment variables
2. Modify config object after loading:
   ```python
   config = SessionConfig.from_env(model_path)
   config.vad.threshold = 0.7  # Override
   ```

### Q: Can I use parts of the new system?

**A:** Yes! You can adopt features incrementally:
- Use `config.py` but keep old logging
- Use `logging_utils` but keep hardcoded values
- Use `vad_helpers` for new code only

### Q: How do I debug configuration issues?

**A:** Run validation script:
```bash
python scripts/validate_config.py

# With custom env file
python scripts/validate_config.py --env-file .env.debug
```

### Q: What if validation fails?

**A:** The validator will tell you exactly what's wrong:
```
❌ VAD Config Invalid: VAD threshold must be in [0.0, 1.0], got 1.5
```

Fix the issue in `.env` and re-validate.

## Getting Help

1. **Configuration errors**: Run `python scripts/validate_config.py`
2. **Performance issues**: Run `python scripts/benchmark.py`
3. **Type errors**: Check `src/type_definitions.py` for available types
4. **Logging questions**: See `DEVELOPMENT.md` logging section
5. **Complex logic**: See `src/vad_helpers.py` examples

## Summary

The DX optimization makes the codebase:
- ✅ **More maintainable** - configuration centralized
- ✅ **Easier to understand** - complex logic simplified
- ✅ **Better documented** - comprehensive guides
- ✅ **Type safe** - better IDE support
- ✅ **Production ready** - proper log levels

Migrate incrementally at your own pace. All features are optional but recommended.
