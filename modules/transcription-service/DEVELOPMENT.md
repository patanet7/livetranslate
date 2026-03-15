# Whisper Service - Development Guide

> **For ML Engineers and Contributors**
> This guide helps you understand the codebase, configure services, and develop new features efficiently.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration System](#configuration-system)
4. [Key Components](#key-components)
5. [Development Workflow](#development-workflow)
6. [Performance Characteristics](#performance-characteristics)
7. [Parameter Tuning Guide](#parameter-tuning-guide)
8. [Testing Strategy](#testing-strategy)
9. [Debugging Tips](#debugging-tips)
10. [Common Pitfalls](#common-pitfalls)

---

## Quick Start

### Setup Development Environment

```bash
# 1. Install dependencies
cd modules/whisper-service
pip install -r requirements.txt

# 2. Copy environment template
cp .env.example .env

# 3. Edit .env with your settings
# Set WHISPER_MODEL_PATH to your model location

# 4. Validate configuration
python scripts/validate_config.py

# 5. Run benchmarks (optional)
python scripts/benchmark.py --component all
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suite
python -m pytest tests/milestone2/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Session-Restart Transcriber                │
│                  (SessionRestartTranscriber)                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  VAD Filter  │   │  LID Probe   │   │   Whisper    │
│   (Silero)   │   │  (Whisper)   │   │ SimulStream  │
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
                            ▼
                   ┌──────────────┐
                   │  Output JSON │
                   └──────────────┘
```

### Processing Pipeline

```python
Audio Chunk (PCM 16kHz)
    │
    ├──> VAD Detection (Silero)
    │    ├─> SPEECH_START: Begin buffering
    │    ├─> SPEECH_END: Process buffer
    │    └─> NO_CHANGE: Continue current state
    │
    ├──> [If speech] Buffer Audio
    │    └─> Accumulate speech-only audio
    │
    ├──> [Every 100ms] LID Detection (Whisper-native)
    │    ├─> Extract mel spectrogram
    │    ├─> Run encoder (zero-cost probe)
    │    ├─> Apply Viterbi smoothing
    │    └─> Check sustained detection
    │
    ├──> [At VAD END] Process Buffer
    │    ├─> Send to Whisper (SimulStreaming)
    │    ├─> Decode transcription
    │    └─> Create segment
    │
    └──> [If language switch] Create New Session
         ├─> Finish current session
         ├─> Create new Whisper instance
         └─> Set new language SOT token
```

---

## Configuration System

### Overview

All magic numbers and tunable parameters are centralized in `src/service_config.py`.

### Configuration Classes

```python
from service_config import WhisperConfig, VADConfig, LIDConfig, SessionConfig

# Load from environment
config = SessionConfig.from_env(model_path="/path/to/model.pt")

# Access sub-configs
vad_threshold = config.vad.threshold
lid_margin = config.lid.confidence_margin
whisper_decoder = config.whisper.decoder_type

# Configure logging
config.configure_logging()
```

### Environment Variables

See `.env.example` for all available settings.

**Critical Settings:**

```bash
# Model configuration
WHISPER_MODEL_PATH=/path/to/model.pt
WHISPER_DECODER_TYPE=greedy
WHISPER_LANGUAGES=en,zh

# VAD tuning
VAD_THRESHOLD=0.5                    # Speech detection sensitivity
VAD_MIN_SILENCE_MS=500               # Pause duration before ending speech

# LID tuning
LID_CONFIDENCE_MARGIN=0.2            # Language switch threshold
LID_MIN_DWELL_MS=250                 # Minimum dwell before switching

# Logging
LOG_LEVEL=INFO                       # DEBUG, INFO, WARNING, ERROR
ENABLE_PERF_LOGGING=true             # Performance metrics
```

### Validation

```bash
# Validate configuration before running
python scripts/validate_config.py

# Validate with custom env file
python scripts/validate_config.py --env-file .env.production
```

---

## Key Components

### 1. VAD Detector (`src/vad_detector.py`)

**Purpose:** Filter silence before Whisper to prevent hallucinations.

**Key Classes:**
- `SileroVAD`: Main VAD wrapper
- `FixedVADIterator`: Handles arbitrary chunk sizes

**Configuration:**
```python
from service_config import VADConfig

config = VADConfig.from_env()
vad = SileroVAD(
    threshold=config.threshold,
    sampling_rate=config.sampling_rate,
    min_silence_duration_ms=config.min_silence_duration_ms
)
```

**Usage:**
```python
result = vad.check_speech(audio_chunk)

if result is None:
    # No change
    pass
elif 'start' in result:
    # Speech started at result['start'] seconds
    begin_buffering()
elif 'end' in result:
    # Speech ended at result['end'] seconds
    process_buffer()
```

**Performance:**
- Target: < 50ms per chunk
- Real-time factor: > 10x

### 2. Language ID Detector (`src/language_id/`)

**Purpose:** Detect language switches at frame level (10Hz).

**Components:**
- `FrameLevelLID`: Whisper-native zero-cost probe
- `LIDSmoother`: Viterbi smoothing for stability
- `SustainedLanguageDetector`: Hysteresis logic

**Configuration:**
```python
from service_config import LIDConfig

config = LIDConfig.from_env()

detector = SustainedLanguageDetector(
    confidence_margin=config.confidence_margin,
    min_dwell_frames=config.min_dwell_frames,
    min_dwell_ms=config.min_dwell_ms
)
```

**Usage:**
```python
# Update with LID probabilities
switch_event = detector.update(
    lid_probs={'en': 0.8, 'zh': 0.2},
    timestamp=1.5
)

if switch_event:
    # Language switch detected!
    print(f"{switch_event.from_language} → {switch_event.to_language}")
    print(f"Margin: {switch_event.confidence_margin:.3f}")
```

### 3. Session Manager (`src/session_restart/session_manager.py`)

**Purpose:** Orchestrate VAD, LID, and Whisper for code-switching transcription.

**Key Method: `process(audio_chunk)`**

**Simplified Logic (with helpers):**
```python
from vad_helpers import get_vad_action_plan, VADStatus

# Get VAD result
vad_result = self.vad.check_speech(audio_chunk)

# Determine actions
should_buffer, should_process, new_status = get_vad_action_plan(
    vad_result,
    self.vad_status
)

# Buffer speech audio
if should_buffer:
    self.vad_audio_buffer = np.concatenate([
        self.vad_audio_buffer,
        audio_chunk
    ])

# Process at speech boundaries
if should_process:
    # Send buffer to Whisper
    self.current_session.processor.insert_audio(buffer)
    token_ids, metadata = self.current_session.processor.infer(is_last=True)

    # Decode transcription
    text = self.tokenizer.decode(token_ids)

self.vad_status = new_status
```

---

## Development Workflow

### 1. Adding New Configuration Parameters

```python
# 1. Add to config.py
@dataclass
class VADConfig:
    new_parameter: float = 1.0  # Add with default

    def __post_init__(self):
        # Add validation
        if self.new_parameter < 0:
            raise ValueError("new_parameter must be >= 0")

# 2. Add environment variable support
@classmethod
def from_env(cls):
    return cls(
        new_parameter=float(os.getenv('VAD_NEW_PARAM', '1.0'))
    )

# 3. Update .env.example
# VAD_NEW_PARAM=1.0

# 4. Validate
python scripts/validate_config.py
```

### 2. Adding Type Hints

```python
# Use TypedDict for return types
from type_definitions import ProcessResult

def process(self, audio_chunk: np.ndarray) -> ProcessResult:
    return {
        'text': 'transcription',
        'language': 'en',
        'is_final': True,
        'segments': [],
        'switch_detected': False,
        'current_language': 'en',
        'candidate_language': None,
        'chunk_id': 0,
        'chunks_since_output': 0,
        'silence_detected': False,
        'statistics': {}
    }
```

### 3. Adding Logging

```python
from logging_utils import get_component_logger, log_audio_stats

logger = get_component_logger('my_component')

# Log audio statistics at DEBUG level
log_audio_stats(audio_chunk, logger, level=logging.DEBUG)

# Log events at INFO level
logger.info(f"Processing chunk {chunk_id}")

# Use performance logger for timing
from logging_utils import PerformanceLogger

perf = PerformanceLogger('my_component')

with perf.measure('operation_name'):
    result = expensive_operation()

perf.log_summary()  # Log timing statistics
```

### 4. Adding Assertions

```python
from vad_helpers import assert_valid_audio_chunk, assert_valid_vad_state

# Validate inputs
assert_valid_audio_chunk(audio_chunk)
assert_valid_vad_state(self.vad_status, len(self.buffer))

# Add custom assertions
assert self.current_session is not None, "Session must be initialized"
assert len(token_ids) > 0, "Whisper must produce tokens"
```

---

## Performance Characteristics

### Latency Targets

| Component | Target | Typical | Notes |
|-----------|--------|---------|-------|
| VAD Detection | < 50ms | ~15ms | Per 500ms chunk |
| LID Detection | < 20ms | ~8ms | Per 100ms frame |
| Whisper Decode | < 200ms | ~100ms | Per segment |
| End-to-End | < 500ms | ~200ms | From audio to text |

### Memory Usage

- **VAD Buffer**: ~1-5 seconds of audio (~32-160KB)
- **Whisper KV Cache**: ~50-200MB (model dependent)
- **Session State**: < 10MB

### Throughput

- **Real-time Factor**: 10-20x (processes 10-20 seconds of audio per second)
- **Concurrent Sessions**: Depends on hardware (typically 4-8 on GPU)

### Benchmarking

```bash
# Run performance benchmarks
python scripts/benchmark.py --component all

# Benchmark with real audio
python scripts/benchmark.py --audio test_audio.wav

# Custom iterations
python scripts/benchmark.py --component vad --iterations 1000
```

---

## Parameter Tuning Guide

### VAD Tuning

**`VAD_THRESHOLD` (0.0-1.0, default: 0.5)**

```bash
# More aggressive (catches more speech, more false positives)
VAD_THRESHOLD=0.3

# More conservative (misses quiet speech, fewer false positives)
VAD_THRESHOLD=0.7
```

**When to adjust:**
- Noisy environment → increase threshold
- Quiet speakers → decrease threshold
- Hallucinations → increase threshold

**`VAD_MIN_SILENCE_MS` (default: 500)**

```bash
# Shorter pauses trigger speech end (more segments)
VAD_MIN_SILENCE_MS=300

# Longer pauses required (fewer, longer segments)
VAD_MIN_SILENCE_MS=800
```

**When to adjust:**
- Fast speakers → decrease
- Natural pauses → increase
- Real-time requirements → decrease

### LID Tuning

**`LID_CONFIDENCE_MARGIN` (0.1-0.5, default: 0.2)**

```bash
# More sensitive (catches subtle switches, more false switches)
LID_CONFIDENCE_MARGIN=0.15

# More conservative (only clear switches, fewer false switches)
LID_CONFIDENCE_MARGIN=0.3
```

**When to adjust:**
- Frequent code-switching → decrease
- Language flapping issues → increase
- Similar languages (en/es) → increase

**`LID_MIN_DWELL_MS` (default: 250)**

```bash
# Faster switching (more responsive)
LID_MIN_DWELL_MS=150

# Slower switching (more stable)
LID_MIN_DWELL_MS=400
```

**When to adjust:**
- Rapid code-switching → decrease
- Stability issues → increase
- Long monolingual segments → decrease

### Whisper Tuning

**`WHISPER_CHUNK_SIZE` (0.5-5.0, default: 1.2)**

```bash
# Lower latency, less context
WHISPER_CHUNK_SIZE=0.8

# Higher quality, more latency
WHISPER_CHUNK_SIZE=2.0
```

**When to adjust:**
- Real-time requirements → decrease
- Accuracy critical → increase
- Network streaming → decrease

---

## Testing Strategy

### Unit Tests

```bash
# Test individual components
pytest tests/unit/test_vad_detector.py
pytest tests/unit/test_sustained_detector.py
```

### Integration Tests

```bash
# Test complete pipeline
pytest tests/milestone2/test_real_code_switching.py
```

### Performance Tests

```bash
# Benchmark key components
python scripts/benchmark.py --component all
```

### Smoke Tests

```bash
# Quick validation
pytest tests/smoke/ -v
```

---

## Debugging Tips

### Enable Debug Logging

```bash
# Set in .env
LOG_LEVEL=DEBUG
ENABLE_DEBUG_AUDIO=true
ENABLE_PERF_LOGGING=true
```

### Check Audio Quality

```python
from logging_utils import log_audio_stats

log_audio_stats(audio_chunk, logger, level=logging.INFO)
```

**Expected values:**
- RMS: 0.01 - 0.3 (typical speech)
- Max: 0.1 - 1.0 (normalized)

**Problem indicators:**
- RMS < 0.001: Audio too quiet or silence
- Max > 1.0: Audio clipping
- RMS = 0: Empty/zero audio

### Trace VAD Events

```python
from logging_utils import log_vad_event

log_vad_event(vad_result, logger, level=logging.INFO)
```

### Monitor Language Switches

```python
from logging_utils import log_language_switch

log_language_switch(
    from_lang, to_lang, margin, frames, duration_ms, logger
)
```

### Performance Profiling

```python
from logging_utils import PerformanceLogger

perf = PerformanceLogger('component_name')

with perf.measure('operation'):
    # ... code ...

perf.log_summary()  # Logs avg/min/max timings
```

---

## Common Pitfalls

### 1. Filtering by `is_final`

**❌ Wrong:**
```python
# DON'T DO THIS - loses most transcription!
final_segments = [seg for seg in segments if seg['is_final']]
```

**✅ Correct:**
```python
# Collect ALL segments
all_segments = [seg for seg in segments if seg.get('text')]
```

**Why:** `is_final` marks punctuation boundaries, not completion status. See `CLAUDE.md` for details.

### 2. Buffering Silence

**❌ Wrong:**
```python
# Buffering all audio causes hallucinations
buffer.append(audio_chunk)
```

**✅ Correct:**
```python
# Only buffer speech audio
if should_buffer_audio(vad_event, vad_status):
    buffer.append(audio_chunk)
```

### 3. Magic Numbers

**❌ Wrong:**
```python
if chunks_without_output > 10:  # What is 10?
    declare_silence()
```

**✅ Correct:**
```python
from service_config import VADConfig

config = VADConfig.from_env()
if chunks_without_output > config.silence_threshold_chunks:
    declare_silence()
```

### 4. High-Frequency INFO Logs

**❌ Wrong:**
```python
logger.info(f"Processing chunk {i}")  # Logs every 500ms!
```

**✅ Correct:**
```python
logger.debug(f"Processing chunk {i}")  # Use DEBUG for high-frequency
```

### 5. Missing Type Hints

**❌ Wrong:**
```python
def process(self, audio):
    return result
```

**✅ Correct:**
```python
from type_definitions import ProcessResult

def process(self, audio: np.ndarray) -> ProcessResult:
    return {...}
```

---

## Additional Resources

- **Architecture**: See `WHISPER_LID_ARCHITECTURE.md`
- **Feedback**: See `FEEDBACK.md` for design decisions
- **Status**: See `STATUS.md` for current progress
- **API**: See `README.md` for API documentation

---

## Questions?

For ML engineering questions:
- Review `FEEDBACK.md` for design rationale
- Check `STATUS.md` for implementation status
- Run benchmarks to understand performance
- Enable debug logging to trace execution

For configuration issues:
- Run `python scripts/validate_config.py`
- Check `.env.example` for all options
- Review parameter tuning guide above
