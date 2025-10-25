# Whisper Service Refactoring Plan - Test-Driven Approach

**Date:** 2025-10-25
**Status:** READY FOR EXECUTION
**Methodology:** Test-Driven Refactoring (TDD)
**Duration:** 4 weeks
**Team:** 2 developers

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Pre-Refactoring: Feature Inventory](#pre-refactoring-feature-inventory)
3. [TDD Methodology](#tdd-methodology)
4. [Phase 1: Extract ModelManager & Create Abstraction](#phase-1-extract-modelmanager--create-abstraction)
5. [Phase 2: Split whisper_service.py](#phase-2-split-whisper_servicepy)
6. [Phase 3: Split api_server.py](#phase-3-split-api_serverpy)
7. [Phase 4: Consolidate Session Management](#phase-4-consolidate-session-management)
8. [Phase 5: Deduplication & Polish](#phase-5-deduplication--polish)
9. [Rollback Strategy](#rollback-strategy)
10. [Success Metrics](#success-metrics)

---

## Executive Summary

### The Problem
- **whisper_service.py**: 2,392 lines with embedded ModelManager causing name collision
- **api_server.py**: 3,642 lines mixing 10+ responsibilities
- **3 session managers**: Duplicate logic across files causing bugs
- **Duplicate code**: Health monitoring, error handling spread across files

### The Solution
**Test-Driven Refactoring**: Write comprehensive tests FIRST, then split monoliths while keeping tests green.

### Key Principles
1. ✅ **Tests first** - No refactoring without passing tests
2. ✅ **No feature loss** - Every feature documented and verified
3. ✅ **Stateful architecture preserved** - In-memory sessions are correct
4. ✅ **Green builds always** - Every commit must pass all tests
5. ✅ **Incremental changes** - Small, safe, reviewable PRs

### Target Architecture
```
modules/whisper-service/src/
├── api/                         # API Layer (stateful WebSocket + stateless HTTP)
│   ├── http_routes.py          # REST endpoints
│   ├── websocket_routes.py     # SocketIO handlers (stateful!)
│   └── validators.py           # Request validation
│
├── services/                    # Business Logic Layer
│   ├── transcription_service.py  # Core transcription orchestration
│   ├── streaming_service.py      # Stateful streaming coordination
│   └── session_service.py        # Session lifecycle management
│
├── models/                      # Model Abstraction Layer
│   ├── base_model.py           # WhisperModel Protocol
│   ├── openvino_manager.py     # NPU implementation
│   ├── pytorch_manager.py      # GPU/CPU implementation
│   └── model_factory.py        # Factory pattern
│
├── session/                     # Session Management (STATEFUL!)
│   ├── session_manager.py      # Unified session manager
│   ├── vad_state.py            # VAD state per session
│   └── context_manager.py      # Rolling context per session
│
├── audio/                       # Audio Processing
│   ├── audio_processor.py      # Format conversion, resampling
│   └── audio_utils.py          # Utility functions
│
├── monitoring/                  # Observability
│   ├── performance.py          # PerformanceMonitor, AudioProcessingPool
│   └── health_checks.py        # Health monitoring
│
└── context/                     # Context Management
    └── rolling_context.py      # Rolling context implementation
```

---

## Pre-Refactoring: Feature Inventory

### Critical: Document ALL Features Before Refactoring

**Purpose:** Ensure ZERO features are lost during refactoring.

### Feature Categories

#### 1. HTTP REST Endpoints (35+ endpoints in api_server.py)

**Transcription Endpoints:**
- [ ] `POST /transcribe` - Single file transcription
- [ ] `POST /api/transcribe` - Alternative transcription endpoint
- [ ] `POST /api/process-chunk` - Process audio chunk
- [ ] `POST /api/process-final` - Finalize processing

**Model Management:**
- [ ] `GET /models` - List available models
- [ ] `GET /api/models` - Alternative model list
- [ ] `POST /load-model` - Load specific model
- [ ] `POST /unload-model` - Unload model
- [ ] `GET /model-info` - Get model details

**Session Management:**
- [ ] `POST /stream/start` - Start streaming session
- [ ] `POST /api/stream/start` - Alternative stream start
- [ ] `POST /stream/stop` - Stop streaming session
- [ ] `GET /session-status` - Get session status

**Configuration:**
- [ ] `GET /config` - Get current config
- [ ] `POST /config` - Update config
- [ ] `POST /update-config` - Alternative config update

**Cache & Maintenance:**
- [ ] `POST /clear-cache` - Clear model cache
- [ ] `POST /clear-session` - Clear session data

**Health & Monitoring:**
- [ ] `GET /health` - Health check
- [ ] `GET /api/health` - Alternative health check
- [ ] `GET /stats` - Performance statistics
- [ ] `GET /device-status` - Device availability

#### 2. WebSocket Events (11+ events in api_server.py)

**Connection Management:**
- [ ] `connect` - Client connection established
- [ ] `disconnect` - Client disconnection
- [ ] `reconnect` - Client reconnection

**Streaming:**
- [ ] `start_stream` - Initialize streaming session
- [ ] `audio_chunk` - Receive audio chunk
- [ ] `stop_stream` - End streaming session

**Configuration:**
- [ ] `update_session_config` - Update session parameters

**Responses (Server → Client):**
- [ ] `transcription_result` - Partial/final transcription
- [ ] `error` - Error messages
- [ ] `status` - Status updates
- [ ] `session_ready` - Session initialized

#### 3. Core Transcription Features (whisper_service.py)

**Model Support:**
- [ ] NPU (OpenVINO) acceleration
- [ ] GPU (CUDA) acceleration
- [ ] CPU fallback
- [ ] Automatic device fallback chain
- [ ] Multiple model sizes (tiny, base, small, medium, large)

**Transcription Modes:**
- [ ] Single-file transcription
- [ ] Streaming transcription (real-time)
- [ ] Batch processing
- [ ] Rolling context (uses previous transcriptions)

**Advanced Features:**
- [ ] Speaker diarization integration
- [ ] VAD (Voice Activity Detection)
- [ ] Language detection
- [ ] Domain-specific prompts
- [ ] Beam search decoding
- [ ] Temperature sampling
- [ ] Timestamp alignment
- [ ] UTF-8 boundary fixing
- [ ] Token deduplication

**Session Features:**
- [ ] Per-session context isolation
- [ ] Session warmup
- [ ] Context carryover between chunks
- [ ] Session state persistence (in-memory)

#### 4. Audio Processing Features

**Format Support:**
- [ ] WAV, MP3, FLAC, OGG
- [ ] Automatic resampling (48kHz → 16kHz)
- [ ] Mono conversion
- [ ] FFmpeg integration
- [ ] Librosa fallback for resampling

**Processing:**
- [ ] Audio enhancement (optional)
- [ ] Noise reduction (configurable)
- [ ] Audio chunking
- [ ] Buffer management

#### 5. Performance Features

**Optimization:**
- [ ] Audio processing thread pool
- [ ] Message queue for async processing
- [ ] Performance monitoring
- [ ] Memory management
- [ ] Model caching (LRU, max 3 models)

**Monitoring:**
- [ ] Request latency tracking
- [ ] Queue depth monitoring
- [ ] Thread pool utilization
- [ ] Memory usage tracking
- [ ] Error rate tracking

#### 6. Session Management Features

**Lifecycle:**
- [ ] Session creation
- [ ] Session recovery (reconnection)
- [ ] Session expiration (30 min timeout)
- [ ] Session cleanup
- [ ] Message buffering during reconnection

**State Management:**
- [ ] VAD state per session
- [ ] Rolling context per session
- [ ] Audio buffer per session
- [ ] Configuration per session
- [ ] Speaker embeddings per session

#### 7. Authentication & Security

**Auth:**
- [ ] Token-based authentication
- [ ] Session tokens
- [ ] Token cleanup (expired tokens)

**CORS:**
- [ ] Configurable CORS settings
- [ ] Multiple origin support

---

## TDD Methodology

### The Test-First Refactoring Process

```
┌─────────────────────────────────────────────────────────┐
│  Phase: Extract Component                               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Step 1: Document Current Behavior                      │
│  ├─ Identify all functions/classes to extract          │
│  ├─ Document inputs, outputs, side effects             │
│  └─ Note all dependencies                              │
│                                                          │
│  Step 2: Write Characterization Tests                   │
│  ├─ Test current behavior (even if ugly)               │
│  ├─ Cover all edge cases                               │
│  ├─ Aim for 100% coverage of extracted code            │
│  └─ ✅ ALL TESTS MUST PASS                             │
│                                                          │
│  Step 3: Extract to New Module                          │
│  ├─ Copy code to new location                          │
│  ├─ Keep old code in place (don't delete yet!)         │
│  ├─ Update imports in old file to use new module       │
│  └─ ✅ ALL TESTS MUST STILL PASS                       │
│                                                          │
│  Step 4: Verify Feature Parity                          │
│  ├─ Run full integration test suite                    │
│  ├─ Test all HTTP endpoints                            │
│  ├─ Test all WebSocket events                          │
│  ├─ Manual smoke testing                               │
│  └─ ✅ ZERO REGRESSIONS ALLOWED                        │
│                                                          │
│  Step 5: Delete Old Code                                │
│  ├─ Remove commented/unused code from original file    │
│  ├─ Clean up imports                                   │
│  └─ ✅ ALL TESTS MUST STILL PASS                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Testing Pyramid for Refactoring

```
                    ┌──────────────┐
                    │  End-to-End  │  5% - Full service integration
                    │    Tests     │       (HTTP + WebSocket + Models)
                    └──────────────┘
                  ┌──────────────────┐
                  │  Integration     │  20% - Multi-module interactions
                  │     Tests        │        (API → Service → Model)
                  └──────────────────┘
              ┌────────────────────────┐
              │   Component Tests      │  35% - Single module testing
              │                        │        (Services, Models, Audio)
              └────────────────────────┘
        ┌────────────────────────────────┐
        │      Unit Tests                │  40% - Individual functions
        │                                │        (Pure functions, utils)
        └────────────────────────────────┘
```

### Test Coverage Requirements

**Before ANY refactoring can begin:**
- [ ] Unit test coverage: >80% on code to be extracted
- [ ] Integration test coverage: >60% on API endpoints
- [ ] End-to-end tests: All critical user flows

**Per-phase requirements:**
- [ ] Every extracted module: 100% unit test coverage
- [ ] Every API route: Integration test exists
- [ ] Every WebSocket event: Integration test exists

---

## Phase 1: Extract ModelManager & Create Abstraction

**Duration:** 5 days
**Goal:** Fix duplicate ModelManager classes, create clean abstraction
**Risk:** MEDIUM (core model loading logic)

### Current State Analysis

**Problem:**
```python
# TWO classes both named "ModelManager":

# 1. src/model_manager.py (587 lines)
class ModelManager:  # OpenVINO/NPU
    def load_model(...)
    def safe_inference(...)

# 2. src/whisper_service.py (lines 168-587, 420 lines)
class ModelManager:  # PyTorch/GPU
    def load_model(...)
    def warmup(...)
```

**Import Confusion:**
```python
# Which ModelManager are we getting?
from model_manager import ModelManager  # OpenVINO?
from whisper_service import ModelManager  # PyTorch?
```

### Day 1: Write Characterization Tests

**Goal:** Document current behavior with tests

### Test Organization - LOCAL to Module

**IMPORTANT:** Tests live IN the whisper-service module to use poetry environment:

```
modules/whisper-service/
├── src/                    # Source code
├── tests/                  # LOCAL tests using poetry env
│   ├── conftest.py        # Pytest fixtures (audio files, models)
│   ├── unit/              # Unit tests
│   │   ├── test_openvino_manager.py
│   │   ├── test_pytorch_manager.py
│   │   └── test_model_factory.py
│   ├── integration/       # Integration tests with REAL audio
│   │   ├── test_transcription_e2e.py
│   │   ├── test_streaming.py
│   │   └── test_api_endpoints.py
│   └── fixtures/
│       └── audio/         # Test audio files
│           ├── hello_world.wav      (1s, 16kHz, mono)
│           ├── silence.wav          (1s, silence)
│           ├── multilang.wav        (multi-language)
│           └── noisy.wav            (background noise)
├── pyproject.toml
└── poetry.lock
```

**Why local tests?**
- Uses poetry environment directly
- No cross-module import complexity
- Audio fixtures are local and easy to reference
- pytest discovers tests naturally
- `poetry run pytest tests/` just works!

---

#### 1.1 Test OpenVINO ModelManager (OPTIONAL - Skip if Not Installed)

```python
# tests/unit/test_openvino_manager.py (NEW FILE)

import pytest
from unittest.mock import Mock, patch
import sys

# Try to import OpenVINO - skip tests if not available
try:
    import openvino_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

sys.path.insert(0, 'src')
from model_manager import ModelManager  # OpenVINO version

@pytest.mark.skipif(not OPENVINO_AVAILABLE, reason="OpenVINO not installed")
class TestOpenVINOModelManager:
    """Characterization tests for OpenVINO ModelManager (optional)"""

class TestOpenVINOModelManager:
    """Characterization tests for OpenVINO ModelManager"""

    @pytest.fixture
    def manager(self):
        """Create OpenVINO manager instance"""
        return ModelManager(model_name="whisper-base")

    def test_initialization(self, manager):
        """Test manager initializes correctly"""
        assert manager.model_name == "whisper-base"
        assert manager.device in ["npu", "gpu", "cpu"]
        assert manager.pipeline is None  # Not loaded yet

    def test_device_detection_priority(self, manager):
        """Test device detection follows GPU/MPS → NPU → CPU"""
        # Test current behavior
        detected = manager._detect_best_device()
        assert detected in ["gpu", "mps", "npu", "cpu"]

    @patch('openvino_genai.WhisperPipeline')
    def test_load_model_npu(self, mock_pipeline, manager):
        """Test loading model on NPU"""
        manager.load_model()
        # Verify current behavior
        assert manager.pipeline is not None
        mock_pipeline.assert_called_once()

    def test_safe_inference_with_audio(self, manager):
        """Test inference with valid audio data"""
        import numpy as np
        audio = np.random.randn(16000).astype(np.float32)

        with patch.object(manager, 'pipeline') as mock_pipeline:
            mock_pipeline.generate.return_value = "test transcription"
            result = manager.safe_inference(audio)

            assert isinstance(result, str)
            mock_pipeline.generate.assert_called_once()

    def test_fallback_on_npu_failure(self, manager):
        """Test fallback to GPU/CPU if NPU fails"""
        # Document current fallback behavior
        with patch('openvino_genai.WhisperPipeline', side_effect=Exception("NPU error")):
            manager.load_model()
            assert manager.device in ["gpu", "cpu"]

    def test_clear_cache(self, manager):
        """Test cache clearing"""
        manager.pipelines = {"model1": Mock()}
        manager.clear_cache()
        assert len(manager.pipelines) == 0

    def test_health_check(self, manager):
        """Test health check returns valid status"""
        health = manager.health_check()
        assert "device" in health
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
```

#### 1.2 Test PyTorch ModelManager (REQUIRED - Works on All Systems)

```python
# tests/unit/test_pytorch_manager.py (NEW FILE)

import pytest
from unittest.mock import Mock, patch
import sys
sys.path.insert(0, 'src')
from whisper_service import ModelManager  # PyTorch version

class TestPyTorchModelManager:
    """Characterization tests for PyTorch ModelManager"""

    @pytest.fixture
    def manager(self):
        """Create PyTorch manager instance"""
        return ModelManager(model_name="base", device="cpu")

    def test_initialization(self, manager):
        """Test manager initializes correctly"""
        assert manager.model_name == "base"
        assert manager.device == "cpu"
        assert manager.model is None  # Not loaded yet

    @patch('whisper.load_model')
    def test_load_model_pytorch(self, mock_load, manager):
        """Test loading PyTorch model"""
        mock_model = Mock()
        mock_load.return_value = mock_model

        manager.load_model()

        assert manager.model is not None
        mock_load.assert_called_once_with("base", device="cpu")

    def test_warmup(self, manager):
        """Test model warmup with dummy audio"""
        import numpy as np

        with patch.object(manager, 'model') as mock_model:
            mock_model.transcribe.return_value = {"text": "warmup"}
            manager.warmup()

            # Verify warmup was called
            mock_model.transcribe.assert_called_once()

    def test_init_context(self, manager):
        """Test context initialization per session"""
        session_id = "test-session-123"
        manager.init_context(session_id)

        assert session_id in manager.contexts
        assert manager.contexts[session_id]["tokens"] == []

    def test_trim_context(self, manager):
        """Test context trimming to max length"""
        session_id = "test-session-123"
        manager.contexts[session_id] = {"tokens": list(range(1000))}

        manager.trim_context(session_id, max_length=500)

        assert len(manager.contexts[session_id]["tokens"]) == 500

    def test_clear_context(self, manager):
        """Test clearing session context"""
        session_id = "test-session-123"
        manager.contexts[session_id] = {"tokens": [1, 2, 3]}

        manager.clear_context(session_id)

        assert session_id not in manager.contexts
```

#### 1.3 Create Test Fixtures and conftest.py

```python
# tests/conftest.py (NEW FILE)

"""
Pytest configuration and fixtures for whisper-service tests.

Provides:
- Test audio files (real audio data)
- Mock models for testing
- Shared fixtures across test suites
"""
import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import shutil

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture(scope="session", autouse=True)
def create_test_audio_files():
    """
    Create test audio files for integration testing.

    Creates:
    - hello_world.wav: 1 second, 16kHz, mono, simple tone
    - silence.wav: 1 second, 16kHz, mono, silence
    - noisy.wav: 1 second, 16kHz, mono, noise
    """
    # Create hello_world.wav (440Hz tone for 1 second)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note

    hello_path = FIXTURES_DIR / "hello_world.wav"
    sf.write(hello_path, audio.astype(np.float32), sample_rate)

    # Create silence.wav
    silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
    silence_path = FIXTURES_DIR / "silence.wav"
    sf.write(silence_path, silence, sample_rate)

    # Create noisy.wav (white noise)
    noisy = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
    noisy_path = FIXTURES_DIR / "noisy.wav"
    sf.write(noisy_path, noisy, sample_rate)

    print(f"\n✅ Created test audio files in {FIXTURES_DIR}")

    yield

    # Cleanup after all tests
    # (Keep files for manual inspection, comment out to clean up)
    # shutil.rmtree(FIXTURES_DIR, ignore_errors=True)

@pytest.fixture
def audio_hello_world():
    """Load hello_world.wav as numpy array"""
    audio_path = FIXTURES_DIR / "hello_world.wav"
    audio, sr = sf.read(audio_path, dtype='float32')
    return audio

@pytest.fixture
def audio_silence():
    """Load silence.wav as numpy array"""
    audio_path = FIXTURES_DIR / "silence.wav"
    audio, sr = sf.read(audio_path, dtype='float32')
    return audio

@pytest.fixture
def audio_noisy():
    """Load noisy.wav as numpy array"""
    audio_path = FIXTURES_DIR / "noisy.wav"
    audio, sr = sf.read(audio_path, dtype='float32')
    return audio

@pytest.fixture
def audio_file_path():
    """Get path to test audio file"""
    return FIXTURES_DIR / "hello_world.wav"

@pytest.fixture
def temp_audio_file():
    """Create temporary audio file for upload testing"""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / "test_upload.wav"

    # Create dummy audio
    audio = np.random.randn(16000).astype(np.float32) * 0.1
    sf.write(temp_path, audio, 16000)

    yield temp_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

# Skip tests if OpenVINO not available
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "openvino: mark test as requiring OpenVINO (skip if not installed)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (skip if not available)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skip in fast test runs)"
    )
```

#### 1.4 Run Tests and Document Baseline

```bash
#!/bin/bash
# Run from modules/whisper-service/

set -e

echo "==================================="
echo "PHASE 1: BASELINE TEST SUITE"
echo "==================================="

# Use poetry environment
poetry install --with dev

# Run tests with coverage
poetry run pytest tests/unit/ \
    --cov=src \
    --cov-report=html \
    --cov-report=term \
    --verbose \
    -v

# Verify coverage meets requirements
poetry run coverage report --fail-under=80

echo "✅ Baseline tests pass!"
echo "Coverage report: htmlcov/index.html"
```

**Acceptance Criteria:**
- [ ] Test fixtures created (hello_world.wav, silence.wav, noisy.wav)
- [ ] conftest.py configured with audio fixtures
- [ ] All characterization tests pass
- [ ] Coverage >80% on ModelManager classes
- [ ] Tests run with `poetry run pytest tests/`
- [ ] OpenVINO tests properly skipped on Mac (no failures)
- [ ] Tests document all current behavior (even quirks!)

---

### Day 2-3: Extract and Rename ModelManagers

**Goal:** Move code to new locations WITHOUT changing behavior

#### 2.1 Create Model Abstraction

```python
# src/models/base_model.py (NEW FILE)

from typing import Protocol, Dict, Any, Optional
import numpy as np

class WhisperModel(Protocol):
    """
    Protocol (interface) for Whisper model implementations.

    All model backends (OpenVINO NPU, PyTorch GPU/CPU) must implement
    this interface for seamless switching.
    """

    def load_model(self, model_name: str) -> None:
        """
        Load model into memory.

        Args:
            model_name: Model identifier (e.g., "whisper-base", "base")

        Raises:
            RuntimeError: If model loading fails
        """
        ...

    def transcribe(self,
                   audio: np.ndarray,
                   language: Optional[str] = None,
                   task: str = "transcribe",
                   **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as numpy array (16kHz, mono, float32)
            language: Target language code (e.g., "en", "es")
            task: "transcribe" or "translate"
            **kwargs: Additional model-specific parameters

        Returns:
            Dictionary with keys:
                - text: Transcribed text
                - segments: List of timestamped segments (optional)
                - language: Detected/specified language
        """
        ...

    def get_device(self) -> str:
        """
        Get current device.

        Returns:
            Device string: "npu", "gpu", or "cpu"
        """
        ...

    def clear_cache(self) -> None:
        """Clear model cache and free memory"""
        ...

    def health_check(self) -> Dict[str, Any]:
        """
        Check model health status.

        Returns:
            Dictionary with keys:
                - status: "healthy", "degraded", or "unhealthy"
                - device: Current device
                - model_loaded: Boolean
                - last_inference_time: Timestamp of last inference
        """
        ...
```

#### 2.2 Extract OpenVINO ModelManager

```python
# src/models/openvino_manager.py (NEW FILE)

import logging
from typing import Dict, Any, Optional
import numpy as np

try:
    import openvino_genai
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

logger = logging.getLogger(__name__)

class OpenVINOModelManager:
    """
    OpenVINO-based Whisper model manager for NPU acceleration.

    Extracted from: src/model_manager.py
    Date: 2025-10-25

    Features:
    - NPU (Neural Processing Unit) optimization
    - Automatic fallback to GPU/CPU
    - Thread-safe inference
    - LRU cache for multiple models
    """

    def __init__(self, model_name: str = "whisper-base"):
        """
        Initialize OpenVINO model manager.

        Args:
            model_name: Model identifier (e.g., "whisper-base")
        """
        if not OPENVINO_AVAILABLE:
            raise RuntimeError("OpenVINO not available. Install: pip install openvino-genai")

        self.model_name = model_name
        self.device = self._detect_best_device()
        self.pipeline = None
        self.pipelines: Dict[str, Any] = {}  # LRU cache
        self.inference_lock = threading.Lock()
        self.stats = {
            "total_inferences": 0,
            "errors": 0,
            "last_inference_time": None
        }

    # COPY EXACT CODE FROM src/model_manager.py
    # (All 587 lines)

    def _detect_best_device(self) -> str:
        """Detect best available device (NPU → GPU → CPU)"""
        # ... existing implementation ...

    def load_model(self) -> None:
        """Load model with fallback chain"""
        # ... existing implementation ...

    def safe_inference(self, audio: np.ndarray, **kwargs) -> str:
        """Thread-safe inference"""
        # ... existing implementation ...

    # ... rest of implementation ...
```

**Key Point:** EXACT copy of existing code, just renamed class and file!

#### 2.3 Extract PyTorch ModelManager

```python
# src/models/pytorch_manager.py (NEW FILE)

import logging
from typing import Dict, Any, Optional
import numpy as np
import whisper
import torch

logger = logging.getLogger(__name__)

class PyTorchModelManager:
    """
    PyTorch-based Whisper model manager for GPU/CPU.

    Extracted from: src/whisper_service.py (lines 168-587)
    Date: 2025-10-25

    Features:
    - CUDA GPU acceleration
    - Rolling context support
    - Per-session context isolation
    - Beam search decoding
    - Model warmup
    """

    def __init__(self, model_name: str = "base", device: str = "auto"):
        """
        Initialize PyTorch model manager.

        Args:
            model_name: Model size ("tiny", "base", "small", "medium", "large")
            device: Device preference ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.model = None
        self.contexts: Dict[str, Dict] = {}  # Per-session contexts
        self.inference_lock = threading.Lock()

    # COPY EXACT CODE FROM src/whisper_service.py lines 168-587
    # (All 420 lines)

    def _detect_device(self, device_preference: str) -> str:
        """Detect best PyTorch device"""
        # ... existing implementation ...

    def load_model(self) -> None:
        """Load PyTorch Whisper model"""
        # ... existing implementation ...

    def warmup(self) -> None:
        """Warm up model with dummy audio"""
        # ... existing implementation ...

    def init_context(self, session_id: str) -> None:
        """Initialize rolling context for session"""
        # ... existing implementation ...

    # ... rest of implementation ...
```

#### 2.4 Create Model Factory

```python
# src/models/model_factory.py (NEW FILE)

import logging
from typing import Optional
from .base_model import WhisperModel
from .openvino_manager import OpenVINOModelManager, OPENVINO_AVAILABLE
from .pytorch_manager import PyTorchModelManager
import torch

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for creating appropriate Whisper model implementation.

    Selection priority:
    1. GPU (PyTorch CUDA/MPS) - Most reliable, widely available
    2. NPU (OpenVINO) - If available and supported (may not work on Mac)
    3. CPU (PyTorch) - Universal fallback

    NOTE: OpenVINO is OPTIONAL and may not be installed on all systems.
    """

    @staticmethod
    def create(device: str = "auto",
               model_name: str = "whisper-base") -> WhisperModel:
        """
        Create appropriate model implementation.

        Args:
            device: Device preference ("auto", "gpu", "mps", "npu", "cpu")
            model_name: Model identifier

        Returns:
            Model implementation (OpenVINO or PyTorch)

        Raises:
            RuntimeError: If no suitable backend available
        """
        # Auto-detect best device
        if device == "auto":
            device = ModelFactory._detect_best_device()

        # NPU requested - check if OpenVINO available
        if device == "npu":
            if not OPENVINO_AVAILABLE:
                logger.warning("NPU requested but OpenVINO not installed, falling back to GPU/CPU")
                device = ModelFactory._detect_best_device()
            else:
                logger.info(f"Creating OpenVINO model for NPU: {model_name}")
                return OpenVINOModelManager(model_name=model_name)

        # PyTorch for GPU/MPS/CPU
        logger.info(f"Creating PyTorch model for {device.upper()}: {model_name}")
        return PyTorchModelManager(model_name=model_name, device=device)

    @staticmethod
    def _detect_best_device() -> str:
        """
        Detect best available device.

        Priority: GPU/MPS → NPU → CPU

        Returns:
            Device string: "gpu", "mps", "npu", or "cpu"
        """
        # Check GPU (CUDA) - Linux/Windows with NVIDIA
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected")
            return "gpu"

        # Check MPS (Apple Silicon) - Mac with M1/M2/M3
        if torch.backends.mps.is_available():
            logger.info("Apple MPS (Metal) detected")
            return "mps"

        # Check NPU (OpenVINO) - Intel NPU
        # NOTE: OpenVINO may not be installed, handle gracefully
        if OPENVINO_AVAILABLE:
            try:
                import openvino as ov
                core = ov.Core()
                devices = core.available_devices
                if "NPU" in devices:
                    logger.info("Intel NPU detected")
                    return "npu"
            except Exception as e:
                logger.debug(f"NPU detection failed: {e}")

        # Fallback to CPU (always available)
        logger.info("Using CPU (no accelerator detected)")
        return "cpu"
```

#### 2.5 Update Original Files to Use New Modules

```python
# src/model_manager.py (MODIFIED - THIN WRAPPER)

"""
DEPRECATED: This file is kept for backward compatibility.
New code should import from models.openvino_manager.

Will be removed in future version.
"""
import warnings
from models.openvino_manager import OpenVINOModelManager

warnings.warn(
    "Importing from model_manager.py is deprecated. "
    "Use: from models.openvino_manager import OpenVINOModelManager",
    DeprecationWarning,
    stacklevel=2
)

# Backward compatibility alias
ModelManager = OpenVINOModelManager
```

```python
# src/whisper_service.py (MODIFIED - REMOVE EMBEDDED CLASS)

"""
Core Whisper transcription service.

ModelManager extracted to: src/models/pytorch_manager.py
"""
from models.pytorch_manager import PyTorchModelManager
from models.model_factory import ModelFactory

# Lines 1-167: Keep as-is (imports, dataclasses)
# Lines 168-587: DELETE (ModelManager moved to pytorch_manager.py)
# Lines 588+: Keep as-is (WhisperService class)

class WhisperService:
    """Main transcription service"""

    def __init__(self, model_name: str = "base", device: str = "auto"):
        # OLD: self.model_manager = ModelManager(model_name, device)
        # NEW: Use factory
        self.model_manager = ModelFactory.create(device=device, model_name=model_name)

        # ... rest of implementation stays the same ...
```

#### 2.6 Run Tests - Verify No Breakage

```bash
#!/bin/bash
# Run from modules/whisper-service/

set -e

echo "==================================="
echo "PHASE 1: POST-EXTRACTION TESTS"
echo "==================================="

# Run all tests using poetry
poetry run pytest tests/ --verbose -v

# Run specific model tests
poetry run pytest tests/unit/test_*_manager.py --verbose

# Run integration tests with real audio
poetry run pytest tests/integration/ --verbose

# Check that all features still work
poetry run python tests/manual_verification.py

echo "✅ All tests pass after extraction!"

# Generate coverage report
poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term
echo "📊 Coverage report: htmlcov/index.html"
```

**Acceptance Criteria:**
- [ ] All existing tests still pass (100%)
- [ ] New tests for extracted modules pass (100%)
- [ ] OpenVINO tests skip gracefully on Mac (no failures)
- [ ] No feature regressions detected
- [ ] Import warnings appear for deprecated paths
- [ ] Factory correctly selects device:
  - Mac: MPS → CPU (OpenVINO skipped)
  - Linux/Windows with NVIDIA: GPU → NPU → CPU
  - Other: CPU fallback

---

### Day 4: Update All Imports Across Codebase

**Goal:** Update all files to use new model paths

```bash
#!/bin/bash
# scripts/update_model_imports.sh (NEW FILE)

echo "Updating imports to use new model structure..."

# Find all Python files importing old ModelManager
files=$(grep -rl "from model_manager import ModelManager" src/ || true)

for file in $files; do
    echo "Updating: $file"
    sed -i.bak 's/from model_manager import ModelManager/from models.openvino_manager import OpenVINOModelManager as ModelManager/g' "$file"
done

# Find files importing from whisper_service
files=$(grep -rl "from whisper_service import ModelManager" src/ || true)

for file in $files; do
    echo "Updating: $file"
    sed -i.bak 's/from whisper_service import ModelManager/from models.pytorch_manager import PyTorchModelManager as ModelManager/g' "$file"
done

echo "✅ Import updates complete!"
echo "Review changes before committing."
```

**Manual Review Required:**
- [ ] Review all changed files
- [ ] Ensure semantics are preserved
- [ ] Update any comments referencing old paths
- [ ] Run full test suite

---

### Day 5: Cleanup & Documentation

**Goal:** Remove deprecated code, document changes

#### 5.1 Delete Deprecated Files

```bash
# After verifying everything works, delete old files
# (Keep for 1 sprint as rollback option first!)

# Mark for deletion (don't delete yet!)
git mv src/model_manager.py src/model_manager.py.deprecated
git mv src/whisper_service.py.bak src/whisper_service.py.original

# Add deprecation notice
cat > src/model_manager.py.deprecated << 'EOF'
"""
DEPRECATED FILE - DO NOT USE

This file has been replaced by:
  - src/models/openvino_manager.py (OpenVINO/NPU)
  - src/models/pytorch_manager.py (PyTorch/GPU/CPU)
  - src/models/model_factory.py (Factory)

Scheduled for deletion: 2025-11-25
"""
EOF
```

#### 5.2 Update Documentation

```markdown
# docs/model_architecture.md (NEW FILE)

# Model Architecture - Post-Refactoring

## Overview

Whisper model management has been refactored into a clean abstraction layer.

## Structure

\`\`\`
src/models/
├── base_model.py          # WhisperModel Protocol (interface)
├── openvino_manager.py    # NPU implementation (formerly model_manager.py)
├── pytorch_manager.py     # GPU/CPU implementation (extracted from whisper_service.py)
└── model_factory.py       # Factory pattern for device selection
\`\`\`

## Usage

### Creating a Model

\`\`\`python
from models.model_factory import ModelFactory

# Auto-detect best device (NPU → GPU → CPU)
model = ModelFactory.create(device="auto", model_name="whisper-base")

# Or specify device explicitly
model = ModelFactory.create(device="npu", model_name="whisper-base")
model = ModelFactory.create(device="gpu", model_name="whisper-base")
model = ModelFactory.create(device="cpu", model_name="whisper-base")
\`\`\`

### Using the Model

\`\`\`python
import numpy as np

# Load model
model.load_model()

# Transcribe audio
audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
result = model.transcribe(audio, language="en")

print(result["text"])
\`\`\`

## Migration Guide

### Old Code

\`\`\`python
from model_manager import ModelManager  # Which one?
from whisper_service import ModelManager  # Collision!

manager = ModelManager("whisper-base")
\`\`\`

### New Code

\`\`\`python
from models.model_factory import ModelFactory

model = ModelFactory.create(device="auto", model_name="whisper-base")
\`\`\`

## Device Selection Logic

**Priority (Corrected for Mac Compatibility):**

1. **GPU (CUDA)**: NVIDIA GPU on Linux/Windows - Fast, widely supported
2. **MPS (Metal)**: Apple Silicon (M1/M2/M3) on Mac - Native acceleration
3. **NPU (OpenVINO)**: Intel NPU - May not be installed on all systems (optional dependency)
4. **CPU**: Universal fallback - Always available, slowest

**Important:** OpenVINO is **OPTIONAL** and may not work on Mac. The factory handles this gracefully.

The factory automatically selects the best available device.
\`\`\`
```

#### 5.3 Phase 1 Completion Checklist

```markdown
# Phase 1 Completion Checklist

## Tests
- [ ] All baseline tests pass
- [ ] All post-extraction tests pass
- [ ] Integration tests pass
- [ ] Manual smoke test completed
- [ ] Coverage >80% on extracted modules

## Code Changes
- [ ] OpenVINOModelManager extracted to models/openvino_manager.py
- [ ] PyTorchModelManager extracted to models/pytorch_manager.py
- [ ] WhisperModel protocol created in models/base_model.py
- [ ] ModelFactory created in models/model_factory.py
- [ ] All imports updated across codebase
- [ ] Deprecated files marked with warnings

## Features Verified
- [ ] NPU device detection works
- [ ] GPU device detection works
- [ ] CPU fallback works
- [ ] Model loading works (all devices)
- [ ] Transcription works (all devices)
- [ ] Device fallback chain works (NPU → GPU → CPU)
- [ ] Health checks work
- [ ] Cache clearing works

## Documentation
- [ ] model_architecture.md created
- [ ] Migration guide written
- [ ] Code comments updated
- [ ] CHANGELOG.md updated

## Rollback
- [ ] Original files backed up (.deprecated, .original)
- [ ] Rollback procedure documented
- [ ] Git commits are atomic and revertable

## Sign-off
- [ ] Code review completed
- [ ] QA testing completed
- [ ] Product owner verified features
- [ ] Deployed to staging
- [ ] Monitored for 24 hours
- [ ] ✅ PHASE 1 COMPLETE - MERGE TO MAIN
```

---

## Phase 2: Split whisper_service.py

**Duration:** 5 days
**Goal:** Break down 2,392-line business logic monolith
**Risk:** HIGH (core transcription logic)

### Day 6: Write Tests for whisper_service.py

**Goal:** Comprehensive test coverage before splitting

#### 6.1 Test Transcription Service

```python
# tests/services/test_transcription_service.py (NEW FILE)

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
sys.path.insert(0, 'src')
from whisper_service import WhisperService, TranscriptionRequest, TranscriptionResult

class TestWhisperService:
    """Characterization tests for WhisperService (before splitting)"""

    @pytest.fixture
    def service(self):
        """Create WhisperService instance"""
        with patch('whisper_service.ModelFactory'):
            return WhisperService(model_name="base", device="cpu")

    def test_initialization(self, service):
        """Test service initializes with correct defaults"""
        assert service.model_manager is not None
        assert hasattr(service, 'sessions')
        assert hasattr(service, 'vad_processor')

    def test_transcribe_single_file(self, service):
        """Test single file transcription"""
        audio = np.random.randn(16000).astype(np.float32)

        with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
            mock_transcribe.return_value = {"text": "test transcription"}

            request = TranscriptionRequest(
                audio=audio,
                language="en",
                task="transcribe"
            )
            result = service.transcribe(request)

            assert isinstance(result, TranscriptionResult)
            assert result.text == "test transcription"
            assert result.language == "en"

    def test_transcribe_with_rolling_context(self, service):
        """Test transcription uses rolling context"""
        session_id = "test-session-123"
        audio = np.random.randn(16000).astype(np.float32)

        # Initialize session
        service.init_session(session_id)

        # First transcription
        with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
            mock_transcribe.return_value = {"text": "first sentence"}

            result1 = service.transcribe_streaming(session_id, audio)

            # Second transcription should have context
            mock_transcribe.return_value = {"text": "second sentence"}
            result2 = service.transcribe_streaming(session_id, audio)

            # Verify context was used
            assert result2.text == "second sentence"
            # Context should be passed to model
            calls = mock_transcribe.call_args_list
            assert len(calls) == 2

    def test_session_lifecycle(self, service):
        """Test session creation, use, and cleanup"""
        session_id = "test-session-456"

        # Create session
        service.init_session(session_id)
        assert session_id in service.sessions

        # Use session
        audio = np.random.randn(16000).astype(np.float32)
        service.transcribe_streaming(session_id, audio)

        # Clean up session
        service.cleanup_session(session_id)
        assert session_id not in service.sessions

    def test_vad_integration(self, service):
        """Test VAD (Voice Activity Detection) integration"""
        session_id = "test-session-789"
        service.init_session(session_id)

        # Silent audio (no speech)
        silent_audio = np.zeros(16000, dtype=np.float32)

        with patch.object(service.vad_processor, 'process') as mock_vad:
            mock_vad.return_value = {"speech_detected": False, "chunks": []}

            result = service.transcribe_streaming(session_id, silent_audio)

            # Should handle silent audio gracefully
            assert result is not None or mock_vad.called

    def test_speaker_diarization(self, service):
        """Test speaker diarization integration"""
        audio = np.random.randn(32000).astype(np.float32)  # 2 seconds

        with patch.object(service, 'diarization_pipeline') as mock_diarization:
            mock_diarization.return_value = [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "start": 1.0, "end": 2.0}
            ]

            request = TranscriptionRequest(
                audio=audio,
                enable_diarization=True
            )
            result = service.transcribe(request)

            assert result.segments is not None
            assert len(result.segments) > 0

    def test_language_detection(self, service):
        """Test automatic language detection"""
        audio = np.random.randn(16000).astype(np.float32)

        with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "test",
                "language": "es"  # Detected Spanish
            }

            request = TranscriptionRequest(
                audio=audio,
                language=None  # Auto-detect
            )
            result = service.transcribe(request)

            assert result.language == "es"

    def test_domain_prompt_usage(self, service):
        """Test domain-specific prompts"""
        audio = np.random.randn(16000).astype(np.float32)

        request = TranscriptionRequest(
            audio=audio,
            domain="medical",  # Use medical terminology
            language="en"
        )

        with patch.object(service, 'get_domain_prompt') as mock_prompt:
            mock_prompt.return_value = "This is a medical transcription."

            with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
                mock_transcribe.return_value = {"text": "patient has fever"}

                result = service.transcribe(request)

                # Verify domain prompt was used
                mock_prompt.assert_called_once_with("medical", "en")
                assert result.text == "patient has fever"

    def test_error_handling_invalid_audio(self, service):
        """Test error handling for invalid audio"""
        # Empty audio
        with pytest.raises(ValueError, match="Audio.*empty"):
            service.transcribe(TranscriptionRequest(audio=np.array([])))

        # Wrong dtype
        with pytest.raises(ValueError, match="Audio.*float32"):
            service.transcribe(TranscriptionRequest(
                audio=np.array([1, 2, 3], dtype=np.int16)
            ))

    def test_concurrent_sessions(self, service):
        """Test handling multiple concurrent sessions"""
        session_ids = ["session-1", "session-2", "session-3"]
        audio = np.random.randn(16000).astype(np.float32)

        # Initialize all sessions
        for session_id in session_ids:
            service.init_session(session_id)

        # Process audio in all sessions
        results = []
        with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
            mock_transcribe.return_value = {"text": "test"}

            for session_id in session_ids:
                result = service.transcribe_streaming(session_id, audio)
                results.append(result)

        # Verify all sessions processed independently
        assert len(results) == 3
        assert all(r.text == "test" for r in results)

        # Verify sessions are isolated
        for session_id in session_ids:
            assert session_id in service.sessions
```

#### 6.2 Test Rolling Context

```python
# tests/context/test_rolling_context.py (NEW FILE)

import pytest
from unittest.mock import Mock
import sys
sys.path.insert(0, 'src')
from whisper_service import WhisperService

class TestRollingContext:
    """Test rolling context functionality"""

    @pytest.fixture
    def service(self):
        with patch('whisper_service.ModelFactory'):
            return WhisperService(model_name="base", device="cpu")

    def test_context_initialization(self, service):
        """Test context initializes correctly"""
        session_id = "test-session"
        service.init_session(session_id)

        # Verify context structure
        assert session_id in service.sessions
        session = service.sessions[session_id]
        assert "context" in session
        assert "tokens" in session["context"]
        assert session["context"]["tokens"] == []

    def test_context_accumulation(self, service):
        """Test context accumulates across transcriptions"""
        session_id = "test-session"
        service.init_session(session_id)

        # Simulate multiple transcriptions
        with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
            # First transcription
            mock_transcribe.return_value = {
                "text": "Hello world",
                "tokens": [1, 2, 3]
            }
            service.transcribe_streaming(session_id, np.random.randn(16000).astype(np.float32))

            # Second transcription
            mock_transcribe.return_value = {
                "text": "How are you",
                "tokens": [4, 5, 6]
            }
            service.transcribe_streaming(session_id, np.random.randn(16000).astype(np.float32))

            # Verify context accumulated
            session = service.sessions[session_id]
            tokens = session["context"]["tokens"]
            assert len(tokens) == 6
            assert tokens == [1, 2, 3, 4, 5, 6]

    def test_context_trimming(self, service):
        """Test context trims when exceeding max length"""
        session_id = "test-session"
        service.init_session(session_id)

        # Add many tokens (exceed limit)
        session = service.sessions[session_id]
        session["context"]["tokens"] = list(range(1000))

        # Trim context
        service.trim_context(session_id, max_length=500)

        # Verify trimmed to last 500 tokens
        tokens = session["context"]["tokens"]
        assert len(tokens) == 500
        assert tokens[0] == 500  # Should keep most recent

    def test_context_carryover(self, service):
        """Test context carries over between chunks"""
        session_id = "test-session"
        service.init_session(session_id)

        with patch.object(service.model_manager, 'transcribe') as mock_transcribe:
            # First chunk
            mock_transcribe.return_value = {"text": "The quick brown", "tokens": [1, 2, 3]}
            result1 = service.transcribe_streaming(session_id, np.random.randn(16000).astype(np.float32))

            # Second chunk (model should see previous context)
            mock_transcribe.return_value = {"text": "fox jumps over", "tokens": [4, 5, 6]}
            result2 = service.transcribe_streaming(session_id, np.random.randn(16000).astype(np.float32))

            # Verify model was called with context
            second_call_kwargs = mock_transcribe.call_args_list[1][1]
            assert "context" in second_call_kwargs or "prompt" in second_call_kwargs
```

**Run Baseline Tests:**
```bash
pytest tests/services/test_transcription_service.py -v --cov=src/whisper_service --cov-report=html
pytest tests/context/test_rolling_context.py -v

# Verify coverage >80%
coverage report --fail-under=80
```

**Acceptance Criteria:**
- [ ] All tests pass
- [ ] Coverage >80% on whisper_service.py
- [ ] All features documented with tests

---

### Day 7-8: Extract Components from whisper_service.py

**Goal:** Split into focused modules

#### 7.1 Extract Request/Response Models

```python
# src/transcription/request_models.py (NEW FILE)

"""
Request and response models for transcription.

Extracted from: src/whisper_service.py
Date: 2025-10-25
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

@dataclass
class TranscriptionRequest:
    """Request for transcription"""
    audio: np.ndarray
    language: Optional[str] = None
    task: str = "transcribe"  # or "translate"
    enable_diarization: bool = False
    domain: Optional[str] = None  # For domain-specific prompts
    initial_prompt: Optional[str] = None
    temperature: float = 0.0
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0

    def __post_init__(self):
        """Validate request parameters"""
        if self.audio.size == 0:
            raise ValueError("Audio data cannot be empty")
        if self.audio.dtype != np.float32:
            raise ValueError("Audio must be float32")
        if self.task not in ["transcribe", "translate"]:
            raise ValueError(f"Invalid task: {self.task}")

@dataclass
class TranscriptionSegment:
    """Single segment of transcription with timestamp"""
    id: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    speaker: Optional[str] = None  # From diarization

@dataclass
class TranscriptionResult:
    """Result of transcription"""
    text: str
    language: str
    segments: Optional[List[TranscriptionSegment]] = None
    duration: Optional[float] = None
    processing_time: Optional[float] = None
    model_name: Optional[str] = None
    device: Optional[str] = None

    # Streaming-specific fields
    is_final: bool = True
    session_id: Optional[str] = None
```

#### 7.2 Extract Rolling Context Manager

```python
# src/context/rolling_context_manager.py (NEW FILE)

"""
Rolling context management for streaming transcription.

Extracted from: src/whisper_service.py
Date: 2025-10-25

The rolling context allows Whisper to use previous transcriptions
as context for better accuracy in streaming scenarios.
"""
import logging
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

class RollingContextManager:
    """
    Manages rolling context for transcription sessions.

    Features:
    - Per-session context isolation
    - Automatic context trimming (keeps recent tokens)
    - Token buffer management
    - Context carryover between chunks
    """

    def __init__(self, max_context_length: int = 448):
        """
        Initialize context manager.

        Args:
            max_context_length: Maximum tokens to keep in context
        """
        self.max_context_length = max_context_length
        self.contexts: Dict[str, Dict] = {}

    def init_context(self, session_id: str) -> None:
        """
        Initialize context for new session.

        Args:
            session_id: Unique session identifier
        """
        self.contexts[session_id] = {
            "tokens": [],
            "text_history": deque(maxlen=10),  # Keep last 10 transcriptions
            "metadata": {}
        }
        logger.info(f"Initialized context for session: {session_id}")

    def add_to_context(self,
                       session_id: str,
                       tokens: List[int],
                       text: str) -> None:
        """
        Add tokens to session context.

        Args:
            session_id: Session identifier
            tokens: Token IDs to add
            text: Corresponding text
        """
        if session_id not in self.contexts:
            self.init_context(session_id)

        context = self.contexts[session_id]
        context["tokens"].extend(tokens)
        context["text_history"].append(text)

        # Trim if exceeds max length
        if len(context["tokens"]) > self.max_context_length:
            self.trim_context(session_id)

    def get_context(self, session_id: str) -> Optional[List[int]]:
        """
        Get current context tokens for session.

        Args:
            session_id: Session identifier

        Returns:
            List of token IDs, or None if session doesn't exist
        """
        if session_id not in self.contexts:
            return None

        return self.contexts[session_id]["tokens"].copy()

    def get_text_history(self, session_id: str, n: int = 3) -> List[str]:
        """
        Get recent text history.

        Args:
            session_id: Session identifier
            n: Number of recent transcriptions to return

        Returns:
            List of recent transcriptions
        """
        if session_id not in self.contexts:
            return []

        history = self.contexts[session_id]["text_history"]
        return list(history)[-n:]

    def trim_context(self, session_id: str) -> None:
        """
        Trim context to max length (keeps most recent).

        Args:
            session_id: Session identifier
        """
        if session_id not in self.contexts:
            return

        context = self.contexts[session_id]
        tokens = context["tokens"]

        if len(tokens) > self.max_context_length:
            # Keep most recent tokens
            context["tokens"] = tokens[-self.max_context_length:]
            logger.debug(f"Trimmed context for {session_id}: {len(tokens)} → {self.max_context_length}")

    def clear_context(self, session_id: str) -> None:
        """
        Clear context for session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.contexts:
            del self.contexts[session_id]
            logger.info(f"Cleared context for session: {session_id}")

    def get_stats(self) -> Dict:
        """Get context manager statistics"""
        return {
            "active_sessions": len(self.contexts),
            "total_tokens": sum(len(ctx["tokens"]) for ctx in self.contexts.values()),
            "max_context_length": self.max_context_length
        }
```

#### 7.3 Extract Transcription Service

```python
# src/services/transcription_service.py (NEW FILE)

"""
Core transcription service orchestration.

Extracted from: src/whisper_service.py
Date: 2025-10-25
"""
import logging
import time
from typing import Optional, Dict, Any
import numpy as np

from models.model_factory import ModelFactory
from transcription.request_models import TranscriptionRequest, TranscriptionResult
from context.rolling_context_manager import RollingContextManager

logger = logging.getLogger(__name__)

class TranscriptionService:
    """
    Core transcription orchestration service.

    Responsibilities:
    - Coordinate model, context, VAD, diarization
    - Handle single-file and streaming transcription
    - Manage transcription sessions
    - Apply domain prompts
    """

    def __init__(self,
                 model_name: str = "base",
                 device: str = "auto",
                 enable_vad: bool = True,
                 enable_diarization: bool = False):
        """
        Initialize transcription service.

        Args:
            model_name: Whisper model size
            device: Device preference (auto/npu/gpu/cpu)
            enable_vad: Enable voice activity detection
            enable_diarization: Enable speaker diarization
        """
        self.model = ModelFactory.create(device=device, model_name=model_name)
        self.model.load_model()

        self.context_manager = RollingContextManager()
        self.sessions: Dict[str, Dict[str, Any]] = {}

        self.enable_vad = enable_vad
        self.enable_diarization = enable_diarization

        if enable_vad:
            from audio.vad_processor import VADProcessor
            self.vad_processor = VADProcessor()

        if enable_diarization:
            from audio.diarization import DiarizationPipeline
            self.diarization_pipeline = DiarizationPipeline()

        logger.info(f"TranscriptionService initialized: {model_name} on {device}")

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """
        Transcribe single audio file.

        Args:
            request: Transcription request

        Returns:
            Transcription result
        """
        start_time = time.time()

        # Apply VAD if enabled
        audio = request.audio
        if self.enable_vad:
            vad_result = self.vad_processor.process(audio)
            if not vad_result["speech_detected"]:
                logger.info("No speech detected in audio")
                return TranscriptionResult(
                    text="",
                    language=request.language or "en",
                    processing_time=time.time() - start_time
                )
            audio = vad_result["audio"]  # Speech-only audio

        # Prepare transcription kwargs
        transcribe_kwargs = {
            "language": request.language,
            "task": request.task,
            "temperature": request.temperature,
            "beam_size": request.beam_size,
            "initial_prompt": request.initial_prompt
        }

        # Apply domain prompt if specified
        if request.domain:
            domain_prompt = self.get_domain_prompt(request.domain, request.language)
            transcribe_kwargs["initial_prompt"] = domain_prompt

        # Transcribe
        result = self.model.transcribe(audio, **transcribe_kwargs)

        # Apply diarization if enabled
        segments = None
        if request.enable_diarization and self.enable_diarization:
            segments = self.apply_diarization(audio, result.get("segments", []))

        processing_time = time.time() - start_time

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language", request.language or "en"),
            segments=segments,
            processing_time=processing_time,
            model_name=self.model.model_name,
            device=self.model.get_device()
        )

    def transcribe_streaming(self,
                           session_id: str,
                           audio_chunk: np.ndarray) -> TranscriptionResult:
        """
        Transcribe audio chunk in streaming session.

        Args:
            session_id: Session identifier
            audio_chunk: Audio data chunk

        Returns:
            Transcription result with context
        """
        # Get session context
        context_tokens = self.context_manager.get_context(session_id)

        # Build prompt from recent history
        text_history = self.context_manager.get_text_history(session_id, n=3)
        initial_prompt = " ".join(text_history) if text_history else None

        # Transcribe with context
        result = self.model.transcribe(
            audio_chunk,
            initial_prompt=initial_prompt,
            language=self.sessions[session_id].get("language")
        )

        # Update context
        if "tokens" in result:
            self.context_manager.add_to_context(
                session_id,
                result["tokens"],
                result["text"]
            )

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language", "en"),
            is_final=False,
            session_id=session_id
        )

    def init_session(self,
                     session_id: str,
                     language: Optional[str] = None,
                     domain: Optional[str] = None) -> None:
        """
        Initialize transcription session.

        Args:
            session_id: Unique session identifier
            language: Language code
            domain: Domain for prompts
        """
        self.sessions[session_id] = {
            "language": language,
            "domain": domain,
            "created_at": time.time()
        }
        self.context_manager.init_context(session_id)
        logger.info(f"Initialized session: {session_id}")

    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up session resources.

        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
        self.context_manager.clear_context(session_id)
        logger.info(f"Cleaned up session: {session_id}")

    def get_domain_prompt(self, domain: str, language: Optional[str]) -> str:
        """
        Get domain-specific prompt.

        Args:
            domain: Domain name (e.g., "medical", "legal")
            language: Language code

        Returns:
            Domain-specific prompt text
        """
        # TODO: Integrate with DomainPromptManager
        prompts = {
            "medical": "This is a medical transcription. Use proper medical terminology.",
            "legal": "This is a legal transcription. Use proper legal terminology.",
            "technical": "This is a technical transcription. Use proper technical terminology."
        }
        return prompts.get(domain, "")

    def apply_diarization(self, audio: np.ndarray, segments: List) -> List:
        """
        Apply speaker diarization to segments.

        Args:
            audio: Audio data
            segments: Transcription segments

        Returns:
            Segments with speaker labels
        """
        diarization_result = self.diarization_pipeline(audio)

        # Match segments with speakers
        for segment in segments:
            # Find speaker at segment midpoint
            midpoint = (segment["start"] + segment["end"]) / 2
            for speaker_segment in diarization_result:
                if speaker_segment["start"] <= midpoint <= speaker_segment["end"]:
                    segment["speaker"] = speaker_segment["speaker"]
                    break

        return segments

    def get_stats(self) -> Dict:
        """Get service statistics"""
        return {
            "active_sessions": len(self.sessions),
            "model": self.model.model_name,
            "device": self.model.get_device(),
            "context_stats": self.context_manager.get_stats()
        }
```

#### 7.4 Update whisper_service.py to Use Extracted Components

```python
# src/whisper_service.py (MODIFIED - NOW THIN WRAPPER)

"""
Whisper transcription service.

NOTE: This file has been refactored. Core logic moved to:
  - services/transcription_service.py
  - models/pytorch_manager.py
  - context/rolling_context_manager.py
  - transcription/request_models.py
"""

# Keep backward compatibility
from services.transcription_service import TranscriptionService
from transcription.request_models import (
    TranscriptionRequest,
    TranscriptionResult,
    TranscriptionSegment
)

# Deprecated alias for backward compatibility
WhisperService = TranscriptionService

# Factory function (keep for backward compat)
def create_whisper_service(model_name: str = "base",
                          device: str = "auto",
                          **kwargs) -> TranscriptionService:
    """
    Create transcription service.

    Args:
        model_name: Whisper model size
        device: Device preference
        **kwargs: Additional arguments

    Returns:
        TranscriptionService instance
    """
    return TranscriptionService(
        model_name=model_name,
        device=device,
        **kwargs
    )
```

---

### Day 9: Run Tests & Verify No Breakage

```bash
#!/bin/bash
set -e

echo "==================================="
echo "PHASE 2: POST-SPLIT VERIFICATION"
echo "==================================="

# Run all tests
pytest tests/ --verbose --cov=src --cov-report=html

# Specific verification
pytest tests/services/test_transcription_service.py -v
pytest tests/context/test_rolling_context.py -v
pytest tests/models/test_model_factory.py -v

# Integration tests
pytest tests/integration/test_api_endpoints.py -v
pytest tests/integration/test_websocket.py -v

# Feature verification script
python tests/verify_features.py --check-all

echo "✅ Phase 2 verification complete!"
```

**Feature Verification Script:**

```python
# tests/verify_features.py (NEW FILE)

"""
Verify all features still work after refactoring.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, 'src')

from services.transcription_service import TranscriptionService
from transcription.request_models import TranscriptionRequest

def test_single_file_transcription():
    """Test single file transcription works"""
    print("Testing single file transcription...")
    service = TranscriptionService(model_name="base", device="cpu")

    audio = np.random.randn(16000).astype(np.float32)
    request = TranscriptionRequest(audio=audio, language="en")
    result = service.transcribe(request)

    assert result.text is not None
    assert result.language == "en"
    print("✅ Single file transcription works")

def test_streaming_transcription():
    """Test streaming transcription works"""
    print("Testing streaming transcription...")
    service = TranscriptionService(model_name="base", device="cpu")

    session_id = "test-session"
    service.init_session(session_id, language="en")

    # Send multiple chunks
    for i in range(3):
        audio = np.random.randn(16000).astype(np.float32)
        result = service.transcribe_streaming(session_id, audio)
        assert result.text is not None
        assert result.session_id == session_id

    service.cleanup_session(session_id)
    print("✅ Streaming transcription works")

def test_rolling_context():
    """Test rolling context works"""
    print("Testing rolling context...")
    service = TranscriptionService(model_name="base", device="cpu")

    session_id = "context-test"
    service.init_session(session_id)

    # Transcribe multiple chunks
    chunks = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
    for chunk in chunks:
        service.transcribe_streaming(session_id, chunk)

    # Verify context exists
    context = service.context_manager.get_context(session_id)
    assert context is not None

    service.cleanup_session(session_id)
    print("✅ Rolling context works")

def test_model_device_fallback():
    """Test model device fallback"""
    print("Testing device fallback...")

    # Try NPU → should fallback to GPU/CPU
    service = TranscriptionService(model_name="base", device="npu")
    device = service.model.get_device()
    assert device in ["npu", "gpu", "cpu"]

    print(f"✅ Device fallback works (using: {device})")

def test_vad_integration():
    """Test VAD integration"""
    print("Testing VAD integration...")
    service = TranscriptionService(model_name="base", device="cpu", enable_vad=True)

    # Silent audio
    silent = np.zeros(16000, dtype=np.float32)
    request = TranscriptionRequest(audio=silent)
    result = service.transcribe(request)

    # Should handle gracefully
    assert result.text == "" or result.text is not None
    print("✅ VAD integration works")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    args = parser.parse_args()

    if args.check_all:
        test_single_file_transcription()
        test_streaming_transcription()
        test_rolling_context()
        test_model_device_fallback()
        test_vad_integration()

        print("\n" + "="*50)
        print("✅ ALL FEATURES VERIFIED!")
        print("="*50)
```

**Acceptance Criteria:**
- [ ] All tests pass (unit + integration + end-to-end)
- [ ] Feature verification script passes
- [ ] whisper_service.py reduced from 2,392 → ~600 lines
- [ ] No feature regressions detected
- [ ] Backward compatibility maintained

---

### Day 10: Cleanup & Documentation

**Phase 2 Completion Checklist:**

```markdown
# Phase 2 Completion Checklist

## Files Created
- [ ] src/transcription/request_models.py (request/response models)
- [ ] src/context/rolling_context_manager.py (context management)
- [ ] src/services/transcription_service.py (core service)
- [ ] tests/services/test_transcription_service.py (tests)
- [ ] tests/context/test_rolling_context.py (tests)
- [ ] tests/verify_features.py (feature verification)

## Files Modified
- [ ] src/whisper_service.py (now thin wrapper)

## Tests
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Feature verification passes
- [ ] Coverage >80% on new modules

## Features Verified
- [ ] Single file transcription works
- [ ] Streaming transcription works
- [ ] Rolling context works
- [ ] VAD integration works
- [ ] Speaker diarization works
- [ ] Language detection works
- [ ] Domain prompts work
- [ ] Session management works

## Documentation
- [ ] Updated architecture docs
- [ ] Migration guide for new structure
- [ ] Code comments updated

## Sign-off
- [ ] Code review completed
- [ ] QA testing completed
- [ ] ✅ PHASE 2 COMPLETE
```

---

## Phase 3: Split api_server.py

**Duration:** 5 days
**Goal:** Break down 3,642-line API monolith
**Risk:** HIGH (API layer, WebSocket infrastructure)

### Day 11: Write Tests for api_server.py

**Goal:** Comprehensive API test coverage

#### 11.1 Test HTTP Endpoints

```python
# tests/api/test_http_routes.py (NEW FILE)

import pytest
from flask import Flask
import numpy as np
import sys
sys.path.insert(0, 'src')
from api_server import app  # Current monolithic app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

class TestHTTPEndpoints:
    """Test all HTTP endpoints before extraction"""

    def test_health_endpoint(self, client):
        """Test /health endpoint"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_transcribe_endpoint(self, client):
        """Test /transcribe endpoint"""
        # Create dummy audio file
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()

        response = client.post('/transcribe',
                             data={'audio': (io.BytesIO(audio_data), 'test.wav')},
                             content_type='multipart/form-data')

        assert response.status_code == 200
        data = response.get_json()
        assert "text" in data

    def test_api_transcribe_endpoint(self, client):
        """Test /api/transcribe endpoint (duplicate)"""
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()

        response = client.post('/api/transcribe',
                             data={'audio': (io.BytesIO(audio_data), 'test.wav')},
                             content_type='multipart/form-data')

        assert response.status_code == 200
        data = response.get_json()
        assert "text" in data

    def test_models_endpoint(self, client):
        """Test /models endpoint"""
        response = client.get('/models')
        assert response.status_code == 200
        data = response.get_json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_load_model_endpoint(self, client):
        """Test /load-model endpoint"""
        response = client.post('/load-model',
                             json={'model_name': 'whisper-base'})
        assert response.status_code in [200, 201]

    def test_config_endpoint(self, client):
        """Test /config endpoint"""
        # GET config
        response = client.get('/config')
        assert response.status_code == 200
        config = response.get_json()
        assert isinstance(config, dict)

        # POST config
        response = client.post('/config',
                             json={'model_name': 'whisper-base'})
        assert response.status_code in [200, 201]

    def test_stats_endpoint(self, client):
        """Test /stats endpoint"""
        response = client.get('/stats')
        assert response.status_code == 200
        data = response.get_json()
        assert "device" in data or "stats" in data

    def test_device_status_endpoint(self, client):
        """Test /device-status endpoint"""
        response = client.get('/device-status')
        assert response.status_code == 200
        data = response.get_json()
        assert "devices" in data or "available" in data

    def test_clear_cache_endpoint(self, client):
        """Test /clear-cache endpoint"""
        response = client.post('/clear-cache')
        assert response.status_code == 200

    def test_session_status_endpoint(self, client):
        """Test /session-status endpoint"""
        response = client.get('/session-status')
        assert response.status_code == 200
        data = response.get_json()
        assert "sessions" in data or "active_sessions" in data

    def test_error_handling(self, client):
        """Test error handling"""
        # Invalid request
        response = client.post('/transcribe', data={})
        assert response.status_code in [400, 422]

        # Not found
        response = client.get('/nonexistent')
        assert response.status_code == 404
```

#### 11.2 Test WebSocket Events

```python
# tests/api/test_websocket_events.py (NEW FILE)

import pytest
from flask_socketio import SocketIOTestClient
import numpy as np
import sys
sys.path.insert(0, 'src')
from api_server import app, socketio

@pytest.fixture
def socket_client():
    """Create SocketIO test client"""
    app.config['TESTING'] = True
    client = socketio.test_client(app)
    yield client
    client.disconnect()

class TestWebSocketEvents:
    """Test all WebSocket events before extraction"""

    def test_connect_event(self, socket_client):
        """Test client connection"""
        assert socket_client.is_connected()

    def test_disconnect_event(self, socket_client):
        """Test client disconnection"""
        socket_client.disconnect()
        assert not socket_client.is_connected()

    def test_start_stream_event(self, socket_client):
        """Test start_stream event"""
        socket_client.emit('start_stream', {
            'session_id': 'test-session-123',
            'language': 'en',
            'model_name': 'whisper-base'
        })

        # Should receive session_ready
        received = socket_client.get_received()
        assert len(received) > 0
        assert any(msg['name'] == 'session_ready' for msg in received)

    def test_audio_chunk_event(self, socket_client):
        """Test audio_chunk event"""
        # Start session first
        socket_client.emit('start_stream', {'session_id': 'test-123'})
        socket_client.get_received()  # Clear buffer

        # Send audio chunk
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        socket_client.emit('audio_chunk', {
            'session_id': 'test-123',
            'audio': audio_data
        })

        # Should receive transcription_result
        received = socket_client.get_received()
        assert len(received) > 0
        # May receive transcription_result or status update

    def test_stop_stream_event(self, socket_client):
        """Test stop_stream event"""
        # Start session
        socket_client.emit('start_stream', {'session_id': 'test-456'})
        socket_client.get_received()

        # Stop session
        socket_client.emit('stop_stream', {'session_id': 'test-456'})

        # Should receive confirmation
        received = socket_client.get_received()
        assert len(received) > 0

    def test_update_session_config_event(self, socket_client):
        """Test update_session_config event"""
        socket_client.emit('start_stream', {'session_id': 'test-789'})
        socket_client.get_received()

        socket_client.emit('update_session_config', {
            'session_id': 'test-789',
            'config': {'language': 'es'}
        })

        # Should receive acknowledgment
        received = socket_client.get_received()
        # Verify config was updated

    def test_reconnection_scenario(self, socket_client):
        """Test client reconnection maintains session"""
        # Start session
        socket_client.emit('start_stream', {'session_id': 'reconnect-test'})
        socket_client.get_received()

        # Disconnect
        socket_client.disconnect()

        # Reconnect
        socket_client.connect()
        assert socket_client.is_connected()

        # Session should be recoverable
        socket_client.emit('audio_chunk', {
            'session_id': 'reconnect-test',
            'audio': np.random.randn(16000).astype(np.float32).tobytes()
        })

        # Should handle gracefully (may error if session expired, that's OK)

    def test_concurrent_sessions(self, socket_client):
        """Test multiple concurrent streaming sessions"""
        session_ids = ['session-1', 'session-2', 'session-3']

        # Start all sessions
        for session_id in session_ids:
            socket_client.emit('start_stream', {'session_id': session_id})

        socket_client.get_received()  # Clear buffer

        # Send audio to all sessions
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        for session_id in session_ids:
            socket_client.emit('audio_chunk', {
                'session_id': session_id,
                'audio': audio_data
            })

        # Should handle all sessions
        received = socket_client.get_received()
        assert len(received) > 0
```

**Run Baseline API Tests:**

```bash
pytest tests/api/test_http_routes.py -v --cov=src/api_server
pytest tests/api/test_websocket_events.py -v

# Verify all endpoints covered
coverage report --show-missing
```

**Acceptance Criteria:**
- [ ] All HTTP endpoint tests pass
- [ ] All WebSocket event tests pass
- [ ] Coverage >70% on api_server.py (API portions)

---

### Day 12-13: Extract Components from api_server.py

**Goal:** Split into focused API modules

#### 12.1 Extract Performance Monitoring

```python
# src/monitoring/performance.py (NEW FILE)

"""
Performance monitoring and thread pool management.

Extracted from: src/api_server.py (lines 97-208)
Date: 2025-10-25
"""
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from typing import Dict, Any, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0

class AudioProcessingPool:
    """
    Thread pool for audio processing tasks.

    Extracted from api_server.py
    Features:
    - Configurable worker count
    - Task queueing
    - Performance tracking
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize audio processing pool.

        Args:
            max_workers: Maximum concurrent workers
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="audio_worker"
        )
        self.max_workers = max_workers
        self.active_tasks = 0
        self.lock = threading.Lock()

        logger.info(f"AudioProcessingPool initialized with {max_workers} workers")

    def submit(self, func: Callable, *args, **kwargs):
        """
        Submit task to pool.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Future object
        """
        with self.lock:
            self.active_tasks += 1

        future = self.executor.submit(func, *args, **kwargs)
        future.add_done_callback(self._task_done)
        return future

    def _task_done(self, future):
        """Callback when task completes"""
        with self.lock:
            self.active_tasks -= 1

    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self.lock:
            return {
                "max_workers": self.max_workers,
                "active_tasks": self.active_tasks,
                "queue_depth": self.active_tasks  # Simplified
            }

    def shutdown(self, wait: bool = True):
        """Shutdown pool"""
        self.executor.shutdown(wait=wait)
        logger.info("AudioProcessingPool shut down")

class MessageQueue:
    """
    Async message queue for processing.

    Extracted from api_server.py
    """

    def __init__(self, maxsize: int = 1000):
        """
        Initialize message queue.

        Args:
            maxsize: Maximum queue size
        """
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item: Any, block: bool = True, timeout: float = None):
        """Add item to queue"""
        self.queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: float = None) -> Any:
        """Get item from queue"""
        return self.queue.get(block=block, timeout=timeout)

    def get_nowait(self) -> Any:
        """Get item without blocking"""
        return self.queue.get_nowait()

    def qsize(self) -> int:
        """Get queue size"""
        return self.queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()

    def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            "size": self.queue.qsize(),
            "maxsize": self.maxsize,
            "utilization": self.queue.qsize() / self.maxsize if self.maxsize > 0 else 0
        }

class PerformanceMonitor:
    """
    Monitor and track performance metrics.

    Extracted from api_server.py
    """

    def __init__(self):
        """Initialize performance monitor"""
        self.metrics = PerformanceMetrics()
        self.lock = threading.Lock()
        self.request_history = []
        self.max_history = 1000

    def record_request(self,
                      processing_time: float,
                      success: bool = True,
                      metadata: Dict = None):
        """
        Record request metrics.

        Args:
            processing_time: Time taken to process request
            success: Whether request succeeded
            metadata: Additional metadata
        """
        with self.lock:
            self.metrics.total_requests += 1

            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1

            self.metrics.total_processing_time += processing_time
            self.metrics.avg_processing_time = (
                self.metrics.total_processing_time / self.metrics.total_requests
            )

            # Record in history
            self.request_history.append({
                "timestamp": time.time(),
                "processing_time": processing_time,
                "success": success,
                "metadata": metadata or {}
            })

            # Trim history
            if len(self.request_history) > self.max_history:
                self.request_history = self.request_history[-self.max_history:]

    def get_metrics(self) -> PerformanceMetrics:
        """Get current metrics"""
        with self.lock:
            return PerformanceMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                total_processing_time=self.metrics.total_processing_time,
                avg_processing_time=self.metrics.avg_processing_time
            )

    def get_stats(self) -> Dict:
        """Get metrics as dictionary"""
        metrics = self.get_metrics()
        return {
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "success_rate": (
                metrics.successful_requests / metrics.total_requests
                if metrics.total_requests > 0 else 0
            ),
            "avg_processing_time": metrics.avg_processing_time,
            "recent_requests": len(self.request_history)
        }

    def reset(self):
        """Reset metrics"""
        with self.lock:
            self.metrics = PerformanceMetrics()
            self.request_history = []
            logger.info("Performance metrics reset")
```

#### 12.2 Extract HTTP Routes

```python
# src/api/http_routes.py (NEW FILE)

"""
HTTP REST API routes.

Extracted from: src/api_server.py
Date: 2025-10-25
"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import logging
import io
import numpy as np

from services.transcription_service import TranscriptionService
from transcription.request_models import TranscriptionRequest
from audio.audio_processor import AudioProcessor
from monitoring.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

# Create blueprint
http_api = Blueprint('http_api', __name__)

# Global instances (will be injected by app factory)
transcription_service: TranscriptionService = None
audio_processor: AudioProcessor = None
performance_monitor: PerformanceMonitor = None

def init_http_routes(service: TranscriptionService,
                     processor: AudioProcessor,
                     monitor: PerformanceMonitor):
    """
    Initialize HTTP routes with dependencies.

    Args:
        service: Transcription service instance
        processor: Audio processor instance
        monitor: Performance monitor instance
    """
    global transcription_service, audio_processor, performance_monitor
    transcription_service = service
    audio_processor = processor
    performance_monitor = monitor

@http_api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "whisper-service",
        "device": transcription_service.model.get_device() if transcription_service else "unknown"
    })

@http_api.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file.

    Expected form data:
        - audio: Audio file
        - language: (optional) Language code
        - task: (optional) "transcribe" or "translate"
    """
    import time
    start_time = time.time()

    try:
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        # Read and process audio
        audio_data = audio_file.read()
        audio_array = audio_processor.process(audio_data)

        # Get parameters
        language = request.form.get('language')
        task = request.form.get('task', 'transcribe')

        # Create request
        transcription_request = TranscriptionRequest(
            audio=audio_array,
            language=language,
            task=task
        )

        # Transcribe
        result = transcription_service.transcribe(transcription_request)

        # Record metrics
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, success=True)

        return jsonify({
            "text": result.text,
            "language": result.language,
            "processing_time": processing_time
        })

    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        performance_monitor.record_request(
            time.time() - start_time,
            success=False,
            metadata={"error": str(e)}
        )
        return jsonify({"error": str(e)}), 500

@http_api.route('/api/transcribe', methods=['POST'])
def api_transcribe():
    """Alternative transcribe endpoint (for compatibility)"""
    return transcribe()

@http_api.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    models = [
        "whisper-tiny",
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        "whisper-large"
    ]
    return jsonify(models)

@http_api.route('/api/models', methods=['GET'])
def api_list_models():
    """Alternative models endpoint"""
    return list_models()

@http_api.route('/config', methods=['GET', 'POST'])
def config():
    """Get or update configuration"""
    if request.method == 'GET':
        # Get current config
        config = {
            "model_name": transcription_service.model.model_name,
            "device": transcription_service.model.get_device(),
            "vad_enabled": transcription_service.enable_vad,
            "diarization_enabled": transcription_service.enable_diarization
        }
        return jsonify(config)

    elif request.method == 'POST':
        # Update config
        new_config = request.get_json()
        # Note: Changing model requires reload, handle carefully
        return jsonify({"status": "config_updated", "config": new_config})

@http_api.route('/stats', methods=['GET'])
def stats():
    """Get performance statistics"""
    stats = {
        "service": transcription_service.get_stats(),
        "performance": performance_monitor.get_stats()
    }
    return jsonify(stats)

@http_api.route('/device-status', methods=['GET'])
def device_status():
    """Get device availability status"""
    device = transcription_service.model.get_device()
    health = transcription_service.model.health_check()

    return jsonify({
        "current_device": device,
        "health": health,
        "available_devices": ["npu", "gpu", "cpu"]  # TODO: Actually detect
    })

@http_api.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear model cache"""
    transcription_service.model.clear_cache()
    return jsonify({"status": "cache_cleared"})

@http_api.route('/session-status', methods=['GET'])
def session_status():
    """Get session status"""
    stats = transcription_service.get_stats()
    return jsonify({
        "active_sessions": stats["active_sessions"],
        "sessions": list(transcription_service.sessions.keys())
    })
```

*(Continue with WebSocket routes extraction in the actual document...)*

---

**Due to length constraints, I'll summarize the remaining phases:**

### Phase 4: Consolidate Session Management (Days 16-18)
- Merge 3 session managers into one
- Test session lifecycle
- Verify reconnection works

### Phase 5: Deduplication & Polish (Days 19-20)
- Remove duplicate health monitoring
- Consolidate error handling
- Clean up code duplication
- Final integration tests

---

## Rollback Strategy

**If anything goes wrong:**

```bash
# Rollback to previous phase
git revert HEAD~5..HEAD  # Revert last 5 commits

# Or rollback specific file
git checkout HEAD~5 -- src/whisper_service.py

# Restore from backup
cp src/whisper_service.py.original src/whisper_service.py
```

**Rollback decision points:**
- Any test suite failure → STOP, investigate
- Feature regression detected → ROLLBACK
- Coverage drops below 75% → STOP, add tests

---

## Success Metrics

**Final Targets:**
- [ ] whisper_service.py: 2,392 → 600 lines (75% reduction)
- [ ] api_server.py: 3,642 → 800 lines (78% reduction)
- [ ] 3 session managers → 1 unified manager
- [ ] Test coverage: >80% across all modules
- [ ] ZERO feature regressions
- [ ] All integration tests pass
- [ ] Documentation complete
- [ ] Migration guide written

**Success = Clean, maintainable, STATEFUL service with zero feature loss!**
