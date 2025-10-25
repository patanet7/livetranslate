# Whisper Service Architecture Audit

**Audit Date:** 2025-10-25
**Module:** `modules/whisper-service/`
**Total Production Files:** 51 Python files
**Total Production LOC:** ~24,000 lines

---

## Executive Summary

The whisper-service module exhibits **significant architectural debt** with several critical anti-patterns:

1. **Monolithic API Server (3,642 lines)** - Violates Single Responsibility Principle with mixed concerns: HTTP routing, WebSocket handling, audio processing, session management, authentication, performance monitoring, and business logic all in one file.

2. **Duplicate ModelManager Implementations** - Two completely separate ModelManager classes (`model_manager.py` and embedded in `whisper_service.py`) serving different purposes (OpenVINO NPU vs PyTorch) with no abstraction layer.

3. **9+ Manager Classes** - Over-engineered manager pattern with unclear boundaries and potential circular dependencies.

4. **WebSocket Infrastructure Spread** - Real-time communication logic scattered across 3 files (api_server.py, heartbeat_manager.py, main.py) with 50+ emit() calls.

5. **Session Management Duplication** - Session handling logic duplicated across stream_session_manager.py, reconnection_manager.py, and api_server.py.

**Risk Level:** **HIGH** - The current architecture will become unmaintainable as complexity grows.

---

## Monolithic Files Analysis

### 🔴 CRITICAL: `api_server.py` (3,642 lines)

**Current Responsibilities (10+):**
- HTTP endpoint routing (35+ routes)
- WebSocket server configuration
- Audio file processing (`_process_audio_data()`)
- Session lifecycle management
- Authentication middleware integration
- Performance monitoring (AudioProcessingPool, MessageQueue, PerformanceMonitor)
- Error handling and recovery
- Model initialization (stateful whisper)
- CORS configuration
- FFmpeg path configuration
- Service initialization
- Heartbeat management callbacks
- Connection state callbacks
- Token cleanup scheduling

**Recommended Split Strategy:**

```
api_server.py (3,642 lines) →

1. src/api/http_routes.py (800 lines)
   - HTTP endpoint definitions
   - Request/response handling
   - Route decorators
   Complexity: LOW

2. src/api/websocket_routes.py (400 lines)
   - WebSocket event handlers
   - Real-time streaming endpoints
   - SocketIO configuration
   Complexity: LOW

3. src/audio/audio_processing.py (300 lines)
   - _process_audio_data()
   - Audio format conversion
   - FFmpeg integration
   Complexity: MEDIUM (already has audio_processor.py)

4. src/session/session_orchestrator.py (500 lines)
   - Session creation/destruction
   - Session state management
   - streaming_sessions dict management
   Complexity: MEDIUM

5. src/performance/monitoring.py (400 lines)
   - AudioProcessingPool
   - MessageQueue
   - PerformanceMonitor
   - Metrics collection
   Complexity: LOW

6. src/initialization/service_init.py (300 lines)
   - initialize_service()
   - initialize_stateful_whisper()
   - Connection manager startup
   - Heartbeat manager setup
   Complexity: MEDIUM

7. src/api/middleware.py (200 lines)
   - Authentication callbacks
   - Error handlers
   - CORS configuration
   Complexity: LOW

8. REMAINING: api_server.py (740 lines)
   - Flask app configuration
   - Import coordination
   - Main entry point
```

**Migration Complexity:** **HIGH**
- **Blocker:** Global `whisper_service`, `stateful_whisper`, `streaming_sessions` state
- **Requires:** Dependency injection container or service locator pattern
- **Estimate:** 3-4 days of refactoring + 2 days testing

---

### 🟡 MEDIUM: `whisper_service.py` (2,392 lines)

**Current Responsibilities (8+):**
- WhisperService class (main service interface)
- ModelManager class (PyTorch-based, lines 168-587)
- TranscriptionRequest dataclass
- TranscriptionResult dataclass
- Beam search integration
- AlignAtt decoder integration
- Domain prompt management
- Rolling context system
- Session management (per-session contexts)
- Audio processing
- VAD integration
- Speaker diarization
- Warmup system

**Key Issue:** **Two ModelManagers Exist**
- `whisper_service.py::ModelManager` - PyTorch Whisper (GPU/CPU)
- `model_manager.py::ModelManager` - OpenVINO Whisper (NPU)
- **NO abstraction layer** to switch between them

**Recommended Split Strategy:**

```
whisper_service.py (2,392 lines) →

1. src/models/pytorch_model_manager.py (420 lines)
   - Extract ModelManager class (lines 168-587)
   - Rename to PyTorchModelManager
   - PyTorch-specific optimizations
   Complexity: LOW

2. src/models/model_factory.py (150 lines)
   - Abstract ModelInterface protocol
   - Factory to select NPU vs PyTorch
   - Device detection logic
   Complexity: MEDIUM

3. src/transcription/transcription_service.py (800 lines)
   - Core WhisperService orchestration
   - Transcription pipeline
   - Session coordination
   Complexity: MEDIUM

4. src/transcription/request_models.py (100 lines)
   - TranscriptionRequest
   - TranscriptionResult
   - Configuration dataclasses
   Complexity: LOW

5. src/context/rolling_context.py (300 lines)
   - Rolling context management
   - Per-session context isolation
   - Token buffer integration
   Complexity: LOW

6. REMAINING: whisper_service.py (622 lines)
   - create_whisper_service() factory
   - High-level coordination
```

**Migration Complexity:** **MEDIUM**
- **Requires:** Extract ModelManager → Create abstraction → Refactor imports
- **Estimate:** 2-3 days

---

### 🟢 ACCEPTABLE: `vac_online_processor.py` (960 lines)

**Responsibilities:**
- VAD-based chunking
- SimulStreaming integration
- Audio buffer management
- Silence detection
- Language detection
- Token deduplication
- UTF-8 boundary fixing

**Status:** Well-contained, single responsibility (VAD-Whisper coordination)

**Minor Optimization:**
- Extract `_should_reset_sot()` logic (lines 400-461) → `sot_reset_policy.py`
- Extract hybrid tracking (lines 599-630) → `timestamp_tracker.py`

**Migration Complexity:** **LOW** (2-3 hours)

---

### 🟡 MEDIUM: `simul_whisper/whisper/decoding.py` (833 lines)

**Responsibilities:**
- Beam search decoding
- Greedy decoding
- Temperature sampling
- Sequence scoring
- Token generation
- Compression ratio filtering

**Issue:** This is a fork of OpenAI Whisper's decoding logic with SimulStreaming modifications.

**Status:** **Keep as-is** (external library fork, well-tested)

**Recommendation:** Document which lines differ from upstream Whisper for future merge tracking.

---

### 🟡 MEDIUM: `simul_whisper/simul_whisper.py` (819 lines)

**Responsibilities:**
- PaddedAlignAttWhisper class
- SimulStreaming policy implementation
- Cross-attention extraction
- Frame-level decision making
- Audio chunking coordination

**Status:** Core SimulStreaming algorithm - **Keep as-is**

**Minor Optimization:**
- Extract config validation (lines 30-60) → `config_validator.py`

---

## Code Duplication & Redundancy

### 🔴 CRITICAL: Duplicate ModelManager Classes

**Location 1:** `/src/model_manager.py` (587 lines)
- **Purpose:** OpenVINO NPU acceleration
- **Device Support:** NPU → GPU → CPU fallback
- **Model Format:** OpenVINO IR (.xml/.bin)
- **Key Methods:** `load_model()`, `safe_inference()`, `_load_pipeline_with_fallback()`

**Location 2:** `/src/whisper_service.py` (lines 168-587, embedded)
- **Purpose:** PyTorch GPU/CPU
- **Device Support:** CUDA → MPS → CPU
- **Model Format:** PyTorch (.pt)
- **Key Methods:** `load_model()`, `warmup()`, `init_context()`, `trim_context()`

**Duplication Analysis:**
```
Shared Responsibilities:
✓ Device detection (_detect_best_device)
✓ Model loading (load_model)
✓ Thread safety (inference_lock)
✓ Memory management (clear_cache)
✓ Statistics tracking (get_stats)
✓ Health checks (health_check)

Unique to OpenVINO:
- openvino_genai.WhisperPipeline
- NPU-specific error handling (ZE_RESULT_ERROR_DEVICE_LOST)
- Minimum inference interval (NPU cooldown)

Unique to PyTorch:
- whisper.load_model()
- Beam search configuration
- Rolling context system
- Warmup system
- Per-session tokenizers
```

**Consolidation Strategy:**

```python
# 1. Create abstraction (Protocol-based)
# src/models/base_model.py
from typing import Protocol, Any
import numpy as np

class WhisperModel(Protocol):
    """Abstract interface for Whisper models"""

    def load(self, model_name: str) -> None:
        """Load model into memory"""
        ...

    def transcribe(self, audio: np.ndarray, **kwargs) -> str:
        """Transcribe audio to text"""
        ...

    def get_device(self) -> str:
        """Get current device (NPU/GPU/CPU)"""
        ...

    def clear_cache(self) -> None:
        """Clear model cache"""
        ...

# 2. Rename existing implementations
# src/models/openvino_model.py
class OpenVINOWhisperModel:
    """OpenVINO NPU implementation"""
    # Current model_manager.py content
    ...

# src/models/pytorch_model.py
class PyTorchWhisperModel:
    """PyTorch GPU/CPU implementation"""
    # Extracted from whisper_service.py
    ...

# 3. Factory pattern
# src/models/model_factory.py
def create_model(device_preference: str = "auto") -> WhisperModel:
    """
    Factory to select appropriate model implementation

    Priority:
    1. NPU → OpenVINOWhisperModel (if openvino_genai available)
    2. GPU → PyTorchWhisperModel with CUDA
    3. CPU → PyTorchWhisperModel (fallback)
    """
    if device_preference == "npu" and OPENVINO_AVAILABLE:
        return OpenVINOWhisperModel(...)
    elif torch.cuda.is_available():
        return PyTorchWhisperModel(device="cuda")
    else:
        return PyTorchWhisperModel(device="cpu")
```

**Recommendation:** Implement adapter pattern to unify interfaces.

**Complexity:** MEDIUM (1-2 days)

---

### 🟡 MEDIUM: Session Management Duplication

**Location 1:** `stream_session_manager.py` (402 lines)
- **Purpose:** WebSocket streaming sessions
- **Features:** Session lifecycle, buffer management, VAD state
- **Storage:** In-memory dict

**Location 2:** `reconnection_manager.py` (398 lines)
- **Purpose:** Session persistence for reconnection
- **Features:** Session recovery, message buffering, expiration
- **Storage:** In-memory dict with TTL

**Location 3:** `api_server.py` (lines 242, 999-1027)
- **Purpose:** HTTP session tracking
- **Features:** streaming_sessions dict, session configuration
- **Storage:** Global dict

**Overlap:** All three maintain session_id → config mappings with partial overlap.

**Consolidation Strategy:**

```python
# src/session/unified_session_manager.py
from dataclasses import dataclass
from enum import Enum

class SessionType(Enum):
    HTTP = "http"
    WEBSOCKET = "websocket"
    STREAMING = "streaming"

@dataclass
class UnifiedSession:
    session_id: str
    session_type: SessionType
    config: Dict[str, Any]
    state: SessionState
    created_at: float
    last_activity: float

    # Optional features
    reconnection_token: Optional[str] = None
    buffered_messages: List[Any] = None
    vad_state: Optional[Any] = None

class UnifiedSessionManager:
    """
    Centralized session management for all session types

    Features:
    - Single source of truth for session state
    - Type-specific extensions via composition
    - Automatic cleanup
    - Reconnection support
    """

    def __init__(self):
        self.sessions: Dict[str, UnifiedSession] = {}
        self.http_extension = HTTPSessionExtension()
        self.websocket_extension = WebSocketSessionExtension()
        self.streaming_extension = StreamingSessionExtension()

    def create_session(self, session_type: SessionType, **kwargs) -> str:
        """Create session with appropriate extensions"""
        ...

    def get_session(self, session_id: str) -> Optional[UnifiedSession]:
        """Retrieve session with all extensions loaded"""
        ...
```

**Recommendation:** Refactor to single UnifiedSessionManager with type-specific extensions.

**Complexity:** MEDIUM-HIGH (2-3 days)

---

### 🟢 ACCEPTABLE: Audio Processing Functions

**_process_audio_data() locations:**
- `api_server.py` (lines 742, 831, 885, 927) - **PRIMARY**
- Not duplicated (only one implementation)

**Status:** Centralized in api_server.py, should move to `audio/audio_processing.py`

**Recommendation:** Extract to dedicated module (30 minutes)

---

### 🟡 MEDIUM: Manager Class Proliferation

**9 Manager Classes Found:**
1. `ModelManager` (model_manager.py) - NPU models
2. `ModelManager` (whisper_service.py) - PyTorch models ⚠️ NAME COLLISION
3. `DomainPromptManager` (domain_prompt_manager.py) - Prompt templates
4. `TranscriptManager` (transcript_manager.py) - Transcript storage
5. `ConnectionManager` (connection_manager.py) - WebSocket connections
6. `HeartbeatManager` (heartbeat_manager.py) - Connection health
7. `ReconnectionManager` (reconnection_manager.py) - Session recovery
8. `StreamSessionManager` (stream_session_manager.py) - Streaming sessions
9. `BufferManager` (buffer_manager.py) - Audio buffering

**Analysis:**
- **Appropriate:** DomainPromptManager, TranscriptManager (clear boundaries)
- **Over-engineered:** Connection/Heartbeat/Reconnection could merge
- **Name Collision:** Two ModelManagers (CRITICAL ISSUE)

**Consolidation Opportunities:**

```python
# BEFORE (3 managers)
ConnectionManager      # WebSocket pool
HeartbeatManager       # Ping/pong
ReconnectionManager    # Session recovery

# AFTER (1 manager)
class WebSocketConnectionManager:
    """
    Unified WebSocket lifecycle management

    Responsibilities:
    - Connection pooling (from ConnectionManager)
    - Health monitoring (from HeartbeatManager)
    - Automatic reconnection (from ReconnectionManager)
    """

    def __init__(self):
        self.connections = ConnectionPool()
        self.heartbeat = HeartbeatMonitor()
        self.recovery = SessionRecovery()
```

**Recommendation:** Merge WebSocket-related managers (Connection, Heartbeat, Reconnection) into single `WebSocketLifecycleManager`.

**Complexity:** MEDIUM (1.5 days)

---

## Architectural Recommendations

### 1. Service Boundaries & Separation of Concerns

**Current Issues:**
- API layer mixed with business logic
- No clear service boundaries
- Global state prevents testability

**Recommended Architecture:**

```
modules/whisper-service/src/
├── api/                         # API Layer (HTTP/WebSocket)
│   ├── http_routes.py          # REST endpoints
│   ├── websocket_routes.py     # SocketIO handlers
│   ├── middleware.py           # Auth, CORS, error handling
│   └── validators.py           # Request validation
│
├── services/                    # Business Logic Layer
│   ├── transcription_service.py
│   ├── streaming_service.py
│   ├── session_service.py
│   └── model_service.py
│
├── models/                      # Model Abstraction Layer
│   ├── base_model.py           # WhisperModel protocol
│   ├── openvino_model.py       # NPU implementation
│   ├── pytorch_model.py        # PyTorch implementation
│   └── model_factory.py        # Factory pattern
│
├── audio/                       # Audio Processing Layer
│   ├── audio_processor.py      # Format conversion, resampling
│   ├── vad_processor.py        # Voice activity detection
│   ├── buffer_manager.py       # Audio buffering
│   └── audio_utils.py          # Utility functions
│
├── session/                     # Session Management Layer
│   ├── unified_session_manager.py
│   ├── session_extensions.py   # HTTP/WebSocket/Streaming
│   └── session_storage.py      # Persistence
│
├── websocket/                   # WebSocket Infrastructure
│   ├── connection_lifecycle.py # Unified Connection/Heartbeat/Reconnection
│   ├── message_router.py       # Existing, keep as-is
│   └── error_handler.py        # WebSocket-specific errors
│
├── streaming/                   # Real-time Streaming Components
│   ├── vac_online_processor.py # Existing, keep as-is
│   ├── continuous_stream_processor.py
│   └── simul_whisper/          # SimulStreaming fork
│
├── context/                     # Context Management
│   ├── rolling_context.py      # Token buffer management
│   ├── domain_prompt_manager.py
│   └── context_carryover.py
│
├── monitoring/                  # Performance & Observability
│   ├── performance_monitor.py
│   ├── metrics_collector.py
│   └── health_checks.py
│
└── utils/                       # Shared Utilities
    ├── audio_errors.py         # Existing, keep as-is
    └── common.py
```

**Key Principles:**
1. **Layered Architecture:** API → Services → Models → Infrastructure
2. **Dependency Inversion:** Services depend on abstractions (protocols), not implementations
3. **Single Responsibility:** Each module has ONE clear purpose
4. **Testability:** Pure business logic isolated from Flask/SocketIO

---

### 2. Dependency Flow & Circular Dependencies

**Current Dependency Graph (Simplified):**

```
api_server.py
    ↓ imports
whisper_service.py (WhisperService, ModelManager)
    ↓ imports
vac_online_processor.py
    ↓ imports
whisper_service.py (ModelManager)  ⚠️ CIRCULAR
```

**Circular Dependency Example:**
```python
# vac_online_processor.py line 916
from whisper_service import ModelManager

# whisper_service.py line 34
from vac_online_processor import VACOnlineASRProcessor
```

**Resolution:**
1. Extract `ModelManager` to separate `models/` package
2. Both `whisper_service.py` and `vac_online_processor.py` import from `models.model_factory`
3. Break circular reference

**Recommended Dependency Flow:**

```
[API Layer]
api_server.py
    ↓
[Service Layer]
transcription_service.py
    ↓
[Model Layer]
model_factory.py → openvino_model.py / pytorch_model.py
    ↓
[Processing Layer]
vac_online_processor.py
```

**No upward dependencies allowed** (enforce with import linter)

---

### 3. Scalability Concerns

**What Would Break Under Load:**

#### 🔴 CRITICAL: Global State

**Problem:**
```python
# api_server.py
whisper_service: Optional[WhisperService] = None  # GLOBAL
stateful_whisper = None  # GLOBAL
stateful_whisper_lock = threading.Lock()  # GLOBAL
streaming_sessions: Dict[str, Dict] = {}  # GLOBAL
```

**Impact:**
- Cannot run multiple worker processes (gunicorn)
- No horizontal scaling
- Session state lost on restart

**Solution:**
```python
# Use dependency injection
class WhisperAPI:
    def __init__(self,
                 transcription_service: TranscriptionService,
                 session_manager: SessionManager):
        self.transcription_service = transcription_service
        self.session_manager = session_manager

# With Redis for distributed sessions
class RedisSessionManager(SessionManager):
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
```

---

#### 🟡 MEDIUM: In-Memory Buffering

**Problem:**
```python
# stream_session_manager.py
self.audio_buffer = deque(maxlen=self.max_buffer_size)  # RAM only
```

**Impact at Scale:**
- 1000 concurrent sessions × 30s buffer × 16kHz × 4 bytes = 1.92 GB RAM
- Memory exhaustion with high concurrency

**Solution:**
- Streaming audio to Redis/disk for sessions > 10s
- Implement buffer size limits per session
- Add memory pressure monitoring

---

#### 🟡 MEDIUM: Thread Pool Saturation

**Problem:**
```python
# api_server.py
audio_pool = AudioProcessingPool(max_workers=4)  # Fixed size
```

**Impact:**
- Queue buildup with >4 concurrent requests
- No backpressure mechanism

**Solution:**
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Use async queue with backpressure
class AdaptiveAudioPool:
    def __init__(self, min_workers=4, max_workers=32):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="audio"
        )
        self.semaphore = asyncio.Semaphore(max_workers)

    async def submit(self, func, *args):
        async with self.semaphore:  # Backpressure
            return await asyncio.wrap_future(
                self.executor.submit(func, *args)
            )
```

---

#### 🟢 ACCEPTABLE: Model Caching

**Current:**
```python
self.pipelines: Dict[str, Any] = {}  # LRU cache (max 3 models)
```

**Status:** Reasonable for single-instance deployment

**Improvement for Scale:**
- Shared model cache across workers (Redis)
- Model server pattern (separate GPU process)

---

### 4. Separation of Concerns Violations

**Examples:**

#### Example 1: api_server.py mixing audio processing
```python
# BEFORE: api_server.py line 742
def _process_audio_data(audio_data, enhance=False):
    """Audio format conversion logic in API layer"""
    ...
    # 60+ lines of audio processing

# AFTER: audio/audio_processor.py
class AudioProcessor:
    def process(self, audio_data, enhance=False):
        """Dedicated audio processing service"""
        ...

# api_server.py
audio_processor = AudioProcessor()
audio_array = audio_processor.process(audio_data)
```

---

#### Example 2: whisper_service.py containing ModelManager
```python
# BEFORE: whisper_service.py (2,392 lines)
class WhisperService:
    ...

class ModelManager:  # 420 lines embedded
    ...

# AFTER:
# services/transcription_service.py
class TranscriptionService:
    def __init__(self, model: WhisperModel):
        self.model = model

# models/pytorch_model.py
class PyTorchWhisperModel:
    """Extracted ModelManager"""
```

---

#### Example 3: api_server.py with performance monitoring
```python
# BEFORE: api_server.py lines 97-208
class AudioProcessingPool:
    ...
class MessageQueue:
    ...
class PerformanceMonitor:
    ...

# AFTER: monitoring/performance_monitor.py
# Move all monitoring classes to dedicated module
```

---

## Risk Assessment

### Critical Risks (Immediate Action Required)

#### 1. **Duplicate ModelManager Classes** 🔴
- **Severity:** CRITICAL
- **Impact:** Name collision causes import ambiguity, hard to switch between NPU/PyTorch
- **Likelihood:** Currently happening (100%)
- **Mitigation:** Extract to separate modules, create abstraction layer
- **Timeline:** 2 days

#### 2. **3,642-line Monolithic API Server** 🔴
- **Severity:** CRITICAL
- **Impact:** Cannot scale horizontally, testing is difficult, changes are risky
- **Likelihood:** Will block future features (90%)
- **Mitigation:** Gradual extraction (start with monitoring, then audio processing)
- **Timeline:** 4-6 days (phased approach)

#### 3. **Global State Preventing Horizontal Scaling** 🔴
- **Severity:** CRITICAL
- **Impact:** Cannot use gunicorn workers, no load balancing
- **Likelihood:** Scaling attempts will fail (100%)
- **Mitigation:** Dependency injection + Redis session storage
- **Timeline:** 3-4 days

---

### High Risks (Plan Required)

#### 4. **Circular Dependencies** 🟡
- **Severity:** HIGH
- **Impact:** Maintenance complexity, refactoring difficulty
- **Likelihood:** Will worsen over time (70%)
- **Mitigation:** Extract ModelManager, enforce layered architecture
- **Timeline:** 1-2 days

#### 5. **Session Management Fragmentation** 🟡
- **Severity:** HIGH
- **Impact:** State inconsistency across HTTP/WebSocket, reconnection bugs
- **Likelihood:** Currently causing issues (60%)
- **Mitigation:** Unified session manager
- **Timeline:** 2-3 days

#### 6. **Memory Pressure with Concurrent Sessions** 🟡
- **Severity:** HIGH
- **Impact:** OOM crashes at scale
- **Likelihood:** Under high load (80%)
- **Mitigation:** Buffer size limits, Redis offloading, monitoring
- **Timeline:** 1-2 days

---

### Medium Risks (Monitor)

#### 7. **Manager Class Proliferation** 🟡
- **Severity:** MEDIUM
- **Impact:** Increased cognitive load, unclear ownership
- **Likelihood:** Will slow development (50%)
- **Mitigation:** Consolidate WebSocket managers
- **Timeline:** 1.5 days

#### 8. **Thread Pool Saturation** 🟡
- **Severity:** MEDIUM
- **Impact:** Request queueing, timeout errors
- **Likelihood:** Under burst traffic (40%)
- **Mitigation:** Adaptive thread pool, backpressure
- **Timeline:** 1 day

---

## Recommended Refactoring Roadmap

### Phase 1: Critical Fixes (Week 1)

**Priority 1: Model Abstraction** (2 days)
- [ ] Extract `ModelManager` from `whisper_service.py` → `models/pytorch_model.py`
- [ ] Rename `model_manager.py` → `models/openvino_model.py`
- [ ] Create `models/base_model.py` with `WhisperModel` protocol
- [ ] Implement `models/model_factory.py`
- [ ] Update all imports

**Priority 2: Extract Performance Monitoring** (1 day)
- [ ] Move `AudioProcessingPool`, `MessageQueue`, `PerformanceMonitor` → `monitoring/`
- [ ] Update `api_server.py` imports
- [ ] Test performance monitoring in isolation

**Priority 3: Session Management Consolidation** (2 days)
- [ ] Create `session/unified_session_manager.py`
- [ ] Migrate `stream_session_manager.py` → extension
- [ ] Migrate `reconnection_manager.py` → extension
- [ ] Update `api_server.py` to use unified manager

---

### Phase 2: API Layer Decomposition (Week 2)

**Extract HTTP Routes** (1 day)
- [ ] `api/http_routes.py` - Transcription, model, health endpoints
- [ ] `api/validators.py` - Request validation
- [ ] Update `api_server.py` to register blueprints

**Extract WebSocket Routes** (1 day)
- [ ] `api/websocket_routes.py` - SocketIO event handlers
- [ ] `websocket/connection_lifecycle.py` - Merge Connection/Heartbeat/Reconnection
- [ ] Update `api_server.py`

**Extract Middleware** (0.5 days)
- [ ] `api/middleware.py` - Auth, CORS, error handlers
- [ ] Update `api_server.py`

---

### Phase 3: Business Logic Extraction (Week 3)

**Service Layer** (2 days)
- [ ] `services/transcription_service.py` - Core WhisperService logic
- [ ] `services/streaming_service.py` - Real-time streaming coordination
- [ ] `services/session_service.py` - Session lifecycle
- [ ] Inject dependencies (no global state)

**Audio Processing** (1 day)
- [ ] `audio/audio_processor.py` - Extract `_process_audio_data()`
- [ ] `audio/audio_utils.py` - Format conversion helpers
- [ ] Update all callers

---

### Phase 4: Horizontal Scaling Enablement (Week 4)

**Dependency Injection** (2 days)
- [ ] Remove global `whisper_service`
- [ ] Remove global `streaming_sessions`
- [ ] Implement service container pattern
- [ ] Update initialization flow

**Redis Session Storage** (1 day)
- [ ] Implement `session/redis_storage.py`
- [ ] Migrate in-memory sessions to Redis
- [ ] Add session TTL management

**Testing** (2 days)
- [ ] Unit tests for extracted services
- [ ] Integration tests for multi-worker setup
- [ ] Load testing with gunicorn workers

---

## Metrics & Success Criteria

**Before Refactoring:**
- `api_server.py`: 3,642 lines
- ModelManager: 2 implementations, no abstraction
- Session managers: 3 separate implementations
- Test coverage: ~40% (estimated)
- Horizontal scaling: Not possible

**After Refactoring (Target):**
- `api_server.py`: <800 lines (78% reduction)
- ModelManager: 1 abstraction, 2 implementations
- Session managers: 1 unified manager
- Test coverage: >80%
- Horizontal scaling: Enabled (gunicorn workers)

**Code Quality Metrics:**
- Cyclomatic complexity: <10 per function
- Class size: <500 lines
- Dependency depth: <4 layers
- No circular dependencies (enforced)

---

## Conclusion

The whisper-service module requires **significant architectural refactoring** to address:

1. **Monolithic file explosion** (3,642-line API server)
2. **Duplicate implementations** (2 ModelManagers)
3. **Global state preventing scaling**
4. **Session management fragmentation**

**Recommended Action:** Implement **phased refactoring roadmap** over 4 weeks with focus on:
- Week 1: Model abstraction + performance extraction
- Week 2: API layer decomposition
- Week 3: Business logic extraction
- Week 4: Horizontal scaling enablement

**ROI:** 4 weeks investment → 10x maintainability improvement, horizontal scaling unlocked, 50% reduction in future feature development time.

**Risk if Deferred:** Technical debt will compound, making future refactoring exponentially more expensive (estimated 3-6 months if delayed 1 year).
