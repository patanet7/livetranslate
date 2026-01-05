# Whisper Service Multi-Language Architecture Analysis

## Executive Summary

The whisper-service module has a **SESSION-ISOLATED ARCHITECTURE** that supports per-session language configuration. However, there are several architectural patterns and design decisions that impact how language is managed and whether concurrent multi-language sessions can work effectively.

**Key Finding**: Language is NOT globally locked, but rather managed at the SESSION level within the code, while rolling context, model loading, and VAD state are properly isolated per session.

---

## 1. SESSION MANAGEMENT ARCHITECTURE

### 1.1 Session Data Structure

Location: `src/api_server.py` (lines 999-1027) and `src/whisper_service.py` (lines 867-901)

**Streaming Sessions Dictionary** (API Server):
```python
streaming_sessions: Dict[str, Dict] = {}

# Each session contains:
streaming_config = {
    "session_id": session_id,              # Unique session ID
    "model_name": "large-v3-turbo",        # Per-session model
    "language": None,                      # Per-session LANGUAGE ✓
    "buffer_duration": 6.0,
    "inference_interval": 3.0,
    "enable_vad": True,
    "created_at": datetime.now().isoformat()
}
```

**SessionManager** (WhisperService):
```python
class SessionManager:
    sessions: Dict[str, Dict] = {}  # session_id → session metadata
    
    # Session config can include:
    session_config = {
        "session_id": session_id,
        "created_at": datetime,
        "config": config,              # Can include language
        "stats": {...}
    }
```

**Per-Session Audio Buffers** (WhisperService):
```python
self.session_audio_buffers: Dict[str, List[torch.Tensor]] = {}  # session_id → audio chunks
self.session_buffers_lock = threading.Lock()                    # Thread-safe access
```

### 1.2 Session Lifecycle

From `src/api_server.py` (lines 964-1111):

1. **CREATE** (`/stream/configure` or `/api/realtime/start`):
   - Generate unique session_id
   - Store session config in `streaming_sessions[session_id]`
   - Create buffer: `session_audio_buffers[session_id] = []`
   - Language is set at this point

2. **PROCESS** (`/stream/audio` or `/api/realtime/audio`):
   - Retrieve session config: `config = streaming_sessions[session_id]`
   - Add audio to session buffer: `add_audio_chunk(audio_chunk, session_id=session_id)`
   - Session config includes language for transcription

3. **STOP** (`/stream/stop` or `/api/realtime/stop`):
   - Mark session inactive
   - Call `whisper_service.close_session(session_id)`
   - Clean up buffer: `del session_audio_buffers[session_id]`
   - Delete stability tracker: `del session_stability_trackers[session_id]`

### ✅ **Session Isolation Assessment**: 
- **GOOD**: Audio buffers are per-session and isolated
- **GOOD**: Session config including language is per-session
- **GOOD**: VAD state is per-session: `session_vad_states[session_id]`
- **GOOD**: Stability trackers are per-session: `session_stability_trackers[session_id]`

---

## 2. LANGUAGE CONFIGURATION

### 2.1 Where Language is Set

**Primary Location**: API endpoint parameter

```python
# From api_server.py line 1021
"language": data.get('language', streaming_sessions[session_id].get('language'))
```

**Session Creation** (`src/api_server.py` lines 964-992):
```python
def configure_streaming():
    streaming_config = {
        "language": data.get('language'),  # ← Language from request
        # ... other config
    }
    streaming_sessions[session_id] = streaming_config
```

**Session Start** (`src/api_server.py` lines 1030-1044):
```python
def api_realtime_start():
    data = request.get_json() or {}
    config = _ensure_streaming_session(data)  # Gets/creates session
    # Language was set during create if provided
```

### 2.2 Language During Transcription

From `src/whisper_service.py` (lines 1593-1606):

```python
stream_request = TranscriptionRequest(
    audio_data=full_audio,
    model_name=request.model_name,
    language=request.language,              # ← Language from session
    session_id=session_id,
    # ...
)
```

From `src/whisper_service.py` (lines 682-708):

```python
# Language configuration
if language:
    decode_options["language"] = language
    logger.info(f"[INFERENCE] Language: {language}")

# Task handling
if task == 'translate':
    if target_language.lower() in ['en', 'eng', 'english']:
        decode_options["task"] = "translate"  # Whisper → English only
    else:
        decode_options["task"] = "transcribe"
        # External translation service handles target language
```

### ✅ **Language Configuration Assessment**:
- **GOOD**: Language is per-session, not globally locked
- **GOOD**: Language is passed to each TranscriptionRequest
- **GOOD**: Language is available in decode_options for Whisper model
- **POTENTIAL ISSUE**: No validation to prevent language switching within a session (could break context carryover)

---

## 3. MODEL LOADING & MEMORY FOOTPRINT

### 3.1 Model Loading Architecture

Location: `src/model_manager.py` and `src/whisper_service.py`

**ModelManager (PyTorch version in whisper_service.py)**:
```python
class ModelManager:
    def __init__(self):
        self.models: Dict[str, Any] = {}      # Global model cache
        self.inference_lock = threading.Lock()
        self.request_queue = Queue(maxsize=10)
        self.last_inference_time = 0
        self.min_inference_interval = 0.1  # 100ms minimum
```

**Model Loading Flow**:
```python
def load_model(self, model_name: str):
    if model_name not in self.models:
        logger.info(f"Loading model: {model_name}")
        model = whisper.load_model(
            name=model_name,
            device=self.device,
            download_root=self.models_dir
        )
        self.models[model_name] = model  # Cache globally
    return self.models[model_name]
```

### 3.2 Memory Implications

**Current Design**:
- Models are SHARED across all sessions
- Multiple sessions using different models can load multiple models into memory
- Memory management via LRU eviction: `max_cached_models = 3` (from model_manager.py line 104)

**Language Impact on Memory**:
- Language does NOT affect memory footprint (same Whisper model used for all languages)
- Whisper models are multilingual - one model handles 99 languages
- Model size: `large-v3` ≈ 2.7GB GPU memory

### 3.3 Memory for Context

From `src/token_buffer.py`:
```python
class TokenBuffer:
    def __init__(self):
        self._text = text          # Stored as Python string (minimal memory)
        self.tokenizer = tokenizer # Shared across sessions
        self.prefix_token_ids = [] # Small list
```

Rolling context size is minimal:
- Max context tokens: 223 (from model_manager.py line 225)
- ~600-800 bytes per session for rolling context storage

### ✅ **Model Loading Assessment**:
- **GOOD**: Models cached globally (efficient for multiple sessions)
- **GOOD**: Language doesn't increase memory footprint
- **LIMITATION**: Only 3 models cached simultaneously - concurrent English + Chinese sessions using different models might cause evictions
- **GOOD**: Rolling context is per-session and minimal overhead

---

## 4. AUDIO PROCESSING PIPELINE

### 4.1 Complete Audio Flow

From `src/whisper_service.py`:

**Input → Buffer → Transcription**:
```
1. Frontend sends audio_chunk
   ↓
2. API endpoint: /stream/audio or /api/realtime/audio
   ↓
3. add_audio_chunk(audio_chunk, session_id)
   - VAD pre-filter (optional): filters silence
   - Store in session_audio_buffers[session_id].append(audio_tensor)
   ↓
4. Periodic transcription (inference_interval = 3.0s)
   ↓
5. transcribe_stream(session_id)
   - Retrieve full buffer for THIS session
   - Call model.transcribe(audio, language=..., task=...)
   ↓
6. Stability tracking (per-session StabilityTracker)
   - Identifies stable vs unstable text
   - Deduplicates unchanged segments
   ↓
7. Emit result via WebSocket
```

### 4.2 VAD (Voice Activity Detection)

From `src/whisper_service.py` (lines 1722-1815):

**Per-Session VAD State**:
```python
self.session_vad_states: Dict[session_id] = {}  # 'voice' or 'nonvoice'

# Each session tracks its own VAD state
if enable_vad_prefilter and self.vad is not None:
    vad_result = self.vad.check_speech(audio_chunk)
    # Update: self.session_vad_states[session_id] = 'voice' | 'nonvoice'
```

**VAD Behavior**:
- VAD is OPTIONAL (controlled by `enable_vad_prefilter` flag)
- When enabled, filters silence chunks before adding to buffer
- Per-session state tracking prevents cross-session interference
- VAD is LANGUAGE-INDEPENDENT (uses acoustic features, not language)

### ✅ **Audio Pipeline Assessment**:
- **EXCELLENT**: Per-session buffers prevent cross-contamination
- **EXCELLENT**: VAD is per-session and language-independent
- **GOOD**: Stability tracking is per-session
- **GOOD**: Language changes during transcription (transcribe vs translate task)

---

## 5. TRANSLATION INTEGRATION

### 5.1 Translation Task Configuration

From `src/whisper_service.py` (lines 686-708):

**Task Parameter**:
```python
# From TranscriptionRequest dataclass (line 93)
task: str = "transcribe"              # "transcribe" or "translate"
target_language: str = "en"           # Target language for external translation

# During inference (lines 695-708)
if task == 'translate':
    if target_language.lower() in ['en', 'eng', 'english']:
        # Use Whisper's built-in translation to English
        decode_options["task"] = "translate"
    else:
        # Transcribe and use external translation service
        decode_options["task"] = "transcribe"
        # External service handles: source → target_language
```

### 5.2 Language Lock at Session Level

**Critical Finding**: Language is NOT locked at session creation, but:
1. **Session config stores language**: `streaming_config["language"]`
2. **Each request uses session language**: `language=config.get('language')`
3. **Task (translate/transcribe) can be set per-request**: `request.task`
4. **Target language is flexible**: `target_language=request.target_language`

**Potential Issue**: Rolling context assumes SAME language across consecutive transcriptions:
```python
def append_to_context(self, text: str):
    # Appends previous transcription to context for next inference
    # But rolling_context is per-ModelManager, NOT per-session!
```

### ✅ **Translation Integration Assessment**:
- **GOOD**: Language is session-configurable
- **GOOD**: Task (transcribe vs translate) is flexible
- **⚠️ WARNING**: Rolling context is GLOBAL at ModelManager level, not per-session
  - If two sessions change language between requests, context will be wrong

---

## 6. CONFIGURATION MANAGEMENT

### 6.1 Session Configuration

From `src/api_server.py` (lines 999-1027):

```python
def _ensure_streaming_session(data: Dict[str, Any]) -> Dict[str, Any]:
    if session_id not in streaming_sessions:
        streaming_config = {
            "session_id": session_id,
            "model_name": data.get('model_name', 'large-v3-turbo'),
            "language": data.get('language'),
            "buffer_duration": data.get('buffer_duration', 6.0),
            "inference_interval": data.get('inference_interval', 3.0),
            "enable_vad": data.get('enable_vad', True),
        }
        streaming_sessions[session_id] = streaming_config
```

**Configuration Sources**:
1. Request parameter (highest priority)
2. Session stored config
3. Default value

### 6.2 Device & Model Configuration

From `src/whisper_service.py` (lines 1946-1988):

```python
config = {
    "models_dir": os.getenv("WHISPER_MODELS_DIR", ...),
    "default_model": os.getenv("WHISPER_DEFAULT_MODEL", "large-v3-turbo"),
    "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
    "device": os.getenv("OPENVINO_DEVICE"),
    "orchestration_mode": os.getenv("ORCHESTRATION_MODE", "false").lower() == "true",
}
```

### ✅ **Configuration Assessment**:
- **GOOD**: Per-session configuration is flexible
- **GOOD**: Environment variables provide service-level defaults
- **GOOD**: Request parameters can override session config

---

## 7. CURRENT ARCHITECTURE LIMITATIONS

### 7.1 Global ModelManager Rolling Context

**Location**: `src/whisper_service.py` (lines 220-226)

```python
# GLOBAL at ModelManager level
self.rolling_context = None  # TokenBuffer, initialized by init_context()
self.static_prompt = static_prompt or ""
self.max_context_tokens = max_context_tokens
```

**Problem**:
- Rolling context is PER-ModelManager instance, not per-session
- Only one global rolling context exists for all sessions
- If Session A and Session B both use transcription with context, they share the same rolling context
- Language switching between sessions will corrupt context

**Impact for Multi-Language**:
- Session A transcribes English: "Hello world" → added to rolling context
- Session B transcribes Chinese (same session_id or different): context is now mixed
- Next transcription will have corrupted context

### 7.2 Global Model Cache

**Location**: `src/model_manager.py` (lines 87-88)

```python
self.pipelines: Dict[str, Any] = {}  # Global model cache
self.models: Dict[str, Any] = {}     # Global model cache (PyTorch)
```

**Implication**:
- Models are shared across sessions (efficient)
- But if two sessions load different models simultaneously, only 3 are kept in memory
- Third session loading another model causes LRU eviction
- Not a language problem specifically, but a resource constraint

### 7.3 Language Codec/Tokenizer

**Location**: `src/whisper_service.py` (lines 364-366)

```python
tokenizer = whisper.tokenizer.get_tokenizer(
    multilingual=model.is_multilingual
)
```

**Finding**:
- Tokenizer is retrieved per-request (good)
- Whisper's tokenizer is MULTILINGUAL by default
- Can handle 99 languages with same tokenizer
- No language locking observed

### 7.4 No Session-Level Rolling Context

**Current Design**: Rolling context is at ModelManager level
- Could affect concurrent sessions if using context carryover
- Session-specific context needs to be implemented for true multi-language support

---

## 8. REQUIREMENTS FOR CONCURRENT MULTI-LANGUAGE SESSIONS

### 8.1 English Session Example
```
Session ID: "en-session-001"
Config: {
    "language": "en",
    "model_name": "large-v3-turbo",
    "task": "transcribe"
}

Audio Flow:
chunk1 → session_audio_buffers["en-session-001"]
chunk2 → session_audio_buffers["en-session-001"]
combined → transcribe(..., language="en", task="transcribe")
result: "Hello, how are you?"
```

### 8.2 Chinese Session Example
```
Session ID: "zh-session-001"
Config: {
    "language": "zh",
    "model_name": "large-v3-turbo",
    "task": "transcribe"
}

Audio Flow:
chunk1 → session_audio_buffers["zh-session-001"]
chunk2 → session_audio_buffers["zh-session-001"]
combined → transcribe(..., language="zh", task="transcribe")
result: "你好，你好吗？"
```

### 8.3 What Works Currently ✅
- Per-session audio buffers (isolated)
- Per-session VAD state (isolated)
- Per-session stability trackers (isolated)
- Per-session session configuration (language configurable)
- Language parameter in transcription request
- Whisper model is multilingual (handles any language)

### 8.4 What Needs Changes ⚠️
- **Session-specific rolling context** (currently global at ModelManager)
- **Session-specific model tokenizer** (currently loaded globally)
- **Validation to prevent language switching** within same session

---

## 9. CHANGES NEEDED FOR FULL MULTI-LANGUAGE SUPPORT

### 9.1 Session-Level Rolling Context

**Current Code** (whisper_service.py line 226):
```python
self.rolling_context = None  # Global
```

**Should Be**:
```python
self.session_rolling_contexts: Dict[str, TokenBuffer] = {}  # Per-session

def get_rolling_context(self, session_id: str) -> TokenBuffer:
    if session_id not in self.session_rolling_contexts:
        self.session_rolling_contexts[session_id] = TokenBuffer(...)
    return self.session_rolling_contexts[session_id]
```

**Impact**: Each session maintains its own context without cross-contamination.

### 9.2 Language Consistency Validation

**New Method**:
```python
def validate_language_consistency(self, session_id: str, new_language: str) -> bool:
    session = self.sessions.get(session_id)
    if session and session.get("config", {}).get("language"):
        if session["config"]["language"] != new_language:
            logger.warning(f"Language mismatch: {session['config']['language']} → {new_language}")
            return False
    return True
```

**Implementation Point**: Before transcription request, verify language matches session config.

### 9.3 Session Metadata Enhancement

**Extend SessionManager**:
```python
session = {
    "session_id": session_id,
    "config": {
        "language": "en",  # Store language in config
        "model": "large-v3-turbo"
    },
    "metadata": {
        "rolling_context": TokenBuffer(...),  # Per-session
        "language_confirmed": True,
        "created_at": datetime
    }
}
```

---

## 10. MEMORY & PERFORMANCE IMPLICATIONS

### 10.1 Memory Footprint: English + Chinese Concurrent Sessions

**Scenario**: 2 concurrent sessions (1 English, 1 Chinese)

**Memory Usage**:
```
Shared:
- Whisper Model (large-v3): 2.7 GB
- Tokenizer: 5 MB
- Service infrastructure: 100 MB
Subtotal: 2.805 GB

Per-Session (English):
- Audio buffer (6 sec @ 16kHz): 192 KB
- Rolling context (223 tokens): 1 KB
- VAD state: 100 bytes
- Stability tracker: 10 KB
Subtotal: ~200 KB

Per-Session (Chinese):
- Audio buffer (6 sec @ 16kHz): 192 KB
- Rolling context (223 tokens): 1 KB
- VAD state: 100 bytes
- Stability tracker: 10 KB
Subtotal: ~200 KB

TOTAL: ~2.805 GB + 400 KB ≈ 2.805 GB
```

**Key Finding**: Language does NOT significantly impact memory. Both English and Chinese use the same Whisper model.

### 10.2 Processing Latency

**Current Flow**:
```
Session 1 (English) chunk1 → transcribe @ 3s → result → stability → emit
Session 2 (Chinese) chunk2 → (queued behind Session 1)
```

**Bottleneck**: Single inference_lock in ModelManager

From `src/model_manager.py` (line 92):
```python
self.inference_lock = threading.RLock()  # Single lock for all sessions
```

**Issue**: Only one session can transcribe at a time
- Session 1 transcribe (2s)
- Wait for lock release
- Session 2 transcribe (2s)
- Total latency: 4s for both sessions

**Mitigation**: Could use per-session inference locks, but would need separate model instances.

### 10.3 Processing Throughput

**Current Limitation**: `max_queue_size = 10` (model_manager.py line 94)

With concurrent sessions:
- Session 1 fills queue with 5 requests
- Session 2 tries to add → waits or queues
- Bottleneck at inference lock

---

## 11. CONCLUSIONS & RECOMMENDATIONS

### 11.1 Current Capability Assessment

| Feature | Status | Details |
|---------|--------|---------|
| Per-Session Audio Buffers | ✅ WORKING | Each session has isolated audio buffer |
| Language Configuration | ✅ WORKING | Language is per-session, not globally locked |
| Concurrent Sessions | ✅ WORKING | Multiple sessions can exist simultaneously |
| Language-Specific Context | ⚠️ LIMITED | Rolling context is global, not per-session |
| Multi-Language Transcription | ✅ WORKING | Whisper model handles all 99 languages |
| Simultaneous English + Chinese | ⚠️ PARTIAL | Works, but context will be shared if enabled |

### 11.2 To Support Concurrent English + Chinese Sessions

**Minimum Changes Required**:

1. **Implement Session-Level Rolling Context** (Priority: HIGH)
   - Move rolling context from ModelManager to per-session tracking
   - Update `append_to_context()` to use session_id
   - Update `get_inference_context()` to retrieve session-specific context

2. **Add Language Validation** (Priority: MEDIUM)
   - Prevent language changes within same session
   - Log warnings for language mismatches
   - Optional: Auto-reset context on language change

3. **Document Language Behavior** (Priority: MEDIUM)
   - Clarify that language is per-session
   - Specify that language should not change during session lifetime
   - Provide examples of concurrent multi-language usage

4. **Consider Per-Session Tokenizers** (Priority: LOW)
   - Currently works fine (tokenizer is language-agnostic)
   - Only needed if future optimizations use language-specific tokenizers

### 11.3 Optional Enhancements

1. **Session-Level Language Lock**:
   ```python
   session_config["language_locked"] = True
   # Prevents accidental language changes
   ```

2. **Per-Session Model Instances**:
   - Would enable true parallel inference
   - Requires 2x model memory (not practical for large models)

3. **Async Inference Processing**:
   - Replace threading.RLock with asyncio locks
   - Enable true parallel inference for different sessions

4. **Language Detection Confidence**:
   - Track detected language during transcription
   - Warn if detected language differs from configured language

### 11.4 Quick Reference: Implementation Locations

| Component | File | Lines |
|-----------|------|-------|
| Session Config | `api_server.py` | 999-1027 |
| Audio Buffers | `whisper_service.py` | 1015-1018 |
| Rolling Context | `whisper_service.py` | 220-226 |
| Language Config | `api_server.py` | 1020-1021 |
| Model Manager | `model_manager.py` | 53-434 |
| Token Buffer | `token_buffer.py` | Full file |
| Session Manager | `whisper_service.py` | 867-901 |
| Transcription | `whisper_service.py` | 1226-1482 |

---

## 12. ARCHITECTURE DIAGRAM

```
┌────────────────────────────────────────────────────────────────┐
│                         API Layer (Flask)                      │
│  /stream/configure  /stream/audio  /stream/stop                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                   WhisperService (Main Class)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SessionManager                                          │  │
│  │  - sessions[session_id] → session_config                │  │
│  │  - Config includes: language, model, enable_vad, etc    │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Per-Session Audio Buffers (✅ Isolated)                │  │
│  │  - session_audio_buffers[session_id] = [audio_chunks]   │  │
│  │  - Thread-safe with session_buffers_lock                │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Per-Session VAD State (✅ Isolated)                    │  │
│  │  - session_vad_states[session_id] = 'voice'/'nonvoice'  │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Per-Session Stability Trackers (✅ Isolated)           │  │
│  │  - session_stability_trackers[session_id]                │  │
│  │  - Tracks stable vs unstable text tokens                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                     ModelManager (Shared)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Models Cache (Global)                                  │  │
│  │  - self.models[model_name] = loaded_model               │  │
│  │  - Shared across all sessions                           │  │
│  │  - Max 3 models cached (LRU eviction)                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Rolling Context (⚠️ Global, NOT per-session)           │  │
│  │  - self.rolling_context = TokenBuffer                   │  │
│  │  - Shared across ALL sessions (LIMITATION)              │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Inference Lock (Single Lock)                           │  │
│  │  - self.inference_lock = threading.RLock()              │  │
│  │  - Only one session can infer at a time                 │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                    Whisper Model (Multilingual)                │
│  - Handles 99 languages with single model                      │
│  - Same model for English and Chinese                          │
│  - No language-specific loading                                │
└────────────────────────────────────────────────────────────────┘
```

---

## FINAL ANSWER: Can It Support Concurrent English + Chinese?

### ✅ YES - Technically Working
- Audio buffers are per-session and isolated
- Language is configurable per-session
- Whisper model handles both languages
- VAD and stability tracking are per-session

### ⚠️ WITH CAVEATS
- **Rolling context**: Global, not per-session (needs refactoring)
- **Inference lock**: Only one session transcribes at a time (design choice)
- **No validation**: Language can change within session (document as not allowed)

### 🎯 IMPLEMENTATION EFFORT
- **Minimal**: 1-2 day effort to make rolling context per-session
- **Recommended**: Also add language validation (1 hour)
- **Optional**: Async inference for parallel transcription (2-3 days)

