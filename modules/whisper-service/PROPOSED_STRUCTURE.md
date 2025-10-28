# Professional Directory Structure Proposal

**Goal**: Clean, maintainable, self-documenting structure
**Principle**: Group by functionality, not by implementation detail
**Naming**: Clear, professional terms (no "simul", "online", etc.)

---

## Proposed Structure

```
whisper-service/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Service entry point
│   │
│   ├── server/                    # API Layer (WebSocket + REST)
│   │   ├── __init__.py
│   │   ├── app.py                 # Flask + SocketIO application
│   │   ├── routes.py              # REST API endpoints
│   │   ├── websocket.py           # WebSocket event handlers
│   │   ├── middleware.py          # Auth, logging, metrics
│   │   └── errors.py              # Error handling utilities
│   │
│   ├── core/                      # Core Business Logic
│   │   ├── __init__.py
│   │   ├── service.py             # Main WhisperService class
│   │   ├── streaming.py           # Streaming inference engine
│   │   ├── inference.py           # Model inference logic
│   │   └── response.py            # Response formatting
│   │
│   ├── models/                    # Model Management
│   │   ├── __init__.py
│   │   ├── manager.py             # Model loading/caching
│   │   ├── factory.py             # Model factory pattern
│   │   ├── pytorch.py             # PyTorch Whisper implementation
│   │   ├── openvino.py            # OpenVINO NPU implementation
│   │   └── base.py                # Base model interface
│   │
│   ├── audio/                     # Audio Processing
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Resampling, normalization
│   │   ├── vad.py                 # Voice Activity Detection
│   │   ├── buffers.py             # Audio buffer management
│   │   └── utils.py               # Audio utilities
│   │
│   ├── streaming/                 # Real-Time Streaming Components
│   │   ├── __init__.py
│   │   ├── processor.py           # Main streaming processor (VAC replacement)
│   │   ├── decoder.py             # Beam search decoder
│   │   ├── attention.py           # Attention-based policies
│   │   └── chunking.py            # Audio chunking strategies
│   │
│   ├── multilang/                 # Multi-Language Support
│   │   ├── __init__.py
│   │   ├── detection.py           # Language identification
│   │   ├── code_switching.py      # Code-switching logic
│   │   ├── tokenization.py        # Language-specific tokenizers
│   │   └── prompts.py             # Domain/language prompts
│   │
│   ├── transcription/             # Transcription Post-Processing
│   │   ├── __init__.py
│   │   ├── segmentation.py        # Sentence segmentation
│   │   ├── timestamps.py          # Timestamp alignment
│   │   ├── stability.py           # Stability tracking
│   │   ├── text_analysis.py       # Hallucination detection
│   │   └── formatting.py          # Output formatting
│   │
│   ├── session/                   # Session Management
│   │   ├── __init__.py
│   │   ├── manager.py             # Session lifecycle
│   │   ├── state.py               # Per-session state
│   │   └── persistence.py         # Session persistence
│   │
│   ├── connection/                # Connection Management
│   │   ├── __init__.py
│   │   ├── manager.py             # Connection pooling
│   │   ├── heartbeat.py           # Health monitoring
│   │   ├── reconnection.py        # Reconnection handling
│   │   └── routing.py             # Message routing
│   │
│   ├── config/                    # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py            # Service configuration
│   │   ├── models.py              # Model configurations
│   │   └── loader.py              # Config file loading
│   │
│   ├── utils/                     # Shared Utilities
│   │   ├── __init__.py
│   │   ├── logging.py             # Logging setup
│   │   ├── errors.py              # Error definitions
│   │   ├── metrics.py             # Performance metrics
│   │   └── validation.py          # Input validation
│   │
│   └── integration/               # External Integration
│       ├── __init__.py
│       ├── orchestration.py       # Orchestration service client
│       ├── translation.py         # Translation service client
│       └── diarization.py         # Speaker diarization
│
├── tests/
│   ├── unit/                      # Unit tests by module
│   ├── integration/               # Integration tests
│   └── fixtures/                  # Test data
│
├── legacy/                        # Working legacy implementations
│   ├── README.md
│   └── *.py
│
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   └── DEPLOYMENT.md
│
└── scripts/
    ├── warmup.py
    └── benchmark.py
```

---

## Key Design Decisions

### 1. **server/** - Clean API Layer
**Why**: Separates HTTP/WebSocket handling from business logic
**Contents**:
- `app.py` - Flask/SocketIO setup
- `websocket.py` - Socket event handlers (join_session, transcribe_stream)
- `routes.py` - REST endpoints (/transcribe, /health, /models)
- `middleware.py` - Auth, metrics, logging
- `errors.py` - HTTP error responses

**No more**: Giant `api_server.py` (3600 lines) doing everything

---

### 2. **core/** - Business Logic
**Why**: Core transcription logic independent of transport layer
**Contents**:
- `service.py` - Main `WhisperService` class (orchestration)
- `streaming.py` - Streaming inference coordinator
- `inference.py` - Model inference calls
- `response.py` - Response formatting

**Replaces**: Scattered logic between `whisper_service.py` and `api_server.py`

---

### 3. **streaming/** - Real-Time Processing
**Why**: All streaming-specific logic in one place
**Contents**:
- `processor.py` - Main streaming processor (replaces `vac_online_processor.py`)
- `decoder.py` - Beam search (replaces `beam_decoder.py`, `alignatt_decoder.py`)
- `attention.py` - Attention-based policies (AlignAtt)
- `chunking.py` - Audio chunking strategies

**Replaces**:
- `vac_online_processor.py` (rename to `processor.py`)
- `continuous_stream_processor.py` (dead code?)
- `simul_whisper/` folder (merge into proper locations)

**No more**: "simul", "online", "VAC" naming

---

### 4. **multilang/** - Multi-Language Support
**Why**: Code-switching is a first-class feature
**Contents**:
- `detection.py` - Language ID (from Whisper)
- `code_switching.py` - Code-switching logic (sustained detection, etc.)
- `tokenization.py` - Multi-language tokenizers
- `prompts.py` - Domain/language-specific prompts

**Replaces**:
- `text_language_detector.py`
- `sliding_lid_detector.py`
- `domain_prompt_manager.py`
- Parts of `simul_whisper/simul_whisper.py`

**No more**: Scattered language detection files

---

### 5. **audio/** - Audio Processing
**Why**: All audio manipulation in one place
**Contents**:
- `preprocessing.py` - Resampling, normalization (from `audio/audio_utils.py`)
- `vad.py` - Voice Activity Detection (consolidate all VAD files)
- `buffers.py` - Audio buffer management
- `utils.py` - Helper functions

**Replaces**:
- `audio_processor.py` (root)
- `audio/vad_processor.py`
- `vad_detector.py`
- `silero_vad_iterator.py`
- `buffer_manager.py` (both copies)

**No more**: 3-4 VAD files in different locations

---

### 6. **transcription/** - Post-Processing
**Why**: Clear separation from real-time streaming
**Contents**:
- Keep existing: `segmentation.py`, `timestamps.py`, `text_analysis.py`
- Add: `stability.py` (from `stability_tracker.py`)
- Add: `formatting.py` (from `orchestration/response_formatter.py`)
- Remove: Duplicate `buffer_manager.py`

**Principle**: Final output processing, not real-time logic

---

### 7. **connection/** - WebSocket Infrastructure
**Why**: Connection management is complex enough to deserve own module
**Contents**:
- `manager.py` (from `connection_manager.py`)
- `heartbeat.py` (from `heartbeat_manager.py`)
- `reconnection.py` (from `reconnection_manager.py`)
- `routing.py` (from `message_router.py`)

**No more**: Root-level connection files

---

### 8. **models/** - Model Management
**Why**: Already well-organized
**Keep as-is**:
- `manager.py` (from `pytorch_manager.py`)
- `factory.py`
- `base.py`
- `openvino.py`
- `pytorch.py`

**Fix**: Add `current_model` property

---

### 9. **session/** - Session Management
**Why**: Already clean
**Keep**:
- `manager.py` (from `session/session_manager.py`)
- `state.py` (new - per-session state)

**Remove**: `stream_session_manager.py` (duplicate?)

---

### 10. **integration/** - External Services
**Why**: Clear boundary for external dependencies
**New module** for:
- Orchestration service client
- Translation service client
- Speaker diarization client

---

## File Mapping: Current → Proposed

### Root Level (33 files → 1 file)

| Current | Proposed | Notes |
|---------|----------|-------|
| `main.py` | `main.py` | Keep |
| `api_server.py` | `server/app.py` + `server/websocket.py` | Split 3600 lines |
| `whisper_service.py` | `core/service.py` | Rename |
| `vac_online_processor.py` | `streaming/processor.py` | Rename, no "VAC" |
| `continuous_stream_processor.py` | **DELETE** | Dead code |
| `alignatt_decoder.py` | `streaming/decoder.py` | Merge |
| `beam_decoder.py` | `streaming/decoder.py` | Merge |
| `audio_processor.py` | `audio/preprocessing.py` | Move to audio/ |
| `vad_detector.py` | `audio/vad.py` | Consolidate |
| `silero_vad_iterator.py` | `audio/vad.py` | Consolidate |
| `buffer_manager.py` | `audio/buffers.py` | Move |
| `text_language_detector.py` | `multilang/detection.py` | Move |
| `sliding_lid_detector.py` | `multilang/code_switching.py` | Move |
| `domain_prompt_manager.py` | `multilang/prompts.py` | Move |
| `token_deduplicator.py` | `streaming/processor.py` | Merge |
| `utf8_boundary_fixer.py` | `utils/encoding.py` | Move |
| `eow_detection.py` | `streaming/decoder.py` | Merge |
| `segment_timestamper.py` | `transcription/timestamps.py` | Move |
| `sentence_segmenter.py` | `transcription/segmentation.py` | Move |
| `stability_tracker.py` | `transcription/stability.py` | Move |
| `transcript_manager.py` | `transcription/formatting.py` | Move |
| `speaker_diarization.py` | `integration/diarization.py` | Move |
| `pipeline_integration.py` | **DELETE** | Unclear purpose |
| `connection_manager.py` | `connection/manager.py` | Move |
| `heartbeat_manager.py` | `connection/heartbeat.py` | Move |
| `reconnection_manager.py` | `connection/reconnection.py` | Move |
| `message_router.py` | `connection/routing.py` | Move |
| `simple_auth.py` | `server/middleware.py` | Merge |
| `error_handler.py` | `utils/errors.py` | Move |
| `websocket_stream_server.py` | **DELETE** | Duplicate of api_server? |
| `stream_session_manager.py` | **DELETE** | Duplicate |
| `token_buffer.py` | `streaming/buffers.py` | Merge |

### simul_whisper/ Folder → streaming/ + multilang/

| Current | Proposed | Notes |
|---------|----------|-------|
| `simul_whisper/simul_whisper.py` | `streaming/processor.py` + `multilang/code_switching.py` | Split |
| `simul_whisper/beam.py` | `streaming/decoder.py` | Merge |
| `simul_whisper/eow_detection.py` | `streaming/decoder.py` | Merge (duplicate) |
| `simul_whisper/generation_progress.py` | `streaming/decoder.py` | Merge |
| `simul_whisper/config.py` | `config/models.py` | Move |
| `simul_whisper/whisper/` | `models/whisper/` | Move to models |

### audio/ Folder → Expanded

| Current | Proposed | Notes |
|---------|----------|-------|
| `audio/audio_utils.py` | `audio/preprocessing.py` | Rename |
| `audio/vad_processor.py` | `audio/vad.py` | Consolidate |

### transcription/ Folder → Cleaned

| Current | Proposed | Notes |
|---------|----------|-------|
| `transcription/request_models.py` | `server/models.py` | Move (API models) |
| `transcription/result_parser.py` | `transcription/formatting.py` | Rename |
| `transcription/text_analysis.py` | `transcription/text_analysis.py` | Keep |
| `transcription/domain_prompt_helper.py` | `multilang/prompts.py` | Move |
| `transcription/buffer_manager.py` | **DELETE** | Duplicate |

### Other Folders - Keep Structure

| Current | Proposed | Notes |
|---------|----------|-------|
| `models/*` | `models/*` | Keep (add `current_model` property) |
| `config/*` | `config/*` | Keep |
| `session/*` | `session/*` | Keep |
| `orchestration/*` | `integration/*` | Rename folder |
| `utils/*` | `utils/*` | Expand |

---

## Benefits of New Structure

### 1. **Self-Documenting**
- `multilang/` - Obviously handles multi-language
- `streaming/` - Obviously handles real-time
- `server/` - Obviously the API layer
- `core/` - Obviously the main logic

### 2. **No Confusing Names**
- ❌ `simul_whisper` - What does "simul" mean?
- ❌ `vac_online_processor` - What is "VAC"?
- ❌ `text_language_detector` vs `sliding_lid_detector` - What's the difference?
- ✅ `streaming/processor` - Clear!
- ✅ `multilang/detection` - Clear!

### 3. **No Duplicates**
- One `buffer_manager` in `audio/buffers.py`
- One `eow_detection` in `streaming/decoder.py`
- One VAD implementation in `audio/vad.py`
- One decoder in `streaming/decoder.py`

### 4. **Logical Grouping**
- All audio processing together
- All streaming logic together
- All multi-language together
- Clear separation of concerns

### 5. **Easier Testing**
```python
from whisper_service.streaming import StreamingProcessor
from whisper_service.multilang import CodeSwitchingHandler
from whisper_service.audio import VAD
```
Clear imports, easy to mock

---

## Migration Strategy

### Phase 1: Critical Fixes (Do First)
1. Fix the 3 critical regressions IN PLACE
2. Add `current_model` property
3. Test everything works

### Phase 2: Gradual Refactoring (After Fixes)
1. Create new folder structure (empty)
2. Move files one module at a time:
   - Start with `connection/` (least dependencies)
   - Then `audio/`
   - Then `multilang/`
   - Then `streaming/`
   - Finally `server/` (most dependencies)
3. Update imports incrementally
4. Run tests after each module migration
5. Delete old files only after new structure verified

### Phase 3: Documentation
1. Update all import paths
2. Document new structure in `docs/ARCHITECTURE.md`
3. Create migration guide for external users

---

## File Count Reduction

| Location | Current | Proposed | Reduction |
|----------|---------|----------|-----------|
| Root level | 33 | 1 | -32 |
| Folders | 9 | 10 | +1 |
| Total files | 76 | ~50 | -26 |
| Duplicates | 8 | 0 | -8 |

**Net result**: ~26 fewer files, 0 duplicates, 10 well-organized modules

---

## Next Steps

1. **Review this proposal** - Get approval on structure
2. **Apply critical fixes** - Fix regressions FIRST
3. **Create skeleton** - Empty folders with `__init__.py`
4. **Migrate gradually** - One module at a time
5. **Update tests** - After each migration
6. **Document** - Final architecture docs

---

## Questions to Answer

1. ✅ **Naming**: Professional, clear terms?
2. ✅ **Grouping**: Logical separation of concerns?
3. ✅ **Duplicates**: All identified and removed?
4. ✅ **Dependencies**: Can we migrate incrementally?
5. ❓ **Impact**: Any external services importing these files?
6. ❓ **Timeline**: Fix bugs first, or refactor immediately?

---

## Recommendation

**DO NOT REFACTOR NOW**

1. Apply critical fixes to current structure
2. Verify code-switching works
3. THEN gradually migrate to new structure
4. Test continuously during migration

**Why**: Refactoring while broken = high risk of introducing more bugs
