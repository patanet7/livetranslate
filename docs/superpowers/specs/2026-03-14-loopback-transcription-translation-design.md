# Real-Time Loopback Transcription & Translation System

**Date:** 2026-03-14
**Status:** Design
**Approach:** Refactor In Place (Approach 1)
**Reviewed by:** MLOps Engineer, Microservices Architect, Media Streaming Specialist, Architect Reviewer

## Overview

Build a real-time loopback system for live meetings: mic + system audio → language-agnostic transcription (best model per language, GPU-accelerated on RTX 4090) → translation → multi-mode display in SvelteKit frontend. Ephemeral by default, promotable to full meeting sessions with continuous recording and persistence — the same meeting pipeline that meeting bots plug into.

## Goals

1. **Pluggable transcription backends** — not just Whisper. Best model per language (SenseVoice, FunASR, faster-whisper, etc.) with auto-detection and user override.
2. **All GPU compute on thomas-pc** (RTX 4090 via Tailscale) — transcription service runs remotely, Ollama/vLLM handles translation (already running on thomas-pc). Local machine handles mic capture, orchestration, and display.
3. **SvelteKit loopback page** — live captions + translation with split/subtitle/transcript display modes.
4. **Unified meeting pipeline** — one recording/persistence system that both loopback sessions and meeting bots feed into.
5. **Translation rolling window** — context-aware translation using recent transcript history for quality.

## Non-Goals

- Meeting bot UI changes (bots already work, just need to plug into unified pipeline)
- SeamlessM4T integration (removed — using separate transcription + translation instead)
- Multi-machine load balancing (single RTX 4090 for now)
- Meeting summary generation (future enhancement)

---

## Architecture

### Deployment Topology

```
Local Machine                          thomas-pc (RTX 4090, Tailscale)
┌─────────────────────┐               ┌──────────────────────────────┐
│ dashboard-service    │               │ transcription-service        │
│ (SvelteKit frontend) │               │ ├─ VAD / chunking            │
│                      │               │ ├─ language detection         │
│ orchestration-service│──Tailscale───▶│ ├─ model registry            │
│ ├─ WebSocket hub     │   (~1-5ms)    │ ├─ backend manager (VRAM)    │
│ ├─ meeting pipeline  │               │ ├─ backend: whisper          │
│ ├─ audio recording   │               │ ├─ backend: sensevoice       │
│ ├─ downsampling      │               │ └─ dedup / rolling window    │
│ ├─ translation module│               │                              │
│ │  ├─ context window │──Tailscale───▶│ Ollama / vLLM (GPU)          │
│ │  ├─ backpressure   │  (HTTP POST)  │ (already running, model      │
│ │  └─ LLM client     │               │  router handles serving)     │
│ └─ session mgmt      │               └──────────────────────────────┘
│                      │
│ PostgreSQL           │
│ (session data)       │
└─────────────────────┘
```

**Translation service absorbed into orchestration.** The previous `modules/translation-service/` was a thin HTTP relay (~150 lines of core logic) between orchestration and Ollama/vLLM. Since thomas-pc already runs Ollama with a model router, orchestration now calls Ollama's OpenAI-compatible API directly. Rolling context, prompt construction, and backpressure all live in orchestration. `modules/translation-service/` is archived.

### Audio Flow

```
Browser mic/system audio (48kHz+ native quality)
    │
    ▼
Orchestration Service (local)
    ├──[meeting mode]──▶ Save original quality to disk (FLAC, 48kHz+)
    ├──▶ Downsample to 16kHz mono (librosa/scipy)
    └──▶ Send 16kHz mono via WebSocket (binary frames) to transcription-service
            │
            ▼
        Transcription Service (thomas-pc)
        VAD → Language detect → Registry lookup → Backend inference
            │
            ▼
        Dedup / rolling window → clean text
            │
            ▼
        Return to orchestration (text frame)
            │
            ▼
        Orchestration: translate only is_final segments
        Translation module calls Ollama/vLLM directly (thomas-pc, HTTP POST)
            ├─ Rolling context window (last N sentences)
            └─ Return translation
            │
            ▼
        Orchestration broadcasts via WebSocket to frontend
            ├─ [meeting mode] persist transcript + translation to DB
            └─ Frontend displays in selected mode
```

### Key Architectural Decisions

**1. Binary WebSocket frames for audio.** Browser sends raw `Float32Array` buffers as binary WebSocket frames to orchestration (not base64-encoded JSON). Text frames used for control messages. Eliminates 33% base64 overhead and GC pressure.

**2. Downsample in orchestration before Tailscale.** Orchestration forks the audio stream: one path saves native quality (48kHz+ stereo FLAC) to disk for recording, the other downsamples to 16kHz mono and forwards to thomas-pc. This reduces inter-machine bandwidth by 6x.

**3. VAD runs on the transcription service.** Audio streams continuously from orchestration to the transcription service. VAD, chunking, and all inference happen on thomas-pc. This keeps VAD co-located with inference (avoids the "two VAD" drift problem), lets each `BackendConfig` control its own VAD threshold, and the bandwidth cost is trivial (~32KB/s for 16kHz mono over Tailscale).

**4. WebSocket for orchestration→transcription link.** Persistent WebSocket connection (not HTTP POST per chunk). Supports bidirectional streaming: audio in, interim/final results out. Replaces the current Socket.IO client with raw WebSocket for lower overhead.

**5. Translate only `is_final` segments.** Interim/draft transcription results are broadcast to the frontend for display but NOT sent to translation. Only finalized segments go to translation to avoid overloading the GPU with redundant translation work. The frontend shows interim captions in original language (italic/faded) and adds translations when they arrive.

**6. Graceful degradation.**
- Transcription service down: continue recording audio (meeting mode), show "transcription unavailable" banner, no live captions. Audio is preserved for batch processing when service recovers.
- Ollama/vLLM down or unreachable: show original-language captions only, with "translation unavailable" indicator. Transcription continues normally.
- Tailscale link drops: transcription service and Ollama both unreachable. Use a single shared circuit breaker for the remote machine. Buffer audio to disk if in meeting mode.

---

## Plan 1: Transcription Service Refactor

### Rename

`modules/whisper-service/` → `modules/transcription-service/`

All references updated: imports, Docker configs, env vars, orchestration client, CLAUDE.md, pyproject.toml.

### Whisper Library Decision

**Decision: Use faster-whisper (CTranslate2) as the Whisper backend, not openai-whisper.**

Rationale: faster-whisper provides 4x+ speedup on GPU via CTranslate2, supports batched inference, and has built-in VAD. The existing SimulStreaming module (`simul_whisper/`) is built against openai-whisper internals and will NOT be ported — it will be replaced by faster-whisper's native streaming capabilities. The token deduplicator and text deduplicator remain (they operate on text output, not model internals).

Impact: `beam_decoder.py`, `alignatt_decoder.py`, and the `simul_whisper/` module are retired. The WhisperBackend adapter wraps faster-whisper's `WhisperModel.transcribe()` directly.

### TranscriptionBackend Protocol

```python
class TranscriptionBackend(Protocol):
    """Any transcription engine implements this."""

    async def transcribe(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> TranscriptionResult: ...

    async def transcribe_stream(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> AsyncIterator[TranscriptionResult]: ...

    def supports_language(self, lang: str) -> bool: ...

    def get_model_info(self) -> ModelInfo: ...

    async def load_model(self, model_name: str, device: str = "cuda") -> None: ...

    async def unload_model(self) -> None: ...

    async def warmup(self) -> None: ...

    def vram_usage_mb(self) -> int: ...
```

Key additions from review:
- `transcribe_stream()` — async generator for producing interim/partial results
- `warmup()` — first-class warm-up to eliminate cold-start latency
- `vram_usage_mb()` — reports current VRAM consumption for the BackendManager

### Shared Types (in `livetranslate-common`)

```python
@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    segments: list[Segment]       # with timestamps
    stable_text: str              # confirmed prefix
    unstable_text: str            # still-forming tail
    is_final: bool                # segment boundary reached
    is_draft: bool                # incremental update
    speaker_id: str | None
    should_translate: bool        # has enough stable text for translation

@dataclass
class ModelInfo:
    name: str
    backend: str
    languages: list[str]
    vram_mb: int
    compute_type: str             # "float16", "int8", etc.
```

### Backend Implementations

Located in `src/backends/`:

| Backend | File | Primary Languages | Notes |
|---------|------|-------------------|-------|
| Whisper (faster-whisper) | `whisper.py` | Universal, strong English | CTranslate2, fastest Whisper variant on GPU |
| SenseVoice | `sensevoice.py` | Chinese, Japanese, Korean, English | FunAudioLLM, strong CJK |
| FunASR (Paraformer) | `funasr.py` | Chinese (Mandarin) | Alibaba DAMO, best Mandarin accuracy |

First implementation: port existing Whisper code into `whisper.py` backend using faster-whisper for GPU performance. Add SenseVoice or FunASR as second backend. Additional backends added by implementing the `TranscriptionBackend` protocol.

### BackendManager (VRAM Budget)

Sits between the ModelRegistry and backend instances. Manages GPU memory on the shared RTX 4090.

```python
class BackendManager:
    max_vram_mb: int = 10000      # Budget for transcription (leave ~14GB for translation)
    loaded_backends: dict[str, TranscriptionBackend]  # keyed by "backend:model"
    lru_order: list[str]          # least-recently-used tracking

    async def get_backend(self, config: BackendConfig) -> TranscriptionBackend:
        """Load and return backend, evicting LRU if over budget."""

    async def evict_lru(self) -> None:
        """Unload least-recently-used backend to free VRAM."""
```

**VRAM budget:** The RTX 4090 has 24GB. Translation model (Qwen 2.5 7B at int4) uses ~8GB. CUDA context overhead ~1GB. That leaves ~15GB for transcription, but we budget conservatively at 10GB to leave headroom for activation memory and KV caches. The BackendManager enforces this:
- Tracks VRAM per loaded backend via `vram_usage_mb()`
- LRU eviction: when loading a new backend would exceed the budget, unload the least-recently-used backend first
- Serializes `load_model()` calls to prevent concurrent loading races

### ModelRegistry

Configurable mapping of language → backend + model + processing parameters. Stored as a YAML config file with a version field. Supports hot-reload via `POST /api/registry/reload` or SIGHUP.

```python
@dataclass
class BackendConfig:
    backend: str           # "whisper", "sensevoice", "funasr"
    model: str             # "large-v3-turbo", "SenseVoiceSmall", etc.
    compute_type: str      # "float16", "int8", "int8_float16"
    chunk_duration_s: float  # Maximum chunk size for this model
    stride_s: float        # Inference stride (chunk_duration - overlap)
    overlap_s: float       # Overlap between chunks for context continuity
    vad_threshold: float   # VAD sensitivity
    beam_size: int         # Beam search width
    prebuffer_s: float     # Minimum audio before first inference (0.3-0.5s for low latency)
    batch_profile: str     # "realtime" or "batch" — controls quality vs latency tradeoffs
```

**`chunk_duration_s` vs `stride_s` clarification:** `chunk_duration_s` is the total audio window fed to the model. `stride_s` is how far the window advances between inferences (`chunk_duration_s - overlap_s`). The overlap carries acoustic context across chunk boundaries. For example: `chunk_duration_s=5.0, overlap_s=0.5` means a 5s window advances by 4.5s between inferences.

**`batch_profile`:** Controls whether the model is configured for real-time streaming (lower latency, smaller chunks, greedy decoding) or batch post-processing (higher quality, longer context, beam search). Post-meeting re-transcription uses batch profile with the full recording.

The registry is the single source of truth for "what model handles what language with what parameters." Changeable without code changes.

### Language Detection (Authoritative LID)

The current `SlidingLIDDetector` is passive/tracking-only and cannot serve as the authoritative signal for registry routing. The refactored LID system:

1. **First chunk:** Use faster-whisper's built-in language detection on the first ~1 second of audio. This produces a high-confidence language code that keys into the registry.
2. **Ongoing:** The `SlidingLIDDetector` continues monitoring for language switches. When it detects a sustained language change (>3 seconds), it triggers a registry re-lookup and potential backend switch.
3. **Language code normalization:** A normalization layer maps from LID output (which may include regional variants like `zh-CN`, `zh-TW`, `yue`) to registry key space (`zh`, `en`, `ja`, etc.).

### Model Selection Flow

```
Audio arrives
    → VAD detects speech
    → Language detector: faster-whisper LID on first chunk
    → Normalize language code (zh-CN → zh)
    → Check: user override? → use specified model
    → Check: registry[detected_language]? → use registered config
    → Fallback: registry["*"] → default model
    → BackendManager.get_backend(config) → load/return (LRU evict if needed)
    → TranscriptionBackend.transcribe() with config params
    → On sustained language change: re-lookup, potentially switch backend
```

### Infrastructure Preserved (Above Backend Layer)

These modules remain, operating on backend output regardless of which engine produced it:

- `token_deduplicator.py` — token-level dedup at chunk boundaries
- `continuous_stream_processor.py` — text-level dedup across inference windows
- `vac_online_processor.py` — VAD chunking (Silero, backend-independent), refactored to implement proper overlap (retain tail of previous chunk's audio buffer instead of clearing entirely)
- `token_buffer.py` — rolling context buffer
- `sentence_segmenter.py` — **refactored to be language-universal** (not just CJK)

**VAC processor improvements:** The current `VACOnlineASRProcessor` clears its buffer entirely after each inference call, losing acoustic context at chunk boundaries. The refactored version:
- Retains the last `overlap_s` seconds of audio in the buffer after inference
- Uses `asyncio.Queue` or ring buffer instead of `is_processing` flag + `pending_chunks` list (eliminates serialization between inference and audio ingestion)
- First inference fires at `prebuffer_s` (0.3-0.5s for fast time-to-first-text), subsequent inferences at `stride_s` intervals

### Sentence Segmenter (Language-Universal)

Current segmenter handles CJK fullwidth punctuation. Refactored to support Latin (`.` `!` `?`) and CJK (`。` `！` `？`) initially. Additional scripts (Arabic, Thai, Devanagari) added as needed via configurable per-language rules — the architecture supports it but initial scope is Latin + CJK.

### VAD/Chunking Strategy

**Decision:** Use the current Silero VAD architecture as the foundation. Integrate faster-whisper's CTranslate2 backend for the Whisper implementation (significant GPU speedup over openai-whisper). Each backend's `BackendConfig` in the registry controls chunking parameters, so different models get model-appropriate chunking without changing the VAD layer.

**Pre-task (first step of Plan 1):** Produce a comparison table of VAD/chunking approaches used by WhisperX (pyannote VAD + forced alignment), faster-whisper (built-in VAD + batched inference), and FasterWhisperX (merged approach). Deliverable: a markdown document in `docs/research/` with a recommendation on which patterns to adopt. This must complete before implementing backend adapters, as it may refine the `BackendConfig` fields. This pre-task also determines the latency benchmark design (streaming time-to-first-token vs batch inference latency are fundamentally different measurements).

### Transcription Service HTTP API Contract

```
GET  /health                    → { status, loaded_backends, vram_usage_mb }
GET  /api/models                → [{ name, backend, languages, vram_mb, compute_type }]
GET  /api/registry              → current registry YAML as JSON
POST /api/registry/reload       → reload registry from disk

WebSocket /api/stream
  Client sends:
    - binary frames: 16kHz mono float32 PCM audio
    - text frames: { type: "config", language?: str, backend?: str, model?: str }
    - text frames: { type: "end" }
  Server sends:
    - text frames: { type: "segment", ...TranscriptionResult fields }
    - text frames: { type: "interim", text, confidence }
    - text frames: { type: "language_detected", language, confidence }
    - text frames: { type: "backend_switched", from, to, reason }

POST /api/transcribe            → batch transcription (file upload, post-meeting)
  Request: multipart form with audio file + { language?, backend?, model?, profile: "batch" }
  Response: { text, segments[], language, confidence }
```

### Transcription Benchmarking Harness

**Location:** `modules/transcription-service/benchmarks/`
**Interface:** CLI tool — `uv run python -m benchmarks.run --backend whisper --language en --dataset librispeech-test-clean`
**Prerequisite:** Transcription service must be stopped (benchmarks use the GPU exclusively to avoid contention).
**Test data:** Standard ASR datasets (LibriSpeech for English, AISHELL-1 for Chinese) downloaded on first run. Custom test sets can be added as WAV + reference transcript pairs in `benchmarks/data/`.

Metrics collected per run:
- Accuracy: WER (word error rate) for alphabetic languages, CER (character error rate) for CJK
- Latency: time-to-first-token (streaming profile), total inference time (batch profile)
- GPU memory: peak VRAM usage during inference
- Throughput: utterances/second at batch sizes 1, 4, 8

Each result JSON includes a `system_info` block: GPU model, driver version, CUDA version, Python package versions for the backend, and model file checksum (SHA256) for reproducibility.

Results written to `benchmarks/results/{backend}_{language}_{timestamp}.json` for comparison.

---

## Plan 2: SvelteKit Loopback Page

### Route

`modules/dashboard-service/src/routes/(app)/loopback/+page.svelte`

### Top Toolbar

| Element | Purpose |
|---------|---------|
| Audio source selector | Mic device, system audio (loopback device), both |
| Source language | Auto-detected with manual override dropdown |
| Target language | Translation target |
| Model override | Optional — defaults to registry's best for detected language |
| Display mode switcher | Split \| Subtitle \| Transcript |
| Start/End Meeting button | Promotes ephemeral → meeting session |
| Connection status | Dot showing transcription + translation service health |

### Display Modes

**Split view (default):**
- Left panel: original-language captions, scrolling
- Right panel: translations, scrolling
- Speaker colors consistent across panels
- Interim text: lower opacity, italic (original language only — translations arrive for final segments)

**Subtitle overlay:**
- Captions pinned to bottom of screen
- Original language top line (gold), translation below (green)
- Configurable font size, background opacity
- Can pop out as separate browser window for screen-sharing

**Transcript mode:**
- Scrolling feed per utterance
- Speaker name + color badge, timestamp
- Original text + translation per entry
- Good for review and note-taking

### Audio Capture

- `navigator.mediaDevices.getUserMedia()` for microphone
- System audio: primary approach is `getUserMedia()` with a virtual loopback device (BlackHole on macOS, PulseAudio monitor on Linux). `getDisplayMedia()` is a fallback for environments without a loopback device but requires user screen-share consent. The audio source selector in the toolbar lists available devices including virtual loopback devices.
- AudioWorklet processes audio, posts `Float32Array` buffers to main thread
- Captures at **native quality** (48kHz+ stereo)
- Sends via WebSocket (`/api/audio/stream`) as **binary frames** to orchestration

### WebSocket Protocol (Browser ↔ Orchestration)

**Version:** Include `protocol_version: 1` in the `connected` response. Clients check version compatibility.

**Binary frames (browser → orchestration):** Raw `Float32Array` audio buffers at native sample rate.

**Text frames (browser → orchestration):**
- `{ type: "start_session", sample_rate: 48000, channels: 2, device_id?: string }`
- `{ type: "end_session" }`
- `{ type: "promote_to_meeting" }`
- `{ type: "end_meeting" }`

**Text frames (orchestration → browser):**
- `{ type: "connected", protocol_version: 1, session_id: string }`
- `{ type: "segment", ...TranscriptionResult fields }` — finalized transcription
- `{ type: "interim", text: string, confidence: float }` — work-in-progress (original language only)
- `{ type: "translation", text: string, source_lang: string, target_lang: string, transcript_id: int }`
- `{ type: "meeting_started", session_id: string, started_at: string }`
- `{ type: "recording_status", recording: bool, chunks_written: int }`
- `{ type: "service_status", transcription: "up"|"down", translation: "up"|"down" }`

### Meeting Mode UI

When "Start Meeting" clicked:
- Button becomes red "End Meeting"
- Pulsing red dot + elapsed timer appears
- Session info bar: session ID, start time, detected languages
- "End Meeting" shows confirmation dialog, then triggers cleanup

---

## Plan 3: Unified Meeting Pipeline

### Core Principle

One meeting pipeline. Multiple input sources. The loopback page and meeting bots are just different audio sources feeding the same system.

### MeetingAudioStream Interface (Shared Contract)

Defined in `livetranslate-common`:

```python
class MeetingAudioStream(Protocol):
    """Interface that any audio source implements to feed the meeting pipeline."""

    source_type: str               # "loopback", "google_meet_bot", etc.
    sample_rate: int               # e.g., 48000
    channels: int                  # e.g., 2 (stereo)
    encoding: str                  # "float32", "int16"

    async def read_chunk(self) -> AudioChunk | None:
        """Returns next audio chunk, or None when stream ends."""
        ...

@dataclass
class AudioChunk:
    data: bytes                    # raw PCM audio
    timestamp_ms: int              # monotonic timestamp
    sequence_number: int           # for gap detection
    source_id: str                 # identifies which source (mic, system, bot)
```

### Audio Sources

| Source | How audio arrives | Quality |
|--------|------------------|---------|
| Loopback page (mic) | WebSocket binary frames from browser | Native (48kHz+ stereo) |
| Loopback page (system audio) | WebSocket binary frames from browser | Native |
| Google Meet bot | Browser audio capture | Whatever Chrome provides |

All sources implement `MeetingAudioStream`. A single session can have multiple audio sources simultaneously (e.g., mic + system audio). Sources are tracked via `source_id` within the session.

### Session Lifecycle

```
┌─────────────────┐
│   Ephemeral      │ ◄── Default state
│   (no persistence)│
└────────┬────────┘
         │ "Start Meeting" / bot session start
         ▼
┌─────────────────┐
│   Meeting        │ ◄── Full persistence
│   (recording +   │
│    transcripts +  │
│    translations)  │
└────────┬────────┘
         │ "End Meeting" / bot disconnect
         ▼
┌─────────────────┐
│   Post-Processing│ ◄── Background task (async, non-blocking)
└─────────────────┘
```

### Continuous Recording (Meeting Mode)

- **Format**: FLAC (lossless) at original sample rate (48kHz+ stereo)
- **Chunking**: 30-second segment files, flush-on-write
- **Path**: `recordings/{session_id}/chunk_{sequence:06d}_{timestamp}.flac`
- **Manifest file**: `recordings/{session_id}/manifest.json` — tracks chunk sequence, sample counts per chunk, total samples, and timestamps. Enables crash recovery with gap detection and gapless concatenation.
- **16kHz downsampling**: happens in orchestration before forwarding to transcription service — never touches recorded files
- **Sample-exact continuity**: monotonic sample counter across chunks — no gaps, no overlaps between recording segments

### Crash Safety

1. **Session metadata**: written to DB on "Start Meeting" (session exists even if process dies)
2. **Audio chunks**: flushed to disk as they arrive (lose at most current ~30s chunk). Manifest file updated per chunk.
3. **Transcripts/translations**: persisted to DB row-by-row as they arrive (not buffered)
4. **Orphan detection — two mechanisms:**
   - On startup: find sessions marked "active" that never got an "end" event → mark as "interrupted"
   - Periodic heartbeat: if no audio chunks arrive for a session within 120 seconds, mark as "interrupted." Covers cases where browser tab closes without sending `end_session`.
5. **Untranslated recovery**: on recovery, query for `meeting_transcripts` rows without corresponding `meeting_translations` rows and re-submit them for translation.

### Database Schema

**Migration strategy:** Additive, not destructive.
1. Create `meeting_sessions` table alongside existing `bot_sessions`
2. Copy/backfill existing data from `bot_sessions` into `meeting_sessions` with `source_type = 'google_meet_bot'`
3. Switch all writes to `meeting_sessions`
4. Update foreign key references in bot code
5. Deprecate reads from `bot_sessions`
6. Drop `bot_sessions` after a safe period

Implemented as an Alembic migration.

```sql
-- Unified meeting session (works for both loopback and bot)
CREATE TABLE meeting_sessions (
    id UUID PRIMARY KEY,
    source_type TEXT NOT NULL,  -- 'loopback', 'google_meet_bot', etc.
    status TEXT NOT NULL,       -- 'ephemeral', 'active', 'completed', 'interrupted'
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    source_languages TEXT[],
    target_languages TEXT[],
    recording_path TEXT,
    metadata JSONB,
    last_activity_at TIMESTAMPTZ DEFAULT NOW()  -- for heartbeat orphan detection
);

-- Transcript entries (real-time persisted)
CREATE TABLE meeting_transcripts (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES meeting_sessions(id),
    timestamp_ms BIGINT NOT NULL,
    speaker_id TEXT,
    speaker_name TEXT,
    source_language TEXT,
    source_id TEXT,                -- which audio source within the session
    text TEXT NOT NULL,
    confidence FLOAT,
    is_final BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Translation entries (real-time persisted)
CREATE TABLE meeting_translations (
    id BIGSERIAL PRIMARY KEY,
    transcript_id BIGINT REFERENCES meeting_transcripts(id),
    target_language TEXT NOT NULL,
    translated_text TEXT NOT NULL,
    model_used TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Post-Meeting Processing

Runs as a **background task** — "End Meeting" returns immediately to the user. Processing:

1. Concatenate audio chunks → single FLAC file (guided by manifest for gapless output)
2. Optionally run batch re-transcription (batch profile: higher quality, full-file context, beam search)
3. Run full diarization pass if not done in real-time
4. Status updates sent to frontend via WebSocket as processing progresses

---

## Plan 4: Translation Module (Absorbed into Orchestration) & Benchmarking

### Architecture Change: Translation Service Absorbed

The previous `modules/translation-service/` was a thin HTTP relay (~150 lines of core logic) between orchestration and Ollama/vLLM. Since thomas-pc already runs Ollama with a model router, the translation logic is absorbed into the orchestration service. `modules/translation-service/` is archived.

**What moves to orchestration (`src/translation/`):**
- `llm_client.py` — ported from `OpenAICompatibleTranslator` (~150 lines): constructs HTTP POST to `/v1/chat/completions`, parses response, retry with exponential backoff
- `config.py` — LLM backend configuration (Ollama URL, model name, timeout, compute type)
- `context.py` — rolling context window (new, per this spec)
- `service.py` — combines client + config + context into a clean interface

**What is deleted:**
- `modules/translation-service/api_server_fastapi.py` — the proxy layer
- `clients/translation_service_client.py` (~900 lines) — the over-complex client
- `internal_services/translation.py` (~430 lines) — the `sys.path` hack facade
- Legacy backends (`llama_translator.py`, `nllb_translator.py`, etc.) — dead code

### Rolling Context Window

The orchestration service maintains a per-session ring buffer of recent finalized (text, translation) pairs. Each translation request includes the last N sentences as context, giving the LLM continuity for pronouns, terminology, and tone.

**Types (in `livetranslate-common`):**

```python
@dataclass
class TranslationContext:
    text: str                 # original text
    translation: str          # previous translation

@dataclass
class TranslationRequest:
    text: str
    source_language: str
    target_language: str
    context: list[TranslationContext]  # last N sentences
    context_window_size: int = 5

@dataclass
class TranslationResponse:
    translated_text: str
    source_language: str
    target_language: str
    model_used: str
    latency_ms: float
```

These types are defined in `livetranslate-common` so the translation module, orchestration pipeline, and benchmarking harness all share the same schema.

**Context eviction:** By count (`context_window_size`) AND by token count (`max_context_tokens`), whichever limit is hit first. Failed translations are NOT added to the context window.

**Backpressure:** If translation falls behind transcription, use a bounded queue (max depth 10). When full, drop the oldest pending request (stale translations are less valuable). The frontend shows a brief "translation catching up" indicator.

The LLM prompt includes prior context so "他" (he) resolves to "Manager Zhang" rather than a generic "he."

### LLM Client

Orchestration calls Ollama/vLLM directly via the OpenAI-compatible API:

```python
# Orchestration calls thomas-pc's Ollama directly
POST http://thomas-pc:11434/v1/chat/completions
{
    "model": "qwen3.5:7b",
    "messages": [
        {"role": "system", "content": "You are a translator. ..."},
        {"role": "user", "content": "Context:\n...\n\nTranslate: ..."}
    ],
    "temperature": 0.3
}
```

No intermediate service. The `llm_client.py` module handles retry logic and response parsing (~150 lines, ported from `OpenAICompatibleTranslator`).

### Translation Benchmarking Harness

**Location:** `tools/translation-benchmark/`
**Interface:** CLI tool — `uv run python -m tools.translation_benchmark.run --model qwen3.5:7b --lang-pair zh-en --dataset flores-test`
**Prerequisite:** The benchmark calls Ollama/vLLM directly — no separate service needed. Just needs Ollama running on thomas-pc.
**Test data:** FLORES-200 dataset for multilingual benchmarks, plus custom domain-specific test sets as source/reference text pairs in `tools/translation-benchmark/data/`.

Metrics collected per run:
- **Quality**: BLEU and COMET scores against reference translations per language pair
- **Latency**: time per translation request, measured with and without context window at different window sizes
- **Throughput**: concurrent translation requests per second
- **Model comparison**: run identical inputs through multiple models, output a comparison table with quality + speed rankings

Each result JSON includes a `system_info` block for reproducibility (GPU, driver, CUDA, package versions, model checksum).

Results written to `tools/translation-benchmark/results/{model}_{lang_pair}_{timestamp}.json`.

### Configurable Parameters

Translation config in orchestration's environment/settings:

```python
{
    "llm_base_url": "http://thomas-pc:11434/v1",  # Ollama endpoint
    "model": "qwen3.5:7b",           # or any model Ollama serves
    "context_window_size": 5,         # sentences of context
    "max_context_tokens": 500,        # token limit for context
    "temperature": 0.3,
    "timeout_s": 10
}
```

---

## Configuration Summary

### Environment Variables (Orchestration)

```bash
TRANSCRIPTION_SERVICE_URL=ws://thomas-pc:5001/api/stream  # WebSocket
LLM_BASE_URL=http://thomas-pc:11434/v1   # Ollama (OpenAI-compatible)
LLM_MODEL=qwen3.5:7b                     # or any model Ollama serves
RECORDING_PATH=./recordings
RECORDING_FORMAT=flac
RECORDING_CHUNK_DURATION_S=30
DATABASE_URL=postgresql://localhost:5432/livetranslate

# Timeout profiles (per-operation, not global)
TRANSCRIPTION_STREAM_TIMEOUT_S=10    # per-chunk timeout for streaming
TRANSCRIPTION_BATCH_TIMEOUT_S=300    # batch file transcription
LLM_TIMEOUT_S=10                     # per-request translation timeout
LLM_CONTEXT_WINDOW_SIZE=5            # sentences of context for translation
LLM_MAX_CONTEXT_TOKENS=500           # token limit for context
```

### Model Registry (Transcription Service)

```yaml
# config/model_registry.yaml
version: 1  # schema version — service rejects incompatible versions

backends:
  whisper:
    module: src.backends.whisper
    class: WhisperBackend

  sensevoice:
    module: src.backends.sensevoice
    class: SenseVoiceBackend

  funasr:
    module: src.backends.funasr
    class: FunASRBackend

vram_budget_mb: 10000  # max VRAM for transcription backends

language_routing:
  zh:
    backend: sensevoice
    model: SenseVoiceSmall
    compute_type: float16
    chunk_duration_s: 5.0       # 5s window, NOT 10s — latency matters
    stride_s: 4.0               # advance 4s between inferences
    overlap_s: 1.0
    vad_threshold: 0.45
    beam_size: 5
    prebuffer_s: 0.5            # fast first result
    batch_profile: realtime

  en:
    backend: whisper
    model: large-v3-turbo
    compute_type: float16
    chunk_duration_s: 5.0
    stride_s: 4.5
    overlap_s: 0.5
    vad_threshold: 0.5
    beam_size: 1                # greedy for speed in realtime
    prebuffer_s: 0.3
    batch_profile: realtime

  ja:
    backend: sensevoice
    model: SenseVoiceSmall
    compute_type: float16
    chunk_duration_s: 5.0
    stride_s: 4.0
    overlap_s: 1.0
    vad_threshold: 0.45
    beam_size: 5
    prebuffer_s: 0.5
    batch_profile: realtime

  "*":
    backend: whisper
    model: large-v3-turbo
    compute_type: float16
    chunk_duration_s: 5.0
    stride_s: 4.5
    overlap_s: 0.5
    vad_threshold: 0.5
    beam_size: 1
    prebuffer_s: 0.3
    batch_profile: realtime
```

---

## Parallelization

| Plan | Depends On | Can Start |
|------|-----------|-----------|
| Plan 1: Transcription Service Refactor | VAD/chunking pre-task (blocking first step) | Immediately |
| Plan 2: SvelteKit Loopback Page | Pre-agreed contracts (below) | Immediately |
| Plan 3: Unified Meeting Pipeline | Pre-agreed contracts (below) | Immediately |
| Plan 4: Translation Module & Benchmarking | Pre-agreed contracts (below) | Immediately |

All four plans are independent. They share interfaces (WebSocket message format, API contracts) but not implementation. Can be executed in parallel with separate worktrees/branches.

## Integration Points

After all plans complete, integration work:
- Loopback page (Plan 2) uses meeting pipeline (Plan 3) for "Start Meeting"
- Orchestration points at transcription service (Plan 1) via configured URL
- Translation module (Plan 4) in orchestration calls Ollama/vLLM directly with rolling context
- Google Meet bot plugs into unified meeting pipeline (Plan 3) instead of its own session management
- Archive `modules/translation-service/` and delete `clients/translation_service_client.py` + `internal_services/translation.py`
- Extract `BaseServiceClient` from `AudioServiceClient` to consolidate resilience plumbing (circuit breaker, retry, session management)

### Pre-Agreed Contracts (Must Be Defined Before Parallel Work Begins)

These interfaces are shared across plans and must be stable before independent work starts. All defined as Pydantic models in `livetranslate-common`:

1. **WebSocket message schema** — all text frame message types with their fields, versioned with `protocol_version` (Plan 2 ↔ Plan 3)
2. **`MeetingAudioStream` + `AudioChunk` types** — the interface that any audio source implements to feed the pipeline (Plan 2 ↔ Plan 3)
3. **`TranscriptionResult` + `ModelInfo` types** — transcription output shape (Plan 1 ↔ orchestration)
4. **Transcription service WebSocket API** — binary/text frame protocol for `/api/stream` (Plan 1 ↔ orchestration)
5. **`TranslationRequest` + `TranslationResponse` + `TranslationContext` types** — shared types used by the translation module and benchmarking harness (Plan 4)

### Implementation Notes from Reviews

These items should be addressed during implementation but do not require spec-level decisions:

- **Clock drift:** When capturing mic + system audio simultaneously, sample clocks drift. Record as separate tracks, align in post-processing.
- **Drop embedded service fallback:** The refactored orchestration clients should remove `_embedded_enabled()` fallback paths — services are remote-only in this topology.
- **SSL not needed over Tailscale:** All traffic is already encrypted by WireGuard. Use plain HTTP/WS to Tailscale IPs.
- **Fix `sys.path.append` in `audio_service_client.py`:** Use proper UV workspace imports via `livetranslate-common`.
- **`deque` for pending chunks:** Replace `list.pop(0)` (O(n)) with `collections.deque.popleft()` (O(1)) in the audio processing pipeline.
