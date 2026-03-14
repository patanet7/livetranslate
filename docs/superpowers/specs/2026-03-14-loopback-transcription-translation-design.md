# Real-Time Loopback Transcription & Translation System

**Date:** 2026-03-14
**Status:** Design
**Approach:** Refactor In Place (Approach 1)

## Overview

Build a real-time loopback system for live meetings: mic + system audio → language-agnostic transcription (best model per language, GPU-accelerated on RTX 4090) → translation → multi-mode display in SvelteKit frontend. Ephemeral by default, promotable to full meeting sessions with continuous recording and persistence — the same meeting pipeline that meeting bots plug into.

## Goals

1. **Pluggable transcription backends** — not just Whisper. Best model per language (SenseVoice, FunASR, faster-whisper, etc.) with auto-detection and user override.
2. **All GPU compute on thomas-pc** (RTX 4090 via Tailscale) — transcription + translation services run remotely. Local machine handles mic capture + display only.
3. **SvelteKit loopback page** — live captions + translation with split/subtitle/transcript display modes.
4. **Unified meeting pipeline** — one recording/persistence system that both loopback sessions and meeting bots feed into.
5. **Translation rolling window** — context-aware translation using recent transcript history for quality.

## Non-Goals

- Meeting bot UI changes (bots already work, just need to plug into unified pipeline)
- SeamlessM4T integration (removed — using separate transcription + translation instead)
- Multi-machine load balancing (single RTX 4090 for now)

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
│ ├─ WebSocket hub     │   (~1-5ms)    │ ├─ backend: whisper          │
│ ├─ meeting pipeline  │               │ ├─ backend: sensevoice       │
│ ├─ audio recording   │               │ ├─ backend: funasr           │
│ └─ session mgmt      │               │ └─ dedup / rolling window    │
│                      │               │                              │
│ PostgreSQL           │               │ translation-service          │
│ (session data)       │               │ ├─ Ollama / vLLM (GPU)       │
└─────────────────────┘               │ └─ rolling context window    │
                                       └──────────────────────────────┘
```

### Audio Flow

```
Browser mic/system audio (48kHz+ native quality)
    │
    ▼
Orchestration Service
    ├──[meeting mode]──▶ Save original quality to disk (FLAC, 48kHz+)
    ├──▶ Downsample to 16kHz mono (inference only)
    └──▶ Send to transcription-service (thomas-pc)
            │
            ▼
        TranscriptionBackend.transcribe()
            │
            ▼
        Dedup / rolling window → clean text
            │
            ▼
        Return to orchestration
            │
            ▼
        Send to translation-service (thomas-pc)
            ├─ Rolling context window (last N sentences)
            └─ Return translation
            │
            ▼
        Orchestration broadcasts via WebSocket
            ├─ [meeting mode] persist transcript + translation to DB
            └─ Frontend displays in selected mode
```

---

## Plan 1: Transcription Service Refactor

### Rename

`modules/whisper-service/` → `modules/transcription-service/`

All references updated: imports, Docker configs, env vars, orchestration client, CLAUDE.md.

### TranscriptionBackend Protocol

```python
class TranscriptionBackend(Protocol):
    """Any transcription engine implements this."""

    async def transcribe(
        self, audio: np.ndarray, language: str | None = None, **kwargs
    ) -> TranscriptionResult: ...

    def supports_language(self, lang: str) -> bool: ...

    def get_model_info(self) -> ModelInfo: ...

    async def load_model(self, model_name: str, device: str = "cuda") -> None: ...

    async def unload_model(self) -> None: ...
```

### Backend Implementations

Located in `src/backends/`:

| Backend | File | Primary Languages | Notes |
|---------|------|-------------------|-------|
| Whisper (faster-whisper) | `whisper.py` | Universal, strong English | CTranslate2, fastest Whisper variant on GPU |
| SenseVoice | `sensevoice.py` | Chinese, Japanese, Korean, English | FunAudioLLM, strong CJK |
| FunASR (Paraformer) | `funasr.py` | Chinese (Mandarin) | Alibaba DAMO, best Mandarin accuracy |
| More over time | ... | ... | ... |

First implementation: port existing Whisper code into `whisper.py` backend using faster-whisper for GPU performance. Add SenseVoice or FunASR as second backend.

### ModelRegistry

Configurable mapping of language → backend + model + processing parameters. Stored as a config file, editable at runtime.

```python
@dataclass
class BackendConfig:
    backend: str           # "whisper", "sensevoice", "funasr"
    model: str             # "large-v3-turbo", "SenseVoiceSmall", etc.
    chunk_duration_s: float  # Audio chunk size for this model
    overlap_s: float       # Overlap between chunks
    vad_threshold: float   # VAD sensitivity
    beam_size: int         # Beam search width
    prebuffer_s: float     # Minimum audio before first inference
    # ... extensible

# Registry: language code → BackendConfig
# "*" = fallback for unmatched languages
registry: dict[str, BackendConfig]
```

The registry is the single source of truth for "what model handles what language with what parameters." Changeable without code changes.

### Model Selection Flow

```
Audio arrives
    → VAD detects speech
    → Language detector analyzes audio segment
    → Check: user override? → use specified model
    → Check: registry[detected_language]? → use registered config
    → Fallback: registry["*"] → default model
    → Load backend if not already loaded
    → TranscriptionBackend.transcribe() with config params
```

### Infrastructure Preserved (Above Backend Layer)

These modules remain, operating on backend output regardless of which engine produced it:

- `token_deduplicator.py` — token-level dedup at chunk boundaries
- `continuous_stream_processor.py` — text-level dedup across inference windows
- `vac_online_processor.py` — VAD chunking (Silero, backend-independent)
- `token_buffer.py` — rolling context buffer
- `sliding_lid_detector.py` — language detection (feeds registry lookup)
- `sentence_segmenter.py` — **refactored to be language-universal** (not just CJK)

### Sentence Segmenter (Language-Universal)

Current segmenter handles CJK fullwidth punctuation. Refactored to support:

- Latin: `.` `!` `?` and variants
- CJK: `。` `！` `？`
- Arabic/Hebrew: `۔` `؟`
- Thai/Lao: space-based segmentation
- Devanagari: `।`
- Configurable per-language rules

### Research Task

Before finalizing the VAD/chunking layer, evaluate how these frameworks handle it:

- **WhisperX**: pyannote VAD for pre-segmentation, forced alignment for word timestamps
- **faster-whisper**: CTranslate2 backend, batched inference, built-in VAD
- **FasterWhisperX**: merges both approaches
- **SenseVoice**: its own chunking recommendations

Adopt best practices rather than reinventing. This may influence the VAD/chunking architecture.

### Benchmarking Harness

Built-in benchmarking for comparing backends:

- Accuracy (WER/CER) per language with reference transcripts
- Latency (time-to-first-token, total inference time)
- GPU memory usage
- Throughput (concurrent streams)

---

## Plan 2: SvelteKit Loopback Page

### Route

`modules/dashboard-service/src/routes/(app)/loopback/+page.svelte`

### Top Toolbar

| Element | Purpose |
|---------|---------|
| Audio source selector | Mic device, system audio, both |
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
- Interim text: lower opacity, italic

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
- `getDisplayMedia()` or loopback device (BlackHole/Soundflower) for system audio
- AudioWorklet for processing
- Captures at **native quality** (48kHz+ stereo)
- Sends via WebSocket (`/api/audio/stream`) to orchestration

### WebSocket Messages

Sends:
- `audio_chunk`: base64-encoded audio at native quality
- `start_session` / `end_session`: session lifecycle
- `promote_to_meeting`: upgrades ephemeral → meeting

Receives:
- `segment`: transcription result (text, speaker, language, confidence, is_final, is_draft)
- `translation`: translated text (text, source_lang, target_lang)
- `interim_caption`: work-in-progress transcription for real-time display
- `meeting_started`: confirmation of promotion with session_id
- `recording_status`: recording health indicator

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

### Audio Sources

| Source | How audio arrives | Quality |
|--------|------------------|---------|
| Loopback page (mic) | WebSocket from browser | Native (48kHz+ stereo) |
| Loopback page (system audio) | WebSocket from browser | Native |
| Google Meet bot | Browser audio capture | Whatever Chrome provides |
| Future: other bots | Same WebSocket contract | Varies |

All sources produce a `MeetingAudioStream` that the pipeline consumes identically.

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
│   Post-Processing│ ◄── Concatenate audio, optional batch re-transcription
└─────────────────┘
```

### Continuous Recording (Meeting Mode)

- **Format**: FLAC (lossless) at original sample rate (48kHz+ stereo)
- **Chunking**: 30-second segment files, flush-on-write
- **Path**: `recordings/{session_id}/chunk_{timestamp}.flac`
- **16kHz downsampling**: happens only at the inference boundary, never touches recorded files

### Crash Safety

1. **Session metadata**: written to DB on "Start Meeting" (session exists even if process dies)
2. **Audio chunks**: flushed to disk as they arrive (lose at most current ~30s chunk)
3. **Transcripts/translations**: persisted to DB row-by-row as they arrive (not buffered)
4. **Orphan detection**: on startup, find sessions marked "active" that never got an "end" event → mark as "interrupted" with all captured data intact

### Database Schema

Extends existing `bot_sessions` schema (or unified `meeting_sessions`):

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
    metadata JSONB
);

-- Transcript entries (real-time persisted)
CREATE TABLE meeting_transcripts (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES meeting_sessions(id),
    timestamp_ms BIGINT NOT NULL,
    speaker_id TEXT,
    speaker_name TEXT,
    source_language TEXT,
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

When meeting ends:
1. Concatenate audio chunks → single FLAC file
2. Optionally run batch transcription (higher quality, full-file context)
3. Run full diarization pass if not done in real-time
4. Generate meeting summary (future — LLM-based)

---

## Plan 4: Translation Rolling Window & Benchmarking

### Rolling Context Window

Translation service sends the last N sentences as context with each translation request. This gives the LLM continuity for pronouns, terminology, and tone.

```python
# Translation request with context
{
    "text": "他说明天会来。",
    "target_language": "en",
    "context": [
        {"text": "我们在讨论项目进度。", "translation": "We were discussing project progress."},
        {"text": "张经理提到了一些问题。", "translation": "Manager Zhang mentioned some issues."}
    ],
    "context_window_size": 5  # configurable
}
```

The LLM prompt includes prior context so "他" (he) resolves to "Manager Zhang" rather than a generic "he."

### Translation Benchmarking

Built-in benchmarking harness:

- **Quality**: BLEU/COMET scores against reference translations per language pair
- **Latency**: time per translation, with and without context window
- **Throughput**: concurrent translation requests
- **Model comparison**: run same inputs through different models, compare quality + speed

### Configurable Parameters

Per-model translation config in registry:

```python
{
    "model": "qwen2.5:7b",
    "context_window_size": 5,      # sentences of context
    "max_context_tokens": 500,     # token limit for context
    "temperature": 0.3,
    "timeout_s": 10
}
```

---

## Configuration Summary

### Environment Variables (Orchestration)

```bash
TRANSCRIPTION_SERVICE_URL=http://thomas-pc:5001
TRANSLATION_SERVICE_URL=http://thomas-pc:5003
RECORDING_PATH=./recordings
RECORDING_FORMAT=flac
RECORDING_CHUNK_DURATION_S=30
DATABASE_URL=postgresql://localhost:5432/livetranslate
```

### Model Registry (Transcription Service)

```yaml
# config/model_registry.yaml
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

language_routing:
  zh:
    backend: sensevoice
    model: SenseVoiceSmall
    chunk_duration_s: 10.0
    overlap_s: 1.0
    vad_threshold: 0.45
    beam_size: 5
    prebuffer_s: 2.0

  en:
    backend: whisper
    model: large-v3-turbo
    chunk_duration_s: 5.0
    overlap_s: 0.5
    vad_threshold: 0.5
    beam_size: 5
    prebuffer_s: 1.0

  ja:
    backend: sensevoice
    model: SenseVoiceSmall
    chunk_duration_s: 8.0
    overlap_s: 1.0
    vad_threshold: 0.45
    beam_size: 5
    prebuffer_s: 2.0

  "*":
    backend: whisper
    model: large-v3-turbo
    chunk_duration_s: 5.0
    overlap_s: 0.5
    vad_threshold: 0.5
    beam_size: 5
    prebuffer_s: 1.0
```

---

## Parallelization

| Plan | Depends On | Can Start |
|------|-----------|-----------|
| Plan 1: Transcription Service Refactor | None | Immediately |
| Plan 2: SvelteKit Loopback Page | WebSocket contract (already exists) | Immediately |
| Plan 3: Unified Meeting Pipeline | None (defines shared schema) | Immediately |
| Plan 4: Translation Rolling Window | None | Immediately |

All four plans are independent. They share interfaces (WebSocket message format, API contracts) but not implementation. Can be executed in parallel with separate worktrees/branches.

## Integration Points

After all plans complete, integration work:
- Loopback page (Plan 2) uses meeting pipeline (Plan 3) for "Start Meeting"
- Orchestration points at transcription service (Plan 1) via configured URL
- Translation service (Plan 4) receives text from orchestration with rolling context
- Google Meet bot plugs into unified meeting pipeline (Plan 3) instead of its own session management
