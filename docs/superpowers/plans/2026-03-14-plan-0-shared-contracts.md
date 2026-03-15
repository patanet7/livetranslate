# Plan 0: Shared Contracts & VAD/Chunking Research

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define all shared Pydantic models in `livetranslate-common` that Plans 1–4 depend on, plus produce a VAD/chunking research document before Plan 1's backend implementation begins.

**Architecture:** Shared types live in `modules/shared/src/livetranslate_common/models/`. All services import from `livetranslate_common.models`. WebSocket message schemas are also defined here so both Python (orchestration) and TypeScript (dashboard) can share the same contract. VAD research is a standalone markdown deliverable.

**Tech Stack:** Python 3.12+, Pydantic v2, UV workspace, faster-whisper (research only)

**Spec:** `docs/superpowers/specs/2026-03-14-loopback-transcription-translation-design.md`

**Design note:** The spec describes contracts using `@dataclass` syntax for brevity. This plan uses **Pydantic v2 `BaseModel`** instead — Pydantic gives us JSON serialization, field validation, and `model_dump_json()` for free, which is critical for WebSocket transport. The field names and semantics are identical to the spec.

**Cross-plan dependency:** `TranscriptionBackend` (the async Protocol for pluggable backends) is defined in **Plan 1**, not here. Plan 0 only defines the *data types* that cross service boundaries. Plan 1 defines the *behavior contracts* internal to the transcription service.

**Existing system integrations (must be preserved):**
- **AIConnection** (`ai_connections` table): Plan 4's translation module must resolve LLM endpoints from this table (engine, URL, API key, priority, timeout) — not hardcode Ollama URLs. See `modules/orchestration-service/src/database/ai_connection.py`.
- **Glossary** (`glossaries` + `glossary_entries` tables): Domain terms for translation prompt injection AND transcription `initial_prompt`. See `modules/orchestration-service/src/services/glossary_service.py`.
- **SpeakerProfile** (`speaker_profiles` table): Voice embeddings for cross-meeting speaker identification. Plan 3's pipeline should optionally match diarization output against known profiles.
- **Meeting tables** (`meetings`, `meeting_chunks`, `meeting_sentences`, `meeting_translations`): Already exist for Fireflies with `source = "fireflies"`. Plan 3 should evaluate using these tables with `source = "loopback"` instead of creating parallel `meeting_sessions` tables — or document why separate tables are preferred.

---

## Chunk 0: Architecture Documentation

### Task 0: Create ARCHITECTURE.md

**Files:**
- Create: `ARCHITECTURE.md`

The existing architecture docs (`docs/archive/root-reports/analysis-audit/ARCHITECTURE_ANALYSIS.md`) are outdated — they describe 4 services (including the standalone translation service) and the old React frontend. The new architecture has 3 services + external Ollama, a SvelteKit dashboard, and a new loopback/meeting pipeline.

- [ ] **Step 1: Write ARCHITECTURE.md**

Create `ARCHITECTURE.md` at the repo root with:

```markdown
# LiveTranslate Architecture

## System Overview

LiveTranslate is a real-time speech transcription and translation system for live meetings.
Audio flows from browser → orchestration → transcription → translation → display.

## Service Topology

```
┌─────────────────────────────────────────────────────────┐
│  MacBook (local)                                        │
│  ┌──────────────┐    ┌───────────────────────────────┐  │
│  │  Dashboard    │◄──►│  Orchestration Service       │  │
│  │  (SvelteKit)  │    │  (FastAPI)                   │  │
│  │  :5173        │    │  :3000                       │  │
│  └──────────────┘    │  - WebSocket hub              │  │
│                       │  - Meeting pipeline           │  │
│                       │  - Translation (via Ollama)   │  │
│                       │  - FLAC recording             │  │
│                       │  - Audio downsampling         │  │
│                       └─────────────┬─────────────────┘  │
└─────────────────────────────────────┼────────────────────┘
                                      │ Tailscale (16kHz mono)
┌─────────────────────────────────────┼────────────────────┐
│  thomas-pc (RTX 4090)               │                    │
│  ┌──────────────────────────────────▼─────────────────┐  │
│  │  Transcription Service (faster-whisper)            │  │
│  │  :5001                                             │  │
│  │  - Pluggable backends (BackendManager)             │  │
│  │  - VRAM budget (10GB of 24GB)                      │  │
│  │  - ModelRegistry (YAML)                            │  │
│  │  - Authoritative LID                               │  │
│  │  - Silero VAD                                      │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Ollama (LLM inference)                            │  │
│  │  :11434                                            │  │
│  │  - qwen3.5:7b (translation)                        │  │
│  │  - OpenAI-compatible API                           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Audio Flow

1. **Browser** captures mic audio via AudioWorklet at native sample rate (48kHz+)
2. **Binary WebSocket** sends Float32Array frames to orchestration (no base64)
3. **Orchestration** forks: native quality → FLAC disk, 16kHz mono → transcription
4. **Transcription** runs VAD → LID → backend inference → segments back as text frames
5. **Translation** receives final segments, applies rolling context + glossary, calls Ollama
6. **Display** renders in split/subtitle/transcript modes via Svelte 5 runes

## Shared Contracts

All services share Pydantic models from `livetranslate-common` (`modules/shared/`):
- `TranscriptionResult`, `Segment`, `ModelInfo` — transcription output
- `AudioChunk`, `MeetingAudioStream` — audio pipeline types
- `TranslationRequest/Response/Context` — translation with rolling context + glossary
- `BackendConfig` — model registry entries
- WebSocket message schemas — typed protocol with versioning

TypeScript equivalents live in `modules/dashboard-service/src/lib/types/`.

## Meeting Pipeline

Sessions start **ephemeral** (stream-through, no persistence). "Start Meeting" promotes
to **active** (recording + DB persistence). Crash safety via flush-on-write FLAC chunks,
row-by-row DB persistence, manifest tracking, and 120s heartbeat orphan detection.

## Glossary System

Domain-specific terms stored in PostgreSQL via `GlossaryService`:
- **Translation**: glossary terms injected into LLM prompt for consistent terminology
- **Transcription**: glossary terms fed as Whisper's `initial_prompt` to bias recognition

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Dashboard | SvelteKit (Svelte 5 runes), TypeScript |
| Orchestration | FastAPI, SQLAlchemy, Alembic, FLAC (soundfile) |
| Transcription | faster-whisper (CTranslate2), Silero VAD |
| Translation | httpx → Ollama OpenAI-compatible API |
| Shared | Pydantic v2, UV workspace monorepo |
| Database | PostgreSQL |
| IPC | Tailscale VPN |
```

- [ ] **Step 2: Update CLAUDE.md service architecture section**

Update the "Service Architecture" section in the root `CLAUDE.md` to reference `ARCHITECTURE.md` and reflect the new 3-service topology (transcription replaces whisper, translation absorbed into orchestration).

- [ ] **Step 3: Commit**

```bash
git add ARCHITECTURE.md CLAUDE.md
git commit -m "docs: add ARCHITECTURE.md documenting new 3-service topology"
```

---

## Chunk 1: Transcription Types

### Task 1: TranscriptionResult and supporting types

**Files:**
- Create: `modules/shared/src/livetranslate_common/models/__init__.py`
- Create: `modules/shared/src/livetranslate_common/models/transcription.py`
- Create: `modules/shared/tests/test_models_transcription.py`
- Modify: `modules/shared/src/livetranslate_common/__init__.py`

- [ ] **Step 1: Write the failing test for Segment model**

```python
# modules/shared/tests/test_models_transcription.py
"""Tests for shared transcription models."""
import pytest
from livetranslate_common.models.transcription import Segment, TranscriptionResult, ModelInfo


class TestSegment:
    def test_segment_creation(self):
        seg = Segment(
            text="Hello world",
            start_ms=0,
            end_ms=1500,
            confidence=0.95,
        )
        assert seg.text == "Hello world"
        assert seg.start_ms == 0
        assert seg.end_ms == 1500
        assert seg.confidence == 0.95
        assert seg.speaker_id is None

    def test_segment_with_speaker(self):
        seg = Segment(
            text="你好",
            start_ms=1000,
            end_ms=2000,
            confidence=0.88,
            speaker_id="SPEAKER_00",
        )
        assert seg.speaker_id == "SPEAKER_00"

    def test_segment_duration_ms(self):
        seg = Segment(text="hi", start_ms=500, end_ms=2000, confidence=0.9)
        assert seg.duration_ms == 1500
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_transcription.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'livetranslate_common.models'`

- [ ] **Step 3: Write the Segment model**

```python
# modules/shared/src/livetranslate_common/models/__init__.py
"""Shared Pydantic models for all LiveTranslate services."""

from livetranslate_common.models.transcription import (
    ModelInfo,
    Segment,
    TranscriptionResult,
)

__all__ = [
    "ModelInfo",
    "Segment",
    "TranscriptionResult",
]
```

```python
# modules/shared/src/livetranslate_common/models/transcription.py
"""Transcription result types shared across services.

These are the canonical types for transcription output. The transcription
service produces them, orchestration consumes them, and the frontend
receives a JSON-serialized form via WebSocket.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    """A single transcribed segment with timestamps."""

    text: str
    start_ms: int
    end_ms: int
    confidence: float = Field(ge=0.0, le=1.0)
    speaker_id: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


class TranscriptionResult(BaseModel):
    """Complete transcription result returned by any backend."""

    text: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    segments: list[Segment] = Field(default_factory=list)
    stable_text: str = ""
    unstable_text: str = ""
    is_final: bool = False
    is_draft: bool = True
    speaker_id: str | None = None
    should_translate: bool = False
    context_text: str = ""  # previous transcription text fed as context (condition_on_previous_text)


class ModelInfo(BaseModel):
    """Metadata about a loaded transcription model."""

    name: str
    backend: str
    languages: list[str]
    vram_mb: int
    compute_type: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_transcription.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Write tests for TranscriptionResult and ModelInfo**

```python
# Append to modules/shared/tests/test_models_transcription.py

class TestTranscriptionResult:
    def test_minimal_result(self):
        result = TranscriptionResult(
            text="Hello",
            language="en",
            confidence=0.92,
        )
        assert result.text == "Hello"
        assert result.is_final is False
        assert result.is_draft is True
        assert result.should_translate is False
        assert result.segments == []

    def test_final_result_with_segments(self):
        seg = Segment(text="Hello", start_ms=0, end_ms=1000, confidence=0.95)
        result = TranscriptionResult(
            text="Hello",
            language="en",
            confidence=0.95,
            segments=[seg],
            stable_text="Hello",
            unstable_text="",
            is_final=True,
            is_draft=False,
            should_translate=True,
        )
        assert result.is_final is True
        assert result.should_translate is True
        assert len(result.segments) == 1

    def test_roundtrip_json(self):
        """Verify JSON serialization roundtrip — critical for WebSocket transport."""
        seg = Segment(text="你好", start_ms=0, end_ms=1200, confidence=0.88)
        original = TranscriptionResult(
            text="你好",
            language="zh",
            confidence=0.88,
            segments=[seg],
            is_final=True,
            should_translate=True,
        )
        json_str = original.model_dump_json()
        restored = TranscriptionResult.model_validate_json(json_str)
        assert restored == original


class TestModelInfo:
    def test_model_info(self):
        info = ModelInfo(
            name="large-v3-turbo",
            backend="whisper",
            languages=["en", "zh", "ja"],
            vram_mb=6000,
            compute_type="float16",
        )
        assert info.name == "large-v3-turbo"
        assert "zh" in info.languages
        assert info.vram_mb == 6000
```

- [ ] **Step 6: Run all transcription model tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_transcription.py -v`
Expected: PASS (7 tests)

- [ ] **Step 7: Update livetranslate_common __init__.py to re-export models**

Add to `modules/shared/src/livetranslate_common/__init__.py`:

```python
from livetranslate_common.models import (
    ModelInfo,
    Segment,
    TranscriptionResult,
)
```

And add `"ModelInfo"`, `"Segment"`, `"TranscriptionResult"` to `__all__`.

- [ ] **Step 8: Commit**

```bash
git add modules/shared/src/livetranslate_common/models/ modules/shared/tests/test_models_transcription.py modules/shared/src/livetranslate_common/__init__.py
git commit -m "feat(common): add TranscriptionResult, Segment, ModelInfo shared types"
```

---

### Task 2: Audio types (AudioChunk, MeetingAudioStream)

**Files:**
- Create: `modules/shared/src/livetranslate_common/models/audio.py`
- Create: `modules/shared/tests/test_models_audio.py`
- Modify: `modules/shared/src/livetranslate_common/models/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/shared/tests/test_models_audio.py
"""Tests for shared audio models."""
from livetranslate_common.models.audio import AudioChunk


class TestAudioChunk:
    def test_audio_chunk_creation(self):
        chunk = AudioChunk(
            data=b"\x00" * 320,
            timestamp_ms=1000,
            sequence_number=42,
            source_id="mic_0",
        )
        assert chunk.timestamp_ms == 1000
        assert chunk.sequence_number == 42
        assert chunk.source_id == "mic_0"
        assert len(chunk.data) == 320

    def test_audio_chunk_json_roundtrip(self):
        """AudioChunk.data is bytes — verify base64 encoding in JSON."""
        chunk = AudioChunk(
            data=b"\x01\x02\x03\x04",
            timestamp_ms=500,
            sequence_number=1,
            source_id="system_audio",
        )
        json_str = chunk.model_dump_json()
        restored = AudioChunk.model_validate_json(json_str)
        assert restored.data == chunk.data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_audio.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write AudioChunk model**

```python
# modules/shared/src/livetranslate_common/models/audio.py
"""Audio stream types shared across services.

AudioChunk is the unit of audio data flowing through the meeting pipeline.
MeetingAudioStream is a Protocol that any audio source (loopback mic,
system audio, Google Meet bot) implements to feed the pipeline.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class AudioChunk(BaseModel):
    """A single chunk of raw PCM audio data.

    NOTE: This is for internal Python-to-Python pipeline use only.
    WebSocket audio transport uses raw binary frames (Float32Array),
    not JSON-serialized AudioChunk instances.
    """

    data: bytes
    timestamp_ms: int
    sequence_number: int
    source_id: str


@runtime_checkable
class MeetingAudioStream(Protocol):
    """Interface that any audio source implements to feed the meeting pipeline.

    Implementors: LoopbackAudioStream, GoogleMeetBotStream, etc.
    """

    source_type: str  # "loopback", "google_meet_bot", etc.
    sample_rate: int  # e.g. 48000
    channels: int  # e.g. 2 (stereo)
    encoding: str  # "float32", "int16"

    async def read_chunk(self) -> AudioChunk | None:
        """Returns next audio chunk, or None when stream ends."""
        ...
```

Update `modules/shared/src/livetranslate_common/models/__init__.py` to add:

```python
from livetranslate_common.models.audio import AudioChunk, MeetingAudioStream
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_audio.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write MeetingAudioStream protocol compliance test**

```python
# Append to modules/shared/tests/test_models_audio.py
import asyncio
from livetranslate_common.models.audio import MeetingAudioStream


class FakeLoopbackStream:
    """Test double that satisfies the MeetingAudioStream protocol."""

    source_type = "loopback"
    sample_rate = 48000
    channels = 2
    encoding = "float32"

    def __init__(self, chunks: list[AudioChunk]):
        self._chunks = iter(chunks)

    async def read_chunk(self) -> AudioChunk | None:
        try:
            return next(self._chunks)
        except StopIteration:
            return None


class TestMeetingAudioStream:
    def test_protocol_compliance(self):
        stream = FakeLoopbackStream([])
        assert isinstance(stream, MeetingAudioStream)

    def test_read_chunks_until_none(self):
        chunks = [
            AudioChunk(data=b"\x00" * 160, timestamp_ms=i * 10, sequence_number=i, source_id="mic")
            for i in range(3)
        ]
        stream = FakeLoopbackStream(chunks)

        results = []
        loop = asyncio.new_event_loop()
        while True:
            chunk = loop.run_until_complete(stream.read_chunk())
            if chunk is None:
                break
            results.append(chunk)
        loop.close()
        assert len(results) == 3
```

- [ ] **Step 6: Run all audio tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_audio.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add modules/shared/src/livetranslate_common/models/audio.py modules/shared/tests/test_models_audio.py modules/shared/src/livetranslate_common/models/__init__.py
git commit -m "feat(common): add AudioChunk and MeetingAudioStream protocol"
```

---

### Task 3: Translation types

**Files:**
- Create: `modules/shared/src/livetranslate_common/models/translation.py`
- Create: `modules/shared/tests/test_models_translation.py`
- Modify: `modules/shared/src/livetranslate_common/models/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/shared/tests/test_models_translation.py
"""Tests for shared translation models."""
from livetranslate_common.models.translation import (
    TranslationContext,
    TranslationRequest,
    TranslationResponse,
)


class TestTranslationContext:
    def test_context_creation(self):
        ctx = TranslationContext(
            text="你好世界",
            translation="Hello world",
        )
        assert ctx.text == "你好世界"
        assert ctx.translation == "Hello world"


class TestTranslationRequest:
    def test_request_with_context(self):
        ctx = TranslationContext(text="之前的句子", translation="Previous sentence")
        req = TranslationRequest(
            text="这是新的句子",
            source_language="zh",
            target_language="en",
            context=[ctx],
        )
        assert req.text == "这是新的句子"
        assert req.context_window_size == 5  # default
        assert req.max_context_tokens == 500  # default dual eviction budget
        assert len(req.context) == 1

    def test_request_no_context(self):
        req = TranslationRequest(
            text="Hello",
            source_language="en",
            target_language="zh",
            context=[],
        )
        assert req.context == []
        assert req.glossary_terms == {}

    def test_request_with_glossary(self):
        req = TranslationRequest(
            text="Deploy the API to Kubernetes",
            source_language="en",
            target_language="zh",
            glossary_terms={"API": "API", "Kubernetes": "Kubernetes"},
            speaker_name="John",
        )
        assert req.glossary_terms["API"] == "API"
        assert req.speaker_name == "John"

    def test_request_json_roundtrip(self):
        ctx = TranslationContext(text="one", translation="一")
        req = TranslationRequest(
            text="two",
            source_language="en",
            target_language="zh",
            context=[ctx],
            context_window_size=3,
        )
        restored = TranslationRequest.model_validate_json(req.model_dump_json())
        assert restored == req


class TestTranslationResponse:
    def test_response_creation(self):
        resp = TranslationResponse(
            translated_text="Hello world",
            source_language="zh",
            target_language="en",
            model_used="qwen3.5:7b",
            latency_ms=245.3,
        )
        assert resp.translated_text == "Hello world"
        assert resp.model_used == "qwen3.5:7b"
        assert resp.latency_ms == 245.3
        assert resp.quality_score is None  # optional, not always present

    def test_response_with_quality_score(self):
        resp = TranslationResponse(
            translated_text="Hello",
            source_language="zh",
            target_language="en",
            model_used="qwen3.5:7b",
            latency_ms=200.0,
            quality_score=0.85,
        )
        assert resp.quality_score == 0.85
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_translation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write translation models**

```python
# modules/shared/src/livetranslate_common/models/translation.py
"""Translation types shared across orchestration and benchmarking.

TranslationRequest carries rolling context (last N sentence pairs) so
the LLM can resolve pronouns and maintain terminology consistency.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TranslationContext(BaseModel):
    """A previous (source, translation) pair for context continuity."""

    text: str
    translation: str


class TranslationRequest(BaseModel):
    """Request to translate a single segment with rolling context.

    Dual eviction: context is trimmed by both context_window_size (count)
    and max_context_tokens (token budget). See spec line 604.

    glossary_terms: domain-specific term mappings (source→target) from the
    existing GlossaryService. Injected into the LLM prompt to enforce
    consistent terminology. See services/glossary_service.py.
    """

    text: str
    source_language: str
    target_language: str
    context: list[TranslationContext] = Field(default_factory=list)
    context_window_size: int = 5
    max_context_tokens: int = 500
    glossary_terms: dict[str, str] = Field(default_factory=dict)  # {source_term: target_term}
    speaker_name: str | None = None  # for speaker-aware context


class TranslationResponse(BaseModel):
    """Response from the translation module."""

    translated_text: str
    source_language: str
    target_language: str
    model_used: str
    latency_ms: float
    quality_score: float | None = None  # optional BLEU/COMET score from benchmarking
```

Update `modules/shared/src/livetranslate_common/models/__init__.py`:

```python
from livetranslate_common.models.translation import (
    TranslationContext,
    TranslationRequest,
    TranslationResponse,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_translation.py -v`
Expected: PASS (7 tests — 1 context + 4 request + 2 response)

- [ ] **Step 5: Commit**

```bash
git add modules/shared/src/livetranslate_common/models/translation.py modules/shared/tests/test_models_translation.py modules/shared/src/livetranslate_common/models/__init__.py
git commit -m "feat(common): add TranslationRequest, TranslationResponse, TranslationContext types"
```

---

### Task 4: WebSocket message schemas

**Files:**
- Create: `modules/shared/src/livetranslate_common/models/ws_messages.py`
- Create: `modules/shared/tests/test_models_ws_messages.py`
- Modify: `modules/shared/src/livetranslate_common/models/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/shared/tests/test_models_ws_messages.py
"""Tests for WebSocket message schemas.

These schemas define the text-frame protocol between browser ↔ orchestration
and orchestration ↔ transcription service. Both sides must agree on the shape.
"""
import json

from livetranslate_common.models.ws_messages import (
    PROTOCOL_VERSION,
    BackendSwitchedMessage,
    ConfigMessage,
    ConnectedMessage,
    EndMeetingMessage,
    EndMessage,
    EndSessionMessage,
    InterimMessage,
    LanguageDetectedMessage,
    MeetingStartedMessage,
    PromoteToMeetingMessage,
    RecordingStatusMessage,
    SegmentMessage,
    ServiceStatusMessage,
    StartSessionMessage,
    TranslationMessage,
    parse_ws_message,
)


class TestClientMessages:
    def test_start_session(self):
        msg = StartSessionMessage(
            sample_rate=48000,
            channels=2,
            device_id="mic_abc",
        )
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "start_session"
        assert d["sample_rate"] == 48000

    def test_end_session(self):
        msg = EndSessionMessage()
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "end_session"

    def test_promote_to_meeting(self):
        msg = PromoteToMeetingMessage()
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "promote_to_meeting"

    def test_end_meeting(self):
        msg = EndMeetingMessage()
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "end_meeting"


class TestTranscriptionServiceMessages:
    def test_config_message(self):
        msg = ConfigMessage(model="large-v3-turbo", language="en")
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "config"
        assert d["model"] == "large-v3-turbo"

    def test_config_message_auto_detect(self):
        msg = ConfigMessage()
        d = json.loads(msg.model_dump_json())
        assert d["language"] is None  # auto-detect

    def test_end_message(self):
        msg = EndMessage()
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "end"

    def test_language_detected(self):
        msg = LanguageDetectedMessage(language="zh", confidence=0.97)
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "language_detected"
        assert d["language"] == "zh"

    def test_backend_switched(self):
        msg = BackendSwitchedMessage(backend="sensevoice", model="SenseVoiceSmall", language="zh")
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "backend_switched"
        assert d["backend"] == "sensevoice"


class TestServerMessages:
    def test_connected(self):
        msg = ConnectedMessage(session_id="abc-123")
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "connected"
        assert d["protocol_version"] == PROTOCOL_VERSION
        assert d["session_id"] == "abc-123"

    def test_segment(self):
        msg = SegmentMessage(
            text="Hello",
            language="en",
            confidence=0.95,
            stable_text="Hello",
            unstable_text="",
            is_final=True,
            speaker_id=None,
        )
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "segment"
        assert d["is_final"] is True

    def test_interim(self):
        msg = InterimMessage(text="Hel...", confidence=0.7)
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "interim"

    def test_translation(self):
        msg = TranslationMessage(
            text="你好",
            source_lang="en",
            target_lang="zh",
            transcript_id=42,
        )
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "translation"
        assert d["transcript_id"] == 42

    def test_meeting_started(self):
        msg = MeetingStartedMessage(
            session_id="sess-xyz",
            started_at="2026-03-14T10:00:00Z",
        )
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "meeting_started"

    def test_recording_status(self):
        msg = RecordingStatusMessage(recording=True, chunks_written=42)
        d = json.loads(msg.model_dump_json())
        assert d["type"] == "recording_status"
        assert d["recording"] is True
        assert d["chunks_written"] == 42

    def test_service_status(self):
        msg = ServiceStatusMessage(transcription="up", translation="down")
        d = json.loads(msg.model_dump_json())
        assert d["transcription"] == "up"
        assert d["translation"] == "down"


class TestParseMessage:
    def test_parse_start_session(self):
        raw = '{"type": "start_session", "sample_rate": 48000, "channels": 2}'
        msg = parse_ws_message(raw)
        assert isinstance(msg, StartSessionMessage)

    def test_parse_unknown_type_returns_none(self):
        raw = '{"type": "unknown_garbage", "foo": 1}'
        msg = parse_ws_message(raw)
        assert msg is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_ws_messages.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write WebSocket message schemas**

```python
# modules/shared/src/livetranslate_common/models/ws_messages.py
"""WebSocket message schemas for browser ↔ orchestration ↔ transcription.

All text frames are JSON-encoded instances of these models. Binary frames
carry raw Float32Array audio and have no schema here.

PROTOCOL_VERSION is bumped when breaking changes occur. Clients check
this on the ConnectedMessage to verify compatibility.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

PROTOCOL_VERSION = 1

# ── Client → Server ────────────────────────────────────────────

class StartSessionMessage(BaseModel):
    type: Literal["start_session"] = "start_session"
    sample_rate: int
    channels: int
    device_id: str | None = None


class EndSessionMessage(BaseModel):
    type: Literal["end_session"] = "end_session"


class PromoteToMeetingMessage(BaseModel):
    type: Literal["promote_to_meeting"] = "promote_to_meeting"


class EndMeetingMessage(BaseModel):
    type: Literal["end_meeting"] = "end_meeting"


# ── Client → Transcription Service ────────────────────────────

class ConfigMessage(BaseModel):
    """Client sends config to transcription service at session start."""
    type: Literal["config"] = "config"
    model: str | None = None
    language: str | None = None  # None = auto-detect
    initial_prompt: str | None = None  # subject/domain words to bias recognition
    glossary_terms: list[str] | None = None  # domain terms appended to initial_prompt

class EndMessage(BaseModel):
    """Client signals end of audio stream to transcription service."""
    type: Literal["end"] = "end"


# ── Transcription Service → Client ────────────────────────────

class LanguageDetectedMessage(BaseModel):
    """Transcription service reports authoritative LID result."""
    type: Literal["language_detected"] = "language_detected"
    language: str
    confidence: float

class BackendSwitchedMessage(BaseModel):
    """Transcription service reports a backend switch (e.g. language change)."""
    type: Literal["backend_switched"] = "backend_switched"
    backend: str
    model: str
    language: str


# ── Server → Client ────────────────────────────────────────────

class ConnectedMessage(BaseModel):
    type: Literal["connected"] = "connected"
    protocol_version: int = PROTOCOL_VERSION
    session_id: str


class SegmentMessage(BaseModel):
    type: Literal["segment"] = "segment"
    text: str
    language: str
    confidence: float
    stable_text: str
    unstable_text: str
    is_final: bool
    speaker_id: str | None = None


class InterimMessage(BaseModel):
    type: Literal["interim"] = "interim"
    text: str
    confidence: float


class TranslationMessage(BaseModel):
    type: Literal["translation"] = "translation"
    text: str
    source_lang: str
    target_lang: str
    transcript_id: int
    context_used: int = 0  # number of context pairs used for this translation


class MeetingStartedMessage(BaseModel):
    type: Literal["meeting_started"] = "meeting_started"
    session_id: str
    started_at: str


class RecordingStatusMessage(BaseModel):
    type: Literal["recording_status"] = "recording_status"
    recording: bool
    chunks_written: int


class ServiceStatusMessage(BaseModel):
    type: Literal["service_status"] = "service_status"
    transcription: Literal["up", "down"]
    translation: Literal["up", "down"]


# ── Message Parsing ────────────────────────────────────────────

_CLIENT_MESSAGES = {
    "start_session": StartSessionMessage,
    "end_session": EndSessionMessage,
    "promote_to_meeting": PromoteToMeetingMessage,
    "end_meeting": EndMeetingMessage,
    "config": ConfigMessage,
    "end": EndMessage,
}

_SERVER_MESSAGES = {
    "connected": ConnectedMessage,
    "segment": SegmentMessage,
    "interim": InterimMessage,
    "translation": TranslationMessage,
    "meeting_started": MeetingStartedMessage,
    "recording_status": RecordingStatusMessage,
    "service_status": ServiceStatusMessage,
    "language_detected": LanguageDetectedMessage,
    "backend_switched": BackendSwitchedMessage,
}

_ALL_MESSAGES = {**_CLIENT_MESSAGES, **_SERVER_MESSAGES}


def parse_ws_message(raw: str) -> BaseModel | None:
    """Parse a JSON text frame into a typed message, or None if unknown."""
    try:
        data = json.loads(raw)
        msg_type = data.get("type")
        model_cls = _ALL_MESSAGES.get(msg_type)
        if model_cls is None:
            return None
        return model_cls.model_validate(data)
    except (json.JSONDecodeError, Exception):
        return None
```

Update `modules/shared/src/livetranslate_common/models/__init__.py` to add these re-exports:

```python
from livetranslate_common.models.ws_messages import (
    PROTOCOL_VERSION,
    BackendSwitchedMessage,
    ConfigMessage,
    ConnectedMessage,
    EndMeetingMessage,
    EndMessage,
    EndSessionMessage,
    InterimMessage,
    LanguageDetectedMessage,
    MeetingStartedMessage,
    PromoteToMeetingMessage,
    RecordingStatusMessage,
    SegmentMessage,
    ServiceStatusMessage,
    StartSessionMessage,
    TranslationMessage,
    parse_ws_message,
)
```

And add all of these to `__all__`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_ws_messages.py -v`
Expected: PASS (19 tests — 4 client + 5 transcription service + 8 server + 2 parse)

- [ ] **Step 5: Commit**

```bash
git add modules/shared/src/livetranslate_common/models/ws_messages.py modules/shared/tests/test_models_ws_messages.py modules/shared/src/livetranslate_common/models/__init__.py
git commit -m "feat(common): add WebSocket message schemas with protocol versioning"
```

---

### Task 5: BackendConfig for model registry

**Files:**
- Create: `modules/shared/src/livetranslate_common/models/registry.py`
- Create: `modules/shared/tests/test_models_registry.py`
- Modify: `modules/shared/src/livetranslate_common/models/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# modules/shared/tests/test_models_registry.py
"""Tests for BackendConfig — the model registry entry shape."""
import pytest
from livetranslate_common.models.registry import BackendConfig


class TestBackendConfig:
    def test_english_config(self):
        cfg = BackendConfig(
            backend="whisper",
            model="large-v3-turbo",
            compute_type="float16",
            chunk_duration_s=5.0,
            stride_s=4.5,
            overlap_s=0.5,
            vad_threshold=0.5,
            beam_size=1,
            prebuffer_s=0.3,
            batch_profile="realtime",
        )
        assert cfg.backend == "whisper"
        assert cfg.stride_s == cfg.chunk_duration_s - cfg.overlap_s

    def test_chinese_config(self):
        cfg = BackendConfig(
            backend="sensevoice",
            model="SenseVoiceSmall",
            compute_type="float16",
            chunk_duration_s=5.0,
            stride_s=4.0,
            overlap_s=1.0,
            vad_threshold=0.45,
            beam_size=5,
            prebuffer_s=0.5,
            batch_profile="realtime",
        )
        assert cfg.backend == "sensevoice"
        assert cfg.overlap_s == 1.0

    def test_stride_overlap_consistency_enforced(self):
        """stride_s must equal chunk_duration_s - overlap_s."""
        with pytest.raises(ValueError, match="stride_s"):
            BackendConfig(
                backend="whisper",
                model="large-v3-turbo",
                compute_type="float16",
                chunk_duration_s=5.0,
                stride_s=3.0,  # should be 4.5, not 3.0
                overlap_s=0.5,
                vad_threshold=0.5,
                beam_size=1,
                prebuffer_s=0.3,
                batch_profile="realtime",
            )

    def test_zero_chunk_duration_rejected(self):
        """chunk_duration_s must be > 0."""
        with pytest.raises(ValueError):
            BackendConfig(
                backend="whisper",
                model="large-v3-turbo",
                compute_type="float16",
                chunk_duration_s=0,
                stride_s=0,
                overlap_s=0,
                vad_threshold=0.5,
                beam_size=1,
                prebuffer_s=0,
                batch_profile="realtime",
            )

    def test_vad_threshold_out_of_range_rejected(self):
        """vad_threshold must be in [0, 1]."""
        with pytest.raises(ValueError):
            BackendConfig(
                backend="whisper",
                model="large-v3-turbo",
                compute_type="float16",
                chunk_duration_s=5.0,
                stride_s=4.5,
                overlap_s=0.5,
                vad_threshold=1.5,
                beam_size=1,
                prebuffer_s=0.3,
                batch_profile="realtime",
            )

    def test_batch_profile_values(self):
        cfg = BackendConfig(
            backend="whisper",
            model="large-v3-turbo",
            compute_type="float16",
            chunk_duration_s=30.0,
            stride_s=29.0,
            overlap_s=1.0,
            vad_threshold=0.5,
            beam_size=5,
            prebuffer_s=1.0,
            batch_profile="batch",
        )
        assert cfg.batch_profile == "batch"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Write BackendConfig**

```python
# modules/shared/src/livetranslate_common/models/registry.py
"""Model registry types for the transcription service.

BackendConfig is the per-language entry in the YAML model registry.
It controls which backend, model, and chunking parameters are used.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BackendConfig(BaseModel):
    """Configuration for a single backend+model combination in the registry."""

    backend: str
    model: str
    compute_type: str
    chunk_duration_s: float = Field(gt=0)
    stride_s: float = Field(gt=0)
    overlap_s: float = Field(ge=0)
    vad_threshold: float = Field(ge=0.0, le=1.0)
    beam_size: int = Field(ge=1)
    prebuffer_s: float = Field(ge=0)
    batch_profile: Literal["realtime", "batch"] = "realtime"

    @model_validator(mode="after")
    def check_stride_overlap_consistency(self) -> "BackendConfig":
        """Enforce stride_s == chunk_duration_s - overlap_s."""
        expected = self.chunk_duration_s - self.overlap_s
        if abs(self.stride_s - expected) > 1e-6:
            raise ValueError(
                f"stride_s ({self.stride_s}) must equal "
                f"chunk_duration_s ({self.chunk_duration_s}) - overlap_s ({self.overlap_s}) = {expected}"
            )
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_registry.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add modules/shared/src/livetranslate_common/models/registry.py modules/shared/tests/test_models_registry.py modules/shared/src/livetranslate_common/models/__init__.py
git commit -m "feat(common): add BackendConfig for model registry entries"
```

---

## Chunk 2: VAD/Chunking Research

### Task 6: Produce VAD/Chunking Research Document

This is a **research task**, not a coding task. The deliverable is a markdown document.

**Files:**
- Create: `docs/research/vad-chunking-comparison.md`

**Goal:** Compare VAD/chunking approaches used by WhisperX, faster-whisper, and FasterWhisperX. Determine which patterns to adopt for the transcription service's pluggable backend architecture.

- [ ] **Step 1: Research WhisperX's approach**

Read WhisperX source code and documentation. Focus on:
- How it uses pyannote VAD for speech segmentation
- Forced alignment with wav2vec2 for word-level timestamps
- Batch inference strategy
- How chunking parameters are configured

- [ ] **Step 2: Research faster-whisper's approach**

Read faster-whisper source code. Focus on:
- Built-in Silero VAD integration
- `vad_filter` and `vad_parameters` options
- Batched inference via `BatchedInferencePipeline`
- How `chunk_length` and `condition_on_previous_text` work

- [ ] **Step 3: Research FasterWhisperX (if available)**

Check if FasterWhisperX exists as a merged approach. Document findings or note that it's not a maintained project.

- [ ] **Step 4: Write the comparison document**

Create `docs/research/vad-chunking-comparison.md` with:

```markdown
# VAD/Chunking Strategy Comparison

## Date: 2026-03-14
## Status: Research deliverable for Plan 1 (Transcription Service Refactor)

## Purpose
Determine which VAD/chunking patterns to adopt for the pluggable transcription backend architecture.

## Approaches Compared

### 1. WhisperX (pyannote + forced alignment)
[Findings here]

### 2. faster-whisper (Silero VAD + batched inference)
[Findings here]

### 3. FasterWhisperX (merged approach)
[Findings here]

## Comparison Matrix

| Feature | WhisperX | faster-whisper | FasterWhisperX |
|---------|----------|----------------|----------------|
| VAD engine | pyannote | Silero | ? |
| Streaming support | No | Yes (via segments) | ? |
| Batch inference | Yes | Yes | ? |
| Word timestamps | Yes (forced alignment) | Yes (built-in) | ? |
| GPU memory overhead | Higher (pyannote + wav2vec2) | Lower | ? |
| Latency (realtime) | Higher | Lower | ? |

## Recommendation

[Which patterns to adopt and why, with specific implications for BackendConfig fields]

## Impact on BackendConfig

[Any new fields needed, or refinements to existing fields]
```

- [ ] **Step 5: Commit**

```bash
git add docs/research/vad-chunking-comparison.md
git commit -m "docs: VAD/chunking strategy comparison for transcription service refactor"
```

---

## Chunk 3: TypeScript Type Definitions (for Dashboard)

### Task 7: Generate TypeScript types from Pydantic models

**Files:**
- Create: `modules/dashboard-service/src/lib/types/ws-messages.ts`
- Create: `modules/dashboard-service/src/lib/types/transcription.ts`

The TypeScript types must match the Pydantic models exactly. These are hand-written (not auto-generated) to keep the dashboard-service dependency-free from Python tooling.

- [ ] **Step 1: Write TypeScript WebSocket message types**

```typescript
// modules/dashboard-service/src/lib/types/ws-messages.ts

export const PROTOCOL_VERSION = 1;

// ── Client → Server ────────────────────────────────

export interface StartSessionMessage {
  type: 'start_session';
  sample_rate: number;
  channels: number;
  device_id?: string;
}

export interface EndSessionMessage {
  type: 'end_session';
}

export interface PromoteToMeetingMessage {
  type: 'promote_to_meeting';
}

export interface EndMeetingMessage {
  type: 'end_meeting';
}

// ── Client → Transcription Service ─────────────

export interface ConfigMessage {
  type: 'config';
  model?: string;
  language?: string;  // null/undefined = auto-detect
  initial_prompt?: string;  // subject/domain words to bias recognition
  glossary_terms?: string[];  // domain terms appended to initial_prompt
}

export interface EndTranscriptionMessage {
  type: 'end';
}

export type ClientMessage =
  | StartSessionMessage
  | EndSessionMessage
  | PromoteToMeetingMessage
  | EndMeetingMessage
  | ConfigMessage
  | EndTranscriptionMessage;

// ── Server → Client ────────────────────────────────

export interface ConnectedMessage {
  type: 'connected';
  protocol_version: number;
  session_id: string;
}

export interface SegmentMessage {
  type: 'segment';
  text: string;
  language: string;
  confidence: number;
  stable_text: string;
  unstable_text: string;
  is_final: boolean;
  speaker_id: string | null;
}

export interface InterimMessage {
  type: 'interim';
  text: string;
  confidence: number;
}

export interface TranslationMessage {
  type: 'translation';
  text: string;
  source_lang: string;
  target_lang: string;
  transcript_id: number;
  context_used: number;  // number of context pairs used
}

export interface MeetingStartedMessage {
  type: 'meeting_started';
  session_id: string;
  started_at: string;
}

export interface RecordingStatusMessage {
  type: 'recording_status';
  recording: boolean;
  chunks_written: number;
}

export interface ServiceStatusMessage {
  type: 'service_status';
  transcription: 'up' | 'down';
  translation: 'up' | 'down';
}

// ── Transcription Service → Client ─────────────

export interface LanguageDetectedMessage {
  type: 'language_detected';
  language: string;
  confidence: number;
}

export interface BackendSwitchedMessage {
  type: 'backend_switched';
  backend: string;
  model: string;
  language: string;
}

export type ServerMessage =
  | ConnectedMessage
  | SegmentMessage
  | InterimMessage
  | TranslationMessage
  | MeetingStartedMessage
  | RecordingStatusMessage
  | ServiceStatusMessage
  | LanguageDetectedMessage
  | BackendSwitchedMessage;

export function parseServerMessage(raw: string): ServerMessage | null {
  try {
    const data = JSON.parse(raw);
    if (!data || typeof data.type !== 'string') return null;
    // Type-narrow based on known types
    const knownTypes = [
      'connected', 'segment', 'interim', 'translation',
      'meeting_started', 'recording_status', 'service_status',
      'language_detected', 'backend_switched',
    ];
    if (!knownTypes.includes(data.type)) return null;
    return data as ServerMessage;
  } catch {
    return null;
  }
}
```

- [ ] **Step 2: Write TypeScript transcription types**

```typescript
// modules/dashboard-service/src/lib/types/transcription.ts

export interface Segment {
  text: string;
  start_ms: number;
  end_ms: number;
  confidence: number;
  speaker_id: string | null;
}

export interface TranscriptionResult {
  text: string;
  language: string;
  confidence: number;
  segments: Segment[];
  stable_text: string;
  unstable_text: string;
  is_final: boolean;
  is_draft: boolean;
  speaker_id: string | null;
  should_translate: boolean;
  context_text: string;  // previous transcription text fed as context
}
```

- [ ] **Step 3: Update barrel file**

If `modules/dashboard-service/src/lib/types/index.ts` exists, add re-exports:

```typescript
// modules/dashboard-service/src/lib/types/index.ts
export * from './ws-messages';
export * from './transcription';
```

If it doesn't exist, create it with the above content.

- [ ] **Step 4: Typecheck**

Run: `cd modules/dashboard-service && npx tsc --noEmit`
Expected: No type errors. If `tsconfig.json` doesn't include `src/lib/types/`, add the path.

- [ ] **Step 5: Commit**

```bash
git add modules/dashboard-service/src/lib/types/
git commit -m "feat(dashboard): add TypeScript types matching shared Pydantic contracts"
```

---

## Chunk 4: Cross-Module Integration Test

### Task 8: Verify top-level imports work end-to-end

**Files:**
- Create: `modules/shared/tests/test_models_integration.py`

- [ ] **Step 1: Write the cross-module import test**

```python
# modules/shared/tests/test_models_integration.py
"""Integration test: verify all shared models are importable from the top-level package."""


def test_top_level_imports():
    """All shared types must be importable from livetranslate_common directly."""
    from livetranslate_common import (
        AudioChunk,
        BackendConfig,
        ModelInfo,
        Segment,
        TranscriptionResult,
        TranslationContext,
        TranslationRequest,
        TranslationResponse,
    )
    # Verify they're the real classes, not None
    assert TranscriptionResult.__name__ == "TranscriptionResult"
    assert BackendConfig.__name__ == "BackendConfig"


def test_ws_message_imports():
    """WebSocket messages must be importable from livetranslate_common.models.ws_messages."""
    from livetranslate_common.models.ws_messages import (
        PROTOCOL_VERSION,
        BackendSwitchedMessage,
        ConfigMessage,
        ConnectedMessage,
        LanguageDetectedMessage,
        parse_ws_message,
    )
    assert PROTOCOL_VERSION >= 1
    assert parse_ws_message('{"type": "connected", "session_id": "x"}') is not None


def test_models_subpackage_reexports():
    """The models subpackage __init__ must re-export all types."""
    from livetranslate_common.models import (
        AudioChunk,
        BackendConfig,
        ModelInfo,
        Segment,
        TranscriptionResult,
        TranslationContext,
        TranslationRequest,
        TranslationResponse,
    )
    # Quick smoke: construct one from each module
    seg = Segment(text="hi", start_ms=0, end_ms=100, confidence=0.9)
    chunk = AudioChunk(data=b"\x00", timestamp_ms=0, sequence_number=0, source_id="test")
    ctx = TranslationContext(text="a", translation="b")
    assert seg.text == "hi"
    assert chunk.source_id == "test"
    assert ctx.translation == "b"
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/shared/tests/test_models_integration.py -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Commit**

```bash
git add modules/shared/tests/test_models_integration.py
git commit -m "test(common): add cross-module integration test for shared model imports"
```

---

## Summary

After completing Plan 0:
- `ARCHITECTURE.md` documents the new 3-service topology
- `livetranslate-common` exports: `TranscriptionResult`, `Segment`, `ModelInfo`, `AudioChunk`, `MeetingAudioStream`, `TranslationRequest`, `TranslationResponse`, `TranslationContext`, `BackendConfig`, plus all WebSocket message types (including glossary/subject words support)
- `dashboard-service` has matching TypeScript definitions
- VAD/chunking research document exists in `docs/research/`
- All 4 parallel plans can begin immediately — they import from stable, tested contracts

**Total estimated tasks:** 9 tasks (Task 0-8), ~33 steps
**Branch:** `plan-0/shared-contracts`
