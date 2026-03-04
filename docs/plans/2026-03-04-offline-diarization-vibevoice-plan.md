# Offline Diarization with VibeVoice-ASR — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add offline speaker diarization powered by Microsoft VibeVoice-ASR (served via vLLM on a local GPU box), with auto-trigger rules, speaker name mapping, transcript merge, and full dashboard UI.

**Architecture:** VibeVoice runs as a thin vLLM inference server (OpenAI-compatible API) on a LAN GPU box. All business logic — job queue, speaker mapping, transcript merge, auto-rules — lives in the orchestration service. Dashboard provides job management, speaker enrollment, rule config, and transcript comparison.

**Tech Stack:** FastAPI, AsyncOpenAI client, Alembic/PostgreSQL, SvelteKit, Pydantic v2, structlog

**Design doc:** `docs/plans/2026-03-04-offline-diarization-vibevoice-design.md`

---

## Phase 1: Database & Config Foundation

### Task 1: Alembic Migration — diarization_jobs + speaker_profiles tables

**Files:**
- Create: `modules/orchestration-service/alembic/versions/010_diarization_tables.py`

**Context:** Current chain head is `009_meeting_retry_cols`. Revision IDs MUST be <=32 chars. See `modules/orchestration-service/CLAUDE.md` for migration rules.

**Step 1: Verify migration chain health**

Run: `cd modules/orchestration-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic history`
Expected: Chain ending at `009_meeting_retry_cols (head)`

Run: `grep -n "^revision\|^down_revision" alembic/versions/*.py | tail -4`
Expected: Confirm `009_meeting_retry_cols` is the current head revision string.

**Step 2: Create migration file**

```python
"""Add diarization jobs and speaker profiles tables

Revision ID: 010_diarization_tables
Revises: 009_meeting_retry_cols
Create Date: 2026-03-04
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "010_diarization_tables"
down_revision = "009_meeting_retry_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "diarization_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("meeting_id", sa.Integer, sa.ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("triggered_by", sa.String(20), nullable=False, server_default="manual"),
        sa.Column("rule_matched", JSONB, nullable=True),
        sa.Column("audio_url", sa.Text, nullable=True),
        sa.Column("audio_size_bytes", sa.BigInteger, nullable=True),
        sa.Column("raw_segments", JSONB, nullable=True),
        sa.Column("detected_language", sa.String(10), nullable=True),
        sa.Column("num_speakers_detected", sa.Integer, nullable=True),
        sa.Column("processing_time_seconds", sa.Float, nullable=True),
        sa.Column("speaker_map", JSONB, nullable=True),
        sa.Column("unmapped_speakers", JSONB, nullable=True),
        sa.Column("merge_applied", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("merge_applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
    )
    op.create_index("ix_diarization_jobs_meeting", "diarization_jobs", ["meeting_id"])
    op.create_index("ix_diarization_jobs_status", "diarization_jobs", ["status"])

    op.create_table(
        "speaker_profiles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("embedding", JSONB, nullable=True),
        sa.Column("enrollment_source", sa.String(50), nullable=False, server_default="manual"),
        sa.Column("sample_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_speaker_profiles_email", "speaker_profiles", ["email"])
    op.create_index("ix_speaker_profiles_name", "speaker_profiles", ["name"])


def downgrade() -> None:
    op.drop_index("ix_speaker_profiles_name", table_name="speaker_profiles")
    op.drop_index("ix_speaker_profiles_email", table_name="speaker_profiles")
    op.drop_table("speaker_profiles")
    op.drop_index("ix_diarization_jobs_status", table_name="diarization_jobs")
    op.drop_index("ix_diarization_jobs_meeting", table_name="diarization_jobs")
    op.drop_table("diarization_jobs")
```

**Step 3: Run migration**

Run: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic upgrade head`
Expected: `INFO  [alembic.runtime.migration] Running upgrade 009_meeting_retry_cols -> 010_diarization_tables`

**Step 4: Verify**

Run: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic current`
Expected: `010_diarization_tables (head)`

**Step 5: Commit**

```bash
git add modules/orchestration-service/alembic/versions/010_diarization_tables.py
git commit -m "feat(db): add diarization_jobs and speaker_profiles tables"
```

---

### Task 2: Pydantic Models for Diarization

**Files:**
- Create: `modules/orchestration-service/src/models/diarization.py`
- Test: `modules/orchestration-service/tests/diarization/test_diarization_models.py`

**Step 1: Write failing test**

Create `modules/orchestration-service/tests/diarization/__init__.py` (empty).

```python
"""Tests for diarization Pydantic models."""

from models.diarization import (
    DiarizationJobCreate,
    DiarizationJobResponse,
    DiarizationJobStatus,
    DiarizationRules,
    SpeakerMapEntry,
    SpeakerMergeRequest,
    SpeakerProfileCreate,
    SpeakerProfileResponse,
    TranscribeResponse,
    TranscribeSegment,
    TranscriptCompareResponse,
)


def test_transcribe_segment_from_vibevoice():
    seg = TranscribeSegment(speaker=0, start=0.52, end=3.21, text="Hello world")
    assert seg.speaker == 0
    assert seg.start == 0.52
    assert seg.end == 3.21
    assert seg.text == "Hello world"


def test_transcribe_response_parsing():
    resp = TranscribeResponse(
        segments=[
            TranscribeSegment(speaker=0, start=0.0, end=1.0, text="Hi"),
            TranscribeSegment(speaker=1, start=1.0, end=2.0, text="Hey"),
        ],
        detected_language="en",
        num_speakers=2,
        duration_seconds=120.0,
        processing_time_seconds=45.2,
    )
    assert len(resp.segments) == 2
    assert resp.num_speakers == 2


def test_diarization_job_create():
    job = DiarizationJobCreate(meeting_id=42)
    assert job.meeting_id == 42
    assert job.hotwords is None


def test_diarization_job_create_with_hotwords():
    job = DiarizationJobCreate(meeting_id=42, hotwords=["sprint", "deploy"])
    assert job.hotwords == ["sprint", "deploy"]


def test_diarization_job_status_enum():
    assert DiarizationJobStatus.QUEUED == "queued"
    assert DiarizationJobStatus.PROCESSING == "processing"
    assert DiarizationJobStatus.COMPLETED == "completed"
    assert DiarizationJobStatus.FAILED == "failed"


def test_speaker_map_entry():
    entry = SpeakerMapEntry(name="Eric", confidence=0.92, method="voice_profile")
    assert entry.name == "Eric"
    assert entry.confidence == 0.92


def test_diarization_rules():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=["eric@*"],
        title_patterns=["dev weekly*"],
        min_duration_minutes=5,
    )
    assert rules.enabled is True
    assert rules.exclude_empty is True  # default


def test_speaker_profile_create():
    profile = SpeakerProfileCreate(name="Eric Chen", email="eric@example.com")
    assert profile.name == "Eric Chen"


def test_speaker_merge_request():
    req = SpeakerMergeRequest(source_id=3, target_id=1)
    assert req.source_id == 3
    assert req.target_id == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.diarization'`

**Step 3: Write models**

```python
"""Pydantic models for diarization service."""

from enum import StrEnum

from pydantic import BaseModel, Field


class DiarizationJobStatus(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    MAPPING = "mapping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# --- VibeVoice inference models ---


class TranscribeSegment(BaseModel):
    """Single segment from VibeVoice-ASR output."""

    speaker: int = Field(description="Speaker ID (0-indexed)")
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed text")


class TranscribeResponse(BaseModel):
    """Parsed response from VibeVoice-ASR inference."""

    segments: list[TranscribeSegment]
    detected_language: str = "en"
    num_speakers: int = 0
    duration_seconds: float = 0.0
    processing_time_seconds: float = 0.0


# --- Speaker mapping models ---


class SpeakerMapEntry(BaseModel):
    """Mapping from a VibeVoice speaker ID to a named person."""

    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(description="How this mapping was made: 'fireflies_crossref', 'voice_profile', 'manual'")


class SpeakerProfileCreate(BaseModel):
    name: str = Field(max_length=255)
    email: str | None = Field(default=None, max_length=255)


class SpeakerProfileResponse(BaseModel):
    id: int
    name: str
    email: str | None
    enrollment_source: str
    sample_count: int


class SpeakerMergeRequest(BaseModel):
    """Merge source speaker profile into target (keeps target, deletes source)."""

    source_id: int
    target_id: int


# --- Diarization job models ---


class DiarizationJobCreate(BaseModel):
    meeting_id: int
    hotwords: list[str] | None = None


class DiarizationJobResponse(BaseModel):
    id: int
    meeting_id: int
    status: DiarizationJobStatus
    triggered_by: str
    detected_language: str | None = None
    num_speakers_detected: int | None = None
    processing_time_seconds: float | None = None
    speaker_map: dict[str, SpeakerMapEntry] | None = None
    unmapped_speakers: list[int] | None = None
    merge_applied: bool = False
    error_message: str | None = None
    created_at: str | None = None
    completed_at: str | None = None


# --- Rule models ---


class DiarizationRules(BaseModel):
    enabled: bool = True
    participant_patterns: list[str] = Field(default_factory=list)
    title_patterns: list[str] = Field(default_factory=list)
    min_duration_minutes: int = 5
    exclude_empty: bool = True


# --- Transcript comparison ---


class TranscriptCompareResponse(BaseModel):
    """Side-by-side comparison of Fireflies vs VibeVoice transcript."""

    meeting_id: int
    fireflies_sentences: list[dict]
    vibevoice_segments: list[dict]
    speaker_map: dict[str, SpeakerMapEntry] | None = None
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_models.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/models/diarization.py \
       modules/orchestration-service/tests/diarization/__init__.py \
       modules/orchestration-service/tests/diarization/test_diarization_models.py
git commit -m "feat: add diarization Pydantic models with tests"
```

---

### Task 3: DiarizationSettings Config

**Files:**
- Modify: `modules/orchestration-service/src/config.py` (add `DiarizationSettings` class)
- Test: `modules/orchestration-service/tests/diarization/test_diarization_config.py`

**Context:** Follow the `FirefliesSettings` pattern at `config.py:311-388`. Uses `BaseSettings` with `env_prefix`, `Field(default=..., description="...")`, and `ConfigDict`.

**Step 1: Write failing test**

```python
"""Tests for DiarizationSettings configuration."""

import os

from config import DiarizationSettings


def test_default_settings():
    settings = DiarizationSettings()
    assert settings.vibevoice_url == "http://localhost:8000/v1"
    assert settings.enabled is False
    assert settings.max_concurrent_jobs == 1
    assert settings.auto_apply_threshold == 0.85


def test_env_override(monkeypatch):
    monkeypatch.setenv("DIARIZATION_VIBEVOICE_URL", "http://192.168.1.50:8000/v1")
    monkeypatch.setenv("DIARIZATION_ENABLED", "true")
    settings = DiarizationSettings()
    assert settings.vibevoice_url == "http://192.168.1.50:8000/v1"
    assert settings.enabled is True


def test_hotwords_from_comma_string(monkeypatch):
    monkeypatch.setenv("DIARIZATION_HOTWORDS", "sprint,deploy,LiveTranslate")
    settings = DiarizationSettings()
    assert settings.hotwords == ["sprint", "deploy", "LiveTranslate"]


def test_has_vibevoice_url():
    settings = DiarizationSettings()
    assert settings.has_vibevoice_url() is True

    settings.vibevoice_url = ""
    assert settings.has_vibevoice_url() is False
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'DiarizationSettings' from 'config'`

**Step 3: Add DiarizationSettings to config.py**

Add after the `FirefliesSettings` class (around line 388):

```python
class DiarizationSettings(BaseSettings):
    """Offline diarization with VibeVoice-ASR configuration."""

    enabled: bool = Field(default=False, description="Enable offline diarization service")
    vibevoice_url: str = Field(
        default="http://localhost:8000/v1",
        description="VibeVoice vLLM server URL (LAN address)",
    )
    hotwords: list[str] = Field(
        default_factory=list,
        description="Domain-specific hotwords for better recognition",
    )
    max_concurrent_jobs: int = Field(default=1, description="Max concurrent diarization jobs")
    auto_apply_threshold: float = Field(
        default=0.85,
        description="Min speaker mapping confidence to auto-apply without manual review",
    )
    min_confidence_auto_assign: float = Field(
        default=0.80,
        description="Min confidence for auto-assigning speaker names",
    )
    auto_enroll_speakers: bool = Field(
        default=True,
        description="Auto-enroll speaker profiles from manual assignments",
    )
    fireflies_crossref_enabled: bool = Field(
        default=True,
        description="Cross-reference Fireflies participant list for speaker names",
    )

    @field_validator("hotwords", mode="before")
    @classmethod
    def parse_hotwords(cls, v):
        if isinstance(v, str):
            return [w.strip() for w in v.split(",") if w.strip()]
        return v

    def has_vibevoice_url(self) -> bool:
        return bool(self.vibevoice_url and self.vibevoice_url.strip())

    model_config = ConfigDict(
        env_prefix="DIARIZATION_",
        env_file=".env",
        extra="ignore",
    )
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/config.py \
       modules/orchestration-service/tests/diarization/test_diarization_config.py
git commit -m "feat: add DiarizationSettings config class"
```

---

## Phase 2: VibeVoice Client & Core Pipeline

### Task 4: VibeVoice Client

**Files:**
- Create: `modules/orchestration-service/src/clients/vibevoice_client.py`
- Test: `modules/orchestration-service/tests/diarization/test_vibevoice_client.py`

**Context:** The VibeVoice vLLM server exposes an OpenAI-compatible API at `/v1/chat/completions`. We use `openai.AsyncOpenAI` to talk to it, same pattern as translation service. The client sends audio and receives parsed diarized segments.

**Step 1: Write failing test**

```python
"""Tests for VibeVoice client."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from clients.vibevoice_client import VibeVoiceClient, VibeVoiceError
from models.diarization import TranscribeResponse


@pytest.fixture
def client():
    return VibeVoiceClient(base_url="http://localhost:8000/v1")


def test_client_init(client):
    assert client.base_url == "http://localhost:8000/v1"


def test_client_init_strips_trailing_slash():
    c = VibeVoiceClient(base_url="http://localhost:8000/v1/")
    assert c.base_url == "http://localhost:8000/v1"


def test_parse_segments_from_vibevoice_output():
    """Test parsing VibeVoice's structured output into TranscribeResponse."""
    raw_output = json.dumps([
        {"speaker": 0, "start": 0.5, "end": 3.2, "text": "Hello there"},
        {"speaker": 1, "start": 3.5, "end": 7.0, "text": "Hi, how are you"},
    ])
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output(raw_output, duration_seconds=60.0, processing_time=10.0)
    assert isinstance(result, TranscribeResponse)
    assert len(result.segments) == 2
    assert result.segments[0].speaker == 0
    assert result.segments[0].text == "Hello there"
    assert result.num_speakers == 2
    assert result.duration_seconds == 60.0


def test_parse_segments_deduplicates_speakers():
    raw_output = json.dumps([
        {"speaker": 0, "start": 0.0, "end": 1.0, "text": "A"},
        {"speaker": 1, "start": 1.0, "end": 2.0, "text": "B"},
        {"speaker": 0, "start": 2.0, "end": 3.0, "text": "C"},
    ])
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output(raw_output, duration_seconds=3.0, processing_time=1.0)
    assert result.num_speakers == 2


def test_parse_empty_output():
    client = VibeVoiceClient(base_url="http://localhost:8000/v1")
    result = client.parse_vibevoice_output("[]", duration_seconds=0.0, processing_time=0.0)
    assert len(result.segments) == 0
    assert result.num_speakers == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_vibevoice_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clients.vibevoice_client'`

**Step 3: Write client**

```python
"""
VibeVoice-ASR vLLM Client

Thin HTTP client for the VibeVoice-ASR model served via vLLM.
The vLLM server exposes an OpenAI-compatible API at /v1/chat/completions.

This client handles:
- Sending audio data to the inference server
- Parsing the structured (speaker, timestamp, text) output
- Health checks
"""

import json
import time
from typing import Any

import aiohttp
from livetranslate_common.logging import get_logger
from models.diarization import TranscribeResponse, TranscribeSegment

logger = get_logger()


class VibeVoiceError(Exception):
    """Error from VibeVoice inference."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class VibeVoiceClient:
    """Client for VibeVoice-ASR served via vLLM (OpenAI-compatible API)."""

    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> dict[str, Any]:
        """Check if VibeVoice vLLM server is healthy."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {"status": "healthy", "models": data}
                return {"status": "unhealthy", "status_code": resp.status}
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    async def transcribe(
        self,
        audio_path: str,
        hotwords: list[str] | None = None,
        language: str | None = None,
    ) -> TranscribeResponse:
        """Send audio file to VibeVoice for transcription + diarization.

        Args:
            audio_path: Path to audio file on disk.
            hotwords: Optional domain-specific terms for better recognition.
            language: Optional language hint (default: auto-detect).

        Returns:
            TranscribeResponse with parsed segments.
        """
        session = await self._get_session()
        start_time = time.monotonic()

        # Build multipart form data
        data = aiohttp.FormData()
        data.add_field("file", open(audio_path, "rb"), filename=audio_path.split("/")[-1])
        if hotwords:
            data.add_field("hotwords", ",".join(hotwords))
        if language:
            data.add_field("language", language)

        logger.info("vibevoice_transcribe_start", audio_path=audio_path, hotwords=hotwords)

        try:
            # Long timeout — 60min audio could take 15+ minutes to process
            timeout = aiohttp.ClientTimeout(total=1800)
            async with session.post(
                f"{self.base_url}/chat/completions",
                data=data,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise VibeVoiceError(f"VibeVoice inference failed: {error_text}", status_code=resp.status)

                result = await resp.json()
                processing_time = time.monotonic() - start_time

                # Extract the model's text output from OpenAI-compatible response
                raw_output = result.get("choices", [{}])[0].get("message", {}).get("content", "[]")

                # Parse audio duration from response metadata if available
                duration = result.get("usage", {}).get("audio_duration", 0.0)

                parsed = self.parse_vibevoice_output(raw_output, duration, processing_time)
                logger.info(
                    "vibevoice_transcribe_complete",
                    segments=len(parsed.segments),
                    speakers=parsed.num_speakers,
                    detected_language=parsed.detected_language,
                    processing_time=round(processing_time, 1),
                )
                return parsed

        except VibeVoiceError:
            raise
        except Exception as e:
            raise VibeVoiceError(f"VibeVoice request failed: {e}") from e

    def parse_vibevoice_output(
        self,
        raw_output: str,
        duration_seconds: float = 0.0,
        processing_time: float = 0.0,
    ) -> TranscribeResponse:
        """Parse VibeVoice's structured JSON output into TranscribeResponse.

        VibeVoice outputs a JSON array of segments:
        [{"speaker": 0, "start": 0.5, "end": 3.2, "text": "..."}, ...]
        """
        try:
            raw_segments = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
        except json.JSONDecodeError:
            logger.warning("vibevoice_parse_failed", raw_output=raw_output[:200])
            raw_segments = []

        if not isinstance(raw_segments, list):
            raw_segments = []

        segments = [
            TranscribeSegment(
                speaker=seg.get("speaker", 0),
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", ""),
            )
            for seg in raw_segments
            if seg.get("text", "").strip()
        ]

        unique_speakers = {seg.speaker for seg in segments}

        return TranscribeResponse(
            segments=segments,
            num_speakers=len(unique_speakers),
            duration_seconds=duration_seconds,
            processing_time_seconds=processing_time,
        )
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_vibevoice_client.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/clients/vibevoice_client.py \
       modules/orchestration-service/tests/diarization/test_vibevoice_client.py
git commit -m "feat: add VibeVoice-ASR vLLM client with output parsing"
```

---

### Task 5: Auto-Trigger Rules Engine

**Files:**
- Create: `modules/orchestration-service/src/services/diarization/__init__.py`
- Create: `modules/orchestration-service/src/services/diarization/rules.py`
- Test: `modules/orchestration-service/tests/diarization/test_diarization_rules.py`

**Context:** After Fireflies syncs a meeting, we evaluate rules to decide if it should be auto-queued for diarization. Rules match on participant email patterns (glob-style) and meeting title patterns.

**Step 1: Write failing test**

```python
"""Tests for diarization auto-trigger rules."""

from models.diarization import DiarizationRules
from services.diarization.rules import evaluate_rules


def test_no_match_when_disabled():
    rules = DiarizationRules(enabled=False, participant_patterns=["*"])
    meeting = {"title": "anything", "participants": ["eric@test.com"], "duration": 60}
    assert evaluate_rules(meeting, rules) is None


def test_match_participant_pattern():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=["eric@*"],
        title_patterns=[],
    )
    meeting = {"title": "standup", "participants": ["eric@company.com", "alice@company.com"], "duration": 30}
    result = evaluate_rules(meeting, rules)
    assert result is not None
    assert result["match_type"] == "participant"
    assert "eric@company.com" in result["matched_value"]


def test_match_title_pattern():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=[],
        title_patterns=["dev weekly*"],
    )
    meeting = {"title": "Dev Weekly Sync - March", "participants": [], "duration": 30}
    result = evaluate_rules(meeting, rules)
    assert result is not None
    assert result["match_type"] == "title"


def test_title_match_case_insensitive():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=[],
        title_patterns=["1:1*"],
    )
    meeting = {"title": "1:1 with Eric", "participants": [], "duration": 10}
    result = evaluate_rules(meeting, rules)
    assert result is not None


def test_skip_short_meetings():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=["eric@*"],
        min_duration_minutes=5,
    )
    meeting = {"title": "quick chat", "participants": ["eric@company.com"], "duration": 180}  # 3 min (in seconds)
    assert evaluate_rules(meeting, rules) is None


def test_skip_empty_meetings():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=["eric@*"],
        exclude_empty=True,
    )
    meeting = {"title": "standup", "participants": ["eric@company.com"], "duration": 600, "sentence_count": 0}
    assert evaluate_rules(meeting, rules) is None


def test_no_rules_no_match():
    rules = DiarizationRules(
        enabled=True,
        participant_patterns=[],
        title_patterns=[],
    )
    meeting = {"title": "standup", "participants": ["eric@co.com"], "duration": 600}
    assert evaluate_rules(meeting, rules) is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_rules.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write rules engine**

Create `modules/orchestration-service/src/services/diarization/__init__.py`:
```python
"""Diarization pipeline services."""
```

Create `modules/orchestration-service/src/services/diarization/rules.py`:

```python
"""Auto-trigger rules for offline diarization.

Evaluates meeting metadata against configured rules to decide
whether a meeting should be auto-queued for diarization.
"""

from fnmatch import fnmatch
from typing import Any

from models.diarization import DiarizationRules


def evaluate_rules(meeting: dict[str, Any], rules: DiarizationRules) -> dict[str, Any] | None:
    """Evaluate a meeting against diarization rules.

    Args:
        meeting: Dict with keys: title, participants (list[str]), duration (seconds),
                 sentence_count (optional).
        rules: Configured diarization rules.

    Returns:
        Dict describing the match (match_type, matched_value, matched_pattern) or None.
    """
    if not rules.enabled:
        return None

    duration_seconds = meeting.get("duration", 0)
    if duration_seconds < rules.min_duration_minutes * 60:
        return None

    if rules.exclude_empty and meeting.get("sentence_count", 1) == 0:
        return None

    # Check participant patterns
    for pattern in rules.participant_patterns:
        for participant in meeting.get("participants", []):
            if fnmatch(participant.lower(), pattern.lower()):
                return {
                    "match_type": "participant",
                    "matched_value": participant,
                    "matched_pattern": pattern,
                }

    # Check title patterns
    title = meeting.get("title", "")
    for pattern in rules.title_patterns:
        if fnmatch(title.lower(), pattern.lower()):
            return {
                "match_type": "title",
                "matched_value": title,
                "matched_pattern": pattern,
            }

    return None
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_rules.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/__init__.py \
       modules/orchestration-service/src/services/diarization/rules.py \
       modules/orchestration-service/tests/diarization/test_diarization_rules.py
git commit -m "feat: add diarization auto-trigger rules engine"
```

---

### Task 6: Speaker Mapper

**Files:**
- Create: `modules/orchestration-service/src/services/diarization/speaker_mapper.py`
- Test: `modules/orchestration-service/tests/diarization/test_speaker_mapper.py`

**Context:** Maps VibeVoice anonymous speaker IDs (0, 1, 2...) to real names using three layered strategies: Fireflies cross-reference, voice profile matching, and manual assignment.

**Step 1: Write failing test**

```python
"""Tests for speaker mapping strategies."""

from models.diarization import SpeakerMapEntry, TranscribeSegment
from services.diarization.speaker_mapper import SpeakerMapper


def _make_segments() -> list[TranscribeSegment]:
    return [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello from speaker zero"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="Hi from speaker one"),
        TranscribeSegment(speaker=0, start=10.0, end=15.0, text="More from zero"),
    ]


def test_crossref_fireflies_by_timestamp_overlap():
    """Match VibeVoice speakers to Fireflies speakers via timestamp overlap."""
    mapper = SpeakerMapper()
    segments = _make_segments()
    fireflies_sentences = [
        {"speaker_name": "Eric Chen", "start_time": 0.0, "end_time": 5.0, "text": "Hello"},
        {"speaker_name": "Thomas", "start_time": 5.0, "end_time": 10.0, "text": "Hi"},
    ]
    result = mapper.crossref_fireflies(segments, fireflies_sentences)
    assert 0 in result
    assert result[0].name == "Eric Chen"
    assert result[0].method == "fireflies_crossref"
    assert 1 in result
    assert result[1].name == "Thomas"


def test_crossref_handles_no_overlap():
    mapper = SpeakerMapper()
    segments = [TranscribeSegment(speaker=0, start=100.0, end=105.0, text="Late")]
    fireflies_sentences = [
        {"speaker_name": "Eric", "start_time": 0.0, "end_time": 5.0, "text": "Early"},
    ]
    result = mapper.crossref_fireflies(segments, fireflies_sentences)
    assert len(result) == 0


def test_crossref_picks_best_overlap():
    """When a VibeVoice speaker overlaps multiple Fireflies speakers, pick the one with most overlap."""
    mapper = SpeakerMapper()
    segments = [TranscribeSegment(speaker=0, start=0.0, end=10.0, text="Long segment")]
    fireflies_sentences = [
        {"speaker_name": "Eric", "start_time": 0.0, "end_time": 3.0, "text": "Short"},
        {"speaker_name": "Thomas", "start_time": 2.0, "end_time": 10.0, "text": "Longer"},
    ]
    result = mapper.crossref_fireflies(segments, fireflies_sentences)
    assert result[0].name == "Thomas"  # more overlap


def test_merge_maps_combines_strategies():
    mapper = SpeakerMapper()
    crossref = {0: SpeakerMapEntry(name="Eric", confidence=0.7, method="fireflies_crossref")}
    voice = {1: SpeakerMapEntry(name="Thomas", confidence=0.9, method="voice_profile")}
    manual = {2: SpeakerMapEntry(name="Alice", confidence=1.0, method="manual")}

    merged = mapper.merge_maps([crossref, voice, manual])
    assert merged[0].name == "Eric"
    assert merged[1].name == "Thomas"
    assert merged[2].name == "Alice"


def test_merge_maps_higher_confidence_wins():
    mapper = SpeakerMapper()
    low = {0: SpeakerMapEntry(name="Unknown", confidence=0.3, method="fireflies_crossref")}
    high = {0: SpeakerMapEntry(name="Eric", confidence=0.95, method="voice_profile")}

    merged = mapper.merge_maps([low, high])
    assert merged[0].name == "Eric"
    assert merged[0].confidence == 0.95


def test_find_unmapped_speakers():
    mapper = SpeakerMapper()
    segments = _make_segments()  # speakers 0 and 1
    speaker_map = {0: SpeakerMapEntry(name="Eric", confidence=0.9, method="manual")}
    unmapped = mapper.find_unmapped(segments, speaker_map)
    assert unmapped == [1]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_speaker_mapper.py -v`
Expected: FAIL

**Step 3: Write speaker mapper**

```python
"""Speaker mapping: VibeVoice anonymous IDs → real names.

Three-strategy layered approach:
1. Fireflies cross-reference (timestamp overlap)
2. Voice profile matching (embedding similarity)
3. Manual assignment (from dashboard)
"""

from collections import defaultdict

from livetranslate_common.logging import get_logger
from models.diarization import SpeakerMapEntry, TranscribeSegment

logger = get_logger()


class SpeakerMapper:
    """Maps VibeVoice speaker IDs to named people."""

    def crossref_fireflies(
        self,
        segments: list[TranscribeSegment],
        fireflies_sentences: list[dict],
    ) -> dict[int, SpeakerMapEntry]:
        """Strategy 1: Match via timestamp overlap with Fireflies sentences.

        For each VibeVoice speaker, find the Fireflies speaker with the most
        overlapping time, weighted by overlap duration.
        """
        if not segments or not fireflies_sentences:
            return {}

        # Accumulate overlap per (vibevoice_speaker, fireflies_speaker) pair
        overlap_map: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for seg in segments:
            for ff_sent in fireflies_sentences:
                ff_speaker = ff_sent.get("speaker_name", "")
                if not ff_speaker:
                    continue
                ff_start = ff_sent.get("start_time", 0.0)
                ff_end = ff_sent.get("end_time", 0.0)

                # Calculate overlap duration
                overlap_start = max(seg.start, ff_start)
                overlap_end = min(seg.end, ff_end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > 0:
                    overlap_map[seg.speaker][ff_speaker] += overlap

        result: dict[int, SpeakerMapEntry] = {}
        for speaker_id, name_overlaps in overlap_map.items():
            if not name_overlaps:
                continue
            best_name = max(name_overlaps, key=name_overlaps.get)
            total_overlap = sum(name_overlaps.values())
            best_overlap = name_overlaps[best_name]
            confidence = best_overlap / total_overlap if total_overlap > 0 else 0.0

            result[speaker_id] = SpeakerMapEntry(
                name=best_name,
                confidence=round(min(confidence, 1.0), 3),
                method="fireflies_crossref",
            )

        return result

    def merge_maps(
        self,
        maps: list[dict[int, SpeakerMapEntry]],
    ) -> dict[int, SpeakerMapEntry]:
        """Merge multiple speaker maps, keeping highest confidence per speaker.

        Maps are processed in order; higher confidence always wins.
        """
        merged: dict[int, SpeakerMapEntry] = {}
        for m in maps:
            for speaker_id, entry in m.items():
                if speaker_id not in merged or entry.confidence > merged[speaker_id].confidence:
                    merged[speaker_id] = entry
        return merged

    def find_unmapped(
        self,
        segments: list[TranscribeSegment],
        speaker_map: dict[int, SpeakerMapEntry],
    ) -> list[int]:
        """Find speaker IDs that have no mapping."""
        all_speakers = {seg.speaker for seg in segments}
        return sorted(all_speakers - set(speaker_map.keys()))
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_speaker_mapper.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/speaker_mapper.py \
       modules/orchestration-service/tests/diarization/test_speaker_mapper.py
git commit -m "feat: add speaker mapper with Fireflies cross-reference strategy"
```

---

### Task 7: Speaker Merge (Over-Segmentation Fix)

**Files:**
- Create: `modules/orchestration-service/src/services/diarization/speaker_merge.py`
- Test: `modules/orchestration-service/tests/diarization/test_speaker_merge.py`

**Context:** VibeVoice may detect more speakers than actually present (over-segmentation). This module detects likely duplicates and merges them.

**Step 1: Write failing test**

```python
"""Tests for speaker merge (over-segmentation detection)."""

from models.diarization import TranscribeSegment
from services.diarization.speaker_merge import detect_merge_candidates, apply_merge


def test_detect_candidates_by_low_word_count():
    """Speakers with very few words are likely fragments of another speaker."""
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=30.0, text="Long speech from main speaker about many topics"),
        TranscribeSegment(speaker=0, start=30.0, end=60.0, text="Still talking about things and stuff"),
        TranscribeSegment(speaker=1, start=60.0, end=90.0, text="Another person talking at length here"),
        TranscribeSegment(speaker=2, start=90.0, end=92.0, text="Yeah"),
    ]
    candidates = detect_merge_candidates(segments, min_word_ratio=0.05)
    assert len(candidates) >= 1
    assert any(c["source"] == 2 for c in candidates)


def test_no_candidates_when_balanced():
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=30.0, text=" ".join(["word"] * 50)),
        TranscribeSegment(speaker=1, start=30.0, end=60.0, text=" ".join(["word"] * 45)),
    ]
    candidates = detect_merge_candidates(segments)
    assert len(candidates) == 0


def test_apply_merge_replaces_speaker_ids():
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello"),
        TranscribeSegment(speaker=2, start=5.0, end=6.0, text="Yeah"),
        TranscribeSegment(speaker=1, start=6.0, end=10.0, text="World"),
    ]
    merged = apply_merge(segments, source_speaker=2, target_speaker=0)
    assert all(seg.speaker != 2 for seg in merged)
    assert merged[1].speaker == 0  # was speaker 2, now 0


def test_apply_merge_preserves_order():
    segments = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="A"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="B"),
    ]
    merged = apply_merge(segments, source_speaker=1, target_speaker=0)
    assert merged[0].start < merged[1].start
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_speaker_merge.py -v`
Expected: FAIL

**Step 3: Write speaker merge**

```python
"""Speaker merge: detect and fix over-segmentation.

VibeVoice may detect more speakers than are actually present.
This module identifies likely duplicate speakers and provides
merge operations.
"""

from collections import defaultdict
from typing import Any

from livetranslate_common.logging import get_logger
from models.diarization import TranscribeSegment

logger = get_logger()


def detect_merge_candidates(
    segments: list[TranscribeSegment],
    min_word_ratio: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect speakers that are likely fragments of another speaker.

    A speaker is a merge candidate if their total word count is below
    min_word_ratio of the total words across all speakers.

    Returns list of {"source": speaker_to_merge, "suggested_target": best_match,
                     "word_count": n, "word_ratio": r}
    """
    if not segments:
        return []

    # Count words per speaker
    words_per_speaker: dict[int, int] = defaultdict(int)
    for seg in segments:
        words_per_speaker[seg.speaker] += len(seg.text.split())

    total_words = sum(words_per_speaker.values())
    if total_words == 0:
        return []

    candidates = []
    # Sort speakers by word count descending for target suggestion
    sorted_speakers = sorted(words_per_speaker.items(), key=lambda x: x[1], reverse=True)

    for speaker_id, word_count in sorted_speakers:
        ratio = word_count / total_words
        if ratio < min_word_ratio and len(sorted_speakers) > 1:
            # Suggest merging into the speaker with most words (that isn't this one)
            target = next(s for s, _ in sorted_speakers if s != speaker_id)
            candidates.append({
                "source": speaker_id,
                "suggested_target": target,
                "word_count": word_count,
                "word_ratio": round(ratio, 4),
            })

    return candidates


def apply_merge(
    segments: list[TranscribeSegment],
    source_speaker: int,
    target_speaker: int,
) -> list[TranscribeSegment]:
    """Replace all occurrences of source_speaker with target_speaker.

    Returns a new list of segments (does not mutate originals).
    """
    merged = []
    for seg in segments:
        if seg.speaker == source_speaker:
            merged.append(TranscribeSegment(
                speaker=target_speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
            ))
        else:
            merged.append(seg)
    return merged
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_speaker_merge.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/speaker_merge.py \
       modules/orchestration-service/tests/diarization/test_speaker_merge.py
git commit -m "feat: add speaker merge for over-segmentation detection and fix"
```

---

### Task 8: Transcript Merge ("Best Of" Alignment)

**Files:**
- Create: `modules/orchestration-service/src/services/diarization/transcript_merge.py`
- Test: `modules/orchestration-service/tests/diarization/test_transcript_merge.py`

**Context:** Aligns VibeVoice segments with Fireflies sentences by timestamp, keeps Fireflies text but replaces speaker labels with VibeVoice diarization. Output format matches Fireflies sentence structure exactly.

**Step 1: Write failing test**

```python
"""Tests for transcript merge (Fireflies + VibeVoice alignment)."""

from models.diarization import SpeakerMapEntry, TranscribeSegment
from services.diarization.transcript_merge import merge_transcripts


def test_basic_merge_replaces_speaker_names():
    fireflies = [
        {"speaker_name": "Speaker 1", "start_time": 0.0, "end_time": 5.0, "text": "Hello there", "index": 0},
        {"speaker_name": "Speaker 2", "start_time": 5.0, "end_time": 10.0, "text": "Hi back", "index": 1},
    ]
    vibevoice = [
        TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello there"),
        TranscribeSegment(speaker=1, start=5.0, end=10.0, text="Hi back"),
    ]
    speaker_map = {
        0: SpeakerMapEntry(name="Eric", confidence=0.9, method="fireflies_crossref"),
        1: SpeakerMapEntry(name="Thomas", confidence=0.85, method="voice_profile"),
    }

    merged = merge_transcripts(fireflies, vibevoice, speaker_map)
    assert merged[0]["speaker_name"] == "Eric"
    assert merged[1]["speaker_name"] == "Thomas"
    # Keeps Fireflies text
    assert merged[0]["text"] == "Hello there"
    assert merged[1]["text"] == "Hi back"


def test_merge_preserves_all_fireflies_fields():
    fireflies = [
        {"speaker_name": "Speaker 1", "start_time": 0.0, "end_time": 5.0, "text": "Hello",
         "index": 0, "raw_text": "Hello", "speaker_id": 1},
    ]
    vibevoice = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Hello")]
    speaker_map = {0: SpeakerMapEntry(name="Eric", confidence=0.9, method="manual")}

    merged = merge_transcripts(fireflies, vibevoice, speaker_map)
    assert merged[0]["index"] == 0
    assert merged[0]["raw_text"] == "Hello"
    assert merged[0]["speaker_name"] == "Eric"
    assert merged[0]["diarization_source"] == "vibevoice"


def test_merge_falls_back_to_speaker_id_when_unmapped():
    fireflies = [
        {"speaker_name": "Speaker 1", "start_time": 0.0, "end_time": 5.0, "text": "Hello", "index": 0},
    ]
    vibevoice = [TranscribeSegment(speaker=3, start=0.0, end=5.0, text="Hello")]
    speaker_map = {}  # no mapping

    merged = merge_transcripts(fireflies, vibevoice, speaker_map)
    assert merged[0]["speaker_name"] == "SPEAKER_3"
    assert merged[0]["diarization_source"] == "vibevoice"


def test_merge_no_match_keeps_fireflies_speaker():
    """When no VibeVoice segment overlaps, keep original Fireflies speaker."""
    fireflies = [
        {"speaker_name": "Eric", "start_time": 100.0, "end_time": 105.0, "text": "Late", "index": 0},
    ]
    vibevoice = [TranscribeSegment(speaker=0, start=0.0, end=5.0, text="Early")]
    speaker_map = {0: SpeakerMapEntry(name="Thomas", confidence=0.9, method="manual")}

    merged = merge_transcripts(fireflies, vibevoice, speaker_map)
    assert merged[0]["speaker_name"] == "Eric"  # unchanged
    assert merged[0].get("diarization_source") is None
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_transcript_merge.py -v`
Expected: FAIL

**Step 3: Write transcript merge**

```python
"""Transcript merge: align Fireflies sentences with VibeVoice diarization.

Keeps Fireflies text (which is typically good), replaces speaker labels
with VibeVoice's more accurate acoustic diarization. Output format
matches Fireflies sentence structure exactly so the dashboard renders
identically.
"""

from typing import Any

from livetranslate_common.logging import get_logger
from models.diarization import SpeakerMapEntry, TranscribeSegment

logger = get_logger()


def merge_transcripts(
    fireflies_sentences: list[dict[str, Any]],
    vibevoice_segments: list[TranscribeSegment],
    speaker_map: dict[int, SpeakerMapEntry],
) -> list[dict[str, Any]]:
    """Merge Fireflies transcript with VibeVoice diarization.

    For each Fireflies sentence, find the best-overlapping VibeVoice segment
    and replace the speaker name with the mapped VibeVoice speaker.

    Args:
        fireflies_sentences: Original Fireflies sentences (dicts with speaker_name,
            start_time, end_time, text, and any other fields).
        vibevoice_segments: Parsed VibeVoice output segments.
        speaker_map: Mapping from VibeVoice speaker IDs to named people.

    Returns:
        New list of sentence dicts with updated speaker_name and a
        'diarization_source' field set to 'vibevoice' where applied.
    """
    merged = []

    for ff_sent in fireflies_sentences:
        result = dict(ff_sent)  # shallow copy — preserve all original fields
        ff_start = ff_sent.get("start_time", 0.0)
        ff_end = ff_sent.get("end_time", 0.0)

        best_overlap = 0.0
        best_speaker_id: int | None = None

        for vv_seg in vibevoice_segments:
            overlap_start = max(ff_start, vv_seg.start)
            overlap_end = min(ff_end, vv_seg.end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker_id = vv_seg.speaker

        if best_speaker_id is not None and best_overlap > 0:
            if best_speaker_id in speaker_map:
                result["speaker_name"] = speaker_map[best_speaker_id].name
            else:
                result["speaker_name"] = f"SPEAKER_{best_speaker_id}"
            result["diarization_source"] = "vibevoice"
            result["diarization_speaker_id"] = best_speaker_id
            result["diarization_confidence"] = (
                speaker_map[best_speaker_id].confidence if best_speaker_id in speaker_map else 0.0
            )

        merged.append(result)

    return merged
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_transcript_merge.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/transcript_merge.py \
       modules/orchestration-service/tests/diarization/test_transcript_merge.py
git commit -m "feat: add transcript merge for Fireflies + VibeVoice alignment"
```

---

### Task 9: Diarization Pipeline (Job Queue + Worker)

**Files:**
- Create: `modules/orchestration-service/src/services/diarization/pipeline.py`
- Test: `modules/orchestration-service/tests/diarization/test_diarization_pipeline.py`

**Context:** Async job queue that processes diarization requests: download audio → run VibeVoice → map speakers → merge transcript → update DB. Uses in-process asyncio queue (GPU processes one job at a time).

**Step 1: Write failing test**

```python
"""Tests for diarization pipeline job management."""

import pytest

from models.diarization import DiarizationJobStatus
from services.diarization.pipeline import DiarizationPipeline


@pytest.fixture
def pipeline():
    return DiarizationPipeline(
        vibevoice_url="http://localhost:8000/v1",
        max_concurrent=1,
    )


def test_pipeline_init(pipeline):
    assert pipeline.max_concurrent == 1
    assert pipeline.active_jobs == {}


def test_create_job(pipeline):
    job = pipeline.create_job(meeting_id=42, triggered_by="manual")
    assert job["meeting_id"] == 42
    assert job["status"] == DiarizationJobStatus.QUEUED
    assert job["triggered_by"] == "manual"
    assert "job_id" in job


def test_create_job_with_hotwords(pipeline):
    job = pipeline.create_job(meeting_id=42, triggered_by="auto_rule", hotwords=["sprint"])
    assert job["hotwords"] == ["sprint"]


def test_get_job_status(pipeline):
    job = pipeline.create_job(meeting_id=42, triggered_by="manual")
    status = pipeline.get_job(job["job_id"])
    assert status is not None
    assert status["status"] == DiarizationJobStatus.QUEUED


def test_get_nonexistent_job(pipeline):
    assert pipeline.get_job("nonexistent") is None


def test_cancel_queued_job(pipeline):
    job = pipeline.create_job(meeting_id=42, triggered_by="manual")
    result = pipeline.cancel_job(job["job_id"])
    assert result is True
    assert pipeline.get_job(job["job_id"])["status"] == DiarizationJobStatus.CANCELLED


def test_list_jobs(pipeline):
    pipeline.create_job(meeting_id=1, triggered_by="manual")
    pipeline.create_job(meeting_id=2, triggered_by="auto_rule")
    jobs = pipeline.list_jobs()
    assert len(jobs) == 2


def test_list_jobs_by_status(pipeline):
    pipeline.create_job(meeting_id=1, triggered_by="manual")
    j2 = pipeline.create_job(meeting_id=2, triggered_by="manual")
    pipeline.cancel_job(j2["job_id"])
    queued = pipeline.list_jobs(status=DiarizationJobStatus.QUEUED)
    assert len(queued) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_pipeline.py -v`
Expected: FAIL

**Step 3: Write pipeline**

```python
"""Diarization pipeline: async job queue and worker.

Manages the lifecycle of diarization jobs:
  queued → downloading → processing → mapping → completed/failed

The worker loop processes one job at a time (GPU is the bottleneck).
All DB persistence is handled by the caller (router); this module
manages in-memory job state and orchestrates the processing steps.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from livetranslate_common.logging import get_logger
from models.diarization import DiarizationJobStatus

logger = get_logger()


class DiarizationPipeline:
    """In-process diarization job queue and executor."""

    def __init__(
        self,
        vibevoice_url: str = "http://localhost:8000/v1",
        max_concurrent: int = 1,
    ):
        self.vibevoice_url = vibevoice_url
        self.max_concurrent = max_concurrent
        self.active_jobs: dict[str, dict[str, Any]] = {}

    def create_job(
        self,
        meeting_id: int,
        triggered_by: str = "manual",
        hotwords: list[str] | None = None,
        rule_matched: dict | None = None,
    ) -> dict[str, Any]:
        """Create a new diarization job (queued state)."""
        job_id = str(uuid.uuid4())[:12]
        job: dict[str, Any] = {
            "job_id": job_id,
            "meeting_id": meeting_id,
            "status": DiarizationJobStatus.QUEUED,
            "triggered_by": triggered_by,
            "hotwords": hotwords,
            "rule_matched": rule_matched,
            "created_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "error_message": None,
        }
        self.active_jobs[job_id] = job
        logger.info("diarization_job_created", job_id=job_id, meeting_id=meeting_id, triggered_by=triggered_by)
        return job

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self.active_jobs.get(job_id)

    def list_jobs(self, status: DiarizationJobStatus | None = None) -> list[dict[str, Any]]:
        jobs = list(self.active_jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j["status"] == status]
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Returns False if job not found or not cancellable."""
        job = self.active_jobs.get(job_id)
        if not job:
            return False
        if job["status"] not in (DiarizationJobStatus.QUEUED,):
            return False
        job["status"] = DiarizationJobStatus.CANCELLED
        logger.info("diarization_job_cancelled", job_id=job_id)
        return True

    def update_status(self, job_id: str, status: DiarizationJobStatus, **kwargs: Any) -> None:
        """Update job status and optional metadata fields."""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        job["status"] = status
        job.update(kwargs)
        if status in (DiarizationJobStatus.COMPLETED, DiarizationJobStatus.FAILED):
            job["completed_at"] = datetime.now(UTC).isoformat()
        logger.info("diarization_job_status_update", job_id=job_id, status=status)
```

**Step 4: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_pipeline.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/pipeline.py \
       modules/orchestration-service/tests/diarization/test_diarization_pipeline.py
git commit -m "feat: add diarization pipeline with async job queue"
```

---

## Phase 3: Router & App Integration

### Task 10: Diarization Router

**Files:**
- Create: `modules/orchestration-service/src/routers/diarization.py`
- Test: `modules/orchestration-service/tests/diarization/test_diarization_router.py`

**Context:** Follow the pattern in `routers/fireflies.py`. Router prefix will be added in `main_fastapi.py` as `/api/diarization`. Uses `APIRouter(tags=["diarization"])`. Endpoints call the pipeline and DB.

**Step 1: Write the router**

```python
"""Diarization Router — FastAPI endpoints for offline diarization management.

Endpoints:
  POST /jobs              - Submit meeting for diarization
  GET  /jobs              - List diarization jobs
  GET  /jobs/{job_id}     - Get job detail
  POST /jobs/{job_id}/cancel - Cancel a queued job

  GET  /rules             - Get auto-trigger rules
  PUT  /rules             - Update auto-trigger rules

  GET  /speakers          - List speaker profiles
  POST /speakers          - Create speaker profile
  PUT  /speakers/{id}     - Update speaker profile
  POST /speakers/merge    - Merge two speaker profiles
  DELETE /speakers/{id}   - Delete speaker profile

  GET  /meetings/{id}/compare - Side-by-side transcript comparison
  POST /meetings/{id}/apply   - Apply diarization to meeting
"""

from typing import Any

from config import DiarizationSettings
from fastapi import APIRouter, HTTPException, status
from livetranslate_common.logging import get_logger
from models.diarization import (
    DiarizationJobCreate,
    DiarizationJobResponse,
    DiarizationRules,
    SpeakerMergeRequest,
    SpeakerProfileCreate,
    SpeakerProfileResponse,
)
from services.diarization.pipeline import DiarizationPipeline

logger = get_logger()

router = APIRouter(tags=["diarization"])

# Module-level pipeline instance (initialized on first use or at app startup)
_pipeline: DiarizationPipeline | None = None


def get_pipeline() -> DiarizationPipeline:
    global _pipeline
    if _pipeline is None:
        settings = DiarizationSettings()
        _pipeline = DiarizationPipeline(
            vibevoice_url=settings.vibevoice_url,
            max_concurrent=settings.max_concurrent_jobs,
        )
    return _pipeline


# --- Job endpoints ---


@router.post("/jobs", status_code=status.HTTP_201_CREATED)
async def create_diarization_job(req: DiarizationJobCreate) -> dict[str, Any]:
    """Submit a meeting for offline diarization."""
    pipeline = get_pipeline()
    job = pipeline.create_job(
        meeting_id=req.meeting_id,
        triggered_by="manual",
        hotwords=req.hotwords,
    )
    return job


@router.get("/jobs")
async def list_diarization_jobs(status_filter: str | None = None) -> list[dict[str, Any]]:
    """List diarization jobs, optionally filtered by status."""
    pipeline = get_pipeline()
    return pipeline.list_jobs(status=status_filter)


@router.get("/jobs/{job_id}")
async def get_diarization_job(job_id: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    job = pipeline.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel")
async def cancel_diarization_job(job_id: str) -> dict[str, str]:
    pipeline = get_pipeline()
    if not pipeline.cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Job not found or not cancellable")
    return {"status": "cancelled"}


# --- Speaker endpoints ---


@router.get("/speakers")
async def list_speakers() -> list[dict[str, Any]]:
    """List all known speaker profiles."""
    # TODO: Query speaker_profiles table
    return []


@router.post("/speakers", status_code=status.HTTP_201_CREATED)
async def create_speaker(req: SpeakerProfileCreate) -> dict[str, Any]:
    """Create a new speaker profile."""
    # TODO: Insert into speaker_profiles table
    return {"id": 0, "name": req.name, "email": req.email, "enrollment_source": "manual", "sample_count": 0}


@router.put("/speakers/{speaker_id}")
async def update_speaker(speaker_id: int, req: SpeakerProfileCreate) -> dict[str, Any]:
    """Update speaker profile name/email."""
    # TODO: Update speaker_profiles table
    return {"id": speaker_id, "name": req.name, "email": req.email}


@router.post("/speakers/merge")
async def merge_speakers(req: SpeakerMergeRequest) -> dict[str, str]:
    """Merge source speaker profile into target."""
    # TODO: Merge in speaker_profiles table + update diarization_jobs speaker_maps
    return {"status": "merged", "kept": str(req.target_id), "removed": str(req.source_id)}


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: int) -> dict[str, str]:
    """Delete a speaker profile."""
    # TODO: Delete from speaker_profiles table
    return {"status": "deleted"}


# --- Rules endpoints ---


@router.get("/rules")
async def get_rules() -> dict[str, Any]:
    """Get current auto-trigger rules."""
    # TODO: Read from system_config table
    return DiarizationRules().model_dump()


@router.put("/rules")
async def update_rules(rules: DiarizationRules) -> dict[str, Any]:
    """Update auto-trigger rules."""
    # TODO: Write to system_config table
    return rules.model_dump()


# --- Comparison endpoints ---


@router.get("/meetings/{meeting_id}/compare")
async def compare_transcripts(meeting_id: int) -> dict[str, Any]:
    """Side-by-side comparison of Fireflies vs VibeVoice transcript."""
    # TODO: Fetch both transcripts from DB
    return {"meeting_id": meeting_id, "fireflies_sentences": [], "vibevoice_segments": []}


@router.post("/meetings/{meeting_id}/apply")
async def apply_diarization(meeting_id: int) -> dict[str, str]:
    """Apply diarization results to the meeting transcript."""
    # TODO: Run transcript merge and update meeting_sentences
    return {"status": "applied", "meeting_id": str(meeting_id)}
```

**Step 2: Write basic router test**

```python
"""Tests for diarization router endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.diarization import router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/diarization")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_create_job(client):
    resp = client.post("/api/diarization/jobs", json={"meeting_id": 42})
    assert resp.status_code == 201
    data = resp.json()
    assert data["meeting_id"] == 42
    assert data["status"] == "queued"
    assert "job_id" in data


def test_list_jobs(client):
    client.post("/api/diarization/jobs", json={"meeting_id": 1})
    client.post("/api/diarization/jobs", json={"meeting_id": 2})
    resp = client.get("/api/diarization/jobs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_job(client):
    create_resp = client.post("/api/diarization/jobs", json={"meeting_id": 42})
    job_id = create_resp.json()["job_id"]
    resp = client.get(f"/api/diarization/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["meeting_id"] == 42


def test_get_nonexistent_job(client):
    resp = client.get("/api/diarization/jobs/nonexistent")
    assert resp.status_code == 404


def test_cancel_job(client):
    create_resp = client.post("/api/diarization/jobs", json={"meeting_id": 42})
    job_id = create_resp.json()["job_id"]
    resp = client.post(f"/api/diarization/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


def test_get_rules(client):
    resp = client.get("/api/diarization/rules")
    assert resp.status_code == 200
    assert "enabled" in resp.json()


def test_update_rules(client):
    resp = client.put("/api/diarization/rules", json={
        "enabled": True,
        "participant_patterns": ["eric@*"],
        "title_patterns": ["dev weekly*"],
    })
    assert resp.status_code == 200
    assert resp.json()["participant_patterns"] == ["eric@*"]


def test_list_speakers(client):
    resp = client.get("/api/diarization/speakers")
    assert resp.status_code == 200


def test_create_speaker(client):
    resp = client.post("/api/diarization/speakers", json={"name": "Eric Chen", "email": "eric@co.com"})
    assert resp.status_code == 201
    assert resp.json()["name"] == "Eric Chen"
```

**Step 3: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/diarization/test_diarization_router.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/diarization.py \
       modules/orchestration-service/tests/diarization/test_diarization_router.py
git commit -m "feat: add diarization router with job, speaker, and rules endpoints"
```

---

### Task 11: Register Router in Main App

**Files:**
- Modify: `modules/orchestration-service/src/main_fastapi.py`

**Context:** Follow the pattern at lines ~55-250 (import with try/except + routers_status) and lines ~505-665 (include_router with prefix).

**Step 1: Add import block**

In the router import section (around line 55-250), add:

```python
try:
    from routers.diarization import router as diarization_router
    routers_status["diarization_router"] = {"status": "success", "routes": len(diarization_router.routes)}
except Exception as e:
    diarization_router = None
    routers_status["diarization_router"] = {"status": "failed", "error": str(e)}
```

**Step 2: Add include_router**

In the registration section (around lines 505-665), add:

```python
if diarization_router is not None:
    app.include_router(diarization_router, prefix="/api/diarization", tags=["Diarization"])
```

**Step 3: Verify import succeeds**

Run: `cd modules/orchestration-service && uv run python -c "from routers.diarization import router; print(f'Routes: {len(router.routes)}')"`
Expected: `Routes: <number>`

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/main_fastapi.py
git commit -m "feat: register diarization router in main FastAPI app"
```

---

## Phase 4: Fireflies Integration (Auto-Trigger Hook)

### Task 12: Hook Diarization Rules into Fireflies Sync

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py`
- Create: `modules/orchestration-service/src/services/diarization/auto_trigger.py`

**Context:** After `_persist_transcript()` completes during Fireflies sync, evaluate diarization rules. If a meeting matches, auto-queue a diarization job. The trigger function reads rules from `system_config` and calls the pipeline.

**Step 1: Create auto_trigger module**

```python
"""Auto-trigger: evaluate diarization rules after Fireflies sync.

Called from the Fireflies sync flow after a meeting is persisted.
Checks rules and queues diarization if a match is found.
"""

from typing import Any

from livetranslate_common.logging import get_logger
from models.diarization import DiarizationRules
from services.diarization.rules import evaluate_rules

logger = get_logger()


async def maybe_trigger_diarization(
    meeting: dict[str, Any],
    rules: DiarizationRules,
    pipeline: Any,
) -> dict[str, Any] | None:
    """Evaluate rules and queue diarization if meeting matches.

    Args:
        meeting: Dict with title, participants, duration, sentence_count.
        rules: Current diarization rules config.
        pipeline: DiarizationPipeline instance.

    Returns:
        Job dict if queued, None if no match.
    """
    match = evaluate_rules(meeting, rules)
    if match is None:
        return None

    logger.info(
        "diarization_auto_triggered",
        meeting_id=meeting.get("id"),
        match_type=match["match_type"],
        matched_pattern=match["matched_pattern"],
    )

    job = pipeline.create_job(
        meeting_id=meeting["id"],
        triggered_by="auto_rule",
        rule_matched=match,
    )
    return job
```

**Step 2: Add hook point in Fireflies sync flow**

In `routers/fireflies.py`, find `_persist_transcript()` or the sync completion point. Add after successful meeting persistence:

```python
# After meeting is persisted successfully:
# Auto-trigger diarization if rules match
try:
    from services.diarization.auto_trigger import maybe_trigger_diarization
    from services.diarization.pipeline import DiarizationPipeline
    from models.diarization import DiarizationRules
    # TODO: Read rules from system_config table
    # For now, use defaults (disabled) — will be enabled via dashboard
except Exception:
    pass  # Diarization not available — skip silently
```

**Note:** The actual integration point depends on the exact structure of `_persist_transcript`. The implementing engineer should find where `store.update_sync_status(meeting_db_id, "synced", ...)` is called and add the auto-trigger check after it.

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/auto_trigger.py \
       modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: add diarization auto-trigger hook in Fireflies sync flow"
```

---

## Phase 5: Dashboard Frontend

### Task 13: Dashboard API Proxy Routes

**Files:**
- Create: `modules/dashboard-service/src/routes/api/diarization/jobs/+server.ts`
- Create: `modules/dashboard-service/src/routes/api/diarization/rules/+server.ts`
- Create: `modules/dashboard-service/src/routes/api/diarization/speakers/+server.ts`
- Create: `modules/dashboard-service/src/routes/api/diarization/meetings/[meeting_id]/compare/+server.ts`
- Create: `modules/dashboard-service/src/routes/api/diarization/meetings/[meeting_id]/apply/+server.ts`

**Context:** These are thin proxies to the orchestration service, following the pattern in `routes/api/fireflies/transcripts/+server.ts`. Import `ORCHESTRATION_URL` from `$env/static/private`.

**Step 1: Create jobs proxy**

`routes/api/diarization/jobs/+server.ts`:
```typescript
import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
    try {
        const params = new URLSearchParams();
        const status = url.searchParams.get('status');
        if (status) params.set('status_filter', status);

        const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/jobs?${params.toString()}`);
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};

export const POST: RequestHandler = async ({ request }) => {
    try {
        const body = await request.json();
        const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/jobs`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};
```

**Step 2: Create rules proxy**

`routes/api/diarization/rules/+server.ts`:
```typescript
import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
    try {
        const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/rules`);
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};

export const PUT: RequestHandler = async ({ request }) => {
    try {
        const body = await request.json();
        const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/rules`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};
```

**Step 3: Create speakers proxy**

`routes/api/diarization/speakers/+server.ts`:
```typescript
import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
    try {
        const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/speakers`);
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};

export const POST: RequestHandler = async ({ request }) => {
    try {
        const body = await request.json();
        const res = await fetch(`${ORCHESTRATION_URL}/api/diarization/speakers`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};
```

**Step 4: Create compare and apply proxies**

`routes/api/diarization/meetings/[meeting_id]/compare/+server.ts`:
```typescript
import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ params }) => {
    try {
        const res = await fetch(
            `${ORCHESTRATION_URL}/api/diarization/meetings/${params.meeting_id}/compare`
        );
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};
```

`routes/api/diarization/meetings/[meeting_id]/apply/+server.ts`:
```typescript
import { ORCHESTRATION_URL } from '$env/static/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

export const POST: RequestHandler = async ({ params }) => {
    try {
        const res = await fetch(
            `${ORCHESTRATION_URL}/api/diarization/meetings/${params.meeting_id}/apply`,
            { method: 'POST' }
        );
        const data = await res.json();
        return json(data, { status: res.status });
    } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown error';
        return json({ error: `Proxy failed: ${message}` }, { status: 502 });
    }
};
```

**Step 5: Commit**

```bash
git add modules/dashboard-service/src/routes/api/diarization/
git commit -m "feat: add dashboard API proxy routes for diarization"
```

---

### Task 14: Dashboard API Client Library

**Files:**
- Create: `modules/dashboard-service/src/lib/api/diarization.ts`

**Context:** Follow the pattern in `$lib/api/fireflies.ts` — typed wrapper around fetch calls.

**Step 1: Write API client**

```typescript
/**
 * Diarization API client for dashboard.
 */

const BASE = '/api/diarization';

export interface DiarizationJob {
    job_id: string;
    meeting_id: number;
    status: 'queued' | 'downloading' | 'processing' | 'mapping' | 'completed' | 'failed' | 'cancelled';
    triggered_by: string;
    detected_language?: string;
    num_speakers_detected?: number;
    processing_time_seconds?: number;
    speaker_map?: Record<string, SpeakerMapEntry>;
    unmapped_speakers?: number[];
    merge_applied?: boolean;
    error_message?: string;
    created_at?: string;
    completed_at?: string;
}

export interface SpeakerMapEntry {
    name: string;
    confidence: number;
    method: string;
}

export interface SpeakerProfile {
    id: number;
    name: string;
    email?: string;
    enrollment_source: string;
    sample_count: number;
}

export interface DiarizationRules {
    enabled: boolean;
    participant_patterns: string[];
    title_patterns: string[];
    min_duration_minutes: number;
    exclude_empty: boolean;
}

export interface TranscriptComparison {
    meeting_id: number;
    fireflies_sentences: Record<string, unknown>[];
    vibevoice_segments: Record<string, unknown>[];
    speaker_map?: Record<string, SpeakerMapEntry>;
}

export function diarizationApi(fetchFn: typeof fetch = fetch) {
    return {
        // Jobs
        async createJob(meetingId: number, hotwords?: string[]): Promise<DiarizationJob> {
            const res = await fetchFn(`${BASE}/jobs`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ meeting_id: meetingId, hotwords })
            });
            return res.json();
        },

        async listJobs(status?: string): Promise<DiarizationJob[]> {
            const params = status ? `?status=${status}` : '';
            const res = await fetchFn(`${BASE}/jobs${params}`);
            return res.json();
        },

        async getJob(jobId: string): Promise<DiarizationJob> {
            const res = await fetchFn(`${BASE}/jobs/${jobId}`);
            return res.json();
        },

        async cancelJob(jobId: string): Promise<{ status: string }> {
            const res = await fetchFn(`${BASE}/jobs/${jobId}/cancel`, { method: 'POST' });
            return res.json();
        },

        // Speakers
        async listSpeakers(): Promise<SpeakerProfile[]> {
            const res = await fetchFn(`${BASE}/speakers`);
            return res.json();
        },

        async createSpeaker(name: string, email?: string): Promise<SpeakerProfile> {
            const res = await fetchFn(`${BASE}/speakers`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, email })
            });
            return res.json();
        },

        async mergeSpeakers(sourceId: number, targetId: number): Promise<{ status: string }> {
            const res = await fetchFn(`${BASE}/speakers/merge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ source_id: sourceId, target_id: targetId })
            });
            return res.json();
        },

        // Rules
        async getRules(): Promise<DiarizationRules> {
            const res = await fetchFn(`${BASE}/rules`);
            return res.json();
        },

        async updateRules(rules: DiarizationRules): Promise<DiarizationRules> {
            const res = await fetchFn(`${BASE}/rules`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(rules)
            });
            return res.json();
        },

        // Comparison
        async compareTranscripts(meetingId: number): Promise<TranscriptComparison> {
            const res = await fetchFn(`${BASE}/meetings/${meetingId}/compare`);
            return res.json();
        },

        async applyDiarization(meetingId: number): Promise<{ status: string }> {
            const res = await fetchFn(`${BASE}/meetings/${meetingId}/apply`, { method: 'POST' });
            return res.json();
        }
    };
}
```

**Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/api/diarization.ts
git commit -m "feat: add diarization TypeScript API client library"
```

---

### Task 15: Diarization Hub Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/diarization/+page.server.ts`
- Create: `modules/dashboard-service/src/routes/(app)/diarization/+page.svelte`

**Context:** New page at `/diarization` with four tabs: Active Jobs, History, Speakers, Rules. Follow existing dashboard page patterns (e.g., `/fireflies`).

**Step 1: Create page server (data loading)**

`+page.server.ts`:
```typescript
import { diarizationApi } from '$lib/api/diarization';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ fetch }) => {
    const api = diarizationApi(fetch);

    const [jobs, speakers, rules] = await Promise.all([
        api.listJobs().catch(() => []),
        api.listSpeakers().catch(() => []),
        api.getRules().catch(() => ({
            enabled: false,
            participant_patterns: [],
            title_patterns: [],
            min_duration_minutes: 5,
            exclude_empty: true
        }))
    ]);

    return { jobs, speakers, rules };
};
```

**Step 2: Create page component**

`+page.svelte` — This will be a substantial Svelte component. The implementing engineer should:

1. Use the existing dashboard UI patterns (check `modules/dashboard-service/src/routes/(app)/fireflies/+page.svelte` for reference)
2. Create four tab panels: Jobs, Speakers, Rules, History
3. Jobs tab: list with status badges, progress indicators, cancel buttons
4. Speakers tab: table with name, email, enrollment source, sample count, edit/delete actions
5. Rules tab: form with pattern inputs, toggle, save button
6. Include "Diarize" action button that calls `api.createJob(meetingId)`

**Note:** The exact Svelte component code depends heavily on the existing UI component library and styling patterns in the dashboard. The implementing engineer should reference existing pages for consistent look and feel.

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/diarization/
git commit -m "feat: add diarization hub page with jobs, speakers, and rules tabs"
```

---

### Task 16: Meeting Detail — Diarize Button + Compare View

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.svelte`
- Modify: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.server.ts`

**Context:** Add a "Diarize" button to the meeting detail page. When diarization is complete, show a source toggle (Fireflies / Diarized) and a "Compare" view.

**Step 1: Update page server to load diarization status**

In `+page.server.ts`, add to the load function:

```typescript
// Check if this meeting has a diarization job
const diarizationJobs = await diarizationApi(fetch)
    .listJobs()
    .then(jobs => jobs.filter(j => j.meeting_id === meetingId))
    .catch(() => []);
```

Return `diarizationJobs` alongside existing data.

**Step 2: Add Diarize button to page component**

In `+page.svelte`, add:
- A "Diarize" button (disabled if job already queued/processing)
- Job status indicator when a job exists
- Source toggle (Fireflies / Diarized) when job is completed
- "Compare" link that opens side-by-side view

**Note:** The exact placement and styling depends on the existing meeting detail layout. The implementing engineer should study the current page structure first.

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/meetings/\[id\]/
git commit -m "feat: add diarize button and compare view to meeting detail page"
```

---

## Phase 6: VibeVoice Docker Setup

### Task 17: Docker Compose for VibeVoice vLLM

**Files:**
- Create: `docker/docker-compose.vibevoice.yml`

**Step 1: Create compose file**

```yaml
# VibeVoice-ASR vLLM Inference Server
# Run on GPU box: docker compose -f docker/docker-compose.vibevoice.yml up -d
#
# Requires: NVIDIA GPU with 18+ GB VRAM (BF16) or 11+ GB (8-bit) or 7+ GB (4-bit)
# Access: http://<gpu-box-ip>:8000/v1

services:
  vibevoice:
    image: vllm/vllm-openai:v0.14.1
    container_name: vibevoice-asr
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - VIBEVOICE_FFMPEG_MAX_CONCURRENCY=64
      - PYTORCH_ALLOC_CONF=expandable_segments:True
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - vibevoice-models:/root/.cache/huggingface
    ipc: host
    entrypoint: bash
    command: -c "pip install vibevoice && python3 -m vibevoice.vllm_plugin.scripts.start_server"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v1/models"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  vibevoice-models:
```

**Step 2: Commit**

```bash
git add docker/docker-compose.vibevoice.yml
git commit -m "feat: add Docker Compose for VibeVoice-ASR vLLM server"
```

---

## Summary

| Phase | Tasks | What it builds |
|-------|-------|---------------|
| 1 | 1-3 | DB tables, Pydantic models, config |
| 2 | 4-9 | VibeVoice client, rules, speaker mapper, merge, pipeline |
| 3 | 10-11 | FastAPI router, app registration |
| 4 | 12 | Auto-trigger hook in Fireflies sync |
| 5 | 13-16 | Dashboard API proxies, client lib, hub page, meeting detail |
| 6 | 17 | Docker deployment for VibeVoice |

**Total tasks:** 17
**Estimated commits:** 17 (one per task)

After all tasks are complete, the system supports:
- Manual diarization trigger from dashboard
- Auto-trigger based on participant/title rules
- VibeVoice-ASR inference via vLLM on LAN GPU
- Three-strategy speaker name mapping
- Over-segmentation detection and merge
- "Best of" transcript merge (Fireflies text + VibeVoice speakers)
- Side-by-side transcript comparison in dashboard
- Speaker profile enrollment and management
- Configurable rules from dashboard
