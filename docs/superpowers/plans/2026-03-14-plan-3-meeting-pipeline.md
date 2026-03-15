# Plan 3: Unified Meeting Pipeline

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one meeting pipeline that both loopback sessions and meeting bots feed into. Ephemeral by default, promotable to full meeting sessions with continuous FLAC recording, crash-safe persistence, and heartbeat orphan detection.

**Architecture:** The meeting pipeline lives in orchestration service. Any audio source (loopback mic, system audio, Google Meet bot) implements the `MeetingAudioStream` protocol from Plan 0. Sessions start ephemeral (stream-through, no persistence). "Start Meeting" promotes to active (recording + DB persistence). Audio is saved at native quality (48kHz+ FLAC), downsampled to 16kHz only for transcription. Crash safety via flush-on-write chunks, row-by-row DB persistence, manifest tracking, and 120s heartbeat orphan detection.

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy + Alembic, FLAC (soundfile), Pydantic v2, asyncio

**Spec:** `docs/superpowers/specs/2026-03-14-loopback-transcription-translation-design.md` — Plan 3 section

**Depends on:** Plan 0 (shared contracts: `MeetingAudioStream`, `AudioChunk`, WebSocket message types)

---

## Chunk 1: Database Migration

### Task 1: Alembic migration for meeting tables

**Files:**
- Create: `modules/orchestration-service/alembic/versions/013_meeting_tables.py`
- Create: `modules/orchestration-service/tests/test_meeting_migration.py`

**Important Alembic rules (from CLAUDE.md):**
- Revision IDs MUST be ≤32 characters
- `down_revision` must match the parent's `revision` exactly
- Check current chain with `grep -n "^revision\|^down_revision" alembic/versions/*.py`

**Alembic chain status (verified 2026-03-14):**
The current main chain ends at `012_ai_connections` (down_revision: `011_chat_tables`). There is also a detached migration `5f3bcf8a26da_add_all_missing_tables_and_indexes.py` with `down_revision = "001_initial"` that creates a "multiple heads" situation. This migration must NOT be used as the parent.

- [ ] **Step 1: Verify Alembic chain head and check for multiple heads**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && grep -n "^revision\|^down_revision" alembic/versions/*.py | tail -6
```

Verify the chain head is `012_ai_connections`. Also run:

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic heads
```

If "multiple heads" is reported due to the detached `5f3bcf8a26da` migration, resolve it before proceeding (e.g., merge heads or remove the detached migration). This new migration MUST chain off `012_ai_connections` only.

- [ ] **Step 2: Write the migration**

```python
# modules/orchestration-service/alembic/versions/013_meeting_tables.py
"""Create meeting_sessions, meeting_transcripts, meeting_translations tables.

Additive migration: creates new tables alongside existing bot_sessions.
Does NOT drop or modify bot_sessions.

Revision ID: 013_meeting_tables
Revises: 012_ai_connections
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# Revision identifiers
revision = "013_meeting_tables"
down_revision = "012_ai_connections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "meeting_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="ephemeral"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_languages", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("target_languages", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("recording_path", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "last_activity_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_meeting_sessions_status", "meeting_sessions", ["status"])
    op.create_index("ix_meeting_sessions_started_at", "meeting_sessions", ["started_at"])

    op.create_table(
        "meeting_transcripts",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("meeting_sessions.id"),
            nullable=False,
        ),
        sa.Column("timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("speaker_id", sa.Text(), nullable=True),
        sa.Column("speaker_name", sa.Text(), nullable=True),
        sa.Column("source_language", sa.Text(), nullable=True),
        sa.Column("source_id", sa.Text(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_final", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_meeting_transcripts_session", "meeting_transcripts", ["session_id"])
    op.create_index("ix_meeting_transcripts_ts", "meeting_transcripts", ["timestamp_ms"])

    op.create_table(
        "meeting_translations",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "transcript_id",
            sa.BigInteger(),
            sa.ForeignKey("meeting_transcripts.id"),
            nullable=False,
        ),
        sa.Column("target_language", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=False),
        sa.Column("model_used", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_meeting_translations_transcript",
        "meeting_translations",
        ["transcript_id"],
    )


def downgrade() -> None:
    op.drop_table("meeting_translations")
    op.drop_table("meeting_transcripts")
    op.drop_table("meeting_sessions")
```

- [ ] **Step 3: Run the migration**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic upgrade head
```
Expected: Migration applies successfully

- [ ] **Step 4: Verify tables exist**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run python -c "
from sqlalchemy import create_engine, inspect
import os
engine = create_engine(os.environ['DATABASE_URL'])
inspector = inspect(engine)
tables = inspector.get_table_names()
for t in ['meeting_sessions', 'meeting_transcripts', 'meeting_translations']:
    assert t in tables, f'{t} missing'
    print(f'{t}: {[c[\"name\"] for c in inspector.get_columns(t)]}')
print('All meeting tables verified')
"
```

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/alembic/versions/013_meeting_tables.py
git commit -m "feat(orchestration): add meeting_sessions, transcripts, translations tables"
```

---

### Task 2: SQLAlchemy models for meeting tables

**Files:**
- Create: `modules/orchestration-service/src/database/meeting_models.py`
- Create: `modules/orchestration-service/tests/test_meeting_models.py`

- [ ] **Step 1: Write failing test**

```python
# modules/orchestration-service/tests/test_meeting_models.py
"""Tests for meeting SQLAlchemy models — behavioral tests against real DB."""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.meeting_models import MeetingSession, MeetingTranscript, MeetingTranslation


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingSession:
    async def test_create_session(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="loopback",
            status="ephemeral",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        result = await db_session.get(MeetingSession, session_id)
        assert result is not None
        assert result.source_type == "loopback"
        assert result.status == "ephemeral"

    async def test_promote_to_active(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="loopback",
            status="ephemeral",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        session.status = "active"
        session.recording_path = f"recordings/{session_id}"
        await db_session.commit()

        result = await db_session.get(MeetingSession, session_id)
        assert result.status == "active"
        assert result.recording_path is not None


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingTranscript:
    async def test_add_transcript(self, db_session: AsyncSession):
        session_id = uuid.uuid4()
        session = MeetingSession(
            id=session_id,
            source_type="loopback",
            status="active",
            started_at=datetime.now(timezone.utc),
        )
        db_session.add(session)
        await db_session.commit()

        transcript = MeetingTranscript(
            session_id=session_id,
            timestamp_ms=1000,
            text="Hello world",
            source_language="en",
            confidence=0.95,
            is_final=True,
        )
        db_session.add(transcript)
        await db_session.commit()

        result = await db_session.execute(
            select(MeetingTranscript).where(MeetingTranscript.session_id == session_id)
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].text == "Hello world"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_meeting_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write SQLAlchemy models**

```python
# modules/orchestration-service/src/database/meeting_models.py
"""SQLAlchemy ORM models for the unified meeting pipeline.

These map to the tables created in migration 013_meeting_tables.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.base import Base


class MeetingSession(Base):
    __tablename__ = "meeting_sessions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="ephemeral")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    source_languages: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    target_languages: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    recording_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transcripts: Mapped[list[MeetingTranscript]] = relationship(back_populates="session")


class MeetingTranscript(Base):
    __tablename__ = "meeting_transcripts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("meeting_sessions.id"), nullable=False
    )
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    speaker_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    speaker_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_language: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_final: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped[MeetingSession] = relationship(back_populates="transcripts")
    translations: Mapped[list[MeetingTranslation]] = relationship(back_populates="transcript")


class MeetingTranslation(Base):
    __tablename__ = "meeting_translations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    transcript_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("meeting_transcripts.id"), nullable=False
    )
    target_language: Mapped[str] = mapped_column(Text, nullable=False)
    translated_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transcript: Mapped[MeetingTranscript] = relationship(back_populates="translations")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_meeting_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/database/meeting_models.py modules/orchestration-service/tests/test_meeting_models.py
git commit -m "feat(orchestration): add MeetingSession, MeetingTranscript, MeetingTranslation ORM models"
```

---

## Chunk 2: Recording & Crash Safety

### Task 3: FLAC chunk recorder

**Files:**
- Create: `modules/orchestration-service/src/meeting/__init__.py`
- Create: `modules/orchestration-service/src/meeting/recorder.py`
- Create: `modules/orchestration-service/tests/test_recorder.py`

- [ ] **Step 0: Create meeting package**

```bash
mkdir -p /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service/src/meeting
touch /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service/src/meeting/__init__.py
```

This `__init__.py` is required for Python to treat `src/meeting/` as a package. All subsequent tasks (recorder, session_manager, downsampler, pipeline, heartbeat) depend on this.

- [ ] **Step 1: Write failing test**

```python
# modules/orchestration-service/tests/test_recorder.py
"""Tests for FLAC chunk recorder — crash-safe continuous recording."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from meeting.recorder import FlacChunkRecorder


class TestFlacChunkRecorder:
    @pytest.fixture
    def rec_dir(self, tmp_path):
        return tmp_path / "recordings" / "test-session"

    def test_start_creates_directory(self, rec_dir):
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=48000,
            channels=2,
            chunk_duration_s=5,  # Short chunks for testing
        )
        recorder.start()
        assert rec_dir.exists()
        manifest = json.loads((rec_dir / "manifest.json").read_text())
        assert manifest["session_id"] == "test-session"
        assert manifest["sample_rate"] == 48000
        recorder.stop()

    def test_write_chunk_creates_flac(self, rec_dir):
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=48000,
            channels=1,
            chunk_duration_s=1,
        )
        recorder.start()

        # Write 1 second of audio (48000 samples)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        recorder.write(audio)

        recorder.stop()

        # Verify FLAC file was written
        flac_files = list(rec_dir.glob("chunk_*.flac"))
        assert len(flac_files) >= 1

        # Verify manifest updated
        manifest = json.loads((rec_dir / "manifest.json").read_text())
        assert len(manifest["chunks"]) >= 1

    def test_manifest_tracks_samples(self, rec_dir):
        recorder = FlacChunkRecorder(
            session_id="test-session",
            base_path=rec_dir.parent,
            sample_rate=16000,
            channels=1,
            chunk_duration_s=2,
        )
        recorder.start()

        # Write 3 seconds (should create 1 full chunk + partial)
        for _ in range(3):
            audio = np.random.randn(16000).astype(np.float32) * 0.1
            recorder.write(audio)

        recorder.stop()

        manifest = json.loads((rec_dir / "manifest.json").read_text())
        total_samples = sum(c["samples"] for c in manifest["chunks"])
        assert total_samples == 48000  # 3 seconds × 16000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_recorder.py -v`
Expected: FAIL

- [ ] **Step 3: Write FlacChunkRecorder**

```python
# modules/orchestration-service/src/meeting/recorder.py
"""FLAC chunk recorder for crash-safe continuous meeting recording.

Writes audio in fixed-duration FLAC chunks with a manifest file
for crash recovery and gapless concatenation.

Key design:
- Flush-on-write: each chunk is a complete FLAC file (lose at most 1 chunk on crash)
- Manifest updated per chunk (tracks sequence, sample counts, timestamps)
- Sample-exact continuity: monotonic counter, no gaps or overlaps
- Native quality: 48kHz+ stereo, NOT downsampled
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf
from livetranslate_common.logging import get_logger

logger = get_logger()


class FlacChunkRecorder:
    def __init__(
        self,
        session_id: str,
        base_path: Path,
        sample_rate: int = 48000,
        channels: int = 2,
        chunk_duration_s: float = 30.0,
    ):
        self.session_id = session_id
        self.session_dir = base_path / session_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_s = chunk_duration_s
        self.chunk_samples = int(chunk_duration_s * sample_rate)

        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._sequence = 0
        self._total_samples = 0
        self._manifest: dict = {}
        self._running = False

    def start(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = {
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunks": [],
            "total_samples": 0,
        }
        self._write_manifest()
        self._running = True
        logger.info("recorder_started", session_id=self.session_id, path=str(self.session_dir))

    def write(self, audio: np.ndarray) -> None:
        """Write audio samples to the recorder. Flushes to FLAC when chunk is full."""
        if not self._running:
            return

        self._buffer.append(audio)
        self._buffer_samples += len(audio)

        while self._buffer_samples >= self.chunk_samples:
            self._flush_chunk()

    def stop(self) -> None:
        """Flush any remaining audio and finalize."""
        if not self._running:
            return

        # Flush partial buffer
        if self._buffer_samples > 0:
            self._flush_chunk()

        self._running = False
        logger.info(
            "recorder_stopped",
            session_id=self.session_id,
            total_samples=self._total_samples,
            chunks=self._sequence,
        )

    def _flush_chunk(self) -> None:
        """Combine buffered audio into one chunk and write to FLAC."""
        if not self._buffer:
            return

        # Combine buffer
        combined = np.concatenate(list(self._buffer))
        self._buffer.clear()
        self._buffer_samples = 0

        # If we have more than a chunk, keep the overflow
        if len(combined) > self.chunk_samples:
            overflow = combined[self.chunk_samples:]
            combined = combined[:self.chunk_samples]
            self._buffer.append(overflow)
            self._buffer_samples = len(overflow)

        # Write FLAC
        timestamp = int(time.time() * 1000)
        filename = f"chunk_{self._sequence:06d}_{timestamp}.flac"
        filepath = self.session_dir / filename

        sf.write(str(filepath), combined, self.sample_rate, format="FLAC")

        # Update manifest
        chunk_info = {
            "sequence": self._sequence,
            "filename": filename,
            "samples": len(combined),
            "timestamp_ms": timestamp,
        }
        self._manifest["chunks"].append(chunk_info)
        self._total_samples += len(combined)
        self._manifest["total_samples"] = self._total_samples
        self._write_manifest()

        self._sequence += 1
        logger.debug("chunk_flushed", filename=filename, samples=len(combined))

    def _write_manifest(self) -> None:
        manifest_path = self.session_dir / "manifest.json"
        manifest_path.write_text(json.dumps(self._manifest, indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_recorder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/meeting/__init__.py modules/orchestration-service/src/meeting/recorder.py modules/orchestration-service/tests/test_recorder.py
git commit -m "feat(orchestration): add FlacChunkRecorder with crash-safe manifest tracking"
```

---

### Task 4: Session manager with heartbeat orphan detection

**Files:**
- Create: `modules/orchestration-service/src/meeting/session_manager.py`
- Create: `modules/orchestration-service/tests/test_session_manager.py`

- [ ] **Step 1: Write failing test**

```python
# modules/orchestration-service/tests/test_session_manager.py
"""Tests for MeetingSessionManager — lifecycle, promotion, heartbeat."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from meeting.session_manager import MeetingSessionManager
from database.meeting_models import MeetingSession


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingSessionManager:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

    async def test_create_ephemeral(self, manager):
        session = await manager.create_session(source_type="loopback")
        assert session.status == "ephemeral"
        assert session.source_type == "loopback"
        assert session.recording_path is None

    async def test_promote_to_meeting(self, manager, db_session):
        session = await manager.create_session(source_type="loopback")
        promoted = await manager.promote_to_meeting(session.id)
        assert promoted.status == "active"
        assert promoted.recording_path is not None

    async def test_end_meeting(self, manager):
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)
        ended = await manager.end_meeting(session.id)
        assert ended.status == "completed"
        assert ended.ended_at is not None

    async def test_detect_orphans(self, manager, db_session):
        """Sessions with no activity for >120s should be marked interrupted."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # Manually set last_activity_at to 200s ago
        result = await db_session.get(MeetingSession, session.id)
        result.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        await db_session.commit()

        orphans = await manager.detect_orphans()
        assert len(orphans) >= 1
        assert any(o.id == session.id for o in orphans)

    async def test_recover_untranslated(self, manager, db_session):
        """Transcripts without translations should be returned for re-submission."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # Add a transcript with no corresponding translation
        transcript = await manager.add_transcript(
            session_id=session.id,
            text="Hello world",
            timestamp_ms=1000,
            language="en",
            confidence=0.95,
            is_final=True,
        )

        untranslated = await manager.recover_untranslated()
        assert len(untranslated) >= 1
        assert any(t.id == transcript.id for t in untranslated)

    async def test_recover_untranslated_excludes_translated(self, manager, db_session):
        """Transcripts that already have translations should NOT be returned."""
        from database.meeting_models import MeetingTranslation

        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        transcript = await manager.add_transcript(
            session_id=session.id,
            text="Hello world",
            timestamp_ms=1000,
            language="en",
            confidence=0.95,
            is_final=True,
        )

        # Add a translation for this transcript
        translation = MeetingTranslation(
            transcript_id=transcript.id,
            target_language="es",
            translated_text="Hola mundo",
            model_used="qwen3.5:7b",
        )
        db_session.add(translation)
        await db_session.commit()

        untranslated = await manager.recover_untranslated()
        assert not any(t.id == transcript.id for t in untranslated)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_session_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Write MeetingSessionManager**

```python
# modules/orchestration-service/src/meeting/session_manager.py
"""MeetingSessionManager — session lifecycle, promotion, heartbeat orphan detection.

Manages the full lifecycle: ephemeral → active → completed/interrupted.
Heartbeat monitoring detects sessions abandoned without an end event.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.meeting_models import MeetingSession, MeetingTranscript, MeetingTranslation
from livetranslate_common.logging import get_logger

logger = get_logger()


class MeetingSessionManager:
    def __init__(
        self,
        db: AsyncSession,
        recording_base_path: Path,
        heartbeat_timeout_s: int = 120,
    ):
        self.db = db
        self.recording_base_path = recording_base_path
        self.heartbeat_timeout_s = heartbeat_timeout_s

    async def create_session(
        self,
        source_type: str,
        sample_rate: int = 48000,
        channels: int = 2,
    ) -> MeetingSession:
        session = MeetingSession(
            id=uuid.uuid4(),
            source_type=source_type,
            status="ephemeral",
            started_at=datetime.now(timezone.utc),
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_created", session_id=str(session.id), source=source_type)
        return session

    async def promote_to_meeting(self, session_id: uuid.UUID) -> MeetingSession:
        session = await self.db.get(MeetingSession, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.status = "active"
        session.recording_path = str(self.recording_base_path / str(session_id))
        session.last_activity_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_promoted", session_id=str(session_id))
        return session

    async def end_meeting(self, session_id: uuid.UUID) -> MeetingSession:
        session = await self.db.get(MeetingSession, session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.status = "completed"
        session.ended_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(session)
        logger.info("session_ended", session_id=str(session_id))
        return session

    async def update_heartbeat(self, session_id: uuid.UUID) -> None:
        await self.db.execute(
            update(MeetingSession)
            .where(MeetingSession.id == session_id)
            .values(last_activity_at=datetime.now(timezone.utc))
        )
        await self.db.commit()

    async def add_transcript(
        self,
        session_id: uuid.UUID,
        text: str,
        timestamp_ms: int,
        language: str,
        confidence: float,
        is_final: bool,
        speaker_id: str | None = None,
        source_id: str | None = None,
    ) -> MeetingTranscript:
        transcript = MeetingTranscript(
            session_id=session_id,
            timestamp_ms=timestamp_ms,
            text=text,
            source_language=language,
            confidence=confidence,
            is_final=is_final,
            speaker_id=speaker_id,
            source_id=source_id,
        )
        self.db.add(transcript)
        await self.db.commit()
        await self.db.refresh(transcript)
        return transcript

    async def detect_orphans(self) -> list[MeetingSession]:
        """Find active sessions with no heartbeat for > timeout seconds."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.heartbeat_timeout_s)
        result = await self.db.execute(
            select(MeetingSession).where(
                MeetingSession.status == "active",
                MeetingSession.last_activity_at < cutoff,
            )
        )
        return list(result.scalars().all())

    async def mark_interrupted(self, session_id: uuid.UUID) -> None:
        await self.db.execute(
            update(MeetingSession)
            .where(MeetingSession.id == session_id)
            .values(status="interrupted", ended_at=datetime.now(timezone.utc))
        )
        await self.db.commit()
        logger.warning("session_interrupted", session_id=str(session_id))

    async def recover_on_startup(self) -> list[MeetingSession]:
        """Find sessions marked 'active' that never got an end event."""
        result = await self.db.execute(
            select(MeetingSession).where(MeetingSession.status == "active")
        )
        orphans = list(result.scalars().all())
        for orphan in orphans:
            await self.mark_interrupted(orphan.id)
        if orphans:
            logger.warning("startup_orphans_recovered", count=len(orphans))
        return orphans

    async def recover_untranslated(self) -> list[MeetingTranscript]:
        """Find finalized transcripts without corresponding translations and return them.

        Used during recovery to re-submit transcripts that were persisted but
        whose translation was lost (e.g., due to crash or translation service outage).
        """
        result = await self.db.execute(
            select(MeetingTranscript)
            .outerjoin(
                MeetingTranslation,
                MeetingTranscript.id == MeetingTranslation.transcript_id,
            )
            .where(
                MeetingTranscript.is_final == True,  # noqa: E712
                MeetingTranslation.id == None,  # noqa: E711
            )
        )
        untranslated = list(result.scalars().all())
        if untranslated:
            logger.info("untranslated_transcripts_found", count=len(untranslated))
        return untranslated
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_session_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/meeting/session_manager.py modules/orchestration-service/tests/test_session_manager.py
git commit -m "feat(orchestration): add MeetingSessionManager with heartbeat orphan detection"
```

---

## Chunk 3: Audio Pipeline Integration

### Task 5: Audio downsampler (48kHz → 16kHz)

**Files:**
- Create: `modules/orchestration-service/src/meeting/downsampler.py`
- Create: `modules/orchestration-service/tests/test_downsampler.py`

- [ ] **Step 1: Write failing test**

```python
# modules/orchestration-service/tests/test_downsampler.py
"""Tests for audio downsampler — native quality → 16kHz mono for transcription."""
import numpy as np
import pytest

from meeting.downsampler import downsample_to_16k


class TestDownsampler:
    def test_48k_to_16k(self):
        # 1 second of 48kHz mono
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=48000)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_44100_to_16k(self):
        audio = np.random.randn(44100).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=44100)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_16k_passthrough(self):
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = downsample_to_16k(audio, source_rate=16000)
        assert len(result) == 16000
        np.testing.assert_array_equal(result, audio)

    def test_stereo_to_mono(self):
        # Stereo: shape (samples, 2)
        stereo = np.random.randn(48000, 2).astype(np.float32) * 0.1
        result = downsample_to_16k(stereo, source_rate=48000, channels=2)
        assert result.ndim == 1
        assert len(result) == 16000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_downsampler.py -v`
Expected: FAIL

- [ ] **Step 3: Write downsampler**

```python
# modules/orchestration-service/src/meeting/downsampler.py
"""Audio downsampler for the meeting pipeline.

Converts native-quality audio (48kHz+ stereo) to 16kHz mono for
transcription service. Uses scipy.signal.resample for quality,
with librosa fallback.

The downsampled audio is ONLY for transcription — recordings stay
at native quality.
"""
from __future__ import annotations

import numpy as np


def downsample_to_16k(
    audio: np.ndarray,
    source_rate: int,
    channels: int = 1,
    target_rate: int = 16000,
) -> np.ndarray:
    """Downsample audio to 16kHz mono float32.

    Args:
        audio: Input audio array. Shape (samples,) for mono or (samples, channels) for stereo.
        source_rate: Source sample rate in Hz.
        channels: Number of channels (1=mono, 2=stereo). Auto-detected from shape if 2D.
        target_rate: Target sample rate (default 16000).

    Returns:
        1D float32 array at target_rate, mono.
    """
    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Passthrough if already at target rate
    if source_rate == target_rate:
        return audio.astype(np.float32)

    # Resample using scipy
    from scipy.signal import resample

    num_samples = int(len(audio) * target_rate / source_rate)
    resampled = resample(audio, num_samples)

    return resampled.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_downsampler.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/meeting/downsampler.py modules/orchestration-service/tests/test_downsampler.py
git commit -m "feat(orchestration): add audio downsampler (48kHz → 16kHz mono)"
```

---

### Task 6: Meeting pipeline coordinator

**Files:**
- Create: `modules/orchestration-service/src/meeting/pipeline.py`
- Create: `modules/orchestration-service/tests/test_pipeline.py`

This is the central coordinator that ties together: audio source → recording fork → downsampling → transcription WebSocket → translation → DB persistence → frontend broadcast.

- [ ] **Step 1: Write the pipeline coordinator**

```python
# modules/orchestration-service/src/meeting/pipeline.py
"""Meeting pipeline coordinator.

Ties together: audio source → recording fork → downsampling →
transcription WebSocket → translation → DB persistence → frontend broadcast.

Handles both ephemeral (stream-through) and active meeting modes.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import numpy as np
from livetranslate_common.logging import get_logger
from livetranslate_common.models import AudioChunk

from meeting.downsampler import downsample_to_16k
from meeting.recorder import FlacChunkRecorder
from meeting.session_manager import MeetingSessionManager

logger = get_logger()


class MeetingPipeline:
    """Coordinates the meeting audio pipeline for a single session.

    In ephemeral mode: audio passes through to transcription only.
    In meeting mode: audio is also recorded to FLAC and persisted to DB.
    """

    def __init__(
        self,
        session_manager: MeetingSessionManager,
        recording_base_path: Path,
        source_type: str = "loopback",
        sample_rate: int = 48000,
        channels: int = 1,
    ):
        self.session_manager = session_manager
        self.recording_base_path = recording_base_path
        self.source_type = source_type
        self.sample_rate = sample_rate
        self.channels = channels

        self._session_id: uuid.UUID | None = None
        self._recorder: FlacChunkRecorder | None = None
        self._is_meeting = False
        self._running = False
        self._last_heartbeat_at: float = 0.0  # monotonic timestamp of last heartbeat update

    @property
    def session_id(self) -> uuid.UUID | None:
        return self._session_id

    @property
    def is_meeting(self) -> bool:
        return self._is_meeting

    async def start(self) -> uuid.UUID:
        """Start an ephemeral session."""
        session = await self.session_manager.create_session(
            source_type=self.source_type,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        self._session_id = session.id
        self._running = True
        logger.info("pipeline_started", session_id=str(session.id), mode="ephemeral")
        return session.id

    async def promote_to_meeting(self) -> None:
        """Promote current ephemeral session to a full meeting."""
        if self._session_id is None:
            raise RuntimeError("No active session to promote")

        session = await self.session_manager.promote_to_meeting(self._session_id)

        self._recorder = FlacChunkRecorder(
            session_id=str(self._session_id),
            base_path=self.recording_base_path,
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
        self._recorder.start()
        self._is_meeting = True

        logger.info("pipeline_promoted", session_id=str(self._session_id))

    async def process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Process incoming audio chunk.

        1. If meeting mode: record at native quality
        2. Downsample to 16kHz mono for transcription
        3. Update heartbeat

        Returns: downsampled audio for forwarding to transcription service.
        """
        if not self._running:
            return np.array([], dtype=np.float32)

        # Fork 1: Record at native quality (meeting mode only)
        if self._is_meeting and self._recorder:
            self._recorder.write(audio)

        # Fork 2: Downsample for transcription
        downsampled = downsample_to_16k(
            audio,
            source_rate=self.sample_rate,
            channels=self.channels,
        )

        # Update heartbeat (throttled: at most once per 30 seconds to reduce DB writes)
        if self._session_id and self._is_meeting:
            now = time.monotonic()
            if now - self._last_heartbeat_at > 30.0:
                await self.session_manager.update_heartbeat(self._session_id)
                self._last_heartbeat_at = now

        return downsampled

    async def end(self) -> None:
        """End the session (whether ephemeral or meeting)."""
        if self._recorder:
            self._recorder.stop()
            self._recorder = None

        if self._session_id and self._is_meeting:
            await self.session_manager.end_meeting(self._session_id)

        self._running = False
        self._is_meeting = False
        logger.info("pipeline_ended", session_id=str(self._session_id))
```

- [ ] **Step 2: Write tests for the pipeline**

```python
# modules/orchestration-service/tests/test_pipeline.py
"""Tests for MeetingPipeline — behavioral tests against real DB.

Uses the same db_session fixture pattern as test_session_manager.py.
NO mocks — real MeetingSessionManager with a real database session.
"""
import json
import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from meeting.pipeline import MeetingPipeline
from meeting.session_manager import MeetingSessionManager
from database.meeting_models import MeetingSession


@pytest.mark.asyncio
@pytest.mark.integration
class TestMeetingPipeline:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

    async def test_start_creates_ephemeral(self, manager, tmp_path):
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
        )
        session_id = await pipeline.start()
        assert session_id is not None
        assert not pipeline.is_meeting

        # Verify session exists in DB
        session = await manager.db.get(MeetingSession, session_id)
        assert session is not None
        assert session.status == "ephemeral"

    async def test_process_audio_returns_downsampled(self, manager, tmp_path):
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
        )
        await pipeline.start()

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = await pipeline.process_audio(audio)
        assert len(result) == 16000  # downsampled to 16kHz

    async def test_promote_starts_recording(self, manager, db_session, tmp_path):
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        assert pipeline.is_meeting is True

        # Verify DB state
        session = await db_session.get(MeetingSession, pipeline.session_id)
        assert session.status == "active"
        assert session.recording_path is not None

    async def test_promote_records_audio_to_flac(self, manager, tmp_path):
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
            sample_rate=48000,
            channels=1,
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()

        # Write enough audio to trigger a chunk flush
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        await pipeline.process_audio(audio)
        await pipeline.end()

        # Verify FLAC files were written
        session_dir = tmp_path / "recordings" / str(pipeline.session_id)
        manifest = json.loads((session_dir / "manifest.json").read_text())
        assert manifest["total_samples"] > 0

    async def test_end_stops_everything(self, manager, db_session, tmp_path):
        pipeline = MeetingPipeline(
            session_manager=manager,
            recording_base_path=tmp_path / "recordings",
        )
        await pipeline.start()
        await pipeline.promote_to_meeting()
        session_id = pipeline.session_id
        await pipeline.end()

        assert pipeline.is_meeting is False

        # Verify DB state
        session = await db_session.get(MeetingSession, session_id)
        assert session.status == "completed"
        assert session.ended_at is not None
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add modules/orchestration-service/src/meeting/pipeline.py modules/orchestration-service/tests/test_pipeline.py
git commit -m "feat(orchestration): add MeetingPipeline coordinator with recording fork"
```

---

### Task 7: Heartbeat background task

**Files:**
- Create: `modules/orchestration-service/src/meeting/heartbeat.py`
- Create: `modules/orchestration-service/tests/test_heartbeat.py`

- [ ] **Step 1: Write heartbeat background task**

```python
# modules/orchestration-service/src/meeting/heartbeat.py
"""Background heartbeat monitor for orphaned meeting sessions.

Runs periodically (every 60s) to check for active sessions that
haven't received audio for > heartbeat_timeout_s. Marks them
as interrupted.
"""
from __future__ import annotations

import asyncio

from livetranslate_common.logging import get_logger
from meeting.session_manager import MeetingSessionManager

logger = get_logger()


async def run_heartbeat_monitor(
    session_manager: MeetingSessionManager,
    check_interval_s: int = 60,
) -> None:
    """Periodically check for orphaned sessions and mark them interrupted."""
    logger.info("heartbeat_monitor_started", interval_s=check_interval_s)

    while True:
        try:
            orphans = await session_manager.detect_orphans()
            for orphan in orphans:
                await session_manager.mark_interrupted(orphan.id)
                logger.warning(
                    "orphan_session_interrupted",
                    session_id=str(orphan.id),
                    source_type=orphan.source_type,
                )
        except Exception:
            logger.exception("heartbeat_check_failed")

        await asyncio.sleep(check_interval_s)
```

- [ ] **Step 2: Write behavioral test for heartbeat**

```python
# modules/orchestration-service/tests/test_heartbeat.py
"""Tests for heartbeat background monitor — behavioral test against real DB."""
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from meeting.heartbeat import run_heartbeat_monitor
from meeting.session_manager import MeetingSessionManager
from database.meeting_models import MeetingSession


@pytest.mark.asyncio
@pytest.mark.integration
class TestHeartbeatMonitor:
    @pytest.fixture
    def manager(self, db_session: AsyncSession, tmp_path):
        return MeetingSessionManager(
            db=db_session,
            recording_base_path=tmp_path / "recordings",
            heartbeat_timeout_s=120,
        )

    async def test_stale_session_marked_interrupted(self, manager, db_session):
        """An active session with stale last_activity_at should be marked interrupted."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # Set last_activity_at to 200 seconds ago (> 120s timeout)
        db_obj = await db_session.get(MeetingSession, session.id)
        db_obj.last_activity_at = datetime.now(timezone.utc) - timedelta(seconds=200)
        await db_session.commit()

        # Run ONE iteration of the heartbeat monitor (use a short interval
        # and cancel after first check completes)
        task = asyncio.create_task(
            run_heartbeat_monitor(manager, check_interval_s=3600)
        )
        # Give it time to run one iteration
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify the session was marked interrupted
        await db_session.refresh(db_obj)
        assert db_obj.status == "interrupted"
        assert db_obj.ended_at is not None

    async def test_active_session_not_interrupted(self, manager, db_session):
        """An active session with recent activity should NOT be interrupted."""
        session = await manager.create_session(source_type="loopback")
        await manager.promote_to_meeting(session.id)

        # last_activity_at is now (fresh), well within the 120s timeout
        task = asyncio.create_task(
            run_heartbeat_monitor(manager, check_interval_s=3600)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        db_obj = await db_session.get(MeetingSession, session.id)
        assert db_obj.status == "active"
```

- [ ] **Step 3: Run tests**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && uv run pytest modules/orchestration-service/tests/test_heartbeat.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add modules/orchestration-service/src/meeting/heartbeat.py modules/orchestration-service/tests/test_heartbeat.py
git commit -m "feat(orchestration): add heartbeat monitor for orphaned meeting sessions"
```

---

## Summary

**Total tasks:** 7 tasks, ~30 steps
**Branch:** `plan-3/meeting-pipeline`

After completing Plan 3:
- `meeting_sessions`, `meeting_transcripts`, `meeting_translations` tables (Alembic migration)
- SQLAlchemy ORM models for meeting data
- `FlacChunkRecorder` — crash-safe continuous recording with manifest
- `MeetingSessionManager` — lifecycle, promotion, heartbeat orphan detection, startup recovery
- `downsample_to_16k` — native quality → 16kHz mono for transcription
- `MeetingPipeline` — coordinator wiring recording fork, downsampling, session management
- Background heartbeat monitor task
- Both loopback and bot audio sources can plug into this pipeline

**Deferred to follow-up plans:**

- **Post-meeting processing** (FLAC concatenation into a single file, batch re-transcription with higher-quality settings, full diarization pass) is not implemented in this plan. The spec describes these as background tasks triggered by "End Meeting" -- they will be covered in a dedicated follow-up plan once the core pipeline is stable.

- **Additive migration steps 2-6** are deferred. This plan implements only step 1 (create `meeting_sessions`, `meeting_transcripts`, `meeting_translations` alongside existing `bot_sessions`). The remaining migration steps are:
  - Step 2: Backfill existing data from `bot_sessions` into `meeting_sessions` with `source_type = 'google_meet_bot'`
  - Step 3: Switch all writes to `meeting_sessions`
  - Step 4: Update foreign key references in bot code
  - Step 5: Deprecate reads from `bot_sessions`
  - Step 6: Drop `bot_sessions` after a safe period

  These steps require coordination with the bot management code and should be planned as a separate migration effort after the meeting pipeline is validated in production.
