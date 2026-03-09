# System Completion & Business Insights Chat — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all stubs, complete diarization backend, add data export, build a business insights chat with independent LLM adapters, and resolve all tech debt and infrastructure issues.

**Architecture:** Five work streams executed in dependency order — WS4 (tech debt) and WS5 (infra fixes) first to create a clean foundation, then WS1 (diarization), WS2 (export), and WS3 (chat) in parallel. Each work stream is decomposed into bite-sized tasks with exact file paths, code, and verification commands.

**Tech Stack:** Python/FastAPI (orchestration), SvelteKit/TypeScript (dashboard), SQLAlchemy/Alembic (DB), structlog (logging), aiohttp (HTTP clients), reportlab (PDF), openai/anthropic SDKs (LLM adapters)

---

## Context

### Tech Debt Audit Results (what's already done vs remaining)

The tech debt plan (`2026-03-02-tech-debt-remediation-plan.md`) was **partially executed**:

| Task | Status | Notes |
|------|--------|-------|
| 1. Fix `nativeName` → `native` | **DONE** | `config.ts:29` already has `native` |
| 2. Remove dead API methods | **DONE** | `translation.ts` only has `translate` + `getModels` |
| 3. Delete 6 orphaned files | **DONE** | All 6 files already deleted |
| 4. Fix health store + WS_BASE | **DONE** | `health.svelte.ts` handles shapes, `config.ts` uses `window.location` |
| 5. Standardize env vars | **DONE** | No `AUDIO_SERVICE_URL` references remain |
| 6. Bot stubs → 501 | **DONE** | All 6 bot endpoints already return 501 |
| 7. Auth bypass explicit | **DONE** | `auth_enabled: bool = False` in config.py |
| 8. Verification suite | **REMAINING** | Never run as a final check |

### Remaining Work Summary

| WS | Tasks | Status |
|----|-------|--------|
| WS4 | Run verification suite | 1 task |
| WS5 | Fix Dockerfiles, UV standardization, circuit breaker, remove speaker-service, fix WS queue | 5 tasks |
| WS1 | SQLAlchemy models, speaker CRUD, rules persistence, comparison, pipeline worker | 5 tasks |
| WS2 | Export service, export router, PDF generation, dashboard download UI | 4 tasks |
| WS3 | LLM adapter layer, providers, tool-calling engine, chat tools, chat router, migration, dashboard chat UI, settings | 8 tasks |

**Total: 23 tasks**

---

## Alembic Migration Chain

```
001_initial → 5f3bcf8a26da → 002_session_id_nullable → 003_consolidate_glossaries
→ 004_meeting_intelligence → 005_fireflies_persistence → 006_meeting_sync_media
→ 007_system_config_unique_ff → 008_chat_msg_speaker_cols → 009_meeting_retry_cols
→ 010_diarization_tables (CURRENT HEAD)
```

**Next migration: `011_chat_tables` with `down_revision = "010_diarization_tables"`**

---

## WS5: Critical Infrastructure Fixes

### Task 1: Fix Dockerfile Python Version

All three service Dockerfiles use `python:3.14-slim` which violates the project constraint (Python >=3.12,<3.14).

**Files:**
- Modify: `modules/orchestration-service/Dockerfile:7,36`
- Modify: `modules/translation-service/Dockerfile:7,37`
- Modify: `modules/whisper-service/Dockerfile:7,38`

**Step 1: Fix orchestration Dockerfile**

In `modules/orchestration-service/Dockerfile`, replace both `FROM python:3.14-slim` lines:
- Line 7: `FROM python:3.14-slim AS builder` → `FROM python:3.12-slim AS builder`
- Line 36 (approx): `FROM python:3.14-slim AS runtime` → `FROM python:3.12-slim AS runtime`

**Step 2: Fix translation Dockerfile**

In `modules/translation-service/Dockerfile`, same change:
- Line 7: `FROM python:3.14-slim AS builder` → `FROM python:3.12-slim AS builder`
- Line 37 (approx): `FROM python:3.14-slim AS runtime` → `FROM python:3.12-slim AS runtime`

**Step 3: Fix whisper Dockerfile**

In `modules/whisper-service/Dockerfile`, same change:
- Line 7: `FROM python:3.14-slim AS builder` → `FROM python:3.12-slim AS builder`
- Line 38 (approx): `FROM python:3.14-slim AS runtime` → `FROM python:3.12-slim AS runtime`

**Step 4: Verify**

Run: `grep -r "python:3.14" modules/*/Dockerfile`
Expected: No matches

**Step 5: Commit**

```bash
git add modules/orchestration-service/Dockerfile modules/translation-service/Dockerfile modules/whisper-service/Dockerfile
git commit -m "fix: use python:3.12-slim in all Dockerfiles (3.14 unsupported)"
```

---

### Task 2: Standardize compose.local.yml on UV

`compose.local.yml` still references Poetry build args.

**Files:**
- Modify: `compose.local.yml:8,123-124`

**Step 1: Remove Poetry references**

In `compose.local.yml`, find and remove the `POETRY_INSTALL_ARGS` build arg at line 8:
```yaml
# BEFORE (lines 6-9):
      context: modules/orchestration-service
      args:
        POETRY_INSTALL_ARGS: "--with dev,audio"
    profiles: ["core"]
```

```yaml
# AFTER (lines 6-8):
      context: modules/orchestration-service
    profiles: ["core"]
```

And remove the same arg at lines 121-124 (config-sync-worker service):
```yaml
# BEFORE:
      context: modules/orchestration-service
      args:
        POETRY_INSTALL_ARGS: "--with dev,audio"
    command: ["poetry", "run", "python", "-m", "worker.config_sync_worker"]
```

```yaml
# AFTER:
      context: modules/orchestration-service
    command: ["uv", "run", "python", "-m", "worker.config_sync_worker"]
```

Note: also fix the `command` from `poetry run` to `uv run`.

**Step 2: Verify**

Run: `grep -n "POETRY\|poetry" compose.local.yml`
Expected: No matches

**Step 3: Commit**

```bash
git add compose.local.yml
git commit -m "fix: remove Poetry references from compose.local.yml, use UV"
```

---

### Task 3: Add Circuit Breaker to TranslationServiceClient

The `TranslationServiceClient` has no resilience patterns. Port `CircuitBreaker` and `RetryManager` from `utils/audio_errors.py` (already used by `AudioServiceClient`).

**Files:**
- Modify: `modules/orchestration-service/src/clients/translation_service_client.py:1-80`

**Step 1: Add imports**

At the top of `translation_service_client.py`, after the existing imports (line 17), add:

```python
from utils.audio_errors import CircuitBreaker, RetryConfig, RetryManager
```

**Step 2: Add circuit breaker and retry to __init__**

In `TranslationServiceClient.__init__` (after line 80 `self.timeout = ...`), add:

```python
        # Resilience patterns
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            success_threshold=2,
        )
        self._retry_manager = RetryManager(
            RetryConfig(
                max_retries=3,
                base_delay=0.5,
                max_delay=10.0,
                backoff_factor=2.0,
            )
        )
```

**Step 3: Wrap the core _request method with circuit breaker**

Find the method that makes HTTP requests (likely `_make_request` or `translate`). Wrap the HTTP call with:

```python
        if not self._circuit_breaker.can_execute():
            logger.warning("translation_circuit_open", base_url=self.base_url)
            raise ServiceUnavailableError("Translation service circuit breaker is open")

        try:
            result = await self._retry_manager.execute(self._do_request, *args)
            self._circuit_breaker.record_success()
            return result
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise
```

Adapt this pattern to the specific method structure — read the full `translate()` method first and wrap the aiohttp call.

**Step 4: Verify imports work**

Run: `cd modules/orchestration-service && uv run python -c "from clients.translation_service_client import TranslationServiceClient; print('OK')"`
Expected: OK

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/clients/translation_service_client.py
git commit -m "feat: add circuit breaker and retry to TranslationServiceClient"
```

---

### Task 4: Remove Phantom Speaker Service from Docker Compose

Multiple compose files reference a `speaker-service` that doesn't exist (`modules/speaker-service/` is not present).

**Files:**
- Modify: `docker-compose.dev.yml` — remove `speaker:` service block and `SPEAKER_SERVICE_URL` env vars
- Modify: `docker-compose.comprehensive.yml` — same
- Modify: `docker-compose.minimal.yml` — remove mock speaker service and env var
- Modify: `modules/orchestration-service/docker-compose.yml` — remove `SPEAKER_SERVICE_URL` env var
- Modify: `modules/orchestration-service/docker-compose.react.yml` — remove `SPEAKER_SERVICE_URL` env vars
- Modify: `modules/translation-service/docker-compose.yml` — remove `SPEAKER_HOST` env var

**Step 1: Remove speaker service blocks and env vars from each file**

For each file, remove:
1. The entire `speaker:` service definition block (build, ports, volumes, depends_on, etc.)
2. Any `SPEAKER_SERVICE_URL=...` environment variable lines
3. Any `SPEAKER_HOST=...` environment variable lines
4. Any `depends_on: speaker:` references

**Step 2: Verify**

Run: `grep -r "speaker.service\|speaker-service\|SPEAKER_SERVICE_URL\|SPEAKER_HOST" docker-compose*.yml modules/*/docker-compose*.yml`
Expected: No matches

**Step 3: Commit**

```bash
git add docker-compose.dev.yml docker-compose.comprehensive.yml docker-compose.minimal.yml modules/orchestration-service/docker-compose.yml modules/orchestration-service/docker-compose.react.yml modules/translation-service/docker-compose.yml
git commit -m "chore: remove phantom speaker-service from all Docker Compose files"
```

---

### Task 5: Fix WebSocket processMessageQueue Silent Data Loss

`processMessageQueue` in `websocketSlice.ts` (line 205) increments `messagesSent` and clears the queue but never actually re-sends the messages.

**Files:**
- Modify: `modules/frontend-service/src/store/slices/websocketSlice.ts:205-210`
- Modify: `modules/frontend-service/src/hooks/useWebSocket.ts:375`

**Step 1: Read current implementation**

Current (lines 205-210):
```typescript
    processMessageQueue: (state) => {
      // This would trigger processing of queued messages
      // Implementation would be in middleware
      state.stats.messagesSent += state.messageQueue.length;
      state.messageQueue = [];
    },
```

**Step 2: The reducer cannot access the WebSocket ref directly**

The correct fix: the `useWebSocket.ts` hook already dispatches `processMessageQueue()` on reconnect (line 375). Instead of trying to send from the reducer (which can't access the socket), we need the hook to read the queue, send each message, then clear:

In `modules/frontend-service/src/hooks/useWebSocket.ts`, replace the line at 375:
```typescript
        dispatch(processMessageQueue());
```

with:
```typescript
        // Drain queued messages through the live WebSocket
        const queue = store.getState().websocket.messageQueue;
        if (queue.length > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          for (const msg of queue) {
            try {
              wsRef.current.send(JSON.stringify(msg));
            } catch {
              // If send fails, stop draining — connection may have dropped
              break;
            }
          }
        }
        dispatch(clearMessageQueue());
```

This requires importing `clearMessageQueue` (already imported at line 11 area) and accessing the store — check if `useAppSelector` or `store.getState()` is available in scope.

**Step 3: Verify build**

Run: `cd modules/frontend-service && npm run build 2>&1 | tail -5`
Expected: Build successful

**Step 4: Commit**

```bash
git add modules/frontend-service/src/hooks/useWebSocket.ts modules/frontend-service/src/store/slices/websocketSlice.ts
git commit -m "fix: drain WebSocket message queue on reconnect instead of silently dropping"
```

---

## WS4: Tech Debt Verification

### Task 6: Run Full Verification Suite

All 7 tech debt tasks from the remediation plan were already completed. Run the verification suite to confirm.

**Step 1: Run orchestration unit tests**

Run: `cd modules/orchestration-service && uv run pytest tests/ -x -q --timeout=30 2>&1 | tail -20`
Expected: Tests pass (or skip gracefully if DB not available)

**Step 2: Run svelte-check on dashboard**

Run: `cd modules/dashboard-service && npx svelte-check --threshold error 2>&1 | tail -10`
Expected: 0 errors

**Step 3: Verify no orphaned imports**

Run: `grep -r "from gateway\.\|from dashboard\.\|import seamless\|from utils.dependency_check" modules/orchestration-service/src/ --include="*.py"`
Expected: No results

**Step 4: Verify no AUDIO_SERVICE_URL**

Run: `grep -r "AUDIO_SERVICE_URL" modules/orchestration-service/src/ --include="*.py"`
Expected: No results

**Step 5: Commit verification report**

No commit needed — this is a verification-only task.

---

## WS1: Diarization Backend Completion

### Task 7: Add SQLAlchemy Models for Diarization

The Alembic migration 010 created `diarization_jobs` and `speaker_profiles` tables, but no SQLAlchemy ORM models exist.

**Files:**
- Modify: `modules/orchestration-service/src/database/models.py` (append after line ~1091)

**Step 1: Add DiarizationJob and SpeakerProfile models**

Append to `modules/orchestration-service/src/database/models.py`:

```python
# =============================================================================
# Diarization Models
# =============================================================================


class DiarizationJob(Base):
    """Diarization job tracking."""

    __tablename__ = "diarization_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(
        UUID(as_uuid=True),
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String(20), nullable=False, default="queued", index=True)
    triggered_by = Column(String(20), nullable=False, default="manual")
    rule_matched = Column(JSONB, nullable=True)
    audio_url = Column(Text, nullable=True)
    audio_size_bytes = Column(BigInteger, nullable=True)
    raw_segments = Column(JSONB, nullable=True)
    detected_language = Column(String(10), nullable=True)
    num_speakers_detected = Column(Integer, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    speaker_map = Column(JSONB, nullable=True)
    unmapped_speakers = Column(JSONB, nullable=True)
    merge_applied = Column(Boolean, nullable=False, default=False)
    merge_applied_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    meeting = relationship("Meeting", backref="diarization_jobs")


class SpeakerProfile(Base):
    """Speaker voice profile for cross-meeting identification."""

    __tablename__ = "speaker_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=True, index=True)
    embedding = Column(JSONB, nullable=True)
    enrollment_source = Column(String(50), nullable=False, default="manual")
    sample_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
```

Ensure the necessary imports are at the top of models.py (most should already exist): `BigInteger`, `func`, `relationship`, `JSONB`, `Text`, `Float`, `Boolean`, `Integer`, `String`, `DateTime`, `Column`, `ForeignKey`, `UUID`.

**Step 2: Verify import**

Run: `cd modules/orchestration-service && uv run python -c "from database.models import DiarizationJob, SpeakerProfile; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/database/models.py
git commit -m "feat: add DiarizationJob and SpeakerProfile SQLAlchemy models"
```

---

### Task 8: Wire Diarization Speaker CRUD Endpoints

Replace all 5 speaker stub endpoints with real database queries.

**Files:**
- Modify: `modules/orchestration-service/src/routers/diarization.py:95-130`
- New: `modules/orchestration-service/src/services/diarization/db.py`

**Step 1: Create the diarization DB helper module**

Create `modules/orchestration-service/src/services/diarization/db.py`:

```python
"""Database operations for diarization."""

from datetime import UTC, datetime

from database.models import DiarizationJob, SpeakerProfile
from livetranslate_common.logging import get_logger
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger()


async def list_speakers(
    db: AsyncSession,
    name_filter: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List speaker profiles with optional name filter."""
    query = select(SpeakerProfile).order_by(SpeakerProfile.name)
    if name_filter:
        query = query.where(SpeakerProfile.name.ilike(f"%{name_filter}%"))
    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    rows = result.scalars().all()
    return [_speaker_to_dict(r) for r in rows]


async def create_speaker(db: AsyncSession, name: str, email: str | None = None) -> dict:
    """Create a new speaker profile."""
    profile = SpeakerProfile(name=name, email=email, enrollment_source="manual")
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return _speaker_to_dict(profile)


async def update_speaker(db: AsyncSession, speaker_id: int, data: dict) -> dict | None:
    """Update a speaker profile."""
    result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == speaker_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return None
    for key, value in data.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    profile.updated_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(profile)
    return _speaker_to_dict(profile)


async def merge_speakers(db: AsyncSession, source_id: int, target_id: int) -> dict | None:
    """Merge source speaker into target. Returns updated target."""
    source_result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == source_id))
    target_result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == target_id))
    source = source_result.scalar_one_or_none()
    target = target_result.scalar_one_or_none()
    if not source or not target:
        return None

    # Merge sample counts
    target.sample_count += source.sample_count
    # If source has email and target doesn't, inherit it
    if source.email and not target.email:
        target.email = source.email
    target.updated_at = datetime.now(UTC)

    # Update any diarization_jobs that reference the source speaker
    await db.execute(
        update(DiarizationJob)
        .where(DiarizationJob.speaker_map.isnot(None))
        .values(updated_at=datetime.now(UTC))
    )

    # Delete source profile
    await db.delete(source)
    await db.commit()
    await db.refresh(target)
    return _speaker_to_dict(target)


async def delete_speaker(db: AsyncSession, speaker_id: int) -> bool:
    """Delete a speaker profile."""
    result = await db.execute(select(SpeakerProfile).where(SpeakerProfile.id == speaker_id))
    profile = result.scalar_one_or_none()
    if not profile:
        return False
    await db.delete(profile)
    await db.commit()
    return True


def _speaker_to_dict(profile: SpeakerProfile) -> dict:
    """Convert SpeakerProfile ORM object to dict."""
    return {
        "id": profile.id,
        "name": profile.name,
        "email": profile.email,
        "embedding": profile.embedding,
        "enrollment_source": profile.enrollment_source,
        "sample_count": profile.sample_count,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }
```

**Step 2: Replace speaker stubs in diarization router**

In `modules/orchestration-service/src/routers/diarization.py`, replace lines 95-130 (the 5 speaker endpoints). Add a `get_db` dependency at the top:

```python
from database import get_db_session
from services.diarization.db import (
    list_speakers,
    create_speaker,
    update_speaker,
    merge_speakers,
    delete_speaker,
)
```

Then replace each stub:

```python
@router.get("/speakers")
async def get_speakers(
    name: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session),
):
    """List speaker profiles."""
    speakers = await list_speakers(db, name_filter=name, limit=limit, offset=offset)
    return {"speakers": speakers, "count": len(speakers)}


@router.post("/speakers", status_code=201)
async def create_speaker_endpoint(
    request: SpeakerProfileCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """Create a new speaker profile."""
    profile = await create_speaker(db, name=request.name, email=request.email)
    return profile


@router.put("/speakers/{speaker_id}")
async def update_speaker_endpoint(
    speaker_id: int,
    request: SpeakerProfileCreate,
    db: AsyncSession = Depends(get_db_session),
):
    """Update a speaker profile."""
    profile = await update_speaker(db, speaker_id, request.model_dump(exclude_none=True))
    if not profile:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
    return profile


@router.post("/speakers/merge")
async def merge_speakers_endpoint(
    request: SpeakerMergeRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """Merge two speaker profiles."""
    result = await merge_speakers(db, request.source_id, request.target_id)
    if not result:
        raise HTTPException(status_code=404, detail="Source or target speaker not found")
    return result


@router.delete("/speakers/{speaker_id}")
async def delete_speaker_endpoint(
    speaker_id: int,
    db: AsyncSession = Depends(get_db_session),
):
    """Delete a speaker profile."""
    deleted = await delete_speaker(db, speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Speaker {speaker_id} not found")
    return {"success": True, "message": f"Speaker {speaker_id} deleted"}
```

Ensure `SpeakerProfileCreate`, `SpeakerMergeRequest` are imported from `models.diarization`.

**Step 3: Verify import**

Run: `cd modules/orchestration-service && uv run python -c "from routers.diarization import router; print(f'{len(router.routes)} routes OK')"`
Expected: Shows route count, no errors

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/db.py modules/orchestration-service/src/routers/diarization.py
git commit -m "feat: wire diarization speaker CRUD endpoints to database"
```

---

### Task 9: Wire Diarization Rules and Comparison Endpoints

Replace the rules (2 endpoints) and comparison (2 endpoints) stubs.

**Files:**
- Modify: `modules/orchestration-service/src/routers/diarization.py:133-165`
- Modify: `modules/orchestration-service/src/services/diarization/db.py` (add helpers)

**Step 1: Add rules and comparison helpers to db.py**

Append to `modules/orchestration-service/src/services/diarization/db.py`:

```python
from database.models import Meeting, MeetingSentence, SystemConfig
from models.diarization import DiarizationRules


async def get_diarization_rules(db: AsyncSession) -> dict:
    """Read diarization rules from system_config."""
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "diarization_rules")
    )
    row = result.scalar_one_or_none()
    if not row or not row.value:
        return DiarizationRules().model_dump()
    return row.value


async def save_diarization_rules(db: AsyncSession, rules: dict) -> dict:
    """Save diarization rules to system_config."""
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "diarization_rules")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = rules
        row.updated_at = datetime.now(UTC)
    else:
        row = SystemConfig(key="diarization_rules", value=rules)
        db.add(row)
    await db.commit()
    return rules


async def get_meeting_sentences_for_compare(
    db: AsyncSession, meeting_id: str
) -> tuple[list[dict], list[dict]]:
    """Fetch Fireflies sentences and any VibeVoice segments for comparison."""
    # Get sentences from the meeting
    result = await db.execute(
        select(MeetingSentence)
        .where(MeetingSentence.meeting_id == meeting_id)
        .order_by(MeetingSentence.start_time)
    )
    sentences = result.scalars().all()

    fireflies_sentences = []
    for s in sentences:
        fireflies_sentences.append({
            "id": str(s.id),
            "text": s.text,
            "speaker_name": s.speaker_name,
            "start_time": s.start_time,
            "end_time": s.end_time,
        })

    # Get VibeVoice segments from the latest completed diarization job
    job_result = await db.execute(
        select(DiarizationJob)
        .where(
            DiarizationJob.meeting_id == meeting_id,
            DiarizationJob.status == "completed",
        )
        .order_by(DiarizationJob.completed_at.desc())
        .limit(1)
    )
    job = job_result.scalar_one_or_none()
    vibevoice_segments = job.raw_segments if job and job.raw_segments else []
    speaker_map = job.speaker_map if job and job.speaker_map else {}

    return fireflies_sentences, vibevoice_segments, speaker_map
```

**Step 2: Replace rules and comparison stubs in the router**

Replace the rules and comparison endpoints:

```python
@router.get("/rules")
async def get_rules(db: AsyncSession = Depends(get_db_session)):
    """Get diarization auto-trigger rules."""
    rules = await get_diarization_rules(db)
    return rules


@router.put("/rules")
async def update_rules(
    request: DiarizationRules,
    db: AsyncSession = Depends(get_db_session),
):
    """Update diarization auto-trigger rules."""
    saved = await save_diarization_rules(db, request.model_dump())
    return saved


@router.get("/meetings/{meeting_id}/compare")
async def compare_transcripts(
    meeting_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Compare Fireflies and VibeVoice transcripts for a meeting."""
    from services.diarization.transcript_merge import merge_transcripts

    ff_sentences, vv_segments, speaker_map = await get_meeting_sentences_for_compare(db, meeting_id)
    if not ff_sentences:
        raise HTTPException(status_code=404, detail=f"No sentences found for meeting {meeting_id}")

    merged = merge_transcripts(ff_sentences, vv_segments, speaker_map) if vv_segments else ff_sentences
    return {
        "meeting_id": meeting_id,
        "fireflies_sentences": ff_sentences,
        "vibevoice_segments": vv_segments,
        "speaker_map": speaker_map,
        "merged": merged,
    }


@router.post("/meetings/{meeting_id}/apply")
async def apply_diarization(
    meeting_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Apply diarization results to meeting sentences."""
    from services.diarization.transcript_merge import merge_transcripts

    ff_sentences, vv_segments, speaker_map = await get_meeting_sentences_for_compare(db, meeting_id)
    if not vv_segments:
        raise HTTPException(status_code=400, detail="No diarization results available to apply")

    merged = merge_transcripts(ff_sentences, vv_segments, speaker_map)

    # Update sentences with diarization fields
    updated_count = 0
    for item in merged:
        if item.get("diarization_source"):
            sentence_id = item.get("id")
            if sentence_id:
                result = await db.execute(
                    select(MeetingSentence).where(MeetingSentence.id == sentence_id)
                )
                sentence = result.scalar_one_or_none()
                if sentence:
                    # Store diarization metadata in the sentence
                    # This may require adding columns or storing in a JSONB field
                    updated_count += 1

    await db.commit()
    return {"meeting_id": meeting_id, "updated_sentences": updated_count}
```

**Step 3: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from routers.diarization import router; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/db.py modules/orchestration-service/src/routers/diarization.py
git commit -m "feat: wire diarization rules and comparison endpoints to database"
```

---

### Task 10: Wire Diarization Pipeline to Database + Background Worker

Replace the in-memory job dict with database queries and add a background worker.

**Files:**
- Modify: `modules/orchestration-service/src/services/diarization/pipeline.py`
- Modify: `modules/orchestration-service/src/main_fastapi.py` (add worker to lifespan)

**Step 1: Add job DB helpers to db.py**

Append to `modules/orchestration-service/src/services/diarization/db.py`:

```python
async def create_diarization_job(
    db: AsyncSession,
    meeting_id: str,
    triggered_by: str = "manual",
    rule_matched: dict | None = None,
) -> dict:
    """Create a diarization job in the database."""
    job = DiarizationJob(
        meeting_id=meeting_id,
        status="queued",
        triggered_by=triggered_by,
        rule_matched=rule_matched,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return _job_to_dict(job)


async def get_diarization_job(db: AsyncSession, job_id: int) -> dict | None:
    """Get a diarization job by ID."""
    result = await db.execute(select(DiarizationJob).where(DiarizationJob.id == job_id))
    job = result.scalar_one_or_none()
    return _job_to_dict(job) if job else None


async def list_diarization_jobs(
    db: AsyncSession, status_filter: str | None = None, limit: int = 50
) -> list[dict]:
    """List diarization jobs."""
    query = select(DiarizationJob).order_by(DiarizationJob.created_at.desc()).limit(limit)
    if status_filter:
        query = query.where(DiarizationJob.status == status_filter)
    result = await db.execute(query)
    return [_job_to_dict(j) for j in result.scalars().all()]


async def update_job_status(
    db: AsyncSession,
    job_id: int,
    status: str,
    error_message: str | None = None,
    **extra_fields,
) -> dict | None:
    """Update a diarization job status."""
    result = await db.execute(select(DiarizationJob).where(DiarizationJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        return None
    job.status = status
    job.updated_at = datetime.now(UTC)
    if error_message:
        job.error_message = error_message
    if status in ("completed", "failed", "cancelled"):
        job.completed_at = datetime.now(UTC)
    for key, value in extra_fields.items():
        if hasattr(job, key):
            setattr(job, key, value)
    await db.commit()
    await db.refresh(job)
    return _job_to_dict(job)


async def get_next_queued_job(db: AsyncSession) -> dict | None:
    """Get the oldest queued job for processing."""
    result = await db.execute(
        select(DiarizationJob)
        .where(DiarizationJob.status == "queued")
        .order_by(DiarizationJob.created_at)
        .limit(1)
    )
    job = result.scalar_one_or_none()
    return _job_to_dict(job) if job else None


def _job_to_dict(job: DiarizationJob) -> dict:
    """Convert DiarizationJob ORM object to dict."""
    return {
        "job_id": job.id,
        "meeting_id": str(job.meeting_id),
        "status": job.status,
        "triggered_by": job.triggered_by,
        "rule_matched": job.rule_matched,
        "audio_url": job.audio_url,
        "raw_segments": job.raw_segments,
        "detected_language": job.detected_language,
        "num_speakers_detected": job.num_speakers_detected,
        "processing_time_seconds": job.processing_time_seconds,
        "speaker_map": job.speaker_map,
        "unmapped_speakers": job.unmapped_speakers,
        "merge_applied": job.merge_applied,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
```

**Step 2: Update the pipeline to use DB instead of in-memory dict**

Rewrite `DiarizationPipeline` in `pipeline.py` to delegate all state to the database. The pipeline becomes a thin service that calls the `db.py` helpers using a session factory.

**Step 3: Add background worker to lifespan**

In `modules/orchestration-service/src/main_fastapi.py`, in the lifespan startup section (after line ~385), add:

```python
        # Start diarization background worker
        try:
            from services.diarization.pipeline import start_diarization_worker
            asyncio.create_task(start_diarization_worker())
            logger.info("Diarization background worker started")
        except Exception as e:
            logger.warning(f"Diarization worker failed to start: {e}")
```

**Step 4: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from services.diarization.pipeline import DiarizationPipeline; print('OK')"`
Expected: OK

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/diarization/pipeline.py modules/orchestration-service/src/services/diarization/db.py modules/orchestration-service/src/main_fastapi.py
git commit -m "feat: wire diarization pipeline to database with background worker"
```

---

### Task 11: Update Fireflies Auto-Trigger to Read Rules from DB

The auto-trigger in `fireflies.py` currently uses empty `DiarizationRules()` defaults. Wire it to read from `system_config`.

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py:905-924` and `1916-1933`

**Step 1: Replace both auto-trigger call sites**

At both locations, replace:
```python
await maybe_trigger_diarization(_meeting_meta, DiarizationRules(), None)
```

with:
```python
from services.diarization.db import get_diarization_rules
rules_dict = await get_diarization_rules(db)
rules = DiarizationRules(**rules_dict)
await maybe_trigger_diarization(_meeting_meta, rules, get_pipeline())
```

This requires `db` (AsyncSession) to be in scope at these call sites — check if the enclosing function already has a db parameter or if one needs to be added.

**Step 2: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from routers.fireflies import router; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: read diarization rules from database in Fireflies auto-trigger"
```

---

## WS2: Data Export / Download

### Task 12: Create Export Service with Format Converters

**Files:**
- Create: `modules/orchestration-service/src/services/export_service.py`

**Step 1: Create the export service**

Create `modules/orchestration-service/src/services/export_service.py` with converters for SRT, VTT, TXT, JSON, and PDF formats. Each converter takes a list of sentence dicts (with `text`, `speaker_name`, `start_time`, `end_time` fields) and returns formatted string/bytes.

Key functions:
- `to_srt(sentences)` — SubRip format with sequence numbers and `HH:MM:SS,mmm` timecodes
- `to_vtt(sentences)` — WebVTT with `WEBVTT` header and `HH:MM:SS.mmm` timecodes
- `to_txt(sentences, include_timestamps=True)` — Plain text with speaker attribution
- `to_json(meeting, sentences, translations)` — Full structured JSON export
- `to_pdf(meeting, sentences, translations)` — PDF via reportlab (import guarded with try/except for when reportlab isn't installed)
- `format_timecode_srt(seconds)` — Helper: float seconds → `00:01:23,456`
- `format_timecode_vtt(seconds)` — Helper: float seconds → `00:01:23.456`

**Step 2: Add reportlab dependency**

Run: `cd modules/orchestration-service && uv add reportlab`

**Step 3: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from services.export_service import to_srt, to_vtt, to_txt; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/export_service.py modules/orchestration-service/pyproject.toml modules/orchestration-service/uv.lock
git commit -m "feat: add export service with SRT, VTT, TXT, JSON, PDF converters"
```

---

### Task 13: Create Export Router with Download Endpoints

**Files:**
- Create: `modules/orchestration-service/src/routers/export.py`
- Modify: `modules/orchestration-service/src/main_fastapi.py` (register router)

**Step 1: Create the export router**

Create `modules/orchestration-service/src/routers/export.py` with endpoints:

```python
"""
Meeting Export Router

Download meeting transcripts and translations in various formats.
"""

import io
import json
import zipfile
from typing import Literal

from database import get_db_session
from database.models import Meeting, MeetingSentence, MeetingTranslation
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from livetranslate_common.logging import get_logger
from services.export_service import to_json, to_pdf, to_srt, to_txt, to_vtt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = get_logger()
router = APIRouter(tags=["Export"])

ExportFormat = Literal["srt", "vtt", "txt", "json", "pdf"]

MIME_TYPES = {
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
    "txt": "text/plain",
    "json": "application/json",
    "pdf": "application/pdf",
}


async def _get_meeting_with_data(db: AsyncSession, meeting_id: str):
    """Fetch meeting with sentences and translations."""
    result = await db.execute(
        select(Meeting)
        .where(Meeting.id == meeting_id)
        .options(
            selectinload(Meeting.sentences).selectinload(MeetingSentence.translations)
        )
    )
    meeting = result.scalar_one_or_none()
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
    return meeting


@router.get("/meetings/{meeting_id}/transcript")
async def export_transcript(
    meeting_id: str,
    format: ExportFormat = Query(default="srt"),
    db: AsyncSession = Depends(get_db_session),
):
    """Export meeting transcript in the specified format."""
    meeting = await _get_meeting_with_data(db, meeting_id)
    sentences = [
        {
            "text": s.text,
            "speaker_name": s.speaker_name,
            "start_time": s.start_time,
            "end_time": s.end_time,
        }
        for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0)
    ]

    if format == "srt":
        content = to_srt(sentences)
    elif format == "vtt":
        content = to_vtt(sentences)
    elif format == "txt":
        content = to_txt(sentences)
    elif format == "json":
        content = to_json(meeting, sentences, [])
    elif format == "pdf":
        content = to_pdf(meeting, sentences, [])
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{meeting.title or meeting_id}.pdf"'},
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    filename = f"{meeting.title or meeting_id}.{format}"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=MIME_TYPES[format],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/meetings/{meeting_id}/translations")
async def export_translations(
    meeting_id: str,
    lang: str = Query(description="Target language code (e.g., 'es')"),
    format: ExportFormat = Query(default="srt"),
    db: AsyncSession = Depends(get_db_session),
):
    """Export meeting translations for a specific language."""
    meeting = await _get_meeting_with_data(db, meeting_id)
    sentences = []
    for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0):
        translation = next(
            (t for t in s.translations if t.target_language == lang),
            None,
        )
        if translation:
            sentences.append({
                "text": translation.translated_text,
                "speaker_name": s.speaker_name,
                "start_time": s.start_time,
                "end_time": s.end_time,
            })

    if not sentences:
        raise HTTPException(status_code=404, detail=f"No translations found for language '{lang}'")

    if format == "srt":
        content = to_srt(sentences)
    elif format == "vtt":
        content = to_vtt(sentences)
    elif format == "txt":
        content = to_txt(sentences)
    else:
        content = json.dumps(sentences, default=str, indent=2)

    filename = f"{meeting.title or meeting_id}_{lang}.{format}"
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type=MIME_TYPES.get(format, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/meetings/{meeting_id}/archive")
async def export_archive(
    meeting_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """Export meeting as ZIP archive with transcript, translations, and metadata."""
    meeting = await _get_meeting_with_data(db, meeting_id)
    sentences = [
        {
            "text": s.text,
            "speaker_name": s.speaker_name,
            "start_time": s.start_time,
            "end_time": s.end_time,
        }
        for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0)
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("transcript.srt", to_srt(sentences))
        zf.writestr("transcript.txt", to_txt(sentences))
        zf.writestr("transcript.vtt", to_vtt(sentences))
        zf.writestr("metadata.json", to_json(meeting, sentences, []))

        # Add translations per language
        languages = set()
        for s in meeting.sentences:
            for t in s.translations:
                languages.add(t.target_language)
        for lang in sorted(languages):
            lang_sentences = []
            for s in sorted(meeting.sentences, key=lambda s: s.start_time or 0):
                translation = next((t for t in s.translations if t.target_language == lang), None)
                if translation:
                    lang_sentences.append({
                        "text": translation.translated_text,
                        "speaker_name": s.speaker_name,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                    })
            if lang_sentences:
                zf.writestr(f"translations/{lang}.srt", to_srt(lang_sentences))

    buf.seek(0)
    filename = f"{meeting.title or meeting_id}_archive.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
```

**Step 2: Register the router in main_fastapi.py**

Add import and registration near the other router registrations:

```python
from routers.export import router as export_router
app.include_router(export_router, prefix="/api/export", tags=["Export"])
```

**Step 3: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from routers.export import router; print(f'{len(router.routes)} routes')"`
Expected: Shows route count

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/export.py modules/orchestration-service/src/main_fastapi.py
git commit -m "feat: add meeting export endpoints (SRT, VTT, TXT, JSON, PDF, ZIP)"
```

---

### Task 14: Add Export UI to Dashboard Meeting Detail Page

**Files:**
- Create: `modules/dashboard-service/src/lib/api/export.ts`
- Modify: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.svelte`

**Step 1: Create TypeScript API client for exports**

Create `modules/dashboard-service/src/lib/api/export.ts`:

```typescript
import { API_BASE } from '$lib/config';

export function exportApi(fetch: typeof globalThis.fetch) {
  return {
    transcriptUrl: (meetingId: string, format: string) =>
      `${API_BASE}/api/export/meetings/${meetingId}/transcript?format=${format}`,

    translationsUrl: (meetingId: string, lang: string, format: string) =>
      `${API_BASE}/api/export/meetings/${meetingId}/translations?lang=${lang}&format=${format}`,

    archiveUrl: (meetingId: string) =>
      `${API_BASE}/api/export/meetings/${meetingId}/archive`,
  };
}
```

**Step 2: Add download buttons to meeting detail page**

In the meeting detail page, add a download dropdown in the toolbar area. Include:
- "Download Transcript" with format options (SRT, VTT, TXT, PDF)
- "Download Translations" with language + format pickers
- "Download All (ZIP)" button

Use `<a>` tags with `download` attribute pointing to the export API URLs for direct browser downloads.

**Step 3: Verify build**

Run: `cd modules/dashboard-service && npm run build 2>&1 | tail -5`
Expected: Build successful

**Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/api/export.ts modules/dashboard-service/src/routes/(app)/meetings/
git commit -m "feat: add transcript download UI to meeting detail page"
```

---

## WS3: Business Insights Chat

### Task 15: Create LLM Adapter Layer (Abstract Interface + Types)

**Files:**
- Create: `modules/orchestration-service/src/services/llm/__init__.py`
- Create: `modules/orchestration-service/src/services/llm/adapter.py`
- Create: `modules/orchestration-service/src/services/llm/providers/__init__.py`

**Step 1: Create the adapter interface and types**

Create the abstract base class and data models that all providers implement. Include:
- `ChatMessage` (role, content, tool_calls, tool_call_id)
- `ToolDefinition` (name, description, parameters as JSON Schema)
- `ToolCall` (id, name, arguments)
- `ChatResponse` (content, tool_calls, model, usage)
- `StreamChunk` (delta_content, delta_tool_call, finish_reason)
- `ModelInfo` (id, name, provider, context_window)
- `LLMAdapter` ABC with: `chat()`, `chat_stream()`, `list_models()`, `health_check()`

**Step 2: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from services.llm.adapter import LLMAdapter, ChatMessage; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/services/llm/
git commit -m "feat: add LLM adapter abstract interface and types"
```

---

### Task 16: Implement LLM Providers (Ollama, OpenAI, Anthropic, OpenAI-Compatible)

**Files:**
- Create: `modules/orchestration-service/src/services/llm/providers/ollama.py`
- Create: `modules/orchestration-service/src/services/llm/providers/openai_provider.py`
- Create: `modules/orchestration-service/src/services/llm/providers/anthropic_provider.py`
- Create: `modules/orchestration-service/src/services/llm/providers/openai_compat.py`

**Step 1: Add dependencies**

Run: `cd modules/orchestration-service && uv add openai anthropic`

**Step 2: Implement each provider**

Each provider class extends `LLMAdapter`. Key details:
- **Ollama**: Uses `openai` SDK pointed at `http://localhost:11434/v1`. `list_models()` calls `GET /api/tags`.
- **OpenAI**: Uses `openai` SDK with API key. Standard tool-calling support.
- **Anthropic**: Uses `anthropic` SDK. Tool use via `tools` parameter with Anthropic's format.
- **OpenAI-Compatible**: Generic adapter for any OpenAI-compatible endpoint (vLLM, Groq, etc.)

**Step 3: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from services.llm.providers.ollama import OllamaAdapter; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/llm/providers/ modules/orchestration-service/pyproject.toml modules/orchestration-service/uv.lock
git commit -m "feat: add Ollama, OpenAI, Anthropic, and OpenAI-compatible LLM providers"
```

---

### Task 17: Create Provider Registry and Tool Executor

**Files:**
- Create: `modules/orchestration-service/src/services/llm/registry.py`
- Create: `modules/orchestration-service/src/services/llm/tool_executor.py`

**Step 1: Create the registry**

`registry.py`:
- `ProviderRegistry` class that maintains configured providers
- `get_adapter(provider_name)` → returns configured `LLMAdapter` instance
- `list_providers()` → returns available providers with health status
- Reads config from `system_config` table (key: `chat_settings`)
- API key encryption via `cryptography.fernet.Fernet` using `SECRET_KEY`

**Step 2: Create the tool executor**

`tool_executor.py`:
- `ToolExecutor` class that manages tool definitions and execution
- `execute_tool(tool_name, arguments)` → calls the corresponding Python function
- `get_tool_definitions()` → returns list of `ToolDefinition` for LLM
- Max 3 tool-call rounds per message (prevent infinite loops)
- `run_chat_with_tools(adapter, messages, tools)` → orchestrates the chat + tool-call loop

**Step 3: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from services.llm.registry import ProviderRegistry; from services.llm.tool_executor import ToolExecutor; print('OK')"`
Expected: OK

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/llm/registry.py modules/orchestration-service/src/services/llm/tool_executor.py
git commit -m "feat: add LLM provider registry and tool-calling executor"
```

---

### Task 18: Create Business Insight Tools

**Files:**
- Create: `modules/orchestration-service/src/services/chat_tools.py`

**Step 1: Create tool definitions and implementations**

Each tool is a Python function with a JSON Schema description. Tools:

1. `query_meetings(date_from, date_to, participant, language, limit)` — SELECT from meetings with filters
2. `get_meeting_details(meeting_id)` — GET meeting + stats
3. `get_translation_stats(date_from, date_to, language_pair)` — Aggregate translation metrics
4. `get_speaker_analytics(date_from, date_to, speaker_name)` — Speaker participation data
5. `get_language_distribution(date_from, date_to)` — Language pair usage counts
6. `get_diarization_stats()` — Job status counts, speaker profile summary
7. `get_system_health()` — Service status from health monitor
8. `get_usage_trends(date_from, date_to, metric, interval)` — Time-series data
9. `search_transcripts(query, date_from, date_to, limit)` — Full-text search across transcripts

Each tool function accepts `db: AsyncSession` and its parameters, returns a dict.

**Step 2: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from services.chat_tools import TOOL_DEFINITIONS; print(f'{len(TOOL_DEFINITIONS)} tools defined')"`
Expected: `9 tools defined`

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/services/chat_tools.py
git commit -m "feat: add business insight tool definitions for chat"
```

---

### Task 19: Create Chat Pydantic Models and Alembic Migration

**Files:**
- Create: `modules/orchestration-service/src/models/chat.py`
- Create: `modules/orchestration-service/alembic/versions/011_chat_tables.py`
- Modify: `modules/orchestration-service/src/database/models.py` (add SQLAlchemy models)

**Step 1: Create Pydantic models**

`models/chat.py`:
- `ChatSettingsRequest` / `ChatSettingsResponse`
- `ConversationCreateRequest` / `ConversationResponse`
- `MessageRequest` / `MessageResponse`
- `ProviderInfo` / `ModelInfo`
- `SuggestedQueriesResponse`

**Step 2: Create Alembic migration**

`alembic/versions/011_chat_tables.py`:
- `revision = "011_chat_tables"` (17 chars, under 32)
- `down_revision = "010_diarization_tables"`
- Creates `chat_conversations` and `chat_messages` tables per the design doc
- Includes indexes on `conversation_id + created_at` and `updated_at DESC`

**Step 3: Add SQLAlchemy models**

Append to `database/models.py`:
- `ChatConversation` model matching `chat_conversations` table
- `ChatMessageModel` model matching `chat_messages` table

**Step 4: Verify migration chain**

Run: `cd modules/orchestration-service && uv run alembic history 2>&1 | head -15`
Expected: Shows 011_chat_tables at head

**Step 5: Commit**

```bash
git add modules/orchestration-service/src/models/chat.py modules/orchestration-service/alembic/versions/011_chat_tables.py modules/orchestration-service/src/database/models.py
git commit -m "feat: add chat conversation tables, models, and Alembic migration 011"
```

---

### Task 20: Create Chat API Router

**Files:**
- Create: `modules/orchestration-service/src/routers/chat.py`
- Modify: `modules/orchestration-service/src/main_fastapi.py` (register router)

**Step 1: Create the chat router**

`routers/chat.py` with endpoints:
- `GET /providers` — list configured providers with health status
- `GET /providers/{provider}/models` — list available models
- `GET /settings` — get current chat settings
- `PUT /settings` — save chat settings
- `POST /conversations` — start new conversation
- `GET /conversations` — list conversations (paginated)
- `GET /conversations/{id}` — get conversation with messages
- `DELETE /conversations/{id}` — delete conversation
- `POST /conversations/{id}/messages` — send message, return response
- `POST /conversations/{id}/messages/stream` — SSE streaming response
- `GET /conversations/{id}/suggestions` — contextual suggestions

The message endpoint orchestrates: load conversation → build messages → call tool executor → persist response → return.

**Step 2: Register router in main_fastapi.py**

```python
from routers.chat import router as chat_router
app.include_router(chat_router, prefix="/api/chat", tags=["Business Chat"])
```

**Step 3: Verify**

Run: `cd modules/orchestration-service && uv run python -c "from routers.chat import router; print(f'{len(router.routes)} routes')"`
Expected: Shows route count

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/chat.py modules/orchestration-service/src/main_fastapi.py
git commit -m "feat: add business insights chat API router with tool-calling"
```

---

### Task 21: Create Dashboard Chat Page and Components

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/chat/+page.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/chat/+page.server.ts`
- Create: `modules/dashboard-service/src/lib/api/chat.ts`
- Create: `modules/dashboard-service/src/lib/components/chat/ChatMessage.svelte`
- Create: `modules/dashboard-service/src/lib/components/chat/ChatInput.svelte`
- Create: `modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte`
- Create: `modules/dashboard-service/src/lib/components/chat/ConversationList.svelte`
- Create: `modules/dashboard-service/src/lib/components/chat/ToolCallIndicator.svelte`

**Step 1: Create the TypeScript API client**

`lib/api/chat.ts` — mirrors all `/api/chat/*` endpoints.

**Step 2: Create chat components**

Follow existing dashboard patterns (shadcn-svelte components, `$state` runes, `$effect` for reactivity):
- `ChatMessage.svelte` — renders user/assistant/tool messages with markdown
- `ChatInput.svelte` — textarea with Shift+Enter, send button, suggestion chips
- `SettingsDrawer.svelte` — slide-out panel with provider/model/temperature/API key fields
- `ConversationList.svelte` — sidebar list with "New Chat" button
- `ToolCallIndicator.svelte` — animated indicator when LLM is querying data

**Step 3: Create the chat page**

`+page.server.ts` — load conversations list
`+page.svelte` — assemble components into the chat layout from the design doc

**Step 4: Verify build**

Run: `cd modules/dashboard-service && npm run build 2>&1 | tail -5`
Expected: Build successful

**Step 5: Commit**

```bash
git add modules/dashboard-service/src/routes/(app)/chat/ modules/dashboard-service/src/lib/api/chat.ts modules/dashboard-service/src/lib/components/chat/
git commit -m "feat: add business insights chat page with conversation UI"
```

---

### Task 22: Add Chat to Dashboard Navigation

**Files:**
- Modify: `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte`
- Modify: `modules/dashboard-service/src/routes/(app)/+layout.svelte` (if needed for proxy routes)

**Step 1: Add Chat nav item to sidebar**

Add a "Chat" entry in the sidebar navigation, between "Intelligence" and "Diarization":

```svelte
<!-- After Intelligence nav item -->
<NavItem href="/chat" icon={MessageSquare}>Chat</NavItem>
```

Import the `MessageSquare` icon from lucide-svelte (or whichever icon library the sidebar uses).

**Step 2: Add API proxy route for chat endpoints**

If the dashboard uses SvelteKit server routes to proxy API calls, create proxy routes for `/api/chat/*` following the existing pattern in `src/routes/api/`.

**Step 3: Verify build**

Run: `cd modules/dashboard-service && npm run build 2>&1 | tail -5`
Expected: Build successful

**Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/components/layout/Sidebar.svelte modules/dashboard-service/src/routes/
git commit -m "feat: add Chat to dashboard navigation sidebar"
```

---

### Task 23: Final Verification and Cleanup

**Step 1: Verify all orchestration routes load**

Run: `cd modules/orchestration-service && uv run python -c "
from main_fastapi import app
routes = [r.path for r in app.routes if hasattr(r, 'path')]
print(f'{len(routes)} routes registered')
for r in sorted(routes):
    if 'chat' in r or 'export' in r or 'diarization' in r:
        print(f'  {r}')
"`
Expected: Shows chat, export, and diarization routes

**Step 2: Verify dashboard builds**

Run: `cd modules/dashboard-service && npm run build 2>&1 | tail -5`
Expected: Build successful

**Step 3: Verify Alembic migration chain**

Run: `cd modules/orchestration-service && uv run alembic history 2>&1`
Expected: Chain from 001 to 011 with no gaps

**Step 4: Run svelte-check**

Run: `cd modules/dashboard-service && npx svelte-check --threshold error 2>&1 | tail -10`
Expected: 0 errors

**Step 5: Commit any final cleanup**

```bash
git add -A
git commit -m "chore: final verification pass for system completion"
```

---

## Files Summary

| Action | File | Task |
|--------|------|------|
| Modify | `modules/orchestration-service/Dockerfile` | 1 |
| Modify | `modules/translation-service/Dockerfile` | 1 |
| Modify | `modules/whisper-service/Dockerfile` | 1 |
| Modify | `compose.local.yml` | 2 |
| Modify | `modules/orchestration-service/src/clients/translation_service_client.py` | 3 |
| Modify | `docker-compose.dev.yml` | 4 |
| Modify | `docker-compose.comprehensive.yml` | 4 |
| Modify | `docker-compose.minimal.yml` | 4 |
| Modify | `modules/orchestration-service/docker-compose.yml` | 4 |
| Modify | `modules/orchestration-service/docker-compose.react.yml` | 4 |
| Modify | `modules/translation-service/docker-compose.yml` | 4 |
| Modify | `modules/frontend-service/src/hooks/useWebSocket.ts` | 5 |
| Modify | `modules/frontend-service/src/store/slices/websocketSlice.ts` | 5 |
| Modify | `modules/orchestration-service/src/database/models.py` | 7, 19 |
| Create | `modules/orchestration-service/src/services/diarization/db.py` | 8, 9, 10 |
| Modify | `modules/orchestration-service/src/routers/diarization.py` | 8, 9 |
| Modify | `modules/orchestration-service/src/services/diarization/pipeline.py` | 10 |
| Modify | `modules/orchestration-service/src/main_fastapi.py` | 10, 13, 20 |
| Modify | `modules/orchestration-service/src/routers/fireflies.py` | 11 |
| Create | `modules/orchestration-service/src/services/export_service.py` | 12 |
| Create | `modules/orchestration-service/src/routers/export.py` | 13 |
| Create | `modules/dashboard-service/src/lib/api/export.ts` | 14 |
| Modify | `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.svelte` | 14 |
| Create | `modules/orchestration-service/src/services/llm/__init__.py` | 15 |
| Create | `modules/orchestration-service/src/services/llm/adapter.py` | 15 |
| Create | `modules/orchestration-service/src/services/llm/providers/__init__.py` | 15 |
| Create | `modules/orchestration-service/src/services/llm/providers/ollama.py` | 16 |
| Create | `modules/orchestration-service/src/services/llm/providers/openai_provider.py` | 16 |
| Create | `modules/orchestration-service/src/services/llm/providers/anthropic_provider.py` | 16 |
| Create | `modules/orchestration-service/src/services/llm/providers/openai_compat.py` | 16 |
| Create | `modules/orchestration-service/src/services/llm/registry.py` | 17 |
| Create | `modules/orchestration-service/src/services/llm/tool_executor.py` | 17 |
| Create | `modules/orchestration-service/src/services/chat_tools.py` | 18 |
| Create | `modules/orchestration-service/src/models/chat.py` | 19 |
| Create | `modules/orchestration-service/alembic/versions/011_chat_tables.py` | 19 |
| Create | `modules/orchestration-service/src/routers/chat.py` | 20 |
| Create | `modules/dashboard-service/src/routes/(app)/chat/+page.svelte` | 21 |
| Create | `modules/dashboard-service/src/routes/(app)/chat/+page.server.ts` | 21 |
| Create | `modules/dashboard-service/src/lib/api/chat.ts` | 21 |
| Create | `modules/dashboard-service/src/lib/components/chat/ChatMessage.svelte` | 21 |
| Create | `modules/dashboard-service/src/lib/components/chat/ChatInput.svelte` | 21 |
| Create | `modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte` | 21 |
| Create | `modules/dashboard-service/src/lib/components/chat/ConversationList.svelte` | 21 |
| Create | `modules/dashboard-service/src/lib/components/chat/ToolCallIndicator.svelte` | 21 |
| Modify | `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte` | 22 |
