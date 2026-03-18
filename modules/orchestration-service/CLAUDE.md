# Orchestration Service

WebSocket hub, audio pipeline coordination, LLM translation, and Google Meet bot management.

## Commands

```bash
uv run pytest modules/orchestration-service/tests/ -v             # All tests
uv run pytest modules/orchestration-service/tests/ -v -m "not e2e"  # Skip E2E (no services needed)
uv run python modules/orchestration-service/src/main_fastapi.py   # Run service
```

## Key Components

### `SessionConfig` (src/routers/audio/websocket_audio.py)
Manages interpreter↔split mode transitions, language save/restore, `lock_language` propagation to transcription service. Methods: `enter_interpreter()`, `leave_interpreter()`, `set_source_language()`, `get_effective_target()`.

### Audio Pipeline (src/routers/audio/websocket_audio.py)
Main WebSocket handler: frontend connects → `start_session` → audio downsample → forward to transcription → segments → translation → frontend.

## Alembic Migrations

**Directory:** `alembic/versions/`
**Connection:** Uses `DATABASE_URL` env var (default in `.env`: `postgresql://postgres:postgres@localhost:5432/livetranslate`)
**Run command:** `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic upgrade head`

### Rules
- **Revision IDs MUST be ≤32 characters** — `alembic_version.version_num` is `varchar(32)`. IDs longer than 32 chars will fail with `StringDataRightTruncationError` at the tracking step (after DDL runs), causing a full transaction rollback.
- **Always run migrations through Alembic, never raw SQL** — If you apply DDL directly, Alembic won't track it. The chain breaks and future migrations fail with `KeyError` on the missing revision. If raw SQL was already applied, use `alembic stamp <revision>` to sync Alembic's state without re-running DDL.
- **`down_revision` must exactly match the parent's `revision` value** — not the filename, not a human-friendly label. Check with `grep -n "^revision\|^down_revision" alembic/versions/*.py` before creating new migrations.
- **Use `alembic history` and `alembic current`** to verify chain health after changes.

### Current Chain (10 migrations)
```
<base> → 001_initial → 5f3bcf8a26da → 002_session_id_nullable → 003_consolidate_glossaries
→ 004_meeting_intelligence → 005_fireflies_persistence → 006_meeting_sync_media
→ 007_system_config_unique_ff → 008_chat_msg_speaker_cols → 009_meeting_retry_cols (head)
```

### Naming Convention
Use format: `NNN_short_description` where NNN is zero-padded sequence number.
Keep the `revision` string short (≤32 chars). Examples:
- `006_meeting_sync_media` (22 chars) ✓
- `008_chat_msg_speaker_cols` (25 chars) ✓
- `008_add_chat_message_speaker_columns` (36 chars) ✗ TOO LONG

## Translation Module (`src/translation/`)

| File | Purpose |
|------|---------|
| `context_store.py` | `DirectionalContextStore` — per-`(source_lang, target_lang)` rolling context windows |
| `context.py` | `RollingContextWindow` — token-budgeted FIFO context window |
| `segment_record.py` | `SegmentRecord` dataclass + `SegmentPhase` enum (draft/final lifecycle) |
| `segment_store.py` | `SegmentStore` — per-session draft/final tracker, sentence accumulation, eviction |
| `config.py` | `TranslationConfig` pydantic-settings (`LLM_` prefix) |
| `service.py` | `TranslationService` — orchestrates LLM calls, draft/final routing |
| `llm_client.py` | HTTP client for Ollama OpenAI-compatible API (streaming + non-streaming) |

### Draft/Final Routing

- **Draft segments** (`is_draft=True`): non-streaming, no context write, provisional context read, drop-if-busy (`_draft_lock`), `LLM_DRAFT_TIMEOUT_S` wall-clock timeout.
- **Non-final finals** (`is_draft=False, is_final=False`): accumulated in `SegmentStore._pending_sentence`, no translation issued.
- **Sentence-boundary finals** (`is_draft=False, is_final=True`): flush `_pending_sentence`, streaming translation, context updated on completion.
- **Eviction**: `segment_store.evict_old(keep_last=50)` called after every draft/final received. Protected: any `segment_id` in `_pending_segment_ids`.

## Fireflies API

- **Rate limits:** Free/Pro = 50 requests/day; Business/Enterprise = 60 requests/minute
- **Bulk query:** Use `transcripts` (plural) GraphQL query — fetches up to 50 transcripts per call. Avoids N+1 pattern.
- **Business-only fields on bulk:** `transcript_url`, `audio_url`, `video_url` on the `transcripts` (plural) query require Business tier. These fields work fine on the singular `transcript(id)` query. All other fields (sentences, summary, analytics, attendance, ai_filters) work on all plans.
- **Boot sync:** Runs once per day (checked via `system_config.last_boot_sync_at`). Beware: `uvicorn --reload` re-triggers lifespan on every file save during development.
- **Webhook payload:** Only sends `{meetingId, eventType, clientReferenceId}` — no transcript data. Must call API to get details.
