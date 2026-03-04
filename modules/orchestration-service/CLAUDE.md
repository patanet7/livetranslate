# Orchestration Service - Backend API & Service Coordination

Work from plan.md in modules. Keep it up to date as you continue to work, update before and after each task. There should be enough context that at any point any engineer should be able to resume what you were doing.

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

## Fireflies API

- **Rate limits:** Free/Pro = 50 requests/day; Business/Enterprise = 60 requests/minute
- **Bulk query:** Use `transcripts` (plural) GraphQL query — fetches up to 50 transcripts per call. Avoids N+1 pattern.
- **Business-only fields on bulk:** `transcript_url`, `audio_url`, `video_url` on the `transcripts` (plural) query require Business tier. These fields work fine on the singular `transcript(id)` query. All other fields (sentences, summary, analytics, attendance, ai_filters) work on all plans.
- **Boot sync:** Runs once per day (checked via `system_config.last_boot_sync_at`). Beware: `uvicorn --reload` re-triggers lifespan on every file save during development.
- **Webhook payload:** Only sends `{meetingId, eventType, clientReferenceId}` — no transcript data. Must call API to get details.
