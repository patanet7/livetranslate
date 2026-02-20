# Fireflies Real-Time Enhancement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 16x chunk duplication, add word-by-word captions, DB persistence, auto-connect, runtime translation config, and full Fireflies data capture.

**Architecture:** Layered Pipeline — three layers (dedup, display, persistence) added to existing pipeline. See `docs/plans/2026-02-20-fireflies-realtime-enhancement-design.md` for full design.

**Tech Stack:** Python/FastAPI, PostgreSQL, Socket.IO, Ollama (qwen2.5:3b), WebSocket, HTML/JS

**Services required:** Orchestration (3000), Translation (5003), PostgreSQL (5432), Redis (6379), Ollama (11434)

---

## Phase 1: Foundation

### Task 1: Database Schema Migration

**Files:**
- Create: `scripts/meeting-schema.sql`

**Step 1: Write the schema file**

Create `scripts/meeting-schema.sql` with the full schema from the design doc. Tables: `meetings`, `meeting_chunks`, `meeting_sentences`, `meeting_translations`, `meeting_insights`, `meeting_speakers`. Include all indexes and full-text search.

See design doc section "Layer 3: Persistence (MeetingStore) > Database Schema" for the complete SQL.

**Step 2: Apply the schema**

Run: `psql postgresql://postgres:postgres@localhost:5432/livetranslate -f scripts/meeting-schema.sql`

Expected: Tables created without errors. Verify with `\dt meeting*`.

**Step 3: Commit**

```bash
git add scripts/meeting-schema.sql
git commit -m "feat: add meeting persistence schema for Fireflies data"
```

---

### Task 2: Chunk Deduplication Layer

**Files:**
- Modify: `modules/orchestration-service/src/clients/fireflies_client.py`

This is the most critical fix. Currently 501 messages produce only 31 unique chunks (16x duplication).

**Step 1: Add `time` import and dedup constants**

At line 18, add `import time`. After the existing reconnection constants (line 49), add:

```python
# Chunk deduplication
# Fireflies sends ~16 interim updates per chunk_id. We buffer by chunk_id
# and only forward to the pipeline when a chunk is finalized (new chunk_id arrives).
CHUNK_FINALIZE_DELAY = 2.0  # seconds - safety timer for silence gaps
```

**Step 2: Add `LiveUpdateCallback` type**

After line 118 (`ErrorCallback`), add:

```python
# Callback for live interim updates (word-by-word captions)
LiveUpdateCallback = Callable[[FirefliesChunk, bool], Awaitable[None]]
# Second param is_final: True when chunk is being finalized
```

**Step 3: Add `on_live_update` to `__init__` signature and dedup state**

Modify `FirefliesRealtimeClient.__init__` (line 335):
- Add `on_live_update: LiveUpdateCallback | None = None` parameter
- Add `self.on_live_update = on_live_update` to callbacks section
- Replace existing dedup state (lines 393-395) with:

```python
# Chunk deduplication state
self._pending_chunks: dict[str, FirefliesChunk] = {}
self._pending_text: dict[str, str] = {}  # chunk_id -> last text (for pure-dup detection)
self._forwarded_chunks: set[str] = set()
self._raw_messages_received: int = 0
```

**Step 4: Rewrite `_handle_transcript` with dedup logic**

Replace the entire `_handle_transcript` method (lines 578-625) with:

```python
async def _handle_transcript(self, message: dict[str, Any]):
    """Handle transcription.broadcast event with deduplication."""
    try:
        self._raw_messages_received += 1

        # Extract chunk data - nested in 'payload' (production) or 'data' (legacy)
        chunk_data = message.get("payload", message.get("data", message)) if isinstance(message, dict) else message

        if not isinstance(chunk_data, dict):
            logger.warning(f"Unexpected transcript data format: {type(chunk_data)}")
            return

        chunk_id = str(chunk_data.get("chunk_id") or chunk_data.get("id") or uuid.uuid4().hex[:12])
        text = chunk_data.get("text", chunk_data.get("content", ""))

        # === DEDUP: Skip pure duplicates (same chunk_id, same text) ===
        if chunk_id in self._pending_text and self._pending_text[chunk_id] == text:
            return

        # Create FirefliesChunk model
        chunk = FirefliesChunk(
            transcript_id=chunk_data.get("transcript_id", self.transcript_id),
            chunk_id=chunk_id,
            text=text,
            speaker_name=chunk_data.get("speaker_name", chunk_data.get("speaker", "Unknown")),
            start_time=float(chunk_data.get("start_time", chunk_data.get("startTime", 0.0))),
            end_time=float(chunk_data.get("end_time", chunk_data.get("endTime", 0.0))),
        )

        is_new_chunk = chunk_id not in self._pending_text

        # === DEDUP: Update pending buffer ===
        self._pending_chunks[chunk_id] = chunk
        self._pending_text[chunk_id] = text

        # === Emit live update (word-by-word captions) ===
        if self.on_live_update:
            try:
                await self.on_live_update(chunk, False)  # is_final=False
            except Exception as e:
                logger.error(f"Error in live update callback: {e}")

        # === FINALIZE: When a new chunk_id arrives, finalize all other pending chunks ===
        if is_new_chunk:
            await self._finalize_other_chunks(chunk_id)

    except Exception as e:
        logger.error(f"Error processing transcript chunk: {e}")

async def _finalize_other_chunks(self, current_chunk_id: str):
    """Forward all pending chunks except the current one as final."""
    to_finalize = [
        cid for cid in self._pending_chunks
        if cid != current_chunk_id and cid not in self._forwarded_chunks
    ]

    for cid in to_finalize:
        chunk = self._pending_chunks.pop(cid)
        self._pending_text.pop(cid, None)
        self._forwarded_chunks.add(cid)

        logger.debug(f"Finalizing chunk {cid}: {chunk.text[:50]}")

        # Emit final live update
        if self.on_live_update:
            try:
                await self.on_live_update(chunk, True)  # is_final=True
            except Exception as e:
                logger.error(f"Error in live update callback: {e}")

        # Forward to pipeline
        if self.on_transcript:
            await self.on_transcript(chunk)

async def _flush_pending_chunks(self):
    """Flush all remaining pending chunks as final (called on disconnect)."""
    for cid in list(self._pending_chunks.keys()):
        if cid not in self._forwarded_chunks:
            chunk = self._pending_chunks.pop(cid)
            self._pending_text.pop(cid, None)
            self._forwarded_chunks.add(cid)

            if self.on_live_update:
                try:
                    await self.on_live_update(chunk, True)
                except Exception:
                    pass

            if self.on_transcript:
                await self.on_transcript(chunk)

    self._pending_chunks.clear()
    self._pending_text.clear()
```

**Step 5: Call flush on disconnect**

In the `disconnect` method (line 561), before the `_set_status` call, add:

```python
await self._flush_pending_chunks()
```

**Step 6: Remove temp debug code**

Delete lines 590-594 (the JSONL dump code):
```python
# DELETE THIS BLOCK:
# TEMP DEBUG: Dump raw message to file for diagnosis
import json as _json
_debug_file = "/tmp/fireflies_raw_chunks.jsonl"
with open(_debug_file, "a") as _f:
    _f.write(_json.dumps({"raw_message": message, "chunk_data": chunk_data}) + "\n")
```

**Step 7: Update `FirefliesClient` (unified client) to pass through `on_live_update`**

Find the `FirefliesClient` class (around line 633). In its `__init__` and `connect_realtime` method, add `on_live_update` parameter passthrough to `FirefliesRealtimeClient`.

**Step 8: Verify dedup logic manually**

Start orchestration, connect to a Fireflies meeting, check that:
- `chunks_received` (unique) is ~30 instead of ~500
- `raw_messages_received` shows the true count
- Captions still appear (via `on_transcript` callback)

**Step 9: Commit**

```bash
git add modules/orchestration-service/src/clients/fireflies_client.py
git commit -m "feat: add chunk deduplication layer to Fireflies client

Fireflies sends ~16 interim updates per chunk_id. Buffer by chunk_id
and only forward to pipeline when finalized (new chunk_id arrives).
Reduces pipeline load by ~16x. Also removes temp debug code."
```

---

### Task 3: PipelineConfig Additions

**Files:**
- Modify: `modules/orchestration-service/src/services/pipeline/config.py:14-84`

**Step 1: Add new config fields**

After line 82 (`intelligence_llm_backend`), add:

```python
# Display modes
display_mode: str = "both"  # "english", "translated", "both"
enable_interim_captions: bool = True  # Word-by-word interim display

# Persistence
enable_persistence: bool = True  # Save chunks/sentences/translations to DB

# Voice commands (experimental)
voice_commands_enabled: bool = False
voice_command_prefix: str = "LiveTranslate"
```

**Step 2: Commit**

```bash
git add modules/orchestration-service/src/services/pipeline/config.py
git commit -m "feat: add display mode, persistence, and voice command config fields"
```

---

## Phase 2: Live Display

### Task 4: Interim Captions in captions.html

**Files:**
- Modify: `modules/orchestration-service/static/captions.html`

**Step 1: Add interim caption handling to WebSocket message handler**

In `captions.html`, find the WebSocket `onmessage` handler. Add a new case for `interim_caption`:

```javascript
case 'interim_caption':
    handleInterimCaption(data);
    break;
case 'set_display_mode':
    currentDisplayMode = data.mode;
    applyDisplayMode();
    break;
```

**Step 2: Add the `handleInterimCaption` function**

```javascript
let currentDisplayMode = new URLSearchParams(window.location.search).get('mode') || 'both';

function handleInterimCaption(data) {
    if (currentDisplayMode === 'translated') return; // Skip interim in translated-only mode

    const chunkId = data.chunk_id;
    let el = document.getElementById('interim-' + chunkId);

    if (el) {
        // Update existing interim caption (grow in place)
        el.querySelector('.caption-text').textContent = data.text;
    } else {
        // Create new interim caption
        el = document.createElement('div');
        el.id = 'interim-' + chunkId;
        el.className = 'caption-box interim-caption';
        el.innerHTML = `
            <span class="speaker-name" style="color: ${data.speaker_color || '#4CAF50'}">${data.speaker_name || 'Speaker'}</span>
            <div class="caption-text">${data.text}</div>
        `;
        captionContainer.appendChild(el);
        enforceMaxCaptions();
    }
}

function applyDisplayMode() {
    document.querySelectorAll('.interim-caption').forEach(el => {
        el.style.display = (currentDisplayMode === 'translated') ? 'none' : '';
    });
    document.querySelectorAll('.final-caption').forEach(el => {
        el.style.display = (currentDisplayMode === 'english') ? 'none' : '';
    });
}
```

**Step 3: Add CSS for interim captions**

```css
.interim-caption {
    opacity: 0.8;
    border-left: 3px solid #FFC107; /* Yellow accent for interim */
}
.interim-caption .caption-text {
    font-style: italic;
    color: #e0e0e0;
}
.final-caption {
    border-left: 3px solid #4CAF50; /* Green accent for final */
}
```

**Step 4: Modify `addCaption` to mark as final and remove corresponding interim**

When a final `caption_added` event arrives, remove any interim captions that correspond to the same chunk_ids. Add `el.className = 'caption-box final-caption'` to the existing `addCaption` function.

**Step 5: Add mode query parameter support**

Add `mode` to the URL params parsing at the top of the script (alongside `session`, `lang`, etc.).

**Step 6: Commit**

```bash
git add modules/orchestration-service/static/captions.html
git commit -m "feat: add interim caption support with grow-in-place display"
```

---

### Task 5: Wire Interim Captions into Fireflies Router

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py:216-230`

**Step 1: Add live update callback in session setup**

In the `_connect_session` method (or wherever `handle_transcript` callback is defined, around line 216), add a new callback for live updates:

```python
async def handle_live_update(chunk: FirefliesChunk, is_final: bool):
    """Broadcast interim caption updates to WebSocket clients."""
    ws_manager = get_ws_manager()
    await ws_manager.broadcast_to_session(
        session_id,
        {
            "event": "interim_caption",
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "speaker_name": chunk.speaker_name,
            "speaker_color": None,  # CaptionBuffer assigns colors
            "is_final": is_final,
        },
    )
```

**Step 2: Pass `on_live_update` when creating the realtime client**

Find where `FirefliesRealtimeClient` or `FirefliesClient` is instantiated and add `on_live_update=handle_live_update`.

**Step 3: Test by connecting to a live meeting**

Verify:
- Open captions.html with `?session=<id>&mode=both`
- Text grows word-by-word in yellow-bordered interim captions
- Final translated captions appear in green-bordered boxes
- No more 16x duplicated text

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: wire interim caption broadcasts through WebSocket"
```

---

### Task 6: Display Mode Switching via Dashboard

**Files:**
- Modify: `modules/orchestration-service/static/fireflies-dashboard.html`
- Modify: `modules/orchestration-service/src/routers/fireflies.py`

**Step 1: Add mode toggle buttons to dashboard**

In the Live Feed section of the dashboard, add three toggle buttons:

```html
<div class="mode-toggle" style="margin: 10px 0;">
    <button onclick="setDisplayMode('english')" id="mode-english">English</button>
    <button onclick="setDisplayMode('both')" id="mode-both" class="active">Both</button>
    <button onclick="setDisplayMode('translated')" id="mode-translated">Translated</button>
</div>
```

**Step 2: Add JavaScript for mode switching**

```javascript
async function setDisplayMode(mode) {
    // Update dashboard UI
    document.querySelectorAll('.mode-toggle button').forEach(b => b.classList.remove('active'));
    document.getElementById('mode-' + mode).classList.add('active');

    // Send to backend
    const sessionId = currentSessionId;
    if (sessionId) {
        await fetch(`/fireflies/sessions/${sessionId}/display-mode`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({mode: mode})
        });
    }
}
```

**Step 3: Add display mode API endpoint**

In `fireflies.py`, add:

```python
@router.put("/sessions/{session_id}/display-mode")
async def set_display_mode(session_id: str, body: dict):
    mode = body.get("mode", "both")
    ws_manager = get_ws_manager()
    await ws_manager.broadcast_to_session(
        session_id,
        {"event": "set_display_mode", "mode": mode},
    )
    return {"success": True, "mode": mode}
```

**Step 4: Commit**

```bash
git add modules/orchestration-service/static/fireflies-dashboard.html modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: add display mode toggle (english/both/translated)"
```

---

## Phase 3: Persistence

### Task 7: MeetingStore Service

**Files:**
- Create: `modules/orchestration-service/src/services/meeting_store.py`

**Step 1: Create the MeetingStore class**

This service handles all PostgreSQL operations for meeting data. Uses `asyncpg` or the existing DB connection from the orchestration service.

Key methods:
```python
class MeetingStore:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.pool = None

    async def initialize(self):
        """Create connection pool."""

    async def create_meeting(self, fireflies_transcript_id: str, title: str | None = None, ...) -> str:
        """Create a new meeting record. Returns meeting UUID."""

    async def store_chunk(self, meeting_id: str, chunk: FirefliesChunk) -> None:
        """Store a deduplicated chunk. Uses UPSERT on (meeting_id, chunk_id)."""

    async def store_sentence(self, meeting_id: str, text: str, speaker: str, ...) -> str:
        """Store an aggregated sentence. Returns sentence UUID."""

    async def store_translation(self, sentence_id: str, translated_text: str, ...) -> None:
        """Store a translation for a sentence."""

    async def store_insight(self, meeting_id: str, insight_type: str, content: dict, source: str = "fireflies") -> None:
        """Store an AI insight (summary, action items, etc.)."""

    async def store_speaker(self, meeting_id: str, speaker_name: str, analytics: dict | None = None) -> None:
        """Upsert speaker metadata."""

    async def complete_meeting(self, meeting_id: str) -> None:
        """Mark meeting as completed."""

    async def get_meeting(self, meeting_id: str) -> dict | None:
        """Get meeting by ID with stats."""

    async def list_meetings(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List meetings with pagination."""

    async def search_meetings(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across chunks and sentences."""

    async def get_meeting_insights(self, meeting_id: str) -> list[dict]:
        """Get all insights for a meeting."""
```

Check the existing DB integration pattern in `modules/orchestration-service/src/database/bot_session_manager.py` for connection pooling approach. Follow the same pattern.

**Step 2: Commit**

```bash
git add modules/orchestration-service/src/services/meeting_store.py
git commit -m "feat: add MeetingStore service for meeting persistence"
```

---

### Task 8: Expanded GraphQL Queries

**Files:**
- Modify: `modules/orchestration-service/src/clients/fireflies_client.py:72-105`

**Step 1: Replace `TRANSCRIPT_DETAIL_QUERY` with full query**

Replace lines 89-105 with the expanded query from the design doc (section "Expanded Fireflies GraphQL Queries"). This adds: `summary` (all fields), `analytics` (sentiments, categories, speakers), `meeting_attendees`, `meeting_attendance`, `sentences` (with `ai_filters`), `transcript_url`, `audio_url`, `video_url`.

**Step 2: Add `download_full_transcript` method to `FirefliesGraphQLClient`**

After `get_transcript_detail` (line 309), add a method that:
1. Calls the expanded query
2. Parses each section (summary, analytics, sentences, etc.)
3. Returns a structured dict ready for `MeetingStore.store_insight`

```python
async def download_full_transcript(self, transcript_id: str) -> dict[str, Any] | None:
    """Download full transcript with all Fireflies AI data."""
    variables = {"id": transcript_id}
    try:
        data = await self.execute_query(TRANSCRIPT_FULL_QUERY, variables)
        transcript = data.get("transcript")
        if not transcript:
            return None

        # Structure insights by type
        insights = []
        if transcript.get("summary"):
            insights.append({"type": "summary", "content": transcript["summary"]})
        if transcript.get("analytics"):
            analytics = transcript["analytics"]
            if analytics.get("sentiments"):
                insights.append({"type": "sentiment", "content": analytics["sentiments"]})
            if analytics.get("speakers"):
                insights.append({"type": "speaker_analytics", "content": analytics["speakers"]})
            if analytics.get("categories"):
                insights.append({"type": "ai_filters", "content": analytics["categories"]})
        if transcript.get("meeting_attendees"):
            insights.append({"type": "attendance", "content": {
                "attendees": transcript["meeting_attendees"],
                "attendance": transcript.get("meeting_attendance", []),
            }})

        media = {}
        for key in ["transcript_url", "audio_url", "video_url"]:
            if transcript.get(key):
                media[key] = transcript[key]
        if media:
            insights.append({"type": "media", "content": media})

        return {
            "transcript": transcript,
            "insights": insights,
            "sentences": transcript.get("sentences", []),
        }
    except Exception as e:
        logger.error(f"Failed to download full transcript: {e}")
        raise
```

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/clients/fireflies_client.py
git commit -m "feat: expand Fireflies GraphQL queries for full data capture"
```

---

### Task 9: Wire Persistence into Fireflies Router

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py`

**Step 1: Initialize MeetingStore**

Add `MeetingStore` import and initialization in `FirefliesSessionManager`:

```python
from services.meeting_store import MeetingStore

# In __init__:
self._meeting_store = MeetingStore(os.environ.get("DATABASE_URL", ""))
```

**Step 2: Create meeting record on session connect**

In the session creation flow, after creating the session, create a meeting record:

```python
meeting_id = await self._meeting_store.create_meeting(
    fireflies_transcript_id=transcript_id,
    title=session.config.transcript_id if session.config else None,
    source="fireflies",
)
session.meeting_db_id = meeting_id  # Store for later use
```

Add `meeting_db_id: str | None = None` field to `FirefliesSession` model.

**Step 3: Store chunks on finalization**

In the `handle_transcript` callback, after pipeline processing, store the chunk:

```python
if session.meeting_db_id and settings.meeting_auto_save:
    await self._meeting_store.store_chunk(session.meeting_db_id, chunk)
```

**Step 4: Store sentences and translations via pipeline callbacks**

Register callbacks on the pipeline coordinator for sentence and translation events:

```python
async def on_sentence(unit):
    if session.meeting_db_id:
        sentence_id = await self._meeting_store.store_sentence(
            session.meeting_db_id, unit.text, unit.speaker_name,
            unit.start_time, unit.end_time, unit.boundary_type, unit.chunk_ids,
        )
        unit._db_sentence_id = sentence_id  # Attach for translation storage

async def on_translation(unit, result):
    if hasattr(unit, '_db_sentence_id') and unit._db_sentence_id:
        await self._meeting_store.store_translation(
            unit._db_sentence_id, result.translated,
            result.target_language, result.confidence,
            result.translation_time_ms, "ollama",
        )

coordinator.on_sentence_ready(on_sentence)
coordinator.on_translation_ready(on_translation)
```

**Step 5: Add env vars to `.env`**

```bash
# Persistence
MEETING_AUTO_SAVE=true
MEETING_DOWNLOAD_ON_COMPLETE=true
MEETING_DOWNLOAD_INSIGHTS=true
```

**Step 6: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py modules/orchestration-service/.env
git commit -m "feat: wire meeting persistence into Fireflies session lifecycle"
```

---

### Task 10: Post-Meeting Full Download

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py`

**Step 1: Add webhook endpoint**

```python
@router.post("/webhook")
async def fireflies_webhook(request: dict, background_tasks: BackgroundTasks):
    """Handle Fireflies post-meeting webhook."""
    event_type = request.get("eventType")
    meeting_id = request.get("meetingId")

    if event_type == "Transcription completed" and meeting_id:
        background_tasks.add_task(_download_meeting_data, meeting_id)
        return {"status": "accepted"}

    return {"status": "ignored"}
```

**Step 2: Add download background task**

```python
async def _download_meeting_data(fireflies_transcript_id: str):
    """Download full transcript and insights from Fireflies."""
    settings = get_settings()
    client = FirefliesClient(api_key=settings.fireflies_api_key)

    result = await client.download_full_transcript(fireflies_transcript_id)
    if not result:
        logger.error(f"Failed to download transcript {fireflies_transcript_id}")
        return

    store = MeetingStore(settings.database_url)
    await store.initialize()

    # Find or create meeting record
    meeting = await store.get_meeting_by_ff_id(fireflies_transcript_id)
    if not meeting:
        meeting_id = await store.create_meeting(
            fireflies_transcript_id=fireflies_transcript_id,
            title=result["transcript"].get("title"),
            source="fireflies",
            status="completed",
        )
    else:
        meeting_id = meeting["id"]
        await store.complete_meeting(meeting_id)

    # Store all insights
    for insight in result["insights"]:
        await store.store_insight(
            meeting_id, insight["type"], insight["content"], source="fireflies",
        )

    # Store sentences with ai_filters
    for sentence in result.get("sentences", []):
        sentence_id = await store.store_sentence(
            meeting_id, sentence.get("text", ""),
            sentence.get("speaker_name", "Unknown"),
            sentence.get("start_time", 0), sentence.get("end_time", 0),
            "fireflies_download",
        )
        if sentence.get("ai_filters"):
            await store.store_insight(
                meeting_id, "sentence_ai_filter",
                {"sentence_id": sentence_id, "filters": sentence["ai_filters"]},
                source="fireflies",
            )

    # Store speaker analytics
    for speaker_data in result.get("insights", []):
        if speaker_data.get("type") == "speaker_analytics":
            for speaker in speaker_data.get("content", []):
                await store.store_speaker(meeting_id, speaker.get("name", "Unknown"), speaker)

    logger.info(f"Successfully downloaded full transcript {fireflies_transcript_id}")
```

**Step 3: Add polling fallback for post-meeting download**

In the auto-connect polling loop (Task 13), when a meeting disappears from `active_meetings`, trigger the same download.

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: add post-meeting webhook and full Fireflies data download"
```

---

## Phase 4: Auto-Connect & Config

### Task 11: Auto-Connect on Startup

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py`

**Step 1: Add auto-connect startup task**

Add a startup event that polls active meetings:

```python
import asyncio

_auto_connect_task: asyncio.Task | None = None
_known_meeting_ids: set[str] = set()

async def _auto_connect_loop(manager: FirefliesSessionManager):
    """Poll for active meetings and auto-connect."""
    settings = get_settings()
    poll_interval = int(os.environ.get("FIREFLIES_POLL_INTERVAL", "30"))

    while True:
        try:
            client = FirefliesClient(api_key=settings.fireflies_api_key)
            meetings = await client.get_active_meetings()

            current_ids = {m.id for m in meetings}

            # New meetings: auto-connect
            for meeting in meetings:
                if meeting.id not in _known_meeting_ids:
                    logger.info(f"Auto-connecting to meeting: {meeting.title} ({meeting.id})")
                    try:
                        await manager.connect_session(
                            transcript_id=meeting.id,
                            api_key=settings.fireflies_api_key,
                            target_languages=[os.environ.get("DEFAULT_TARGET_LANGUAGE", "zh")],
                        )
                    except Exception as e:
                        logger.error(f"Failed to auto-connect to {meeting.id}: {e}")

            # Ended meetings: finalize and download
            ended_ids = _known_meeting_ids - current_ids
            for ended_id in ended_ids:
                logger.info(f"Meeting ended: {ended_id}, triggering download")
                if os.environ.get("MEETING_DOWNLOAD_ON_COMPLETE", "true").lower() == "true":
                    asyncio.create_task(_download_meeting_data(ended_id))

            _known_meeting_ids.clear()
            _known_meeting_ids.update(current_ids)

        except Exception as e:
            logger.error(f"Auto-connect poll error: {e}")

        await asyncio.sleep(poll_interval)
```

**Step 2: Register startup event**

```python
@router.on_event("startup")
async def start_auto_connect():
    global _auto_connect_task
    if os.environ.get("FIREFLIES_AUTO_CONNECT", "true").lower() == "true":
        api_key = os.environ.get("FIREFLIES_API_KEY")
        if api_key:
            manager = _get_session_manager()
            _auto_connect_task = asyncio.create_task(_auto_connect_loop(manager))
            logger.info("Fireflies auto-connect enabled")
```

**Step 3: Add env vars**

```bash
FIREFLIES_AUTO_CONNECT=true
FIREFLIES_POLL_INTERVAL=30
DEFAULT_TARGET_LANGUAGE=zh
```

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py modules/orchestration-service/.env
git commit -m "feat: add Fireflies auto-connect polling on service startup"
```

---

### Task 12: Paste Meeting Link (addToLiveMeeting)

**Files:**
- Modify: `modules/orchestration-service/src/clients/fireflies_client.py`
- Modify: `modules/orchestration-service/src/routers/fireflies.py`

**Step 1: Add `add_to_live_meeting` mutation to GraphQL client**

In `FirefliesGraphQLClient`, add:

```python
ADD_TO_LIVE_MEETING_MUTATION = """
mutation AddToLiveMeeting($meeting_link: String!, $title: String, $duration: Int) {
  addToLiveMeeting(meeting_link: $meeting_link, title: $title, duration: $duration) {
    success
    message
  }
}
"""

async def add_to_live_meeting(self, meeting_link: str, title: str | None = None, duration: int = 60) -> dict:
    variables = {"meeting_link": meeting_link, "title": title, "duration": duration}
    data = await self.execute_query(ADD_TO_LIVE_MEETING_MUTATION, variables)
    return data.get("addToLiveMeeting", {})
```

**Step 2: Add API endpoint**

```python
class InviteBotRequest(BaseModel):
    meeting_link: str = Field(description="Google Meet/Zoom URL")
    title: str | None = Field(default=None, description="Optional meeting title")
    duration: int = Field(default=60, description="Expected duration in minutes (15-120)")

@router.post("/invite-bot")
async def invite_fireflies_bot(request: InviteBotRequest, background_tasks: BackgroundTasks):
    """Invite Fireflies bot to a meeting and auto-connect when ready."""
    settings = get_settings()
    client = FirefliesClient(api_key=settings.fireflies_api_key)

    result = await client.add_to_live_meeting(
        meeting_link=request.meeting_link,
        title=request.title,
        duration=request.duration,
    )

    if result.get("success"):
        # Poll for the transcript_id to become available, then auto-connect
        background_tasks.add_task(_wait_and_connect, settings.fireflies_api_key, request.meeting_link)
        return {"success": True, "message": "Fireflies bot invited. Will auto-connect when ready."}

    raise HTTPException(status_code=400, detail=result.get("message", "Failed to invite bot"))
```

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/clients/fireflies_client.py modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: add invite-bot endpoint for paste-meeting-link flow"
```

---

### Task 13: Runtime Translation Config API

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py`

**Step 1: Add config endpoint**

```python
class TranslationConfigRequest(BaseModel):
    backend: str = Field(default="ollama", description="ollama, vllm, openai, groq")
    model: str = Field(default="qwen2.5:3b")
    base_url: str = Field(default="http://localhost:11434/v1")
    target_language: str = Field(default="zh")
    temperature: float = Field(default=0.3)
    max_tokens: int = Field(default=2048)

@router.put("/config/translation")
async def update_translation_config(config: TranslationConfigRequest):
    """Update translation backend configuration at runtime."""
    # Update in-memory config
    manager = _get_session_manager()
    manager.translation_config = config.model_dump()

    # Forward to translation service for hot-reload
    import httpx
    translation_url = os.environ.get("TRANSLATION_SERVICE_URL", "http://localhost:5003")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{translation_url}/api/config/update", json=config.model_dump())
            if resp.status_code == 200:
                return {"success": True, "config": config.model_dump()}
    except Exception as e:
        logger.warning(f"Could not update translation service config: {e}")

    # Still return success for orchestration-side config
    return {"success": True, "config": config.model_dump(), "warning": "Translation service not updated"}

@router.get("/config/translation")
async def get_translation_config():
    """Get current translation configuration."""
    manager = _get_session_manager()
    return {"config": getattr(manager, 'translation_config', {
        "backend": os.environ.get("OLLAMA_ENABLE", "true") == "true" and "ollama" or "none",
        "model": os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        "target_language": os.environ.get("DEFAULT_TARGET_LANGUAGE", "zh"),
    })}
```

**Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: add runtime translation backend config API"
```

---

## Phase 5: Dashboard UX

### Task 14: Dashboard Single-Click Connect & Meeting Link Input

**Files:**
- Modify: `modules/orchestration-service/static/fireflies-dashboard.html`

**Step 1: Add meeting link input**

In the Connect tab, add a "Paste Meeting Link" input above the existing transcript_id field:

```html
<div class="form-group">
    <label>Paste Meeting Link (Google Meet, Zoom, etc.)</label>
    <div style="display: flex; gap: 8px;">
        <input type="text" id="meetingLink" placeholder="https://meet.google.com/xxx-xxxx-xxx" style="flex:1">
        <button onclick="inviteBot()" class="btn btn-secondary">Invite Bot</button>
    </div>
    <small>Sends Fireflies bot to join the meeting. Auto-connects when ready.</small>
</div>
```

**Step 2: Add inviteBot function**

```javascript
async function inviteBot() {
    const link = document.getElementById('meetingLink').value.trim();
    if (!link) { alert('Please enter a meeting link'); return; }

    const resp = await fetch('/fireflies/invite-bot', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({meeting_link: link, title: document.getElementById('meetingTitle')?.value})
    });
    const data = await resp.json();
    if (data.success) {
        showNotification('Fireflies bot invited! Auto-connecting...', 'success');
    } else {
        showNotification('Failed: ' + (data.detail || data.message), 'error');
    }
}
```

**Step 3: Fix single-click connect**

Modify `connectToMeeting()` to save settings AND connect in one action. Remove the two-step flow.

**Step 4: Add inline caption preview**

Add a `<div id="captionPreview">` in the Live Feed section that shows captions inline (using the same WebSocket connection as the main feed).

**Step 5: Add translation config panel**

Add a settings section with:
- Backend dropdown (Ollama, vLLM, OpenAI, Groq)
- Model text input
- Target language selector
- Save button that calls `PUT /fireflies/config/translation`

**Step 6: Commit**

```bash
git add modules/orchestration-service/static/fireflies-dashboard.html
git commit -m "feat: dashboard UX overhaul - single-click connect, meeting link, caption preview, translation config"
```

---

## Phase 6: Meeting History & Upload

### Task 15: Meetings API Router

**Files:**
- Create: `modules/orchestration-service/src/routers/meetings.py`

**Step 1: Create meetings router**

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.meeting_store import MeetingStore

router = APIRouter(prefix="/meetings", tags=["meetings"])

@router.get("/")
async def list_meetings(limit: int = 50, offset: int = 0):
    """List all meetings with pagination."""

@router.get("/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Get meeting details with insights."""

@router.get("/{meeting_id}/insights")
async def get_meeting_insights(meeting_id: str):
    """Get all AI insights for a meeting."""

@router.get("/{meeting_id}/transcript")
async def get_meeting_transcript(meeting_id: str):
    """Get full transcript (sentences + translations)."""

@router.get("/search")
async def search_meetings(q: str, limit: int = 20):
    """Full-text search across all meetings."""

@router.post("/upload")
async def upload_transcript(file: UploadFile = File(...), title: str | None = None):
    """Upload a transcript file (JSON, TXT, SRT)."""

@router.post("/{meeting_id}/insights/generate")
async def generate_insights(meeting_id: str, insight_types: list[str] | None = None):
    """Generate Ollama insights for a meeting."""
```

**Step 2: Register router in main app**

Add to `main_fastapi.py`:
```python
from routers.meetings import router as meetings_router
app.include_router(meetings_router)
```

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/meetings.py modules/orchestration-service/src/main_fastapi.py
git commit -m "feat: add meetings API router for history, search, and upload"
```

---

### Task 16: Dashboard Meeting History Tab

**Files:**
- Modify: `modules/orchestration-service/static/fireflies-dashboard.html`

**Step 1: Add History tab content**

Populate the existing History tab with:
- Meeting list (cards with title, date, duration, speakers)
- Search bar (full-text search via `/meetings/search`)
- Click to expand: view transcript, translations, insights
- Download buttons for Fireflies data
- Upload area with drag-and-drop

**Step 2: Add upload functionality**

```javascript
async function uploadTranscript(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name);

    const resp = await fetch('/meetings/upload', {method: 'POST', body: formData});
    const data = await resp.json();
    if (data.success) {
        showNotification('Transcript uploaded', 'success');
        loadMeetingHistory();
    }
}
```

**Step 3: Add insight generation UI**

For each meeting, add a "Generate Insights" button that calls `POST /meetings/{id}/insights/generate` and displays results.

**Step 4: Commit**

```bash
git add modules/orchestration-service/static/fireflies-dashboard.html
git commit -m "feat: add meeting history tab with search, upload, and insight generation"
```

---

## Phase 7: Final Integration & Config

### Task 17: Update .env with All Config Vars

**Files:**
- Modify: `modules/orchestration-service/.env`

**Step 1: Add all new config vars**

```bash
# Fireflies Auto-Connect
FIREFLIES_AUTO_CONNECT=true
FIREFLIES_POLL_INTERVAL=30

# Display
DEFAULT_DISPLAY_MODE=both
DEFAULT_TARGET_LANGUAGE=zh

# Persistence
MEETING_AUTO_SAVE=true
MEETING_DOWNLOAD_ON_COMPLETE=true
MEETING_DOWNLOAD_INSIGHTS=true

# Voice Commands (experimental)
VOICE_COMMANDS_ENABLED=false
VOICE_COMMAND_PREFIX=LiveTranslate
```

**Step 2: Commit**

```bash
git add modules/orchestration-service/.env
git commit -m "chore: add all new config vars to .env"
```

---

### Task 18: End-to-End Verification

**No files to create.** This is a manual verification task.

**Step 1: Restart services**

```bash
# Kill existing
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:5003 | xargs kill -9 2>/dev/null

# Apply schema
psql postgresql://postgres:postgres@localhost:5432/livetranslate -f scripts/meeting-schema.sql

# Start translation service
cd modules/translation-service && uv run python src/api_server.py &

# Start orchestration service
cd modules/orchestration-service && uv run python src/main_fastapi.py &
```

**Step 2: Verify auto-connect**

- Start a Google Meet meeting with Fireflies bot present
- Check orchestration logs for "Auto-connecting to meeting"
- Open `http://localhost:3000/static/fireflies-dashboard.html`
- Verify session appears automatically

**Step 3: Verify word-by-word captions**

- Open captions page with `?mode=both`
- Speak in the meeting
- Verify: text grows word-by-word in yellow interim captions
- Verify: final green captions appear with Chinese translation
- Verify: NO duplication

**Step 4: Verify display mode switching**

- Click English/Both/Translated buttons in dashboard
- Verify captions page responds to mode changes

**Step 5: Verify persistence**

- Check `meetings` table has a record
- Check `meeting_chunks` has deduplicated chunks
- Check `meeting_sentences` has aggregated sentences
- Check `meeting_translations` has Chinese translations

**Step 6: Verify post-meeting download**

- End the meeting
- Check logs for "Meeting ended, triggering download"
- Check `meeting_insights` table for Fireflies AI data

**Step 7: Commit any fixes**

```bash
git commit -am "fix: integration fixes from end-to-end verification"
```

---

## Summary

| Phase | Tasks | Key Deliverable |
|-------|-------|----------------|
| 1: Foundation | 1-3 | DB schema, dedup layer, config |
| 2: Live Display | 4-6 | Word-by-word captions, mode toggle |
| 3: Persistence | 7-10 | MeetingStore, expanded queries, webhook |
| 4: Auto-Connect & Config | 11-13 | Auto-connect, invite-bot, runtime config |
| 5: Dashboard UX | 14 | Single-click, meeting link, caption preview |
| 6: History & Upload | 15-16 | Meetings API, history tab, upload |
| 7: Final Integration | 17-18 | Config vars, end-to-end verification |

**Total: 18 tasks across 7 phases.**
