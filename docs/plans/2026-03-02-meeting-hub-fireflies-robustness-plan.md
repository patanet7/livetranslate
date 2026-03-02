# Meeting Hub & Fireflies Robustness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a unified `/meetings` route group in the SvelteKit dashboard with full Fireflies artifact persistence, enhanced WebSocket robustness, rich meeting detail pages, and production polish.

**Architecture:** DB-first meeting hub — all meeting data flows through PostgreSQL via the existing `MeetingStore`. Frontend reads from `/meetings/*` API endpoints (already mostly exist). Live sessions use enhanced WebSocket with heartbeat and session resumption. Post-meeting webhook pulls all Fireflies artifacts.

**Tech Stack:** SvelteKit 5 (runes), shadcn-svelte (bits-ui), Tailwind CSS 4, FastAPI, asyncpg, PostgreSQL, Alembic

**Design Doc:** `docs/plans/2026-03-02-meeting-hub-fireflies-robustness-design.md`

---

## Phase 1: Backend Foundation

### Task 1: Alembic Migration — Add Sync Columns to Meetings Table

**Files:**
- Create: `modules/orchestration-service/alembic/versions/006_add_meeting_sync_and_media_columns.py`

**Step 1: Create the migration file**

```python
"""Add sync status and media URL columns to meetings table.

Revision ID: 006
Revises: 005
"""

from alembic import op
import sqlalchemy as sa

revision = "006_add_meeting_sync_and_media_columns"
down_revision = "005_add_fireflies_meeting_persistence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("meetings", sa.Column("audio_url", sa.Text(), nullable=True))
    op.add_column("meetings", sa.Column("video_url", sa.Text(), nullable=True))
    op.add_column("meetings", sa.Column("transcript_url", sa.Text(), nullable=True))
    op.add_column(
        "meetings",
        sa.Column("sync_status", sa.Text(), server_default="none", nullable=False),
    )
    op.add_column("meetings", sa.Column("sync_error", sa.Text(), nullable=True))
    op.add_column(
        "meetings",
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_meetings_sync_status", "meetings", ["sync_status"])


def downgrade() -> None:
    op.drop_index("ix_meetings_sync_status", table_name="meetings")
    op.drop_column("meetings", "synced_at")
    op.drop_column("meetings", "sync_error")
    op.drop_column("meetings", "sync_status")
    op.drop_column("meetings", "transcript_url")
    op.drop_column("meetings", "video_url")
    op.drop_column("meetings", "audio_url")
```

**Step 2: Verify migration chain**

Run: `cd modules/orchestration-service && uv run alembic heads`
Expected: Shows `006_add_meeting_sync_and_media_columns` as head

**Step 3: Update MeetingStore methods**

Modify: `modules/orchestration-service/src/services/meeting_store.py`

Add method `update_sync_status()` after `complete_meeting()` (after line 149):

```python
async def update_sync_status(
    self,
    meeting_id: str,
    sync_status: str,
    sync_error: str | None = None,
    audio_url: str | None = None,
    video_url: str | None = None,
    transcript_url: str | None = None,
) -> None:
    """Update meeting sync status and media URLs."""
    await self._ensure_pool()
    updates: list[str] = ["sync_status = $2"]
    params: list[Any] = [meeting_id, sync_status]
    idx = 2

    if sync_status == "synced":
        idx += 1
        updates.append(f"synced_at = ${idx}")
        params.append(datetime.now(UTC))

    if sync_error is not None:
        idx += 1
        updates.append(f"sync_error = ${idx}")
        params.append(sync_error)
    elif sync_status == "synced":
        updates.append("sync_error = NULL")

    for col, val in [
        ("audio_url", audio_url),
        ("video_url", video_url),
        ("transcript_url", transcript_url),
    ]:
        if val is not None:
            idx += 1
            updates.append(f"{col} = ${idx}")
            params.append(val)

    query = f"UPDATE meetings SET {', '.join(updates)} WHERE id = $1::uuid"
    await self._pool.execute(query, *params)
    logger.info("meeting_sync_updated", meeting_id=meeting_id, sync_status=sync_status)
```

**Step 4: Commit**

```bash
git add modules/orchestration-service/alembic/versions/006_add_meeting_sync_and_media_columns.py \
       modules/orchestration-service/src/services/meeting_store.py
git commit -m "feat: add sync status and media URL columns to meetings table"
```

---

### Task 2: Backend — Speakers Endpoint and Connect Response Enhancement

**Files:**
- Modify: `modules/orchestration-service/src/routers/meetings.py` (after line 102)
- Modify: `modules/orchestration-service/src/services/meeting_store.py` (add `get_meeting_speakers`)

**Step 1: Add `get_meeting_speakers` to MeetingStore**

Add after `get_meeting_insights()` (after line 402) in `meeting_store.py`:

```python
async def get_meeting_speakers(self, meeting_id: str) -> list[dict[str, Any]]:
    """Get all speakers for a meeting."""
    await self._ensure_pool()
    rows = await self._pool.fetch(
        """
        SELECT * FROM meeting_speakers
        WHERE meeting_id = $1::uuid
        ORDER BY talk_time_seconds DESC
        """,
        meeting_id,
    )
    return [dict(r) for r in rows]
```

**Step 2: Add speakers endpoint to meetings router**

Add after `get_meeting_insights` endpoint (after line 102) in `meetings.py`:

```python
@router.get("/{meeting_id}/speakers")
async def get_meeting_speakers(meeting_id: str) -> dict[str, Any]:
    """Get all speakers for a meeting with analytics."""
    store = await _get_store()
    meeting = await store.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    speakers = await store.get_meeting_speakers(meeting_id)
    return {"meeting_id": meeting_id, "speakers": speakers, "count": len(speakers)}
```

**Step 3: Add `meeting_id` to Fireflies connect response**

Modify: `modules/orchestration-service/src/routers/fireflies.py`

Find the `FirefliesConnectResponse` model and add `meeting_db_id`:

```python
# In the connect response model (look for FirefliesConnectResponse or the connect endpoint response)
# Add meeting_db_id to the response dict returned by the connect endpoint
```

Search for the connect endpoint's return statement — it returns a dict with `session_id`, `connection_status`, etc. Add `meeting_db_id` from the session object:

```python
# In the connect endpoint return, add:
"meeting_id": session.meeting_db_id,
```

The `FirefliesSession` model already has `meeting_db_id: str | None` so this just needs to be included in the response.

**Step 4: Commit**

```bash
git add modules/orchestration-service/src/routers/meetings.py \
       modules/orchestration-service/src/services/meeting_store.py \
       modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: add speakers endpoint and meeting_id in connect response"
```

---

### Task 3: Backend — Enhance Post-Meeting Data Download

**Files:**
- Modify: `modules/orchestration-service/src/routers/fireflies.py` — `_download_meeting_data` function

**Step 1: Find `_download_meeting_data` in fireflies.py**

This function is called by the webhook handler and the auto-connect poller when a meeting ends. It currently calls `client.download_full_transcript()` and stores some data. We need to ensure it stores ALL Fireflies artifacts.

Search for `_download_meeting_data` in the file. It should be around line 1470.

**Step 2: Enhance the function to persist all artifact types**

After the existing data download, add persistence for:
- `audio_url`, `video_url`, `transcript_url` → `store.update_sync_status()`
- Speaker analytics → `store.store_speaker()` for each speaker
- All insight types from the Fireflies `summary` object → `store.store_insight()` for each
- Set `sync_status = 'synced'` on success, `sync_status = 'failed'` on error

Wrap the download in sync status tracking:

```python
# At the start of _download_meeting_data:
if meeting_store and meeting_db_id:
    await meeting_store.update_sync_status(meeting_db_id, "syncing")

# After successful download and storage:
if meeting_store and meeting_db_id:
    await meeting_store.update_sync_status(
        meeting_db_id,
        "synced",
        audio_url=result.get("audio_url"),
        video_url=result.get("video_url"),
        transcript_url=result.get("transcript_url"),
    )

# On error:
if meeting_store and meeting_db_id:
    await meeting_store.update_sync_status(
        meeting_db_id, "failed", sync_error=str(e)
    )
```

Also store the Fireflies `summary` sub-fields as separate insights:

```python
summary = result.get("summary", {})
for insight_type in ["overview", "action_items", "outline", "keywords", "shorthand_bullet"]:
    content = summary.get(insight_type)
    if content:
        await meeting_store.store_insight(
            meeting_id=meeting_db_id,
            insight_type=insight_type,
            content={"text": content} if isinstance(content, str) else content,
            source="fireflies",
        )
```

And store speakers from Fireflies analytics:

```python
analytics = result.get("analytics", {})
for speaker_data in analytics.get("speakers", []):
    await meeting_store.store_speaker(
        meeting_id=meeting_db_id,
        speaker_name=speaker_data.get("name", "Unknown"),
        email=speaker_data.get("email"),
        talk_time_seconds=speaker_data.get("talk_time", 0),
        word_count=speaker_data.get("word_count", 0),
        sentiment_score=speaker_data.get("sentiment", {}).get("score"),
        analytics=speaker_data,
    )
```

**Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/fireflies.py
git commit -m "feat: enhance post-meeting download to persist all Fireflies artifacts"
```

---

## Phase 2: Frontend Foundation

### Task 4: TypeScript Types for Meetings

**Files:**
- Create: `modules/dashboard-service/src/lib/types/meeting.ts`
- Modify: `modules/dashboard-service/src/lib/types/index.ts` (line 4, add export)

**Step 1: Create meeting types**

```typescript
// modules/dashboard-service/src/lib/types/meeting.ts

export interface Meeting {
	id: string;
	fireflies_transcript_id: string | null;
	title: string | null;
	meeting_link: string | null;
	organizer_email: string | null;
	participants: string[];
	start_time: string | null;
	end_time: string | null;
	duration: number | null;
	source: 'fireflies' | 'upload';
	status: 'live' | 'completed' | 'error' | 'archived';
	sync_status: 'none' | 'live' | 'syncing' | 'synced' | 'failed';
	sync_error: string | null;
	synced_at: string | null;
	audio_url: string | null;
	video_url: string | null;
	transcript_url: string | null;
	created_at: string;
	updated_at: string;
	// Computed counts from backend JOIN
	chunk_count: number;
	sentence_count: number;
	translation_count?: number;
	insight_count?: number;
}

export interface MeetingListResponse {
	meetings: Meeting[];
	limit: number;
	offset: number;
}

export interface MeetingSearchResponse {
	results: Meeting[];
	query: string;
	count: number;
}

export interface MeetingSentence {
	id: string;
	meeting_id: string;
	text: string;
	speaker_name: string | null;
	start_time: number;
	end_time: number;
	boundary_type: string | null;
	chunk_ids: string[];
	created_at: string;
	translations: MeetingTranslation[];
}

export interface MeetingTranslation {
	translated_text: string;
	target_language: string;
	confidence: number;
	model_used: string | null;
}

export interface MeetingTranscriptResponse {
	meeting_id: string;
	sentences: MeetingSentence[];
	count: number;
	source?: 'chunks';
}

export interface MeetingInsight {
	id: string;
	meeting_id: string;
	insight_type: string;
	content: Record<string, unknown>;
	source: string;
	model_used: string | null;
	generated_at: string | null;
	created_at: string;
}

export interface MeetingInsightsResponse {
	meeting_id: string;
	insights: MeetingInsight[];
	count: number;
}

export interface MeetingSpeaker {
	id: string;
	meeting_id: string;
	speaker_name: string;
	email: string | null;
	talk_time_seconds: number;
	word_count: number;
	sentiment_score: number | null;
	analytics: Record<string, unknown> | null;
	created_at: string;
}

export interface MeetingSpeakersResponse {
	meeting_id: string;
	speakers: MeetingSpeaker[];
	count: number;
}

export interface InsightGenerateResponse {
	meeting_id: string;
	generated: Array<{ type: string; content: Record<string, unknown> }>;
	count: number;
}
```

**Step 2: Export from types index**

Modify `modules/dashboard-service/src/lib/types/index.ts` — add line:

```typescript
export * from './meeting';
```

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/lib/types/meeting.ts \
       modules/dashboard-service/src/lib/types/index.ts
git commit -m "feat(dashboard): add meeting TypeScript types"
```

---

### Task 5: Meetings API Client

**Files:**
- Create: `modules/dashboard-service/src/lib/api/meetings.ts`

**Step 1: Create the API client**

Pattern: Follow `$lib/api/fireflies.ts` exactly — accepts SvelteKit `fetch`, returns object of typed async methods.

```typescript
// modules/dashboard-service/src/lib/api/meetings.ts

import type {
	MeetingListResponse,
	MeetingSearchResponse,
	Meeting,
	MeetingTranscriptResponse,
	MeetingInsightsResponse,
	MeetingSpeakersResponse,
	InsightGenerateResponse
} from '$lib/types';
import { createApi } from './orchestration';

export function meetingsApi(fetch: typeof globalThis.fetch) {
	const api = createApi(fetch);

	return {
		list: (limit = 50, offset = 0) =>
			api.get<MeetingListResponse>(`/meetings/?limit=${limit}&offset=${offset}`),

		search: (q: string, limit = 20) =>
			api.get<MeetingSearchResponse>(
				`/meetings/search?q=${encodeURIComponent(q)}&limit=${limit}`
			),

		get: (id: string) => api.get<{ meeting: Meeting }>(`/meetings/${id}`),

		getTranscript: (id: string) =>
			api.get<MeetingTranscriptResponse>(`/meetings/${id}/transcript`),

		getInsights: (id: string) =>
			api.get<MeetingInsightsResponse>(`/meetings/${id}/insights`),

		getSpeakers: (id: string) =>
			api.get<MeetingSpeakersResponse>(`/meetings/${id}/speakers`),

		generateInsights: (id: string, types?: string[]) =>
			api.post<InsightGenerateResponse>(`/meetings/${id}/insights/generate`, {
				insight_types: types ?? ['summary', 'action_items', 'keywords']
			}),

		syncNow: (id: string) =>
			api.post<{ success: boolean }>(`/meetings/${id}/sync`)
	};
}
```

**Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/api/meetings.ts
git commit -m "feat(dashboard): add meetings API client"
```

---

### Task 6: Enhanced WebSocket Store

**Files:**
- Modify: `modules/dashboard-service/src/lib/stores/websocket.svelte.ts`

**Step 1: Read the current store**

Read: `modules/dashboard-service/src/lib/stores/websocket.svelte.ts`

Current: 62 lines, basic WebSocket with exponential backoff reconnect, 4 states.

**Step 2: Rewrite with heartbeat, buffering, reconnection ceiling, and resume**

Replace the entire file with:

```typescript
import { browser } from '$app/environment';

export type WsStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export class WebSocketStore {
	url = $state('');
	status = $state<WsStatus>('disconnected');
	reconnectAttempt = $state(0);
	lastCaptionId = $state<string | null>(null);

	#socket: WebSocket | null = null;
	#reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	#heartbeatTimer: ReturnType<typeof setInterval> | null = null;
	#pongTimer: ReturnType<typeof setTimeout> | null = null;
	#outboundBuffer: unknown[] = [];

	#maxReconnectDelay = 30_000;
	#maxReconnectAttempts = 10;
	#heartbeatIntervalMs = 15_000;
	#pongTimeoutMs = 5_000;

	onMessage: ((event: MessageEvent) => void) | null = null;
	onStatusChange: ((status: WsStatus) => void) | null = null;

	connect(url: string) {
		if (!browser) return;
		this.disconnect();
		this.url = url;
		this.#setStatus('connecting');
		this.#openSocket(url);
	}

	send(data: unknown) {
		if (this.#socket?.readyState === WebSocket.OPEN) {
			this.#socket.send(JSON.stringify(data));
		} else if (this.status === 'reconnecting') {
			this.#outboundBuffer.push(data);
		}
	}

	disconnect() {
		this.#clearTimers();
		if (this.#socket) {
			this.#socket.onclose = null;
			this.#socket.close(1000, 'Client disconnect');
			this.#socket = null;
		}
		this.#outboundBuffer = [];
		this.#setStatus('disconnected');
		this.reconnectAttempt = 0;
	}

	retry() {
		if (this.status === 'error' && this.url) {
			this.reconnectAttempt = 0;
			this.#setStatus('connecting');
			this.#openSocket(this.url);
		}
	}

	#openSocket(url: string) {
		try {
			this.#socket = new WebSocket(url);
		} catch {
			this.#setStatus('error');
			return;
		}

		this.#socket.onopen = () => {
			this.#setStatus('connected');

			// Send resume if reconnecting with a known last caption
			if (this.reconnectAttempt > 0 && this.lastCaptionId) {
				this.#socket?.send(
					JSON.stringify({ event: 'resume', last_caption_id: this.lastCaptionId })
				);
			}

			this.reconnectAttempt = 0;
			this.#flushBuffer();
			this.#startHeartbeat();
		};

		this.#socket.onmessage = (event) => {
			// Handle pong
			try {
				const data = JSON.parse(event.data);
				if (data.event === 'pong') {
					this.#clearPongTimer();
					return;
				}
				// Track last caption ID for resume
				if (data.caption?.id) {
					this.lastCaptionId = data.caption.id;
				} else if (data.caption_id) {
					this.lastCaptionId = data.caption_id;
				}
			} catch {
				// Non-JSON message, pass through
			}
			this.onMessage?.(event);
		};

		this.#socket.onclose = (event) => {
			this.#stopHeartbeat();
			if (!event.wasClean) {
				this.#scheduleReconnect();
			} else {
				this.#setStatus('disconnected');
			}
		};

		this.#socket.onerror = () => {
			// onerror is always followed by onclose, so we let onclose handle reconnection
		};
	}

	#startHeartbeat() {
		this.#stopHeartbeat();
		this.#heartbeatTimer = setInterval(() => {
			if (this.#socket?.readyState === WebSocket.OPEN) {
				this.#socket.send(JSON.stringify({ event: 'ping' }));
				this.#pongTimer = setTimeout(() => {
					// No pong received — force reconnect
					this.#socket?.close(4000, 'Heartbeat timeout');
				}, this.#pongTimeoutMs);
			}
		}, this.#heartbeatIntervalMs);
	}

	#stopHeartbeat() {
		if (this.#heartbeatTimer) {
			clearInterval(this.#heartbeatTimer);
			this.#heartbeatTimer = null;
		}
		this.#clearPongTimer();
	}

	#clearPongTimer() {
		if (this.#pongTimer) {
			clearTimeout(this.#pongTimer);
			this.#pongTimer = null;
		}
	}

	#flushBuffer() {
		for (const msg of this.#outboundBuffer) {
			this.send(msg);
		}
		this.#outboundBuffer = [];
	}

	#scheduleReconnect() {
		if (this.reconnectAttempt >= this.#maxReconnectAttempts) {
			this.#setStatus('error');
			return;
		}

		const delay = Math.min(1000 * 2 ** this.reconnectAttempt, this.#maxReconnectDelay);
		this.reconnectAttempt++;
		this.#setStatus('reconnecting');
		this.#reconnectTimer = setTimeout(() => this.#openSocket(this.url), delay);
	}

	#clearTimers() {
		if (this.#reconnectTimer) {
			clearTimeout(this.#reconnectTimer);
			this.#reconnectTimer = null;
		}
		this.#stopHeartbeat();
	}

	#setStatus(status: WsStatus) {
		this.status = status;
		this.onStatusChange?.(status);
	}
}

export const wsStore = new WebSocketStore();
```

**Key changes from current:**
- Added `reconnecting` status (5 states instead of 4)
- Heartbeat ping/pong every 15s with 5s timeout
- Outbound message buffer during reconnection
- Max 10 reconnect attempts → `error` state with manual `retry()`
- `lastCaptionId` tracking for session resumption
- `onStatusChange` callback for connection banner updates
- `resume` event sent on reconnect

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/lib/stores/websocket.svelte.ts
git commit -m "feat(dashboard): enhance WebSocket store with heartbeat, buffering, and resume"
```

---

### Task 7: Reusable Components — ErrorBoundary, SyncBadge, Skeleton

**Files:**
- Create: `modules/dashboard-service/src/lib/components/ErrorBoundary.svelte`
- Create: `modules/dashboard-service/src/lib/components/meetings/SyncBadge.svelte`
- Create: `modules/dashboard-service/src/lib/components/meetings/MeetingSkeleton.svelte`
- Create: `modules/dashboard-service/src/lib/components/meetings/ConnectionBanner.svelte`

**Step 1: ErrorBoundary**

```svelte
<!-- modules/dashboard-service/src/lib/components/ErrorBoundary.svelte -->
<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { toastStore } from '$lib/stores/toast.svelte';

	let { children } = $props();

	let error = $state<Error | null>(null);

	function handleError(e: Error) {
		error = e;
		toastStore.error(`Something went wrong: ${e.message}`);
	}

	function retry() {
		error = null;
	}
</script>

{#if error}
	<div class="flex flex-col items-center justify-center gap-4 py-12 text-center">
		<div class="text-4xl">⚠</div>
		<h3 class="text-lg font-semibold">Something went wrong</h3>
		<p class="text-sm text-muted-foreground max-w-md">{error.message}</p>
		<Button variant="outline" onclick={retry}>Try again</Button>
	</div>
{:else}
	<svelte:boundary onerror={handleError}>
		{@render children()}
	</svelte:boundary>
{/if}
```

**Step 2: SyncBadge**

```svelte
<!-- modules/dashboard-service/src/lib/components/meetings/SyncBadge.svelte -->
<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';

	interface Props {
		status: 'none' | 'live' | 'syncing' | 'synced' | 'failed';
		compact?: boolean;
	}

	let { status, compact = false }: Props = $props();

	const config = $derived(
		({
			none: { label: 'Not synced', variant: 'secondary' as const, class: '' },
			live: { label: 'Live', variant: 'default' as const, class: 'bg-green-600 animate-pulse' },
			syncing: { label: 'Syncing...', variant: 'default' as const, class: 'bg-yellow-600 animate-pulse' },
			synced: { label: 'Synced', variant: 'secondary' as const, class: '' },
			failed: { label: 'Sync failed', variant: 'destructive' as const, class: '' }
		})[status]
	);
</script>

<Badge variant={config.variant} class={config.class}>
	{#if status === 'live'}
		<span class="mr-1 inline-block size-2 rounded-full bg-green-300"></span>
	{/if}
	{#if status === 'synced'}
		<span class="mr-1">✓</span>
	{/if}
	{compact ? status : config.label}
</Badge>
```

**Step 3: MeetingSkeleton**

```svelte
<!-- modules/dashboard-service/src/lib/components/meetings/MeetingSkeleton.svelte -->
<script lang="ts">
	interface Props {
		rows?: number;
	}

	let { rows = 5 }: Props = $props();
</script>

<div class="space-y-3">
	{#each Array(rows) as _, i (i)}
		<div class="flex items-center gap-4 rounded-lg border p-4 animate-pulse">
			<div class="flex-1 space-y-2">
				<div class="h-4 rounded bg-muted" style="width: {60 + Math.random() * 30}%"></div>
				<div class="h-3 rounded bg-muted" style="width: {30 + Math.random() * 20}%"></div>
			</div>
			<div class="h-6 w-16 rounded-full bg-muted"></div>
		</div>
	{/each}
</div>
```

**Step 4: ConnectionBanner**

```svelte
<!-- modules/dashboard-service/src/lib/components/meetings/ConnectionBanner.svelte -->
<script lang="ts">
	import type { WsStatus } from '$lib/stores/websocket.svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		status: WsStatus;
		reconnectAttempt: number;
		maxAttempts?: number;
		onretry?: () => void;
	}

	let { status, reconnectAttempt, maxAttempts = 10, onretry }: Props = $props();
</script>

{#if status === 'connected'}
	<div class="flex items-center gap-2 rounded-md bg-green-500/10 px-3 py-1.5 text-sm text-green-700 dark:text-green-400">
		<span class="inline-block size-2 rounded-full bg-green-500"></span>
		Connected
	</div>
{:else if status === 'connecting'}
	<div class="flex items-center gap-2 rounded-md bg-yellow-500/10 px-3 py-1.5 text-sm text-yellow-700 dark:text-yellow-400 animate-pulse">
		<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
		Connecting...
	</div>
{:else if status === 'reconnecting'}
	<div class="flex items-center gap-2 rounded-md bg-yellow-500/10 px-3 py-1.5 text-sm text-yellow-700 dark:text-yellow-400 animate-pulse">
		<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
		Reconnecting... (attempt {reconnectAttempt}/{maxAttempts})
	</div>
{:else if status === 'error'}
	<div class="flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-1.5 text-sm text-destructive">
		<span class="inline-block size-2 rounded-full bg-destructive"></span>
		Connection lost
		{#if onretry}
			<Button variant="outline" size="sm" class="ml-2 h-6 text-xs" onclick={onretry}>Retry</Button>
		{/if}
	</div>
{:else}
	<div class="flex items-center gap-2 rounded-md bg-muted px-3 py-1.5 text-sm text-muted-foreground">
		<span class="inline-block size-2 rounded-full bg-muted-foreground"></span>
		Disconnected
	</div>
{/if}
```

**Step 5: Commit**

```bash
git add modules/dashboard-service/src/lib/components/ErrorBoundary.svelte \
       modules/dashboard-service/src/lib/components/meetings/
git commit -m "feat(dashboard): add ErrorBoundary, SyncBadge, MeetingSkeleton, ConnectionBanner"
```

---

## Phase 3: Meeting List Page

### Task 8: Meeting List — Route, Server Load, and Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/meetings/+page.server.ts`
- Create: `modules/dashboard-service/src/routes/(app)/meetings/+page.svelte`

**Step 1: Server load function**

```typescript
// modules/dashboard-service/src/routes/(app)/meetings/+page.server.ts

import { meetingsApi } from '$lib/api/meetings';

export async function load({ fetch, url }) {
	const api = meetingsApi(fetch);
	const q = url.searchParams.get('q');
	const limit = Number(url.searchParams.get('limit')) || 50;
	const offset = Number(url.searchParams.get('offset')) || 0;

	try {
		if (q) {
			const result = await api.search(q, limit);
			return { meetings: result.results, query: q, total: result.count, limit, offset };
		}
		const result = await api.list(limit, offset);
		return { meetings: result.meetings, query: null, total: result.meetings.length, limit, offset };
	} catch {
		return { meetings: [], query: q, total: 0, limit, offset };
	}
}
```

**Step 2: Meeting list page**

```svelte
<!-- modules/dashboard-service/src/routes/(app)/meetings/+page.svelte -->
<script lang="ts">
	import { goto } from '$app/navigation';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Badge } from '$lib/components/ui/badge';
	import SyncBadge from '$lib/components/meetings/SyncBadge.svelte';
	import MeetingSkeleton from '$lib/components/meetings/MeetingSkeleton.svelte';
	import type { Meeting } from '$lib/types';

	let { data } = $props();

	let searchQuery = $state(data.query ?? '');
	let statusFilter = $state<'all' | 'live' | 'completed'>('all');

	const filtered = $derived(
		statusFilter === 'all'
			? data.meetings
			: data.meetings.filter((m: Meeting) => m.status === statusFilter)
	);

	function handleSearch(e: Event) {
		e.preventDefault();
		const params = new URLSearchParams();
		if (searchQuery.trim()) params.set('q', searchQuery.trim());
		goto(`/meetings?${params.toString()}`);
	}

	function formatDate(iso: string | null): string {
		if (!iso) return '--';
		return new Date(iso).toLocaleDateString(undefined, {
			month: 'short',
			day: 'numeric',
			year: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	function formatDuration(seconds: number | null): string {
		if (!seconds) return '--';
		const m = Math.floor(seconds / 60);
		const s = seconds % 60;
		return m > 0 ? `${m}m ${s}s` : `${s}s`;
	}
</script>

<PageHeader title="Meetings" description="Browse all meeting transcripts, translations, and insights" />

<!-- Search & Filters -->
<div class="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center">
	<form onsubmit={handleSearch} class="flex flex-1 gap-2">
		<Input
			placeholder="Search meetings..."
			bind:value={searchQuery}
			class="max-w-sm"
		/>
		<Button type="submit" variant="outline">Search</Button>
		{#if data.query}
			<Button variant="ghost" onclick={() => goto('/meetings')}>Clear</Button>
		{/if}
	</form>

	<div class="flex gap-1">
		{#each ['all', 'live', 'completed'] as filter}
			<Button
				variant={statusFilter === filter ? 'default' : 'outline'}
				size="sm"
				onclick={() => (statusFilter = filter as typeof statusFilter)}
			>
				{filter === 'all' ? 'All' : filter.charAt(0).toUpperCase() + filter.slice(1)}
			</Button>
		{/each}
	</div>
</div>

<!-- Search results info -->
{#if data.query}
	<p class="mb-4 text-sm text-muted-foreground">
		{data.total} result{data.total === 1 ? '' : 's'} for "{data.query}"
	</p>
{/if}

<!-- Meeting List -->
{#if filtered.length === 0}
	<Card.Root>
		<Card.Content class="py-16">
			<div class="text-center space-y-4">
				<div class="text-5xl">📋</div>
				<h3 class="text-lg font-semibold">No meetings yet</h3>
				<p class="text-muted-foreground max-w-md mx-auto">
					Connect to a Fireflies transcript to start capturing meeting data, or upload a transcript file.
				</p>
				<div class="flex gap-2 justify-center">
					<Button href="/fireflies">Connect to Fireflies</Button>
				</div>
			</div>
		</Card.Content>
	</Card.Root>
{:else}
	<div class="space-y-3">
		{#each filtered as meeting (meeting.id)}
			<a href="/meetings/{meeting.id}" class="block">
				<Card.Root class="transition-colors hover:bg-accent/50">
					<Card.Content class="flex items-center gap-4 py-4">
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 mb-1">
								<h3 class="font-medium truncate">
									{meeting.title ?? 'Untitled Meeting'}
								</h3>
								<SyncBadge status={meeting.sync_status ?? 'none'} compact />
								<Badge variant="outline" class="text-xs">{meeting.source}</Badge>
							</div>
							<div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
								<span>{formatDate(meeting.created_at)}</span>
								<span>{formatDuration(meeting.duration)}</span>
								<span>{meeting.sentence_count} sentences</span>
								{#if meeting.chunk_count}
									<span>{meeting.chunk_count} chunks</span>
								{/if}
							</div>
						</div>
						<Badge
							variant={meeting.status === 'live' ? 'default' : 'secondary'}
							class={meeting.status === 'live' ? 'bg-green-600 animate-pulse' : ''}
						>
							{meeting.status}
						</Badge>
					</Card.Content>
				</Card.Root>
			</a>
		{/each}
	</div>

	<!-- Pagination -->
	{#if data.total >= data.limit}
		<div class="mt-6 flex justify-center gap-2">
			{#if data.offset > 0}
				<Button
					variant="outline"
					href="/meetings?offset={Math.max(0, data.offset - data.limit)}"
				>
					Previous
				</Button>
			{/if}
			<Button variant="outline" href="/meetings?offset={data.offset + data.limit}">
				Next
			</Button>
		</div>
	{/if}
{/if}
```

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/meetings/
git commit -m "feat(dashboard): add meetings list page with search and filters"
```

---

## Phase 4: Meeting Detail Page

### Task 9: Meeting Detail — Layout and Server Load

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+layout.svelte`
- Create: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.server.ts`

**Step 1: Server load — parallel API calls**

```typescript
// modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.server.ts

import { error } from '@sveltejs/kit';
import { meetingsApi } from '$lib/api/meetings';

export async function load({ params, fetch }) {
	const api = meetingsApi(fetch);

	const meetingResult = await api.get(params.id).catch(() => null);
	if (!meetingResult?.meeting) {
		error(404, 'Meeting not found');
	}

	// Load transcript and insights in parallel (non-critical — don't block on failure)
	const [transcriptResult, insightsResult] = await Promise.all([
		api.getTranscript(params.id).catch(() => ({ meeting_id: params.id, sentences: [], count: 0 })),
		api.getInsights(params.id).catch(() => ({ meeting_id: params.id, insights: [], count: 0 }))
	]);

	return {
		meeting: meetingResult.meeting,
		transcript: transcriptResult,
		insights: insightsResult
	};
}
```

**Step 2: Layout with meeting header**

```svelte
<!-- modules/dashboard-service/src/routes/(app)/meetings/[id]/+layout.svelte -->
<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import SyncBadge from '$lib/components/meetings/SyncBadge.svelte';

	let { data, children } = $props();

	const meeting = $derived(data.meeting);

	function formatDate(iso: string | null): string {
		if (!iso) return '--';
		return new Date(iso).toLocaleDateString(undefined, {
			month: 'short', day: 'numeric', year: 'numeric',
			hour: '2-digit', minute: '2-digit'
		});
	}

	function formatDuration(seconds: number | null): string {
		if (!seconds) return '--';
		const m = Math.floor(seconds / 60);
		return m > 0 ? `${m} min` : `${seconds}s`;
	}
</script>

<!-- Meeting Header -->
<div class="mb-6">
	<div class="mb-2">
		<a href="/meetings" class="text-sm text-muted-foreground hover:text-foreground transition-colors">
			← Back to Meetings
		</a>
	</div>

	<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
		<div>
			<h1 class="text-2xl font-bold">
				{meeting.title ?? 'Untitled Meeting'}
			</h1>
			<div class="flex flex-wrap items-center gap-2 mt-1 text-sm text-muted-foreground">
				<Badge variant={meeting.status === 'live' ? 'default' : 'secondary'}
					class={meeting.status === 'live' ? 'bg-green-600 animate-pulse' : ''}>
					{meeting.status}
				</Badge>
				<SyncBadge status={meeting.sync_status ?? 'none'} />
				<span>{formatDate(meeting.created_at)}</span>
				{#if meeting.duration}
					<span>· {formatDuration(meeting.duration)}</span>
				{/if}
				{#if meeting.sentence_count}
					<span>· {meeting.sentence_count} sentences</span>
				{/if}
			</div>
		</div>

		<div class="flex gap-2">
			{#if meeting.status === 'live'}
				<Button href="/meetings/{meeting.id}/live" variant="default">
					View Live
				</Button>
			{/if}
		</div>
	</div>
</div>

{@render children()}
```

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/meetings/\[id\]/
git commit -m "feat(dashboard): add meeting detail layout and server load"
```

---

### Task 10: Meeting Detail — Tabbed Page with All 5 Tabs

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.svelte`

**Step 1: Create the tabbed detail page**

This is the largest single component. It uses the shadcn-svelte Tabs component (already available at `$lib/components/ui/tabs`).

```svelte
<!-- modules/dashboard-service/src/routes/(app)/meetings/[id]/+page.svelte -->
<script lang="ts">
	import { browser } from '$app/environment';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';
	import { Separator } from '$lib/components/ui/separator';
	import ErrorBoundary from '$lib/components/ErrorBoundary.svelte';
	import { toastStore } from '$lib/stores/toast.svelte';
	import { meetingsApi } from '$lib/api/meetings';
	import type { MeetingSpeaker, MeetingSentence, MeetingInsight } from '$lib/types';

	let { data } = $props();

	const meeting = $derived(data.meeting);
	const sentences: MeetingSentence[] = $derived(data.transcript?.sentences ?? []);
	const insights: MeetingInsight[] = $derived(data.insights?.insights ?? []);

	// --- Transcript tab state ---
	let transcriptSearch = $state('');
	let speakerFilter = $state<string | null>(null);

	const speakers = $derived(
		[...new Set(sentences.map((s) => s.speaker_name).filter(Boolean))] as string[]
	);

	const speakerColors: Record<string, string> = {};
	const palette = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336', '#00BCD4', '#795548', '#607D8B'];

	function getSpeakerColor(name: string): string {
		if (!speakerColors[name]) {
			speakerColors[name] = palette[Object.keys(speakerColors).length % palette.length];
		}
		return speakerColors[name];
	}

	const filteredSentences = $derived(
		sentences.filter((s) => {
			if (speakerFilter && s.speaker_name !== speakerFilter) return false;
			if (transcriptSearch) {
				const q = transcriptSearch.toLowerCase();
				return s.text.toLowerCase().includes(q) ||
					s.translations?.some((t) => t.translated_text.toLowerCase().includes(q));
			}
			return true;
		})
	);

	function formatTimestamp(seconds: number): string {
		if (seconds == null) return '--:--';
		const m = Math.floor(seconds / 60);
		const s = Math.floor(seconds % 60);
		return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
	}

	// --- Insights tab ---
	function getInsightsByType(type: string): MeetingInsight[] {
		return insights.filter((i) => i.insight_type === type);
	}

	function getInsightText(insight: MeetingInsight): string {
		const content = insight.content;
		if (typeof content === 'string') return content;
		if (content?.text && typeof content.text === 'string') return content.text;
		return JSON.stringify(content, null, 2);
	}

	let generating = $state(false);

	async function generateInsights() {
		if (!browser) return;
		generating = true;
		try {
			const api = meetingsApi(fetch);
			await api.generateInsights(meeting.id);
			toastStore.success('Insights generated. Refresh to view.');
		} catch (e) {
			toastStore.error('Failed to generate insights');
		} finally {
			generating = false;
		}
	}

	// --- Speakers tab (lazy loaded) ---
	let speakersData = $state<MeetingSpeaker[]>([]);
	let speakersLoaded = $state(false);
	let loadingSpeakers = $state(false);

	async function loadSpeakers() {
		if (speakersLoaded || !browser) return;
		loadingSpeakers = true;
		try {
			const api = meetingsApi(fetch);
			const result = await api.getSpeakers(meeting.id);
			speakersData = result.speakers;
			speakersLoaded = true;
		} catch {
			toastStore.error('Failed to load speaker data');
		} finally {
			loadingSpeakers = false;
		}
	}

	const totalTalkTime = $derived(
		speakersData.reduce((sum, s) => sum + s.talk_time_seconds, 0)
	);

	// --- Active tab ---
	let activeTab = $state('transcript');
</script>

<Tabs.Root bind:value={activeTab}>
	<Tabs.List>
		<Tabs.Trigger value="transcript">Transcript</Tabs.Trigger>
		<Tabs.Trigger value="translations">Translations</Tabs.Trigger>
		<Tabs.Trigger value="insights">Summary & Insights</Tabs.Trigger>
		<Tabs.Trigger value="speakers" onclick={loadSpeakers}>Speakers</Tabs.Trigger>
		<Tabs.Trigger value="media">Media & Links</Tabs.Trigger>
	</Tabs.List>

	<!-- Tab 1: Transcript -->
	<Tabs.Content value="transcript">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
							<Card.Title>Transcript ({filteredSentences.length})</Card.Title>
							<div class="flex gap-2">
								<Input
									placeholder="Search transcript..."
									bind:value={transcriptSearch}
									class="w-48"
								/>
								{#if speakerFilter}
									<Button variant="ghost" size="sm" onclick={() => (speakerFilter = null)}>
										Clear filter
									</Button>
								{/if}
							</div>
						</div>
						{#if speakers.length > 0}
							<div class="flex flex-wrap gap-1 mt-2">
								{#each speakers as speaker}
									<button
										class="inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors hover:opacity-80"
										style="background-color: {getSpeakerColor(speaker)}20; color: {getSpeakerColor(speaker)}; border: 1px solid {getSpeakerColor(speaker)}40"
										class:ring-2={speakerFilter === speaker}
										onclick={() => (speakerFilter = speakerFilter === speaker ? null : speaker)}
									>
										{speaker}
									</button>
								{/each}
							</div>
						{/if}
					</Card.Header>
					<Card.Content>
						{#if filteredSentences.length === 0}
							<div class="py-12 text-center">
								<p class="text-muted-foreground">
									{sentences.length === 0
										? 'Transcript is still being processed...'
										: 'No results match your search.'}
								</p>
							</div>
						{:else}
							<div class="max-h-[60vh] overflow-y-auto space-y-3">
								{#each filteredSentences as sentence (sentence.id)}
									<div class="text-sm space-y-1">
										<div class="flex items-center gap-2">
											{#if sentence.speaker_name}
												<button
													class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
													style="background-color: {getSpeakerColor(sentence.speaker_name)}20; color: {getSpeakerColor(sentence.speaker_name)}"
													onclick={() => (speakerFilter = sentence.speaker_name)}
												>
													{sentence.speaker_name}
												</button>
											{/if}
											<span class="text-xs text-muted-foreground">
												{formatTimestamp(sentence.start_time)}
												{#if sentence.end_time}
													- {formatTimestamp(sentence.end_time)}
												{/if}
											</span>
										</div>
										<p>{sentence.text}</p>
										{#if sentence.translations?.length}
											{#each sentence.translations as translation}
												<p class="text-primary/80 italic text-xs">
													{translation.translated_text}
													<span class="text-muted-foreground">
														({translation.target_language}
														{#if translation.confidence < 1}
															· {Math.round(translation.confidence * 100)}%
														{/if})
													</span>
												</p>
											{/each}
										{/if}
									</div>
									<Separator />
								{/each}
							</div>
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 2: Translations -->
	<Tabs.Content value="translations">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<Card.Title>Translations</Card.Title>
					</Card.Header>
					<Card.Content>
						{#if sentences.length === 0}
							<p class="py-8 text-center text-muted-foreground">No transcript data available.</p>
						{:else}
							<div class="max-h-[60vh] overflow-y-auto">
								<table class="w-full text-sm">
									<thead class="sticky top-0 bg-background border-b">
										<tr>
											<th class="text-left p-2 w-24">Speaker</th>
											<th class="text-left p-2">Original</th>
											<th class="text-left p-2">Translation</th>
											<th class="text-left p-2 w-20">Confidence</th>
										</tr>
									</thead>
									<tbody>
										{#each sentences as sentence (sentence.id)}
											<tr class="border-b">
												<td class="p-2 align-top">
													<Badge variant="outline">{sentence.speaker_name ?? 'Unknown'}</Badge>
												</td>
												<td class="p-2 align-top">{sentence.text}</td>
												<td class="p-2 align-top text-primary italic">
													{#if sentence.translations?.length}
														{sentence.translations[0].translated_text}
													{:else}
														<span class="text-muted-foreground">--</span>
													{/if}
												</td>
												<td class="p-2 align-top text-xs text-muted-foreground">
													{#if sentence.translations?.length}
														{Math.round(sentence.translations[0].confidence * 100)}%
													{:else}
														--
													{/if}
												</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 3: Summary & Insights -->
	<Tabs.Content value="insights">
		<ErrorBoundary>
			{#snippet children()}
				<div class="space-y-4">
					{#if insights.length === 0}
						<Card.Root>
							<Card.Content class="py-12">
								<div class="text-center space-y-4">
									<div class="text-4xl">💡</div>
									<h3 class="font-semibold">No insights yet</h3>
									<p class="text-muted-foreground text-sm">
										Generate AI insights from the meeting transcript.
									</p>
									<Button onclick={generateInsights} disabled={generating}>
										{generating ? 'Generating...' : 'Generate Insights'}
									</Button>
								</div>
							</Card.Content>
						</Card.Root>
					{:else}
						{#each ['summary', 'overview', 'action_items', 'keywords', 'decisions'] as insightType}
							{@const items = getInsightsByType(insightType)}
							{#if items.length > 0}
								<Card.Root>
									<Card.Header>
										<div class="flex items-center justify-between">
											<Card.Title class="capitalize">
												{insightType.replace(/_/g, ' ')}
											</Card.Title>
											<Badge variant="outline" class="text-xs">
												{items[0].source}
												{#if items[0].model_used}
													· {items[0].model_used}
												{/if}
											</Badge>
										</div>
									</Card.Header>
									<Card.Content>
										{#each items as insight (insight.id)}
											<div class="prose prose-sm dark:prose-invert max-w-none whitespace-pre-wrap">
												{getInsightText(insight)}
											</div>
										{/each}
									</Card.Content>
								</Card.Root>
							{/if}
						{/each}

						<div class="flex justify-center">
							<Button variant="outline" onclick={generateInsights} disabled={generating}>
								{generating ? 'Regenerating...' : 'Regenerate Insights'}
							</Button>
						</div>
					{/if}
				</div>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 4: Speakers -->
	<Tabs.Content value="speakers">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<Card.Title>Speaker Analytics</Card.Title>
					</Card.Header>
					<Card.Content>
						{#if loadingSpeakers}
							<div class="py-8 text-center text-muted-foreground animate-pulse">Loading speakers...</div>
						{:else if speakersData.length === 0}
							<div class="py-12 text-center">
								<p class="text-muted-foreground">Speaker data will appear after the meeting completes and syncs.</p>
							</div>
						{:else}
							<!-- Talk time bar chart -->
							{#if totalTalkTime > 0}
								<div class="mb-6 space-y-2">
									<h4 class="text-sm font-medium">Talk Time Distribution</h4>
									{#each speakersData as speaker (speaker.id)}
										{@const percent = Math.round((speaker.talk_time_seconds / totalTalkTime) * 100)}
										<div class="flex items-center gap-2">
											<span class="w-24 text-sm truncate">{speaker.speaker_name}</span>
											<div class="flex-1 h-5 bg-muted rounded-full overflow-hidden">
												<div
													class="h-full rounded-full transition-all"
													style="width: {percent}%; background-color: {getSpeakerColor(speaker.speaker_name)}"
												></div>
											</div>
											<span class="w-12 text-xs text-muted-foreground text-right">{percent}%</span>
										</div>
									{/each}
								</div>
								<Separator class="my-4" />
							{/if}

							<!-- Speaker cards -->
							<div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
								{#each speakersData as speaker (speaker.id)}
									<Card.Root>
										<Card.Content class="p-4">
											<div class="flex items-center gap-2 mb-2">
												<div
													class="size-3 rounded-full"
													style="background-color: {getSpeakerColor(speaker.speaker_name)}"
												></div>
												<span class="font-medium">{speaker.speaker_name}</span>
											</div>
											{#if speaker.email}
												<p class="text-xs text-muted-foreground mb-2">{speaker.email}</p>
											{/if}
											<div class="grid grid-cols-2 gap-2 text-sm">
												<div>
													<span class="text-muted-foreground">Talk time:</span>
													<span>{Math.round(speaker.talk_time_seconds / 60)}m</span>
												</div>
												<div>
													<span class="text-muted-foreground">Words:</span>
													<span>{speaker.word_count.toLocaleString()}</span>
												</div>
												{#if speaker.sentiment_score != null}
													<div class="col-span-2">
														<span class="text-muted-foreground">Sentiment:</span>
														<span>
															{speaker.sentiment_score > 0.3
																? '😊 Positive'
																: speaker.sentiment_score < -0.3
																	? '😟 Negative'
																	: '😐 Neutral'}
															({speaker.sentiment_score.toFixed(2)})
														</span>
													</div>
												{/if}
											</div>
										</Card.Content>
									</Card.Root>
								{/each}
							</div>
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>

	<!-- Tab 5: Media & Links -->
	<Tabs.Content value="media">
		<ErrorBoundary>
			{#snippet children()}
				<Card.Root>
					<Card.Header>
						<Card.Title>Media & Links</Card.Title>
					</Card.Header>
					<Card.Content class="space-y-4">
						{@const hasMedia = meeting.audio_url || meeting.video_url || meeting.transcript_url || meeting.meeting_link}

						{#if !hasMedia && !meeting.participants?.length && !meeting.organizer_email}
							<div class="py-12 text-center">
								<p class="text-muted-foreground">
									Media links will appear after the meeting syncs from Fireflies.
								</p>
							</div>
						{:else}
							{#if meeting.audio_url}
								<div class="flex items-center justify-between rounded-md border p-3">
									<div>
										<p class="text-sm font-medium">Audio Recording</p>
										<p class="text-xs text-muted-foreground truncate max-w-md">{meeting.audio_url}</p>
									</div>
									<Button variant="outline" size="sm" href={meeting.audio_url} target="_blank">
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.video_url}
								<div class="flex items-center justify-between rounded-md border p-3">
									<div>
										<p class="text-sm font-medium">Video Recording</p>
										<p class="text-xs text-muted-foreground truncate max-w-md">{meeting.video_url}</p>
									</div>
									<Button variant="outline" size="sm" href={meeting.video_url} target="_blank">
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.transcript_url}
								<div class="flex items-center justify-between rounded-md border p-3">
									<div>
										<p class="text-sm font-medium">Fireflies Transcript</p>
										<p class="text-xs text-muted-foreground truncate max-w-md">{meeting.transcript_url}</p>
									</div>
									<Button variant="outline" size="sm" href={meeting.transcript_url} target="_blank">
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.meeting_link}
								<div class="flex items-center justify-between rounded-md border p-3">
									<div>
										<p class="text-sm font-medium">Meeting Link</p>
										<p class="text-xs text-muted-foreground truncate max-w-md">{meeting.meeting_link}</p>
									</div>
									<Button variant="outline" size="sm" href={meeting.meeting_link} target="_blank">
										Open
									</Button>
								</div>
							{/if}

							{#if meeting.organizer_email}
								<Separator />
								<div>
									<p class="text-sm font-medium mb-1">Organizer</p>
									<p class="text-sm text-muted-foreground">{meeting.organizer_email}</p>
								</div>
							{/if}

							{#if meeting.participants?.length}
								<div>
									<p class="text-sm font-medium mb-2">Participants</p>
									<div class="flex flex-wrap gap-1">
										{#each meeting.participants as participant}
											<Badge variant="outline">{participant}</Badge>
										{/each}
									</div>
								</div>
							{/if}
						{/if}
					</Card.Content>
				</Card.Root>
			{/snippet}
		</ErrorBoundary>
	</Tabs.Content>
</Tabs.Root>
```

**Step 2: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/meetings/\[id\]/+page.svelte
git commit -m "feat(dashboard): add 5-tab meeting detail page"
```

---

## Phase 5: Live Session Page

### Task 11: Live Session Page (`/meetings/[id]/live`)

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/meetings/[id]/live/+page.server.ts`
- Create: `modules/dashboard-service/src/routes/(app)/meetings/[id]/live/+page.svelte`

**Step 1: Server load**

```typescript
// modules/dashboard-service/src/routes/(app)/meetings/[id]/live/+page.server.ts

import { error } from '@sveltejs/kit';
import { meetingsApi } from '$lib/api/meetings';
import { firefliesApi } from '$lib/api/fireflies';

export async function load({ params, fetch, url }) {
	const meetingApi = meetingsApi(fetch);
	const ffApi = firefliesApi(fetch);

	const meetingResult = await meetingApi.get(params.id).catch(() => null);
	if (!meetingResult?.meeting) {
		error(404, 'Meeting not found');
	}

	// Try to find a live session for this meeting by checking active sessions
	const sessionId = url.searchParams.get('session');
	let session = null;

	if (sessionId) {
		session = await ffApi.getSession(sessionId).catch(() => null);
	}

	return {
		meeting: meetingResult.meeting,
		session,
		sessionId
	};
}
```

**Step 2: Live session page**

Adapts the current `/fireflies/connect/+page.svelte` pattern but adds connection banner, sync indicator, and uses the enhanced WebSocket store.

```svelte
<!-- modules/dashboard-service/src/routes/(app)/meetings/[id]/live/+page.svelte -->
<script lang="ts">
	import { browser } from '$app/environment';
	import { goto } from '$app/navigation';
	import { page } from '$app/stores';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import CaptionStream from '$lib/components/captions/CaptionStream.svelte';
	import InterimCaption from '$lib/components/captions/InterimCaption.svelte';
	import ConnectionBanner from '$lib/components/meetings/ConnectionBanner.svelte';
	import { wsStore } from '$lib/stores/websocket.svelte';
	import { captionStore } from '$lib/stores/captions.svelte';
	import { toastStore } from '$lib/stores/toast.svelte';
	import { WS_BASE } from '$lib/config';
	import type { CaptionEvent } from '$lib/types';

	let { data } = $props();

	const meeting = $derived(data.meeting);
	const sessionId = $derived(data.sessionId ?? data.session?.session_id);

	let disconnecting = $state(false);

	// Connect WebSocket on mount
	$effect(() => {
		if (!browser || !sessionId) return;

		captionStore.start();
		const wsUrl = `${WS_BASE}/api/captions/stream/${sessionId}`;
		wsStore.onMessage = handleWsMessage;
		wsStore.connect(wsUrl);

		return () => {
			wsStore.disconnect();
			captionStore.stop();
		};
	});

	function handleWsMessage(event: MessageEvent) {
		try {
			const msg: CaptionEvent = JSON.parse(event.data);

			switch (msg.event) {
				case 'connected':
					if (msg.current_captions) {
						for (const c of msg.current_captions) {
							captionStore.addCaption({ ...c, receivedAt: Date.now() });
						}
					}
					break;
				case 'caption_added':
					captionStore.addCaption({ ...msg.caption, receivedAt: Date.now() });
					break;
				case 'caption_updated':
					captionStore.updateCaption(msg.caption);
					break;
				case 'caption_expired':
					captionStore.removeCaption(msg.caption_id);
					break;
				case 'interim_caption':
					captionStore.updateInterim(msg.caption?.text ?? '');
					break;
				case 'session_cleared':
					captionStore.clear();
					break;
			}
		} catch {
			// Ignore malformed messages
		}
	}

	async function handleDisconnect() {
		if (!sessionId) return;
		disconnecting = true;
		try {
			await fetch(`/api/fireflies/disconnect`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ session_id: sessionId })
			});
			wsStore.disconnect();
			toastStore.success('Session disconnected');
			goto(`/meetings/${meeting.id}`);
		} catch {
			toastStore.error('Failed to disconnect');
		} finally {
			disconnecting = false;
		}
	}
</script>

<!-- Connection Banner -->
<ConnectionBanner
	status={wsStore.status}
	reconnectAttempt={wsStore.reconnectAttempt}
	onretry={() => wsStore.retry()}
/>

<div class="mt-4 grid grid-cols-1 lg:grid-cols-4 gap-6">
	<!-- Live Captions -->
	<div class="lg:col-span-3">
		<Card.Root>
			<Card.Header>
				<div class="flex items-center justify-between">
					<Card.Title>Live Captions</Card.Title>
					<Badge variant="outline">
						{captionStore.captions.length} captions
					</Badge>
				</div>
			</Card.Header>
			<Card.Content>
				{#if captionStore.captions.length === 0 && !captionStore.interim}
					<div class="py-12 text-center text-muted-foreground">
						<p>Waiting for captions...</p>
						<p class="text-xs mt-1">Captions will appear here as the meeting progresses.</p>
					</div>
				{:else}
					<CaptionStream />
					{#if captionStore.interim}
						<InterimCaption />
					{/if}
				{/if}
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Session Info Sidebar -->
	<div class="space-y-4">
		<Card.Root>
			<Card.Header>
				<Card.Title>Session Info</Card.Title>
			</Card.Header>
			<Card.Content class="space-y-3 text-sm">
				{#if data.session}
					<div class="flex justify-between">
						<span class="text-muted-foreground">Status</span>
						<Badge variant={data.session.connection_status === 'CONNECTED' ? 'default' : 'secondary'}>
							{data.session.connection_status}
						</Badge>
					</div>
					<div class="flex justify-between">
						<span class="text-muted-foreground">Chunks</span>
						<span>{data.session.chunks_received}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-muted-foreground">Translations</span>
						<span>{data.session.translations_completed}</span>
					</div>
					{#if data.session.speakers_detected?.length}
						<div>
							<span class="text-muted-foreground">Speakers</span>
							<div class="flex flex-wrap gap-1 mt-1">
								{#each data.session.speakers_detected as speaker}
									<Badge variant="outline" class="text-xs">{speaker}</Badge>
								{/each}
							</div>
						</div>
					{/if}
				{:else}
					<p class="text-muted-foreground">No session data available</p>
				{/if}
			</Card.Content>
		</Card.Root>

		<Button
			variant="destructive"
			class="w-full"
			onclick={handleDisconnect}
			disabled={disconnecting}
		>
			{disconnecting ? 'Disconnecting...' : 'Disconnect'}
		</Button>
	</div>
</div>
```

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/meetings/\[id\]/live/
git commit -m "feat(dashboard): add live session page with enhanced WebSocket"
```

---

## Phase 6: Navigation & Route Cleanup

### Task 12: Update Sidebar Navigation

**Files:**
- Modify: `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte`

**Step 1: Add Meetings to the nav items array**

Find the `navItems` array in `Sidebar.svelte`. Add a new top-level item **before** the Fireflies entry:

```typescript
{
    label: 'Meetings',
    href: '/meetings',
    icon: CalendarDays  // or use a relevant Lucide icon
},
```

Also update the Fireflies children to remove the routes being replaced:
- Remove `history` child (replaced by `/meetings`)
- Remove `sessions` child (replaced by `/meetings`)
- Keep: Connect (main `/fireflies`), Live Feed, Glossary

**Step 2: Import the icon**

Add to the Lucide imports at the top of Sidebar.svelte:

```typescript
import { CalendarDays } from '@lucide/svelte';
```

Or use whichever icon naming convention the file already uses.

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/lib/components/layout/Sidebar.svelte
git commit -m "feat(dashboard): add Meetings to sidebar navigation"
```

---

### Task 13: Update Fireflies Connect Form to Redirect to Meeting Hub

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/fireflies/+page.server.ts`

**Step 1: Change the redirect target**

In the `actions.connect()` function, change the redirect from:
```typescript
redirect(303, `/fireflies/connect?session=${result.session_id}`);
```
To:
```typescript
redirect(303, `/meetings/${result.meeting_id}/live?session=${result.session_id}`);
```

This requires the connect response to include `meeting_id` (added in Task 2).

If `meeting_id` is null (e.g., DB not configured), fall back to the old redirect:
```typescript
if (result.meeting_id) {
    redirect(303, `/meetings/${result.meeting_id}/live?session=${result.session_id}`);
} else {
    redirect(303, `/fireflies/connect?session=${result.session_id}`);
}
```

**Step 2: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/fireflies/+page.server.ts
git commit -m "feat(dashboard): redirect Fireflies connect to meeting hub"
```

---

### Task 14: Remove Replaced Routes

**Files:**
- Delete: `modules/dashboard-service/src/routes/(app)/fireflies/history/` (entire directory)
- Delete: `modules/dashboard-service/src/routes/(app)/fireflies/sessions/` (entire directory)
- Keep: `modules/dashboard-service/src/routes/(app)/fireflies/connect/` (fallback for non-DB sessions)

**Step 1: Remove the directories**

```bash
rm -rf modules/dashboard-service/src/routes/\(app\)/fireflies/history
rm -rf modules/dashboard-service/src/routes/\(app\)/fireflies/sessions
```

**Step 2: Verify the build**

Run: `cd modules/dashboard-service && npm run build`
Expected: Build succeeds with no broken imports

**Step 3: Commit**

```bash
git add -u modules/dashboard-service/src/routes/\(app\)/fireflies/history/ \
            modules/dashboard-service/src/routes/\(app\)/fireflies/sessions/
git commit -m "refactor(dashboard): remove routes replaced by /meetings hub"
```

---

## Phase 7: Verification

### Task 15: Build Verification and Smoke Test

**Step 1: Build the dashboard**

```bash
cd modules/dashboard-service && npm run build
```

Expected: Clean build with no errors.

**Step 2: Run existing tests**

```bash
cd modules/dashboard-service && npm test
```

**Step 3: Backend tests**

```bash
uv run pytest modules/orchestration-service/tests/ -v --timeout=30 -x
```

**Step 4: Verify navigation**

Start services and manually verify:
1. `/meetings` — shows meeting list (or empty state if no DB data)
2. `/meetings/{id}` — shows 5-tab detail page
3. `/fireflies` — connect form still works, redirects to `/meetings/{id}/live`
4. Sidebar shows "Meetings" nav item
5. Old routes (`/fireflies/history`, `/fireflies/sessions`) return 404

**Step 5: Final commit if any fixes needed**

```bash
git add -A && git commit -m "fix(dashboard): post-build verification fixes"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1. Backend Foundation | 1-3 | Migration, speakers endpoint, enhanced download |
| 2. Frontend Foundation | 4-7 | Types, API client, WebSocket store, reusable components |
| 3. Meeting List | 8 | List page with search, filters, pagination |
| 4. Meeting Detail | 9-10 | Layout + 5-tab detail page |
| 5. Live Session | 11 | Enhanced live page with connection banner |
| 6. Navigation & Cleanup | 12-14 | Sidebar update, redirect, route removal |
| 7. Verification | 15 | Build, test, smoke test |

**Total: 15 tasks across 7 phases.**

Each task is one commit. Backend tasks can be done in parallel with frontend tasks (Phases 1 and 2 are independent). Frontend tasks within each phase are sequential.
