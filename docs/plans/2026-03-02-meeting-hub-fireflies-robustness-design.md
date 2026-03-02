# Meeting Hub & Fireflies Robustness Design

**Date**: 2026-03-02
**Status**: Approved
**Author**: Thomas Patane + Claude

## Overview

Make the Svelte dashboard frontend robust with Fireflies meeting integrations. Save and sync all meeting data and auto-generated artifacts from Fireflies. Covers reliability, completeness, and polish.

### Goals

1. **Unified Meeting Hub** — Single `/meetings` route group for browsing all meeting data (live and historical)
2. **Full artifact persistence** — Save everything Fireflies produces: transcript, summary, action items, keywords, sentiment, speaker analytics, audio/video URLs
3. **Dual-phase sync** — Real-time chunks during meeting + full Fireflies API pull post-meeting via webhook
4. **Live session robustness** — Heartbeat, message buffering, reconnection ceiling, session resumption
5. **Rich meeting detail page** — 5-tab view with transcript, translations, insights, speakers, media
6. **Production polish** — Skeleton loaders, empty states, error boundaries, responsive design

## Architecture: Meeting Hub

### Route Structure

```
(app)/
  meetings/
    +page.svelte              ← Meeting list (paginated, searchable)
    +page.server.ts           ← Load from GET /meetings/
    [id]/
      +page.svelte            ← Rich meeting detail (5 tabs)
      +page.server.ts         ← Parallel load: meeting + transcript + insights
      +layout.svelte          ← Shared meeting header (title, status, back nav)
      live/
        +page.svelte          ← Live session view (WebSocket captions)
        +page.server.ts       ← Load session data
```

### Route Migration

| Current Route | Fate |
|--------------|------|
| `/fireflies` | **Stays** — Connect form. On success, redirects to `/meetings/[id]/live` |
| `/fireflies/connect` | **Removed** — Replaced by `/meetings/[id]/live` |
| `/fireflies/history` | **Removed** — Replaced by `/meetings` |
| `/fireflies/sessions` | **Removed** — Replaced by `/meetings` (with "Live" status filter) |
| `/fireflies/live-feed` | **Stays** — Standalone caption display (OBS-style) |
| `/fireflies/glossary` | **Stays** — Standalone glossary management |

### Navigation

Sidebar gains a new **Meetings** top-level item:
- Meetings (list all)
- Fireflies (connect entry point)
- Glossary (term management)

## Meeting List Page (`/meetings`)

### Data Source

Uses existing backend endpoint `GET /meetings/` (in `meetings.py:62`). Returns paginated meetings from PostgreSQL.

### Features

- **Search bar**: Wired to `GET /meetings/search?q=...` (full-text search, already exists)
- **Meeting cards/rows**: Title, status badge (live/completed/syncing), date, duration, speaker count, chunk/sentence counts, source badge (fireflies/upload)
- **Filters**: Status (All/Live/Completed), Date range, Source
- **Empty state**: Illustration + CTA → `/fireflies`

### New API Client (`$lib/api/meetings.ts`)

```typescript
export function meetingsApi(fetch: typeof globalThis.fetch) {
  const api = createApi(fetch);
  return {
    list: (limit?: number, offset?: number) =>
      api.get('/meetings/', { limit, offset }),
    search: (q: string, limit?: number) =>
      api.get('/meetings/search', { q, limit }),
    get: (id: string) =>
      api.get(`/meetings/${id}`),
    getTranscript: (id: string) =>
      api.get(`/meetings/${id}/transcript`),
    getInsights: (id: string) =>
      api.get(`/meetings/${id}/insights`),
    getSpeakers: (id: string) =>
      api.get(`/meetings/${id}/speakers`),
    generateInsights: (id: string, types?: string[]) =>
      api.post(`/meetings/${id}/insights/generate`, { insight_types: types }),
  };
}
```

**Backend addition needed**: `GET /meetings/{id}/speakers` endpoint in `meetings.py`.

## Meeting Detail Page (`/meetings/[id]`)

### Layout

**Header** (`+layout.svelte`):
- Back arrow → `/meetings`
- Meeting title (editable inline)
- Status badge (live/completed/syncing)
- Duration | Speaker count | Date
- Actions: "Generate Insights" button, "Export" dropdown (JSON, TXT, SRT), "Sync Now" button

### Tab 1: Transcript

- Searchable, scrollable transcript with speaker-colored names
- Each entry: speaker badge (color-coded), timestamp, original text
- Translation shown below (if available) in accent color
- Click speaker badge to filter to that speaker
- Keyboard shortcut: `Ctrl+F` focuses transcript search

### Tab 2: Translations

- Side-by-side view: Original | Translation
- Language selector (if multiple target languages)
- "Translate All" button for post-meeting bulk translation
- Translation progress bar
- Confidence scores per translation

### Tab 3: Summary & Action Items

- AI summary (from `meeting_data_insights` where `insight_type = 'summary'`)
- Action items list (`insight_type = 'action_items'`)
- Keywords as tags/badges (`insight_type = 'keywords'`)
- Decisions list (`insight_type = 'decisions'`)
- "Generate" button if insights don't exist → `POST /meetings/{id}/insights/generate`
- Each section shows source (Fireflies AI vs Ollama) and model used

### Tab 4: Speaker Analytics

- Speaker cards: name, email, talk time, word count, sentiment score
- Talk time pie chart (CSS-based or lightweight chart lib)
- Sentiment indicators per speaker
- Data from `meeting_speakers` table

### Tab 5: Media & Links

- Audio URL, Video URL (from Fireflies)
- Meeting link
- Transcript URL (Fireflies original)
- Organizer and participant list

### Data Loading

```typescript
// +page.server.ts
export async function load({ params, fetch }) {
  const api = meetingsApi(fetch);
  const [meeting, transcript, insights] = await Promise.all([
    api.get(params.id),
    api.getTranscript(params.id),
    api.getInsights(params.id),
  ]);
  return { meeting, transcript, insights };
}
```

Speakers loaded lazily when tab selected.

## Live Session Robustness (`/meetings/[id]/live`)

### Enhanced WebSocket State Machine

```
disconnected → connecting → connected → reconnecting → disconnected
                    ↓                         ↓
                  error ←─────────────────── error
```

### Key Additions to WebSocketStore

1. **Heartbeat**: Ping every 15s, expect pong within 5s. No pong → immediate reconnect (don't wait for browser's 60s+ `onclose`)
2. **Message buffering**: Queue outbound messages during `reconnecting`. Flush on reconnect.
3. **Reconnection ceiling**: Exponential backoff up to 30s (keep current), max 10 attempts. After 10, transition to `error` with manual "Retry" button.
4. **Session resumption**: On reconnect, send `{ event: "resume", last_caption_id: "..." }`. Backend replays missed captions. (Requires small backend addition to caption stream handler.)

### Live Page Enhancements

**Connection banner** (persistent bar):
- Connected: green dot + "Connected" (subtle)
- Reconnecting: yellow pulse + "Reconnecting... (attempt 3/10)" + countdown
- Error: red + "Connection lost. [Retry]" button
- Disconnected: gray + "Session ended"

**Caption continuity**: Keep existing captions during reconnection. On resume, merge without duplicates using `caption_id`.

**Sync indicator**: Badge showing "Saving..." when chunks are being persisted, with total saved count.

### Flow

```
/fireflies → POST /fireflies/connect → { session_id, meeting_id }
  → redirect to /meetings/{meeting_id}/live
```

## Persistence & Sync Pipeline

### Phase 1: Real-Time (During Meeting)

Already implemented: `FirefliesSessionManager` stores chunks and sentences via `MeetingStore`.

**Verification needed**: Ensure `store_chunk()` and `store_sentence()` are called on every event. Meeting record created with `status = 'live'`.

### Phase 2: Post-Meeting (Webhook + Full Pull)

```
Meeting ends → Fireflies webhook → POST /fireflies/webhook
  → _download_meeting_data(transcript_id)
    → GraphQL TRANSCRIPT_FULL_QUERY
      → Store: summary, action_items, keywords, sentiment, speaker_analytics
      → Store: attendees, attendance, audio_url, video_url, transcript_url
      → Store: speakers with talk_time, word_count, sentiment
      → Update meeting: status = 'completed', sync_status = 'synced'
```

### Backend Changes Needed

**New columns on `meetings` table** (Alembic migration):

```sql
ALTER TABLE meetings ADD COLUMN audio_url TEXT;
ALTER TABLE meetings ADD COLUMN video_url TEXT;
ALTER TABLE meetings ADD COLUMN transcript_url TEXT;
ALTER TABLE meetings ADD COLUMN sync_status TEXT DEFAULT 'none';
ALTER TABLE meetings ADD COLUMN sync_error TEXT;
ALTER TABLE meetings ADD COLUMN synced_at TIMESTAMPTZ;
```

**Sync status values**: `none` | `live` | `syncing` | `synced` | `failed`

**New endpoint**: `GET /meetings/{id}/speakers` — returns speakers from `meeting_speakers` table.

**Enhanced `_download_meeting_data`**: Persist all Fireflies artifact types (attendees, media URLs, full speaker analytics).

### Frontend Sync Awareness

- Meeting list: sync status badges (live/syncing/synced/failed)
- Meeting detail: sync status in header + "Sync Now" button for manual re-trigger
- Sync failed: red badge + retry button

## Polish & Error Handling

### Loading States

Skeleton loaders (shadcn-svelte patterns) replacing all "Loading..." text:
- Meeting list: skeleton card rows (3-5 placeholder rows)
- Meeting detail: skeleton tabs with placeholder content
- Transcript: skeleton lines with varying widths
- Live captions: subtle pulse animation

### Empty States

Each page gets a meaningful empty state with illustration and CTA:
- `/meetings` (no meetings): "No meetings yet" + link to `/fireflies`
- Meeting transcript (no data): "Transcript is still being processed"
- Insights (none generated): "No insights yet" + "Generate Insights" button
- Speakers (none tracked): "Speaker data will appear after meeting completes"

### Error Boundaries

Reusable `ErrorBoundary.svelte`:
- Catches rendering errors
- Shows "Something went wrong" + "Try again" button
- Logs error to toast store
- Wraps each tab content in meeting detail

### Toast Enhancements

- **Action toasts**: "Sync failed. [Retry]" — clicking triggers sync
- **Persistent connection toasts**: WebSocket disconnect shows until reconnected
- **Stacking**: max 3 visible, queue the rest

### Form Validation

Fireflies connect form:
- Client-side transcript ID validation
- Inline field errors (partially implemented)
- Disabled submit while connecting (already done)

### Responsive Design

- Desktop (1024px+): Full layout with sidebar
- Tablet (768px-1024px): Collapsible sidebar, stacked tabs
- Mobile (< 768px): Bottom nav, single-column layout, swipeable tabs

## Summary of Backend Changes

| Change | File | Description |
|--------|------|-------------|
| New endpoint | `meetings.py` | `GET /meetings/{id}/speakers` |
| New columns | Alembic migration 006 | `audio_url`, `video_url`, `transcript_url`, `sync_status`, `sync_error`, `synced_at` on `meetings` |
| Enhance download | `fireflies.py` | `_download_meeting_data` persists all artifact types |
| Session resume | Caption stream handler | Handle `resume` event with `last_caption_id` for caption replay |
| Connect response | `fireflies.py` | Include `meeting_id` in connect response for redirect |

## Summary of Frontend Changes

| Change | Location | Description |
|--------|----------|-------------|
| New route group | `(app)/meetings/` | List, detail (5 tabs), live pages |
| New API client | `$lib/api/meetings.ts` | Full meetings API client |
| Enhanced store | `$lib/stores/websocket.svelte.ts` | Heartbeat, buffering, reconnection ceiling, resume |
| New store | `$lib/stores/meetings.svelte.ts` | Meeting list state, sync status |
| New components | `$lib/components/meetings/` | MeetingCard, TranscriptViewer, InsightsPanel, SpeakerCard, SyncBadge |
| Error boundary | `$lib/components/ErrorBoundary.svelte` | Reusable error wrapper |
| Skeleton loaders | Various | Replace "Loading..." text |
| Empty states | Various | Meaningful empty states with CTAs |
| Navigation | `Sidebar.svelte` | Add Meetings top-level item |
| Route removal | `/fireflies/connect`, `/fireflies/history`, `/fireflies/sessions` | Replaced by `/meetings/*` |
