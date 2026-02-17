# Demo Mode Redesign — Implementation Plan

## Task 1: Add pre-translated templates to demo_server.py

**File**: `src/services/demo_server.py`

Add a `TRANSLATED_TEMPLATES` dict mapping each English conversation template to Spanish. Add a `translated_text` field to `MockChunk`. Update `conversation_scenario()` to accept a `include_translations: bool` param — when True, populate `translated_text` on each chunk.

## Task 2: Add mode parameter to DemoManager

**File**: `src/services/demo_manager.py`

- Add `mode: str` field to `DemoManager` (values: `"passthrough"`, `"pretranslated"`)
- `start()` accepts `mode` parameter, stores it
- When `mode == "pretranslated"`, pass `include_translations=True` to `conversation_scenario()`
- Add `_pretranslated_task` field for the background caption injection task
- Add `_inject_pretranslated_captions()` async method that reads chunks from the scenario and directly calls `caption_buffer.add_caption()` on a timer
- `stop()` cancels the pretranslated task if running

## Task 3: Update demo start endpoint

**File**: `src/routers/fireflies.py`

- `start_demo()` accepts optional `mode` query param (default `"passthrough"`)
- Pass mode to `demo.start()`
- When `mode == "pretranslated"`, after session creation, start the caption injection background task using the session's CaptionBuffer
- Return `mode` in the response

## Task 4: Redesign dashboard demo UX — remove old panel, integrate with Live Feed

**File**: `static/fireflies-dashboard.html`

### Remove:
- The entire `<div id="demoPanel">` section (lines ~518-542)
- `showDemoPanel()`, `hideDemoPanel()`, `openDemoCaptions()` JS functions
- `.demo-panel`, `.demo-iframe-container` CSS

### Add to header:
- Mode dropdown next to Launch Demo button: `<select id="demoMode"><option value="passthrough">Live Passthrough</option><option value="pretranslated">Pre-translated (ES)</option></select>`

### Add to Live Feed tab:
- Demo status badge + Stop button (shown only when demo active)
- Inline at top of the Live Feed card

### Modify `startDemo()`:
1. Read selected mode from dropdown
2. POST `/fireflies/demo/start?mode={mode}`
3. On success: switch to Live Feed tab via `showTab('livefeed')`
4. Auto-populate `feedSessionSelect` with the demo session
5. Auto-call `connectToFeed()` to start WebSocket connection
6. Show demo badge in Live Feed tab

### Modify `stopDemo()`:
1. Disconnect feed WebSocket
2. POST `/fireflies/demo/stop`
3. Hide demo badge
4. Clear feed panels

### Modify `checkDemoStatus()`:
- On page load, if demo active, restore Live Feed connection (not the old iframe panel)

## Task 5: Verify end-to-end

- Start orchestration service: `pdm run python src/main_fastapi.py`
- Open dashboard, click Launch Demo (passthrough) — verify Live Feed shows original text flowing
- Stop Demo — verify cleanup
- Click Launch Demo (pretranslated) — verify Live Feed shows original + Spanish translations
- Stop Demo
- Ctrl+C — verify clean shutdown

## Implementation Order

Tasks 1-3 (backend) can be done in parallel, then Task 4 (frontend), then Task 5 (verify).
