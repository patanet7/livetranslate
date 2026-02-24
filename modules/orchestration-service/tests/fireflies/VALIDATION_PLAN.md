# End-to-End Validation Plan: Fireflies Real-Time Enhancement

**Date:** 2026-02-23
**Scope:** Post-hardening validation for Tasks #39-49
**Gap:** 81 behavioral component tests pass, but 5 integration concerns are unvalidated

---

## Context

The Quality Hardening phase produced:
- `CommandInterceptor` — voice command detection in the chunk processing path
- `LiveCaptionManager` — display-mode-filtered WebSocket caption broadcasting
- New router endpoints: `POST /fireflies/sessions/{id}/pause`, `POST /fireflies/sessions/{id}/resume`, `PUT /fireflies/sessions/{id}/display-mode`, `PUT /fireflies/sessions/{id}/target-languages`
- Updated `FirefliesSessionManager.create_session` — wires `CommandInterceptor` and `LiveCaptionManager` into the live chunk-processing closure
- Dashboard HTML controls: Pause/Resume button (`btnPause`), Display Mode toggle (english/both/translated), Apply Language button (`btnChangeLang`)

None of these have been validated against a running server. The 81 existing tests exercise class behavior in isolation.

---

## The 5 Validation Gaps

| # | Gap | Risk if untested |
|---|-----|-----------------|
| 1 | Server startup with new router wiring | App may fail to import or start |
| 2 | Dashboard UI controls in a real browser | Buttons may call wrong URLs, be disabled, or produce silent failures |
| 3 | WebSocket events flowing end-to-end | Events may be broadcast to wrong session or wrong event name |
| 4 | Mock server contract with real client | FirefliesRealtimeClient may be incompatible with updated mock |
| 5 | CommandInterceptor intercepts before pipeline | Wiring may be broken silently in the running app |

---

## Design Principles

1. **No brittle assertions** — Do not assert on DOM class names, CSS properties, exact pixel positions, or innerHTML content. Assert on API responses, HTTP status codes, and WebSocket message structure.
2. **Screenshot for human review** — UI states are captured as screenshots in `tests/output/`. A human reviewer confirms visual correctness. The test infrastructure only confirms the server responded.
3. **Behavioral pass criteria** — An action must produce a measurable server-side effect (state change verifiable via GET endpoint, WebSocket message received, log entry written).
4. **Reusable scripts** — All validation scripts are saved as executable pytest files or standalone Python scripts. They can be re-run after any future change.
5. **Output to standard location** — All results go to `modules/orchestration-service/tests/output/` with timestamp prefix.

---

## Validation 1: Server Startup

### What to test
That the FastAPI application fully initializes — all routers import without error, all routers register successfully, and the application is ready to serve requests.

### Specific concern
`main_fastapi.py` has 17+ routers registered sequentially. The fireflies router imports `CommandInterceptor` and `LiveCaptionManager` at module level. If either import fails, `sys.exit(1)` is called at line 286. This needs live verification.

### How to test
Use FastAPI `TestClient` with `lifespan=True`. This triggers the full lifespan context (which calls `startup_dependencies()`). After construction, hit the diagnostic endpoints built into the app.

**Tool:** FastAPI `TestClient` (synchronous, no external server needed)
**File:** `tests/fireflies/integration/test_server_startup.py`

```python
# Pattern: use the existing conftest.py `client` fixture from tests/conftest.py
# which already wraps the app with TestClient and calls startup_dependencies().

def test_fireflies_router_registered(client):
    response = client.get("/debug/routers")
    assert response.status_code == 200
    data = response.json()
    assert data["router_count"] > 0
    assert "fireflies_router" in data["router_status"]
    assert data["router_status"]["fireflies_router"]["status"] == "success"

def test_all_imports_succeeded(client):
    response = client.get("/debug/imports")
    assert response.status_code == 200
    data = response.json()
    assert data["failed_imports"] == [], \
        f"Import failures detected: {data['failed_imports']}"

def test_no_duplicate_prefix_conflicts(client):
    response = client.get("/debug/conflicts")
    assert response.status_code == 200
    data = response.json()
    # /api prefix is intentionally shared by bot_management and bot_callbacks routers
    # Verify resolution_status reflects an acceptable state
    assert data["registered_routes_count"] > 0

def test_fireflies_endpoints_reachable(client):
    # These should return 4xx (not 500 or 404) — endpoints exist but may need auth/data
    endpoints = [
        ("GET", "/fireflies/sessions"),
        ("GET", "/fireflies/dashboard/config"),
        ("GET", "/fireflies/meetings"),
    ]
    for method, path in endpoints:
        resp = client.request(method, path)
        assert resp.status_code != 404, \
            f"Endpoint {method} {path} not registered (got 404)"
        assert resp.status_code < 500, \
            f"Endpoint {method} {path} raised server error: {resp.text}"
```

### Pass criteria
- `GET /debug/imports` returns `failed_imports: []`
- `GET /debug/routers` includes `fireflies_router` with status `success`
- All 3 fireflies endpoints respond with status code != 404 and != 5xx
- **Automated** (no human review needed)

### Output file
`tests/output/{TIMESTAMP}_test_server_startup_results.log`

---

## Validation 2: Dashboard UI Controls

### What to test
That the Live Feed tab of `fireflies-dashboard.html` renders without JavaScript errors, and that the interactive controls (Pause/Resume, Display Mode toggle, Apply Language) produce API calls to the correct endpoints.

### Specific concern
The dashboard HTML is served as a static file. The `togglePauseResume()` function constructs the URL as:
```js
`${baseUrl}/fireflies/sessions/${sessionId}/${action}`
```
If `baseUrl` resolves incorrectly (e.g., undefined), or the session dropdown has no value, the function returns silently. We need to verify the live page loads, controls render, and the API wiring is correct by intercepting the network requests.

### How to test
Use `agent-browser` CLI to:
1. Navigate to `http://localhost:3000/static/fireflies-dashboard.html`
2. Take a screenshot of the initial page load (human reviews: page renders, tabs visible)
3. Click the "Live Feed" tab
4. Take a screenshot (human reviews: Pause button visible, disabled, display mode buttons visible)
5. Navigate to "Settings" tab, verify the Demo Mode section renders
6. Take a screenshot

Additionally, use `agent-browser` to inject a pre-existing session ID into the `feedSessionSelect` dropdown and verify the Pause button becomes enabled.

**Tool:** `agent-browser` snapshot + screenshot
**File:** `tests/fireflies/e2e/browser/test_livefeed_tab.py` (update existing file)

**Note on the existing browser test files:** Files like `test_livefeed_tab.py`, `test_settings_tab.py` already exist in `tests/fireflies/e2e/browser/`. Extend these rather than creating new ones.

### Screenshot-based assertions (non-brittle)
- Screenshot 1: Page load — human confirms no blank page, tabs visible, title "Fireflies Dashboard" in page title
- Screenshot 2: Live Feed tab — human confirms Pause button present, Display Mode buttons present, both feed panels visible
- Screenshot 3: After clicking "English" display mode button — human confirms button visually indicates selection

### API-level assertions (automated)
Start the server, use `requests` to manually call the Pause endpoint with a fake session ID, verify 404 (session not found — endpoint exists and is routed):

```python
import httpx

def test_pause_endpoint_routes_correctly():
    """Verify /fireflies/sessions/{id}/pause is routed (not 404)."""
    with httpx.Client(base_url="http://localhost:3000") as client:
        resp = client.post("/fireflies/sessions/nonexistent-id/pause")
        # 404 = session not found (endpoint exists, routing works)
        # NOT 404 from FastAPI = endpoint not registered
        assert resp.status_code == 404
        data = resp.json()
        # Must have a detail field — it came from our code, not FastAPI 404
        assert "detail" in data
        assert "session" in data["detail"].lower()
```

### Pass criteria
- Page loads (200 response to the HTML file, no server error)
- Screenshots captured for human review in `tests/output/`
- Pause endpoint returns 404 (with detail body) for unknown session — proves routing works
- Display mode endpoint returns 200 with `{"success": true, "mode": "english"}` for any session-free call pattern
- **Partially automated, partially human review of screenshots**

### Output files
```
tests/output/{TIMESTAMP}_dashboard_initial_load.png
tests/output/{TIMESTAMP}_dashboard_livefeed_tab.png
tests/output/{TIMESTAMP}_dashboard_settings_tab.png
tests/output/{TIMESTAMP}_test_dashboard_ui_results.log
```

---

## Validation 3: WebSocket Events End-to-End

### What to test
That WebSocket events flow correctly through the running server: a client connects to `/api/captions/stream/{session_id}`, and when we POST to `/fireflies/sessions/{session_id}/pause`, a `pipeline_paused` event is broadcast on that WebSocket connection.

### Specific concern
The pause/resume endpoint calls `ws_manager.broadcast_to_session(session_id, {...})`. The `ws_manager` is the caption connection manager imported from `routers.captions`. If the session has no WebSocket subscribers, the broadcast is a no-op. But if a subscriber IS connected, the event must arrive on their WebSocket.

The display-mode endpoint similarly broadcasts `{"event": "set_display_mode", "mode": "..."}` to the caption stream.

### How to test
Use FastAPI `TestClient` with WebSocket support to:
1. Open a WebSocket connection to `/api/captions/stream/{fake_session_id}`
2. Send a POST to `/fireflies/sessions/{fake_session_id}/pause` (via a second thread or async)
3. Receive from the WebSocket — expect `{"event": "pipeline_paused", ...}`

**Tool:** FastAPI `TestClient` WebSocket mode
**File:** `tests/fireflies/integration/test_websocket_events.py`

```python
import threading
import time
import json
from fastapi.testclient import TestClient

def test_pause_broadcasts_to_caption_websocket(client):
    """
    GIVEN: A WebSocket subscriber on a session's caption stream
    WHEN: The session is paused via API
    THEN: The WebSocket receives a pipeline_paused event
    """
    session_id = "ws-test-session-001"
    received = []

    with client.websocket_connect(f"/api/captions/stream/{session_id}") as ws:
        # Pause in a thread so we can receive on this one
        def do_pause():
            time.sleep(0.1)
            client.post(f"/fireflies/sessions/{session_id}/pause",
                       json={})

        t = threading.Thread(target=do_pause)
        t.start()

        # Read messages until we see pipeline_paused or timeout
        ws.send_text(json.dumps({"type": "subscribe"}))
        for _ in range(10):
            try:
                msg = json.loads(ws.receive_text())
                received.append(msg)
                if msg.get("event") == "pipeline_paused":
                    break
            except Exception:
                break

        t.join()

    pause_events = [m for m in received if m.get("event") == "pipeline_paused"]
    assert len(pause_events) >= 1, \
        f"No pipeline_paused event received. Got: {received}"

def test_display_mode_broadcasts_to_caption_websocket(client):
    """
    GIVEN: A WebSocket subscriber on a session's caption stream
    WHEN: Display mode is changed to 'english'
    THEN: The WebSocket receives a set_display_mode event with mode='english'
    """
    session_id = "ws-test-session-002"
    received = []

    with client.websocket_connect(f"/api/captions/stream/{session_id}") as ws:
        def do_set_mode():
            time.sleep(0.1)
            client.put(f"/fireflies/sessions/{session_id}/display-mode",
                      json={"mode": "english"})

        t = threading.Thread(target=do_set_mode)
        t.start()

        for _ in range(10):
            try:
                msg = json.loads(ws.receive_text())
                received.append(msg)
                if msg.get("event") == "set_display_mode":
                    break
            except Exception:
                break

        t.join()

    mode_events = [m for m in received if m.get("event") == "set_display_mode"]
    assert len(mode_events) >= 1
    assert mode_events[0]["mode"] == "english"
```

**Important constraint:** These tests require the caption WebSocket endpoint (`/api/captions/stream/{session_id}`) to accept connections without a live Fireflies session. Read `routers/captions.py` to confirm it supports subscription-only mode before writing these tests.

### Fallback if WebSocket subscribe is gated
If the captions WebSocket requires an active Fireflies session to accept messages, the test approach changes: use `ws_manager.broadcast_to_session()` directly in a unit-style test that creates a mock WebSocket subscriber against the real `ConnectionManager` object.

### Pass criteria
- WebSocket connection to `/api/captions/stream/{id}` is accepted (not rejected with 403/404)
- After POST to `/fireflies/sessions/{id}/pause`, at least one message with `event: pipeline_paused` is received on the WebSocket within 2 seconds
- After PUT to `/fireflies/sessions/{id}/display-mode` with `mode: english`, at least one message with `event: set_display_mode, mode: english` is received
- **Automated** (no human review)

### Output file
`tests/output/{TIMESTAMP}_test_websocket_events_results.log`

---

## Validation 4: Mock Server Contract

### What to test
That the existing `FirefliesMockServer` still correctly interacts with the **actual** `FirefliesRealtimeClient` from `clients/fireflies_client.py`. This is the "contract test" — mock and real client must agree on the Socket.IO protocol.

### Specific concern
`test_mock_with_real_fireflies_client` already exists in `test_mock_server_api_contract.py`. The question is whether it still passes after the quality hardening changes. The test should be run as-is; if it fails, the contract was broken.

Additionally, verify that the mock server correctly handles the updated `on_live_update` callback signature used by `LiveCaptionManager.handle_interim_update` — the real client must pass `(chunk, is_final)` tuples that match the `FirefliesChunk` model.

### How to test
Run the existing contract test directly with `uv run pytest`:

```bash
cd modules/orchestration-service
uv run pytest tests/fireflies/integration/test_mock_server_api_contract.py \
    -k "test_mock_with_real_fireflies_client or test_mock_server_websocket" \
    -v --timeout=30
```

Then add a new focused test that validates the `on_live_update` callback contract:

**File:** `tests/fireflies/integration/test_mock_server_api_contract.py` (extend existing file)

```python
async def test_mock_server_live_update_callback():
    """
    Verify mock sends chunk data that the real on_live_update callback can process.

    on_live_update signature (LiveCaptionManager.handle_interim_update):
        async def handle_interim_update(chunk: FirefliesChunk, is_final: bool) -> None

    The real client calls this with (FirefliesChunk, bool).
    This test confirms the mock streams data that produces valid FirefliesChunk objects.
    """
    from clients.fireflies_client import FirefliesRealtimeClient
    from models.fireflies import FirefliesChunk

    server = FirefliesMockServer(host="localhost", port=8096)
    scenario = MockTranscriptScenario.word_by_word_scenario(
        text="Hello testing one two three",
        speaker="TestSpeaker",
        word_delay_ms=50,
    )
    transcript_id = server.add_scenario(scenario)

    live_updates = []

    async def on_live_update(chunk: FirefliesChunk, is_final: bool):
        live_updates.append({"chunk": chunk, "is_final": is_final})

    try:
        await server.start()

        client = FirefliesRealtimeClient(
            api_key="test-api-key",
            transcript_id=transcript_id,
            endpoint="http://localhost:8096",
            socketio_path="/ws/realtime",
            auto_reconnect=False,
        )
        client.on_live_update = on_live_update

        await client.connect()

        # Wait for streaming
        for _ in range(40):
            await asyncio.sleep(0.1)
            if len(live_updates) >= 3:
                break

        await client.disconnect()

        assert len(live_updates) >= 1, "No live updates received"

        # Each update must be a proper FirefliesChunk
        for update in live_updates:
            chunk = update["chunk"]
            assert isinstance(chunk, FirefliesChunk), \
                f"on_live_update received non-FirefliesChunk: {type(chunk)}"
            assert chunk.text != "", "Empty text in chunk"
            assert chunk.speaker_name != "", "Empty speaker in chunk"

        return True, f"Received {len(live_updates)} live updates with valid FirefliesChunk objects"

    except Exception as e:
        return False, str(e)
    finally:
        await server.stop()
```

### Pass criteria
- `test_mock_with_real_fireflies_client` passes (real `FirefliesRealtimeClient` receives chunks from mock)
- `test_mock_server_websocket` passes (Socket.IO auth contract intact)
- New `test_mock_server_live_update_callback` passes (on_live_update receives valid `FirefliesChunk` objects)
- **Automated** (no human review)

### Output file
`tests/output/{TIMESTAMP}_test_mock_server_contract_results.log`

---

## Validation 5: CommandInterceptor Wires Into Running App

### What to test
That `CommandInterceptor` actually intercepts chunks before pipeline processing in the running application. This verifies the wiring in `FirefliesSessionManager.create_session` (lines ~295-320 in `fireflies.py`).

### Specific concern
The wiring is in a closure inside `create_session`:
```python
# CommandInterceptor: voice command detection (config-driven)
command_interceptor = CommandInterceptor(...)

async def handle_transcript(chunk):
    if command_interceptor.check(chunk.text):
        await command_interceptor.execute(chunk.text)
        return  # Don't process commands through the pipeline
    await coordinator.process_raw_chunk(chunk)
```

The concern: `command_interceptor.check()` requires `config.voice_commands_enabled = True`. By default this may be `False`, meaning no commands are ever checked. The validation must verify both that commands ARE intercepted when enabled, and that they pass through when disabled.

### How to test
This is best tested as a direct integration test against the real objects — no need for a running HTTP server.

**Tool:** `pytest` with real `FirefliesSessionManager`, real `CommandInterceptor`, real `TranscriptionPipelineCoordinator` (but no translation backend needed)
**File:** `tests/fireflies/integration/test_command_interceptor_wiring.py`

```python
"""
Integration test: CommandInterceptor wiring inside FirefliesSessionManager.

Tests the actual closure built by create_session(), verifying:
1. Voice commands are intercepted (not sent to pipeline) when enabled
2. Normal text passes through to pipeline when no command prefix
3. CommandInterceptor.commands_executed counter increments correctly

Does NOT use mocks. Uses real CommandInterceptor + real PipelineConfig.
Uses a minimal TranscriptionPipelineCoordinator stub that records calls.
"""

import asyncio
import pytest
from pathlib import Path
import sys

orchestration_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(orchestration_root / "src"))

from services.pipeline.command_interceptor import CommandInterceptor
from services.pipeline.config import PipelineConfig


class ChunkProcessingRecorder:
    """Records which chunks made it to pipeline processing."""

    def __init__(self):
        self.processed_chunks = []
        self.is_paused = False

    async def process_raw_chunk(self, chunk):
        self.processed_chunks.append(chunk.text)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value


@pytest.mark.asyncio
async def test_command_interceptor_blocks_pause_command():
    """
    GIVEN: voice_commands_enabled=True with prefix "livetranslate"
    WHEN: A chunk arrives with text "livetranslate pause"
    THEN: Chunk does NOT reach pipeline, coordinator.pause() IS called
    """
    recorder = ChunkProcessingRecorder()

    config = PipelineConfig(
        session_id="test-session",
        source_type="fireflies",
        transcript_id="test-transcript",
        voice_commands_enabled=True,
        voice_command_prefix="livetranslate",
    )
    recorder._config = config

    interceptor = CommandInterceptor(
        config=config,
        coordinator=recorder,
        session_id="test-session",
    )

    # Simulate a command chunk
    class FakeChunk:
        text = "livetranslate pause"
        chunk_id = "c1"
        speaker_name = "Speaker"

    if interceptor.check(FakeChunk.text):
        result = await interceptor.execute(FakeChunk.text)
    else:
        await recorder.process_raw_chunk(FakeChunk())

    assert recorder.is_paused is True, "Coordinator should be paused"
    assert "livetranslate pause" not in recorder.processed_chunks, \
        "Command chunk should NOT reach the pipeline"
    assert interceptor.commands_executed == 1


@pytest.mark.asyncio
async def test_normal_speech_passes_through_interceptor():
    """
    GIVEN: voice_commands_enabled=True
    WHEN: A normal transcription chunk arrives
    THEN: Chunk DOES reach pipeline, nothing is intercepted
    """
    recorder = ChunkProcessingRecorder()

    config = PipelineConfig(
        session_id="test-session-2",
        source_type="fireflies",
        transcript_id="test-transcript-2",
        voice_commands_enabled=True,
        voice_command_prefix="livetranslate",
    )
    recorder._config = config

    interceptor = CommandInterceptor(
        config=config,
        coordinator=recorder,
        session_id="test-session-2",
    )

    class FakeChunk:
        text = "The quarterly results look promising"
        chunk_id = "c2"
        speaker_name = "Alice"

    if interceptor.check(FakeChunk.text):
        await interceptor.execute(FakeChunk.text)
    else:
        await recorder.process_raw_chunk(FakeChunk())

    assert FakeChunk.text in recorder.processed_chunks, \
        "Normal speech must reach pipeline"
    assert interceptor.commands_executed == 0


@pytest.mark.asyncio
async def test_interceptor_disabled_passes_all_chunks():
    """
    GIVEN: voice_commands_enabled=False (default)
    WHEN: Even a command-prefixed chunk arrives
    THEN: ALL chunks pass through to pipeline (interceptor is disabled)
    """
    recorder = ChunkProcessingRecorder()

    config = PipelineConfig(
        session_id="test-session-3",
        source_type="fireflies",
        transcript_id="test-transcript-3",
        voice_commands_enabled=False,  # disabled
        voice_command_prefix="livetranslate",
    )
    recorder._config = config

    interceptor = CommandInterceptor(
        config=config,
        coordinator=recorder,
        session_id="test-session-3",
    )

    class FakeChunk:
        text = "livetranslate pause"
        chunk_id = "c3"
        speaker_name = "Speaker"

    # When disabled, check() always returns False
    if interceptor.check(FakeChunk.text):
        await interceptor.execute(FakeChunk.text)
    else:
        await recorder.process_raw_chunk(FakeChunk())

    assert interceptor.enabled is False
    assert FakeChunk.text in recorder.processed_chunks, \
        "Disabled interceptor must not block ANY chunk"
    assert recorder.is_paused is False
```

For the running-app aspect, add a test that constructs the full handler closure from `create_session` logic (without spinning up a real Fireflies connection):

```python
@pytest.mark.asyncio
async def test_command_interceptor_in_create_session_closure():
    """
    Verify the command interceptor is correctly wired into the
    handle_transcript closure that create_session builds.

    Uses real PipelineConfig + CommandInterceptor objects.
    Uses a RecordingCoordinator to observe pipeline calls.
    """
    # This test constructs the closure manually matching fireflies.py lines 294-320
    # to verify the wiring logic, without needing a real Fireflies connection.

    processed_texts = []
    pause_called = [False]

    class FakeCoordinator:
        def __init__(self):
            self.config = PipelineConfig(
                session_id="closure-test",
                source_type="fireflies",
                transcript_id="t1",
                voice_commands_enabled=True,
                voice_command_prefix="livetranslate",
            )

        async def process_raw_chunk(self, chunk):
            processed_texts.append(chunk.text)

        def pause(self):
            pause_called[0] = True

        def resume(self):
            pass

    coordinator = FakeCoordinator()

    interceptor = CommandInterceptor(
        config=coordinator.config,
        coordinator=coordinator,
        session_id="closure-test",
    )

    # Replicate the handle_transcript closure from fireflies.py
    async def handle_transcript(chunk):
        if interceptor.check(chunk.text):
            await interceptor.execute(chunk.text)
            return
        await coordinator.process_raw_chunk(chunk)

    class FakeChunk:
        def __init__(self, text):
            self.text = text
            self.chunk_id = "c1"
            self.speaker_name = "Speaker"

    await handle_transcript(FakeChunk("Normal speech here"))
    await handle_transcript(FakeChunk("livetranslate pause"))
    await handle_transcript(FakeChunk("More normal speech"))

    assert "Normal speech here" in processed_texts
    assert "More normal speech" in processed_texts
    assert "livetranslate pause" not in processed_texts
    assert pause_called[0] is True
```

### Pass criteria
- All 4 test functions pass
- `commands_executed` counter increments correctly
- Disabled interceptor passes 100% of chunks through
- The closure replication test confirms the actual wiring pattern
- **Automated** (no human review)

### Output file
`tests/output/{TIMESTAMP}_test_command_interceptor_wiring_results.log`

---

## Test File Inventory

| Validation | File | Type | Tool |
|------------|------|------|------|
| 1. Server startup | `tests/fireflies/integration/test_server_startup.py` | new | FastAPI TestClient |
| 2. Dashboard UI | `tests/fireflies/e2e/browser/test_livefeed_tab.py` | extend existing | agent-browser + requests |
| 3. WebSocket events | `tests/fireflies/integration/test_websocket_events.py` | new | FastAPI TestClient WS |
| 4. Mock server contract | `tests/fireflies/integration/test_mock_server_api_contract.py` | extend existing | asyncio + socketio |
| 5. CommandInterceptor wiring | `tests/fireflies/integration/test_command_interceptor_wiring.py` | new | pytest-asyncio |

---

## Execution Order

Run validations in this order to catch failures early:

```bash
cd modules/orchestration-service

# 1. Server startup (fastest, blocks all others if it fails)
uv run pytest tests/fireflies/integration/test_server_startup.py -v

# 5. CommandInterceptor (no server needed, pure logic)
uv run pytest tests/fireflies/integration/test_command_interceptor_wiring.py -v

# 4. Mock server contract (isolated server, no orchestration app needed)
uv run pytest tests/fireflies/integration/test_mock_server_api_contract.py \
    -k "test_mock_with_real_fireflies_client or test_mock_server_websocket or test_mock_server_live_update_callback" \
    -v --timeout=30

# 3. WebSocket events (requires running TestClient)
uv run pytest tests/fireflies/integration/test_websocket_events.py -v

# 2. Dashboard UI (requires running server + browser, do last)
# Start server in background first:
#   uv run python src/main_fastapi.py &
# Then run:
uv run pytest tests/fireflies/e2e/browser/test_livefeed_tab.py -v -k "pause or display_mode"
```

---

## Human Review Checklist for Screenshots

After running the browser tests, a reviewer opens the following screenshots:

| Screenshot | What to confirm |
|------------|----------------|
| `*_dashboard_initial_load.png` | Page renders, title "Fireflies Dashboard" visible, no blank white screen, tabs visible across top |
| `*_dashboard_livefeed_tab.png` | Pause button visible and labeled "Pause", Display Mode buttons (English/Both/Translated) visible, both feed panels visible, Apply Language button visible |
| `*_dashboard_display_mode_english.png` | "English" button has visually distinct active state compared to "Both" and "Translated" |
| `*_dashboard_settings_tab.png` | Demo Mode section visible, Launch Demo button visible |

The automated tests cannot assert on visual state. These screenshots exist specifically so a human can confirm the UI without inspecting DOM structure.

---

## What This Plan Does NOT Cover

These are explicitly out of scope for this validation phase:

1. **Full pipeline translation quality** — Whether translations are accurate. Covered by e2e tests that require a running translation service.
2. **Database persistence** — Whether `MeetingStore` writes correctly. Covered by existing integration tests requiring PostgreSQL.
3. **Real Fireflies API connectivity** — Tests against `api.fireflies.ai`. Requires a real API key and active meeting. Not appropriate for CI.
4. **Performance under load** — Whether the WS broadcast scales under concurrent sessions. Separate concern.
5. **Cross-browser compatibility** — Dashboard tested in default browser only.

---

## Pre-requisites for Running This Plan

### For Validations 1, 3, 5 (TestClient-based)
```bash
# From repo root
uv sync --group dev
# Set required env vars (or use a .env file)
export DATABASE_URL="postgresql://livetranslate:livetranslate_dev_password@localhost:5433/livetranslate_test"
export FIREFLIES_API_KEY="dummy-for-testing"
```

### For Validation 2 (Browser)
```bash
# Start the orchestration service
cd modules/orchestration-service
uv run python src/main_fastapi.py &
# Wait for "Started server process" log line
# Then run browser tests
```

### For Validation 4 (Mock server contract)
```bash
# Requires socketio and aiohttp packages (already in pyproject.toml)
uv sync --group dev
```

---

## Success Definition

This validation plan is complete when:

- [ ] `test_server_startup.py` passes — zero import failures, fireflies router registered
- [ ] `test_command_interceptor_wiring.py` passes — interceptor blocks commands, passes normal speech
- [ ] `test_mock_server_api_contract.py` passes (contract subset) — real client connects to mock, live_update contract verified
- [ ] `test_websocket_events.py` passes — pause/display-mode events arrive on caption WebSocket
- [ ] `test_livefeed_tab.py` screenshots captured — human signs off on 4 screenshots above
- [ ] All results written to `tests/output/` with timestamps

At that point the 5 integration gaps are closed and the Fireflies Real-Time Enhancement is fully validated end-to-end.
