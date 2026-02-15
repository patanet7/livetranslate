# Fireflies Browser E2E Testing with agent-browser

**Date**: 2026-02-15
**Status**: Approved

## Overview

Browser-driven end-to-end test suite for the Fireflies dashboard and captions overlay using `agent-browser` CLI, orchestrated by pytest. Tests the full user-facing experience: navigating tabs, connecting to meetings, watching live captions with speaker attribution, managing glossaries, viewing history, running translations, and validating intelligence features.

## Architecture

```
pytest (orchestrates everything)
├── testcontainers: PostgreSQL 16 + Redis 7
├── Alembic migrations + bot_sessions schema
├── FirefliesMockServer (port 8090, Socket.IO + GraphQL)
├── Orchestration FastAPI (uvicorn, port 3001)
└── agent-browser (headed or headless+stream)

Browser targets:
├── localhost:3001/static/fireflies-dashboard.html (9-tab dashboard)
└── localhost:3001/static/captions.html?session={id} (caption overlay)
```

**Port 3001** to avoid conflicts with a running orchestration service on 3000.

## Viewing Modes

| Mode | Command | Use Case |
|------|---------|----------|
| **Headed** (default) | `agent-browser open <url>` | Local dev — visible Chromium window, watch tests live |
| **Headless + Stream** | `BROWSER_STREAM=1 pytest ...` | CI/remote — `AGENT_BROWSER_STREAM_PORT=9223`, connect viewer at `ws://localhost:9223` |

## Test Infrastructure

### New Files

```
tests/fireflies/e2e/browser/
├── conftest.py              # Fixtures: uvicorn server, mock server, agent-browser lifecycle
├── browser_helpers.py       # Python wrapper around agent-browser CLI
├── test_settings_tab.py     # Settings & API key management
├── test_connect_tab.py      # Connection flow + meeting discovery
├── test_sessions_tab.py     # Session stats & management
├── test_livefeed_tab.py     # Live transcript + translation feed
├── test_captions_overlay.py # Caption rendering, speaker names, timing, fade
├── test_glossary_tab.py     # Glossary CRUD, entries table, import/export
├── test_history_tab.py      # Historical transcripts, date range, viewer modal
├── test_data_logs_tab.py    # Session data viewer, transcripts & translations panels
├── test_translation_tab.py  # Model switching, prompt templates, test translation
└── test_intelligence_tab.py # Meeting notes, insights, Q&A agent
```

### browser_helpers.py — AgentBrowser Wrapper

```python
class AgentBrowser:
    """Thin wrapper around agent-browser CLI via subprocess."""

    def __init__(self, headed: bool = True, stream_port: int | None = None):
        self.headed = headed
        self.stream_port = stream_port

    def open(self, url: str) -> None
    def snapshot(self, interactive: bool = True) -> str     # DOM text with @refs
    def click(self, ref: str) -> None
    def fill(self, ref: str, text: str) -> None
    def select(self, ref: str, values: list[str]) -> None
    def screenshot(self, path: str) -> None
    def wait_for_text(self, text: str, timeout: float = 10) -> bool
    def wait_for_element(self, selector: str, timeout: float = 10) -> bool
    def extract_text(self, selector: str) -> str | None
    def close(self) -> None
```

### conftest.py Fixtures

```python
@pytest.fixture(scope="session")
async def orchestration_server(db_engine, redis_url):
    """Start real uvicorn serving the FastAPI app on port 3001."""

@pytest.fixture(scope="session")
async def mock_fireflies(orchestration_server):
    """Start FirefliesMockServer with Spanish conversation scenario."""

@pytest.fixture
def browser(request, orchestration_server):
    """Launch agent-browser in headed or streaming mode.
    Yields AgentBrowser instance. Closes on teardown."""
```

## Test Specifications — All 9 Dashboard Tabs + Captions Overlay

### 1. Settings Tab (`test_settings_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_save_api_key` | Fill API key → Save → verify masked display shows "Saved" badge |
| `test_test_connection` | Click "Test Connection" → verify status badge turns green (connected) |
| `test_clear_api_key` | Click Clear → verify key removed, display hidden |
| `test_service_status` | Verify service health indicators render (orchestration up, whisper/translation status) |
| `test_activity_log` | Verify log panel shows "[System] Dashboard initialized" entry, clear works |

### 2. Connect Tab (`test_connect_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_connect_to_meeting` | Fill transcript ID → select target language → click Connect → verify success dialog |
| `test_language_selector_populated` | Verify target language `<select>` populated from `/fireflies/dashboard/config` |
| `test_fetch_active_meetings` | Click "Refresh Meetings" → verify meeting list populated from mock GraphQL |
| `test_click_meeting_to_connect` | Click meeting in list → verify transcript ID auto-fills → connect |
| `test_connect_redirects_to_sessions` | After connect, verify tab switches to Sessions with new session visible |

### 3. Sessions Tab (`test_sessions_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_session_stats` | After connecting, verify stat cards: Total Sessions > 0, Connected > 0 |
| `test_session_list_item` | Verify session shows transcript ID, status badge "connected", chunk/translation counts |
| `test_stats_update_as_chunks_arrive` | Wait for mock to send chunks → snapshot → verify Chunks counter increments |
| `test_disconnect_session` | Click disconnect on session → verify status changes, Connected count decrements |
| `test_open_caption_view` | Click "Open Captions" link → verify navigates to captions.html with session param |

### 4. Live Feed Tab (`test_livefeed_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_connect_to_feed` | Select active session → click Connect → verify status badge "Connected" (green) |
| `test_original_transcript_streaming` | Verify left panel shows original Spanish text with speaker names as chunks arrive |
| `test_translation_feed_streaming` | Verify right panel shows English translations appearing alongside originals |
| `test_speaker_attribution_in_feed` | Verify speaker names (Alice, Bob) appear in transcript entries |
| `test_feed_stats` | Verify chunk count and timing stats update in real-time |
| `test_save_feed` | Click "Save Feed" → verify no errors (localStorage save) |
| `test_export_json` | Click "Export JSON" → verify download triggered |

### 5. Captions Overlay (`test_captions_overlay.py`) — Full Validation Suite

| Test | What it validates |
|------|-------------------|
| `test_caption_renders_with_speaker_name` | `.speaker-name` element shows correct speaker from mock scenario |
| `test_speaker_name_colored` | Each speaker gets a distinct color (inline style on `.speaker-name`) |
| `test_original_text_italic` | `.original-text` contains source Spanish text, renders italic |
| `test_translated_text_present` | `.translated-text` contains English translation, larger font size |
| `test_caption_auto_expiry` | Caption visible at T+0, faded/removed after configured duration (~4s default) |
| `test_max_caption_count` | Stream >5 chunks → verify max 5 `.caption-box` elements at any time |
| `test_fade_animation` | Before removal, verify `.caption-box.fading` class applied (opacity transition) |
| `test_connection_status_green` | Verify `.connection-status` indicator is green when WebSocket connected |
| `test_connection_status_red_on_disconnect` | Disconnect → verify indicator turns red |
| `test_multi_speaker_captions` | 2+ speakers in scenario → verify each speaker's caption box has their name + color |
| `test_caption_ordering` | Verify captions appear in chronological order (newest at bottom) |

**Timing validation flow:**
```
T+0s:  Mock sends chunk (Alice: "Hola, bienvenidos")
T+1s:  Snapshot → assert caption present, speaker "Alice", Spanish + English text
T+2s:  Mock sends chunk (Bob: "Gracias por invitarme")
T+3s:  Snapshot → 2 captions visible, Bob's entry present
T+5s:  Snapshot → Alice's caption has class "fading"
T+6s:  Snapshot → Alice's caption removed from DOM, only Bob's remains
```

### 6. Glossary Tab (`test_glossary_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_create_glossary` | Click "+ New" → fill name + domain → Save → verify appears in list |
| `test_load_glossary_entries` | Select glossary → verify entries table populates |
| `test_add_glossary_entry` | Click "+ Add Term" → fill source/target terms → verify row in table |
| `test_delete_glossary` | Click Delete → verify glossary removed from list |
| `test_set_default_glossary` | Click "Set as Default" → verify visual indicator |
| `test_domain_selection` | Verify domain dropdown has options: Medical, Legal, Technology, Business, Finance |

### 7. History Tab (`test_history_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_fetch_past_meetings` | Set date range → click "Fetch Past Meetings" → verify table populates from mock GraphQL |
| `test_past_meeting_columns` | Verify table shows Date, Title, Duration, Speakers, Actions columns |
| `test_open_transcript_viewer` | Click "View" on a meeting → verify modal opens with transcript content |
| `test_translate_full_transcript` | In modal, select target language → "Translate All" → verify progress bar + translations appear |
| `test_import_to_database` | Click "Import to DB" → verify success |
| `test_saved_transcripts_table` | After save, verify appears in "Saved Transcripts" table |

### 8. Data & Logs Tab (`test_data_logs_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_select_session_loads_data` | Select session from dropdown → verify transcripts panel populates |
| `test_transcripts_panel` | Verify left panel shows transcript entries with speaker names and timestamps |
| `test_translations_panel` | Verify right panel shows translation entries with target language |
| `test_empty_state` | With no session selected, verify "Select a session to view" placeholder |

### 9. Translation Tab (`test_translation_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_model_info_displayed` | Verify current model name and backend shown |
| `test_model_selector_populated` | Verify "Available Models" dropdown has options |
| `test_prompt_template_styles` | Switch between Simple/Full/Minimal template styles → verify textarea updates |
| `test_test_translation` | Fill test text → select target language → click Translate → verify result appears in code block |
| `test_save_prompt_template` | Edit prompt template → Save → reload → verify persisted |

### 10. Intelligence Tab (`test_intelligence_tab.py`)

| Test | What it validates |
|------|-------------------|
| `test_session_selector` | Verify session dropdown populated with active sessions |
| `test_add_manual_note` | Type a note → click "Add Note" → verify appears in notes list |
| `test_analyze_prompt` | Enter analysis prompt → click "Analyze" → verify response appears |
| `test_insight_templates_loaded` | Verify insight template dropdown has options |
| `test_generate_insight` | Select session + template → "Generate Insight" → verify result renders |
| `test_meeting_qa_agent` | Type question → Send → verify response in chat area |
| `test_suggested_queries` | Verify suggested query chips render, clickable |

## Output & Evidence

Every test produces:
- **Log file**: `tests/output/TIMESTAMP_test_browser_{tab}_results.log`
- **Screenshots**: `tests/output/TIMESTAMP_test_browser_{tab}_screenshot_{step}.png`
- Key DOM snapshots logged for debugging

## Mock Server Scenarios

The `FirefliesMockServer` will be configured with:

1. **Spanish conversation scenario** (primary): 2 speakers (Alice, Bob), 20 exchanges, 500ms chunk delay — provides enough data for all tabs
2. **GraphQL mock data**: Active meetings list, past transcripts with dates/durations/speakers
3. **Glossary fixtures**: Pre-populated tech glossary with Spanish/French/German terms
4. **Intelligence fixtures**: Mock insight templates and analysis responses

## Running

```bash
# Local dev — headed browser, watch it live
cd modules/orchestration-service
pdm run pytest tests/fireflies/e2e/browser/ -v

# Single tab test
pdm run pytest tests/fireflies/e2e/browser/test_captions_overlay.py -v

# CI / remote — headless with WebSocket stream on port 9223
BROWSER_STREAM=1 pdm run pytest tests/fireflies/e2e/browser/ -v

# Watch stream: connect browser to ws://localhost:9223
```

## Dependencies

- `agent-browser` (already installed: v0.10.0)
- No new Python packages needed — uses subprocess to call CLI
- Existing testcontainers, Alembic, mock server, conftest fixtures reused

## Design Decisions

- **pytest + CLI subprocess** over Node.js API: stays in existing test ecosystem, reuses all fixtures
- **Port 3001** for test uvicorn: avoids conflicts with dev server on 3000
- **Session-scoped server fixtures**: start once per test session, not per test — fast
- **Function-scoped browser fixture**: fresh browser per test for isolation
- **Headed default, streaming opt-in**: natural for local dev, CI-friendly with env var
- **Spanish-to-English baseline**: matches existing 5-min e2e test, framework designed for easy language pair expansion
