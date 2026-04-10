# Meeting Subtitle System — Design Spec

**Date:** 2026-04-10
**Status:** Draft (v2 — post specialist review)
**Scope:** Config-driven modular subtitle pipeline for meeting bots with hybrid rendering

## Overview

A modular meeting subtitle system that renders real-time translated captions into a Google Meet bot's virtual camera feed and browser-based overlays. All behavior is driven by a single `MeetingSessionConfig` — every setting is live-switchable mid-meeting via chat commands from any participant.

### Goals

1. Bot joins Google Meet and displays translated subtitles as its camera feed
2. Same subtitles available as OBS browser source and standalone browser window
3. Config-driven source routing: switch between bot audio capture, Fireflies, or Meet's own captions
4. All display settings (language, font, mode, theme, speakers, original text) changeable live via `/slash` commands in meeting chat
5. Extremely modular — inputs, processing, and outputs are independent layers connected by interfaces

### Non-Goals

- Speaker diarization (handled upstream by transcription service)
- Meeting scheduling/calendar integration
- Recording/playback (handled by existing FLAC recording system)

## Service Boundary: TypeScript Bot vs Python Orchestration

The system spans two services with a clear responsibility split:

**TypeScript (`meeting-bot-service`)** — Owns the browser. All Playwright/DOM interaction:
- Join meeting, manage lifecycle
- Read chat messages (polling-based)
- Parse slash commands
- Type chat responses
- Present virtual camera feed

**Python (`orchestration-service`)** — Owns the data pipeline. All caption/translation/rendering:
- Caption source adapters (bot audio, Fireflies)
- Translation pipeline
- MeetingSessionConfig (single control plane)
- CaptionBuffer (source of truth for display)
- PIL virtual camera frame rendering
- WebSocket broadcast to browser clients

**Communication protocol:** WebSocket between TS bot and Python orchestration (existing `audio_streaming.ts` pattern). New message types:

```typescript
// Bot → Orchestration
{ type: "chat_command", command: "/lang zh", sender: "Alice" }

// Orchestration → Bot
{ type: "chat_response", text: "✓ Translating: auto-detect → Chinese" }
{ type: "config_changed", changes: { target_lang: "zh", source_lang: "auto" } }
```

## Architecture

### Three Independent Layers

```
CAPTION SOURCES          PROCESSING CORE              RENDERERS
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│ BotAudioAdapter  │───→│                     │───→│ PILVirtualCamRenderer│
│ FirefliesAdapter │───→│   CaptionBuffer     │───→│ WebSocketRenderer    │
│                  │    │   TranslationPipe   │───→│ DashboardViews       │
│ ChatCommand (TS) │───→│   CommandDispatcher │───→│ ChatResponder (TS)   │
└─────────────────┘    │   MeetingSessionConfig│    └──────────────────────┘
                       └─────────────────────┘
```

Each layer communicates through defined protocols. No layer knows the internals of any other.

### MeetingSessionConfig — Single Control Plane

**Relationship to existing configs:** This config composes and facades existing config objects rather than duplicating them. It does NOT inherit from Pydantic `BaseModel` — it is a plain class with explicit mutation.

```python
class MeetingSessionConfig:
    """Single control plane for meeting subtitle behavior.
    
    Composes existing configs:
    - Language state delegates to SessionConfig (source_lang, target_lang, lock_language)
    - Display state unifies the two existing WebcamConfig classes
    - Translation toggle delegates to TranslationConfig
    - Source routing is genuinely new (caption_source)
    
    NOT a Pydantic BaseModel. Uses explicit update() for thread-safe 
    mutation with subscriber notification.
    """
    # --- Source Routing (NEW — no existing config covers this) ---
    caption_source: Literal["bot_audio", "fireflies"] = "bot_audio"
    
    # --- Language (delegates to existing SessionConfig) ---
    source_lang: str = "auto"
    target_lang: str = "en"
    translation_enabled: bool = True

    # --- Display (unifies existing WebcamConfig dataclass + Pydantic model) ---
    display_mode: Literal["subtitle", "split", "interpreter"] = "subtitle"
    theme: Literal["dark", "light", "high_contrast", "minimal", "corporate"] = "dark"
    font_size: int = 24
    show_speakers: bool = True
    show_original: bool = False

    # --- Runtime ---
    session_id: str
    bot_id: str | None = None

    def update(self, **changes: Any) -> set[str]:
        """Apply changes atomically, return set of changed field names.
        Thread-safe. Fires subscriber notification once per batch."""
        changed = set()
        with self._lock:
            for field, value in changes.items():
                if getattr(self, field) != value:
                    setattr(self, field, value)
                    changed.add(field)
        if changed:
            self._notify_subscribers(changed)
        return changed
```

**Config unification plan (Phase 0 prerequisite):**
1. Merge the two `WebcamConfig` classes (dataclass in `virtual_webcam.py` + Pydantic in `models/bot.py`) into one canonical Pydantic model
2. Rename `models/bot.py:TranslationConfig` to `TranslationDisplayConfig` to avoid collision with `translation/config.py:TranslationConfig` (LLM runtime settings)
3. Unify `DisplayMode` enums — currently three incompatible sets across PIL, SvelteKit, and bot models. Canonical set: `subtitle`, `split`, `interpreter` (matching dashboard)

### Caption Source Adapters

Two layers of abstraction (not one):

1. **`CaptionSourceAdapter`** (new) — Stateful service with lifecycle. Connects to an input, manages start/stop, emits `CaptionEvent`s.
2. **`ChunkAdapter`** (existing) — Stateless data transformer. Converts raw format → `TranscriptChunk`. Lives in `services/pipeline/adapters/`.

The source adapter *uses* the chunk adapter internally:

```python
class CaptionSourceAdapter(Protocol):
    """Lifecycle-managing source connector. Emits CaptionEvents."""
    async def start(self, config: MeetingSessionConfig) -> None: ...
    async def stop(self) -> None: ...
    on_caption: Callable[[CaptionEvent], Awaitable[None]]

# Example: FirefliesCaptionSource internally uses FirefliesChunkAdapter
# for data conversion, plus adds WebSocket lifecycle, reconnection, etc.
```

**Adapters:**

| Source Adapter | Internal Chunk Adapter | Status |
|---------|-------|--------|
| `BotAudioCaptionSource` | (direct from transcription WebSocket) | Wrap existing audio pipeline |
| `FirefliesCaptionSource` | `FirefliesChunkAdapter` (existing) | Wrap existing adapter + add lifecycle |

When `config.caption_source` changes, the active source adapter is stopped and the new one started. CaptionBuffer receives events identically regardless of source.

### CaptionEvent Schema

```python
@dataclass
class CaptionEvent:
    """Canonical event emitted by source adapters, consumed by renderers."""
    event_type: Literal["added", "updated", "expired", "cleared"]
    caption_id: str
    text: str
    speaker_name: str | None
    speaker_color: str          # Hex string, canonical palette
    source_lang: str
    target_lang: str | None
    translated_text: str | None
    confidence: float
    timestamp: datetime
    expires_at: datetime
    is_draft: bool
```

### CaptionBuffer (Existing — Subscriber Extension)

The existing `CaptionBuffer` already has single-callback slots (`on_caption_added`, `on_caption_updated`, `on_caption_expired`). The extension converts these to multi-subscriber lists, **replacing** (not supplementing) the single-callback pattern. Backward compatibility: constructor-injected callbacks are auto-subscribed.

```python
class CaptionBuffer:
    def subscribe(self, callback: Callable[[CaptionEvent], Awaitable[None]]) -> None: ...
    def unsubscribe(self, callback: ...) -> None: ...
    # Old on_caption_added=fn param still works — auto-subscribed internally
```

### Renderer Protocol

Both renderers implement:

```python
class SubtitleRenderer(Protocol):
    async def on_config_changed(self, changed_fields: set[str]) -> None: ...
    async def on_caption_event(self, event: CaptionEvent) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
```

**PILVirtualCamRenderer:**
- Subscribes to CaptionBuffer events
- Snapshots config at start of each frame (prevents mid-frame tearing from concurrent config changes)
- Renders frames using `VirtualWebcamManager` at 1280x720 (not 1920x1080 — Meet downscales anyway)
- Pre-allocates frame buffer, uses dirty-flag rendering (only re-render when state changes)
- Caches static frames ("Waiting for transcription...")
- Writes frames to `pyvirtualcam` with explicit `fmt=PixelFormat.RGB`
- RGBA overlay mode composites onto opaque background before sending (no alpha in virtual cameras)
- Frame-paced timer (not busy-wait) to prevent 100% CPU usage

**WebSocketRenderer:**
- Subscribes to same CaptionBuffer events
- Pushes JSON (including `expires_at` for client-side expiration sync) to SvelteKit overlay and dashboard clients
- Drops events older than 500ms (staleness check)
- Already largely exists — formalize the interface

### Shared Theme/Color Definitions

Speaker colors, theme colors, and display mode definitions live in ONE canonical location — `livetranslate-common` or a shared JSON file. Both PIL and SvelteKit renderers read from this source.

**Current state (three separate palettes that are already out of sync):**
- `virtual_webcam.py:118` — RGB tuples
- `caption_buffer.py:36` — hex strings
- `loopback.svelte.ts:66` — different hex strings

**Target state:** One canonical palette (hex strings). PIL converts to RGB tuples at render time.

### Virtual Camera Pipeline

**Startup ordering (critical):**
1. Start `pyvirtualcam` device (renders "Connecting..." frame) — MUST happen before browser launch
2. Launch Playwright/Chromium (sees virtual camera as available device)
3. Bot joins meeting with camera enabled
4. Caption pipeline starts
5. `PILVirtualCamRenderer` begins rendering subtitles to the same device

```
pyvirtualcam.Camera(1280, 720, 30, fmt=PixelFormat.RGB)
    ↑
PILVirtualCamRenderer
    ↑ (subscribes to CaptionBuffer + MeetingSessionConfig)
    ↑
VirtualWebcamManager (PIL rendering)
    → frame buffer (pre-allocated, dirty-flag)
    → numpy array (RGB, no RGBA)
    → pyvirtualcam.Camera.send(frame)
    → OS virtual camera device
        macOS: OBS Virtual Camera (OBS must be installed, camera extension activated once)
        Linux: v4l2loopback (host kernel module, --device passthrough to container)
    → Chromium sees as default camera
    → Google Meet shows as bot's video tile
```

**Chrome flags:**

```
# Permission (REQUIRED)
--use-fake-ui-for-media-stream        # Auto-grant camera permission

# DO NOT USE (overrides virtual camera with test pattern):
# --use-fake-device-for-media-stream  # REMOVED — blocks real virtual camera

# Audio capture
--autoplay-policy=no-user-gesture-required
--enable-usermedia-screen-capturing

# Docker/headless
--no-sandbox
--disable-setuid-sandbox
--disable-dev-shm-usage
--disable-gpu
--headless=new
```

**macOS device selection:** If the built-in FaceTime camera is also visible, the bot must use `enumerateDevices()` + `getUserMedia({video: {deviceId: {exact: id}}})` via `page.evaluate()` to select the virtual camera explicitly.

### Chat Command System

**All browser interaction lives in TypeScript (`meeting-bot-service`).**

```
TypeScript Bot (owns Playwright page):
    1. Opens chat panel (aria-label="Chat with everyone")
    2. Polls chat container every 500ms (NOT MutationObserver — more reliable)
    3. Detects new messages, checks for / prefix
    4. Parses command locally (TypeScript CommandParser)
    5. Sends {type: "chat_command"} to orchestration via WebSocket
    6. Receives {type: "chat_response"} from orchestration
    7. Types response into chat (contenteditable div, NOT page.fill)

Python Orchestration (owns config + rendering):
    1. Receives chat_command from bot WebSocket
    2. Validates and applies to MeetingSessionConfig.update()
    3. ConfigChanged event fires to all renderers
    4. Sends chat_response back to bot for display
```

**Chat typing helper** (TypeScript):
```typescript
async function sendChatMessage(page: Page, text: string): Promise<void> {
  // Click chat input (contenteditable div)
  const input = page.locator('[aria-label="Send a message to everyone"]');
  await input.click();
  // Set text via evaluate (page.fill doesn't work on contenteditable)
  await input.evaluate((el, msg) => {
    el.textContent = msg;
    el.dispatchEvent(new InputEvent('input', { inputType: 'insertText', bubbles: true }));
  }, text);
  // Click send button (more reliable than Enter key)
  await page.locator('[aria-label="Send a message"]').click();
}
```

**Bot join message:** On joining, the bot sends: "LiveTranslate bot active. Pin my video for subtitles. Type /help for commands."

**Google Meet selector strategy:**
- Primary: `aria-label` selectors (survive most DOM restructures)
- Secondary: `button:has-text(...)` (works across DOM changes, breaks with locale)
- All selectors live in one `selectors.ts` file — single source of truth, only file that needs updating when Meet changes

**Command Set:**

| Command | Effect | Config field |
|---------|--------|-------------|
| `/lang zh` | Translate auto-detect → Chinese | `source_lang="auto"`, `target_lang="zh"` |
| `/lang zh-en` | Translate Chinese → English | `source_lang="zh"`, `target_lang="en"` |
| `/font up` / `/font down` | Adjust font size by +/-4 | `font_size` |
| `/font 24` | Set exact font size | `font_size` |
| `/mode subtitle` / `split` / `interpreter` | Change display layout | `display_mode` |
| `/theme dark` / `light` / `contrast` | Change visual theme | `theme` |
| `/speakers on` / `off` | Show/hide speaker names | `show_speakers` |
| `/original on` / `off` | Show/hide source language text | `show_original` |
| `/source bot` / `fireflies` | Switch caption source | `caption_source` |
| `/translate on` / `off` | Enable/disable translation | `translation_enabled` |
| `/status` | Bot replies with current config | (read-only) |
| `/help` | Bot lists available commands | (read-only) |

## Phased Implementation

### Phase 0: Validate Foundations

Before building anything new, prove the existing pieces work. Red-green TDD — write the failing test first, then fix/implement until green.

**Prerequisites (before Phase 0 tests):**
- Decide: is `google_meet_automation.py` (Python/Selenium) legacy or active? **Decision: legacy.** The TypeScript `GoogleMeetBot.ts` is canonical. Python Selenium code is reference only.
- Validate macOS virtual camera: install OBS, activate virtual camera once, verify `pyvirtualcam` can write frames. **This is a manual prerequisite.**
- Merge the two `WebcamConfig` classes into one canonical Pydantic model
- Unify `DisplayMode` enums across codebase
- Rename `models/bot.py:TranslationConfig` to `TranslationDisplayConfig`

| # | Validate | Test Type | Pass Criteria |
|---|----------|-----------|---------------|
| 0a | Fix `test_translation_observability.py` import error | Unit | Import resolves, test passes |
| 0b | Fix `test_sustained_detector.py` assertion | Unit | Assertion matches current detector behavior |
| 0c | PIL frame rendering (all 5 modes) at 1280x720 | Integration | Un-skip existing test, frames render without error |
| 0d | CJK font rendering in PIL | Integration | Chinese/Japanese/Korean characters render (not tofu boxes) |
| 0e | `pyvirtualcam` device creation with `PixelFormat.RGB` | Integration | Write RGB frames to virtual camera, verify in OBS/Photo Booth on macOS |
| 0f | CaptionBuffer → WebSocket delivery | Integration | Push caption, browser client receives event with `expires_at` |
| 0g | Fireflies adapter pipeline | Integration | Connect session, captions flow through to CaptionBuffer |
| 0h | Bot joins Meet WITH camera (manual first) | E2E (manual) | Bot appears in Google Meet with video tile showing test frames |
| 0i | Meet chat DOM polling access | E2E (manual) | Polling-based reader captures chat messages from Meet DOM |
| 0j | Docker build and run (without v4l2loopback-dkms) | Docker | `docker build` succeeds, container starts, health check passes |
| 0k | Remove `--use-fake-device-for-media-stream` from all configs | Code cleanup | Flag removed, bot uses real virtual camera device |
| 0l | Clean up `caption_processor.py` TODOs and dataclass→Pydantic | Code cleanup | No TODO comments, Pydantic models throughout |

### Phase 1: Wire Modular Core

1. **MeetingSessionConfig** — Plain class with `update(**changes)` + threading lock + subscriber notification
2. **CaptionSourceAdapter protocol** — Two-layer: source adapter (lifecycle) wraps chunk adapter (data transform)
3. **CaptionBuffer subscriber extension** — Multi-subscriber replaces single-callback slots (backward compat via auto-subscribe)
4. **Shared theme/color JSON** — Canonical palette and theme definitions consumed by both renderers
5. **PILVirtualCamRenderer** — Wire VirtualWebcamManager → pyvirtualcam with frame-paced timer, dirty-flag, config snapshot per frame
6. **Config-driven source routing** — Switching `caption_source` stops/starts the correct adapter
7. **WebSocketRenderer formalization** — Existing WebSocket broadcast conforms to SubtitleRenderer protocol

### Phase 2: Chat Commands & Live Control

1. **Chat DOM poller** (TypeScript) — Opens chat panel, polls every 500ms, detects /commands
2. **CommandParser** (TypeScript) — Pure function, full command set, TDD
3. **WebSocket command protocol** — Bot sends `chat_command`, orchestration sends `chat_response` + `config_changed`
4. **Config mutation dispatch** (Python) — Commands → `MeetingSessionConfig.update()` → change events
5. **ChatResponder** (TypeScript) — Types response into contenteditable div, clicks send button
6. **Selector resilience** — All Meet selectors in single `selectors.ts`, aria-label primary strategy
7. **SvelteKit overlay config sync** — Overlay page receives config changes via WebSocket, updates display reactively

### Phase 3: Polish & Edge Cases

1. **Graceful source switching** — Clean handoff when switching bot_audio ↔ fireflies mid-meeting
2. **Font/CJK improvements** — Line breaking, text wrapping for mixed-script subtitles
3. **Error resilience** — Source drops, reconnection, pyvirtualcam crash recovery, lobby state detection
4. **Export integration** — SRT/VTT export from meeting session with translations
5. **Meet caption scraping** (`MeetCaptionsAdapter`) — DOM scrape Google Meet's built-in CC as a future source option

## Testing Strategy

### Automated Tests (Red-Green TDD)

Every feature starts with a failing test. No implementation code is written before the test exists.

**Unit tests:**
- CommandParser (TypeScript): exhaustive coverage of every command variant, edge cases. Property-based tests with fast-check for Unicode, long strings, embedded newlines
- MeetingSessionConfig reactivity: mutate field, assert subscribers notified with correct field/value. Thread safety: concurrent update + read
- CaptionSourceAdapter: adapter start/stop lifecycle, event emission

**Integration tests:**
- CaptionBuffer → PILVirtualCamRenderer: push caption, assert frame contains expected text
- CaptionBuffer → WebSocketRenderer: push caption, assert WebSocket client receives JSON with `expires_at`
- Config change → both renderers: mutate config, assert both received the change
- Source routing: change `caption_source`, assert old adapter stopped and new one started
- PIL rendering: snapshot tests for all 5 display modes, all 5 themes, CJK text, at 1280x720
- Fireflies adapter: connect to real Fireflies session, verify captions arrive
- WebSocket command protocol: bot sends command, orchestration responds with config_changed

**E2E automated (Playwright):**
- Bot joins Meet, virtual camera shows test frames
- Caption pushed → subtitle appears on bot's camera feed
- Chat command sent → display changes
- Source switch command → captions continue from new source

### Docker Testing

- `docker build` for bot container succeeds (WITHOUT `v4l2loopback-dkms` — removed from Dockerfile)
- Container starts with `--device /dev/videoN` passthrough (host must have v4l2loopback loaded)
- Health check passes
- PIL rendering works inside container (system fonts available, CJK fonts installed)

### Manual Testing (Point-and-Click)

Automated tests prove the code works. Manual tests prove the *product* works.

**Manual test protocol — real meeting, real humans:**

1. **Bot join test:** Start bot, have it join a real Google Meet. Verify it appears with a video tile (not "camera off"). Verify the video shows subtitle overlay (even if just "Waiting for transcription..."). Verify join message appears in chat.

2. **Audio → subtitles test:** Speak in the meeting. Verify transcription appears on the bot's camera within acceptable latency (~150ms end-to-end). Verify translation appears. Watch for: timing, readability, font size, speaker attribution accuracy.

3. **Chat command test:** Type `/lang zh` in meeting chat. Verify bot confirms in chat. Verify subtitles switch to Chinese translation. Test each command: `/font up`, `/mode split`, `/speakers off`, `/original on`, `/source fireflies`, `/theme light`.

4. **Source switch test:** Start with bot audio capture. Type `/source fireflies` in chat. Verify captions switch to Fireflies feed. Type `/source bot` to switch back. Verify no gap or crash.

5. **Multi-participant test:** Two people in the meeting, different languages. Interpreter mode shows both languages. Verify speaker colors are distinct and consistent across virtual camera AND browser overlay. Verify speaker names appear when enabled.

6. **OBS overlay test:** Open SvelteKit overlay page in OBS as browser source. Verify captions appear. Change config via chat commands in the meeting — verify OBS overlay updates too (same config drives both renderers).

7. **Endurance test:** Leave bot in a 30+ minute meeting. Watch for: memory leaks, frame rate degradation, caption buffer overflow, font rendering artifacts, GC stalls.

8. **Agent-browser test:** Use agent-browser to script interactions: join meeting, send chat commands, verify DOM state changes. Bridges the gap between fully manual and fully automated.

9. **Pin test:** Verify participant can pin the bot's video tile. Verify subtitle text is readable at full-size pinned view.

### Test Environments

| Environment | Purpose | Virtual Camera |
|-------------|---------|---------------|
| Local (macOS) | Development | OBS Virtual Camera (OBS installed, camera extension activated) |
| Docker (Linux) | CI and deployment | v4l2loopback (host kernel module + `--device` passthrough) |
| Real Google Meet | Manual and agent-browser E2E | macOS for dev, Linux Docker for CI |

## Files to Create/Modify

### New Files (TypeScript — meeting-bot-service)

| File | Purpose |
|------|---------|
| `modules/meeting-bot-service/src/chat/chat_poller.ts` | Polls Meet chat DOM every 500ms, detects new messages |
| `modules/meeting-bot-service/src/chat/command_parser.ts` | Pure function: parse /slash commands from chat text |
| `modules/meeting-bot-service/src/chat/chat_responder.ts` | Types bot responses into Meet's contenteditable chat input |
| `modules/meeting-bot-service/src/chat/selectors.ts` | Single source of truth for all Google Meet DOM selectors |
| `modules/meeting-bot-service/src/lib/chromium.ts` | Browser context creation with correct Chrome flags (NEW file, not modify) |

### New Files (Python — orchestration-service)

| File | Purpose |
|------|---------|
| `modules/orchestration-service/src/services/meeting_session_config.py` | `MeetingSessionConfig` — plain class, `update()`, thread-safe, subscribers |
| `modules/orchestration-service/src/services/pipeline/adapters/source_adapter.py` | `CaptionSourceAdapter` protocol + `BotAudioCaptionSource`, `FirefliesCaptionSource` |
| `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py` | Wires VirtualWebcamManager → pyvirtualcam, frame-paced, dirty-flag |
| `modules/shared/src/livetranslate_common/theme.py` (or `.json`) | Canonical speaker colors, theme definitions, display mode enum |

### New Test Files

| File | Purpose |
|------|---------|
| `modules/meeting-bot-service/tests/chat/command_parser.test.ts` | CommandParser unit tests (including property-based with fast-check) |
| `modules/orchestration-service/tests/test_meeting_session_config.py` | Config reactivity + thread safety tests |
| `modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py` | PIL → pyvirtualcam → device tests |
| `modules/orchestration-service/tests/integration/test_caption_routing.py` | Source routing integration tests |
| `modules/orchestration-service/tests/e2e/test_bot_meeting_subtitles.py` | Full E2E: bot joins meet, renders subtitles |

### Modified Files

| File | Change |
|------|--------|
| `modules/orchestration-service/src/bot/virtual_webcam.py` | Conform to SubtitleRenderer protocol, config snapshot per frame, 1280x720 default, frame-paced timer |
| `modules/orchestration-service/src/models/bot.py` | Merge two WebcamConfig classes, rename TranslationConfig → TranslationDisplayConfig |
| `modules/orchestration-service/src/services/caption_buffer.py` | Multi-subscriber replaces single-callback, include `expires_at` in events |
| `modules/orchestration-service/src/bot/caption_processor.py` | Clean up TODOs, convert dataclasses to Pydantic, remove dead polling loop |
| `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts` | Join WITH camera, integrate chat poller, remove `--use-fake-device-for-media-stream` |
| `modules/bot-container/Dockerfile` | Remove `v4l2loopback-dkms`, keep only `v4l2loopback-utils` |
| `modules/orchestration-service/tests/meeting/test_translation_observability.py` | Fix broken import |
| `modules/transcription-service/tests/unit/test_sustained_detector.py` | Fix overly strict assertion |
| `modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py` | Un-skip, update to 1280x720 |
| `modules/dashboard-service/src/routes/(overlay)/captions/+page.svelte` | Accept config changes via WebSocket |
| `modules/dashboard-service/src/lib/stores/loopback.svelte.ts` | Use canonical speaker colors from shared theme |

## Resolved Open Questions

1. **macOS virtual camera:** OBS must be installed and the camera extension activated once. OBS does NOT need to be running afterward. `pyvirtualcam` registers directly with CoreMediaIO. **Validated as Phase 0 prerequisite.**

2. **Docker virtual camera:** v4l2loopback is a host-level kernel module. Container needs `--device /dev/videoN` passthrough. `v4l2loopback-dkms` removed from Dockerfile (cannot compile in container). Host setup is a deployment prerequisite.

3. **Google Meet chat DOM stability:** Polling (not MutationObserver) is more reliable. Primary selectors: `aria-label` attributes. All selectors in single `selectors.ts` file. Daily selector validation test recommended for CI.

4. **Fireflies real-time latency:** Must be measured in Phase 0g. If latency is >2s, bot audio capture is preferred for live subtitles. Fireflies remains useful for post-meeting transcript import.

## Remaining Open Questions

1. **macOS device selection:** When both FaceTime camera and OBS Virtual Camera are visible, does Chromium auto-select the virtual camera, or do we need explicit `enumerateDevices()` + `getUserMedia({deviceId: {exact: id}})` via `page.evaluate()`?

2. **`createScriptProcessor` deprecation:** The bot uses deprecated `createScriptProcessor` for audio capture. The dashboard already has an `AudioWorklet` implementation. Should we port the bot to use AudioWorklet in this work, or defer?
