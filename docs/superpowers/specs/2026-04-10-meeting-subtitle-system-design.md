# Meeting Subtitle System — Design Spec

**Date:** 2026-04-10
**Status:** Draft
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

## Architecture

### Three Independent Layers

```
CAPTION SOURCES          PROCESSING CORE              RENDERERS
┌─────────────────┐    ┌─────────────────────┐    ┌──────────────────────┐
│ BotAudioAdapter  │───→│                     │───→│ PILVirtualCamRenderer│
│ FirefliesAdapter │───→│   CaptionBuffer     │───→│ WebSocketRenderer    │
│ MeetCaptionsAdpt │───→│   TranslationPipe   │───→│ DashboardViews       │
│ ChatCommandInput │───→│   CommandParser      │───→│ ChatResponder        │
└─────────────────┘    │   MeetingSessionConfig│    └──────────────────────┘
                       └─────────────────────┘
```

Each layer communicates through defined protocols. No layer knows the internals of any other.

### MeetingSessionConfig — Single Control Plane

Every runtime decision reads from this config. Every mutation (chat command, API call, dashboard setting) writes to it. Both renderers subscribe to changes and react immediately.

```python
class MeetingSessionConfig(BaseModel):
    """Single source of truth for meeting subtitle behavior.
    
    Every field is live-switchable mid-meeting. Mutations trigger
    change events that all subscribers (renderers, adapters) react to.
    """
    # --- Source Routing ---
    caption_source: Literal["bot_audio", "fireflies", "meet_captions"] = "bot_audio"
    translation_enabled: bool = True
    source_lang: str = "auto"          # "auto" = detect, or BCP-47 code
    target_lang: str = "en"

    # --- Display ---
    display_mode: Literal["subtitle", "split", "interpreter"] = "subtitle"
    theme: Literal["dark", "light", "high_contrast", "minimal", "corporate"] = "dark"
    font_size: int = 24
    show_speakers: bool = True
    show_original: bool = False

    # --- Runtime ---
    session_id: str
    bot_id: str | None = None
```

**Reactivity:** The config is observable. On any field change, a `ConfigChanged` event fires with the field name and new value. Subscribers (renderers, adapters) handle only the fields they care about. Implementation: callback list on `__setattr__`, or a lightweight pub/sub.

**Mutation sources:**
- Chat commands (`/lang zh`, `/font up`, `/source fireflies`)
- REST API endpoints (dashboard settings page)
- Programmatic (bot startup defaults)

### Caption Source Adapters

Each input source implements:

```python
class CaptionSourceAdapter(Protocol):
    async def start(self, config: MeetingSessionConfig) -> None: ...
    async def stop(self) -> None: ...
    on_caption: Callable[[CaptionEvent], Awaitable[None]]
```

**Adapters:**

| Adapter | Input | Status |
|---------|-------|--------|
| `BotAudioAdapter` | Bot captures meeting audio → Whisper transcription → segments | Wrap existing audio WebSocket pipeline |
| `FirefliesAdapter` | Fireflies API → chunks via existing `FirefliesChunkAdapter` | Wrap existing adapter |
| `MeetCaptionsAdapter` | DOM scrape Google Meet's built-in closed captions | Future — Phase 3+ |

When `config.caption_source` changes, the active adapter is stopped and the new one started. The CaptionBuffer receives events identically regardless of source.

### CaptionBuffer (Existing — Minor Extension)

The existing `CaptionBuffer` already handles speaker color assignment, time-based expiration, and event broadcasting. 

**Extension:** Add a generic subscriber mechanism so the PIL renderer can listen alongside the existing WebSocket broadcast:

```python
class CaptionBuffer:
    def subscribe(self, callback: Callable[[CaptionEvent], Awaitable[None]]) -> None: ...
    def unsubscribe(self, callback: ...) -> None: ...
```

Events: `caption_added`, `caption_updated`, `caption_expired`, `session_cleared`

### Renderer Protocol

Both renderers implement:

```python
class SubtitleRenderer(Protocol):
    async def on_config_changed(self, config: MeetingSessionConfig) -> None: ...
    async def on_caption_event(self, event: CaptionEvent) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
```

**PILVirtualCamRenderer:**
- Reads caption events from CaptionBuffer subscriber
- Reads display settings from MeetingSessionConfig
- Renders frames using existing `VirtualWebcamManager` (5 display modes, 5 themes, speaker colors)
- Writes frames to `pyvirtualcam` device at 30fps
- Playwright sees virtual camera as default device, Google Meet shows it as bot's webcam

**WebSocketRenderer:**
- Reads same caption events
- Pushes JSON to connected SvelteKit overlay and dashboard clients
- Config changes pushed via WebSocket so browser renderers react immediately
- Already largely exists — formalize the interface

### Virtual Camera Pipeline

```
VirtualWebcamManager (PIL)
    → renders frame (Pillow Image)
    → pyvirtualcam.Camera.send(frame)
    → OS virtual camera device
        macOS: OBS Virtual Camera
        Linux: v4l2loopback kernel module (already in bot Dockerfile)
    → Playwright sees as default camera
    → Google Meet shows as bot's video tile
```

The bot joins WITH camera enabled (not "Continue without microphone and camera"). Playwright launches with `--use-fake-ui-for-media-stream` to auto-grant camera permission. The virtual camera device is the only video input available to the browser, so it's selected by default.

### Chat Command System

```
Meet Chat DOM
    → MutationObserver (injected via page.exposeFunction)
    → Python callback receives message text
    → CommandParser (pure function, stateless)
    → MeetingSessionConfig mutation
    → ConfigChanged event
    → All renderers react
    → ChatResponder types confirmation in Meet chat
```

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

**Parser:** Pure function. Returns a typed `CommandResult` or `None` for non-command messages. Unknown commands get a help nudge in chat.

**Smart defaults:** `/lang zh` implies `source_lang="auto"` (detect what people are speaking) and `target_lang="zh"`. Only use explicit pairs (`/lang zh-en`) when auto-detect isn't desired.

## Phased Implementation

### Phase 0: Validate Foundations

Before building anything new, prove the existing pieces work. Red-green TDD — write the failing test first, then fix/implement until green.

| # | Validate | Test Type | Pass Criteria |
|---|----------|-----------|---------------|
| 0a | Fix `test_translation_observability.py` import error | Unit | Import resolves, test passes |
| 0b | Fix `test_sustained_detector.py` assertion | Unit | Assertion matches current detector behavior |
| 0c | PIL frame rendering (all 5 modes) | Integration | Un-skip existing test, frames render without error |
| 0d | CJK font rendering in PIL | Integration | Chinese/Japanese/Korean characters render (not tofu boxes) |
| 0e | `pyvirtualcam` device creation | Integration | Write frames to virtual camera, verify in OBS/Photo Booth on macOS |
| 0f | CaptionBuffer → WebSocket delivery | Integration | Push caption, browser client receives event |
| 0g | Fireflies adapter pipeline | Integration | Connect session, captions flow through to CaptionBuffer |
| 0h | Bot joins Meet WITH camera | E2E (manual) | Bot appears in Google Meet with video tile showing test frames |
| 0i | Meet chat DOM access | E2E (manual) | Injected MutationObserver captures chat messages |
| 0j | Docker build and run | Docker | `docker build` succeeds, container starts, health check passes |

### Phase 1: Wire Modular Core

Connect validated pieces with the interfaces defined above:

1. **MeetingSessionConfig** — Pydantic model with observable change events
2. **CaptionSourceAdapter protocol** — Wrap existing `BotAudioAdapter` and `FirefliesAdapter`
3. **CaptionBuffer subscriber extension** — Generic subscribe/unsubscribe
4. **PILVirtualCamRenderer** — Wire VirtualWebcamManager → pyvirtualcam → virtual camera device
5. **Config-driven source routing** — Switching `caption_source` stops/starts the correct adapter
6. **WebSocketRenderer formalization** — Existing WebSocket broadcast conforms to SubtitleRenderer protocol

### Phase 2: Chat Commands & Live Control

1. **Chat DOM observer** — Playwright injects MutationObserver on Meet's chat panel
2. **CommandParser** — Pure function, full command set, TDD
3. **Config mutation dispatch** — Commands → MeetingSessionConfig → change events
4. **ChatResponder** — Bot types confirmations in Meet chat via Playwright
5. **SvelteKit overlay config sync** — Overlay page receives config changes via WebSocket, updates display reactively

### Phase 3: Polish & Edge Cases

1. **Graceful source switching** — Clean handoff when switching bot_audio ↔ fireflies mid-meeting
2. **Font/CJK improvements** — Line breaking, text wrapping for mixed-script subtitles
3. **Error resilience** — Source drops, reconnection, fallback behavior
4. **Export integration** — SRT/VTT export from meeting session with translations

## Testing Strategy

### Automated Tests (Red-Green TDD)

Every feature starts with a failing test. No implementation code is written before the test exists.

**Unit tests:**
- CommandParser: exhaustive coverage of every command variant, edge cases (typos, missing args, unknown commands, empty string, non-command messages)
- MeetingSessionConfig reactivity: mutate field, assert subscribers notified with correct field/value
- CaptionSourceAdapter: adapter start/stop lifecycle, event emission

**Integration tests:**
- CaptionBuffer → PILVirtualCamRenderer: push caption, assert frame contains expected text
- CaptionBuffer → WebSocketRenderer: push caption, assert WebSocket client receives JSON
- Config change → both renderers: mutate config, assert both received the change
- Source routing: change `caption_source`, assert old adapter stopped and new one started
- PIL rendering: snapshot tests for all 5 display modes, all 5 themes, CJK text
- Fireflies adapter: connect to real Fireflies session, verify captions arrive

**E2E automated (Playwright):**
- Bot joins Meet, virtual camera shows test frames
- Caption pushed → subtitle appears on bot's camera feed
- Chat command sent → display changes
- Source switch command → captions continue from new source

### Docker Testing

- `docker build` for bot container succeeds with all dependencies (v4l2loopback, pyvirtualcam, Playwright, system fonts)
- Container starts, health check passes
- Virtual camera device creation works inside container
- PIL rendering works inside container (system fonts available)

### Manual Testing (Point-and-Click)

Automated tests prove the code works. Manual tests prove the *product* works.

**Manual test protocol — real meeting, real humans:**

1. **Bot join test:** Start bot, have it join a real Google Meet. Verify it appears with a video tile. Verify the video shows subtitle overlay (even if just "Waiting for transcription...").

2. **Audio → subtitles test:** Speak in the meeting. Verify transcription appears on the bot's camera within acceptable latency. Verify translation appears. Watch for: timing, readability, font size, speaker attribution accuracy.

3. **Chat command test:** Type `/lang zh` in meeting chat. Verify bot confirms in chat. Verify subtitles switch to Chinese translation. Test each command: `/font up`, `/mode split`, `/speakers off`, `/original on`, `/source fireflies`, `/theme light`.

4. **Source switch test:** Start with bot audio capture. Type `/source fireflies` in chat. Verify captions switch to Fireflies feed. Type `/source bot` to switch back. Verify no gap or crash.

5. **Multi-participant test:** Two people in the meeting, different languages. Interpreter mode shows both languages. Verify speaker colors are distinct. Verify speaker names appear when enabled.

6. **OBS overlay test:** Open SvelteKit overlay page in OBS as browser source. Verify captions appear. Change config via chat commands in the meeting — verify OBS overlay updates too.

7. **Endurance test:** Leave bot in a 30+ minute meeting. Watch for: memory leaks, frame rate degradation, caption buffer overflow, font rendering artifacts.

8. **Agent-browser test:** Use agent-browser to script interactions: join meeting, send chat commands, verify DOM state changes. Bridges the gap between fully manual and fully automated.

### Test Environments

| Environment | Purpose |
|-------------|---------|
| Local (macOS) | Development. OBS Virtual Camera for pyvirtualcam. |
| Docker (Linux) | CI and deployment. v4l2loopback for virtual camera. |
| Real Google Meet | Manual and agent-browser E2E tests. |

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `modules/orchestration-service/src/bot/meeting_session_config.py` | `MeetingSessionConfig` with observable change events |
| `modules/orchestration-service/src/bot/caption_source_adapter.py` | Protocol + `BotAudioAdapter`, `FirefliesAdapter` |
| `modules/orchestration-service/src/bot/command_parser.py` | Stateless command parser, pure function |
| `modules/orchestration-service/src/bot/chat_observer.py` | Playwright DOM observer for Meet chat |
| `modules/orchestration-service/src/bot/chat_responder.py` | Bot types replies in Meet chat |
| `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py` | Wires VirtualWebcamManager → pyvirtualcam |
| `modules/orchestration-service/tests/test_command_parser.py` | CommandParser unit tests |
| `modules/orchestration-service/tests/test_meeting_session_config.py` | Config reactivity tests |
| `modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py` | PIL → pyvirtualcam → device tests |
| `modules/orchestration-service/tests/integration/test_caption_routing.py` | Source routing integration tests |
| `modules/orchestration-service/tests/e2e/test_bot_meeting_subtitles.py` | Full E2E: bot joins meet, renders subtitles |

### Modified Files

| File | Change |
|------|--------|
| `modules/orchestration-service/src/bot/virtual_webcam.py` | Conform to SubtitleRenderer protocol, accept config changes |
| `modules/orchestration-service/src/services/caption_buffer.py` | Add generic subscribe/unsubscribe mechanism |
| `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts` | Join WITH camera, inject chat observer, handle commands |
| `modules/meeting-bot-service/src/lib/chromium.ts` | Ensure virtual camera device is available to browser |
| `modules/orchestration-service/tests/meeting/test_translation_observability.py` | Fix broken import |
| `modules/transcription-service/tests/unit/test_sustained_detector.py` | Fix overly strict assertion |
| `modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py` | Un-skip, make pass |
| `modules/dashboard-service/src/routes/(overlay)/captions/+page.svelte` | Accept config changes via WebSocket |

## Open Questions

1. **macOS virtual camera:** Does `pyvirtualcam` work with OBS Virtual Camera on macOS without OBS running? Or does OBS need to be installed and its virtual camera plugin activated?
2. **Docker virtual camera:** Does `v4l2loopback` work inside a Docker container, or does it need host-level kernel module loading with `--device` passthrough?
3. **Google Meet chat DOM stability:** Google updates their DOM frequently. How brittle will the chat MutationObserver be? Do we need a selector resilience layer?
4. **Fireflies real-time latency:** What's the actual latency from speech to Fireflies caption? If it's too high, bot audio capture may always be preferred for live subtitles.
