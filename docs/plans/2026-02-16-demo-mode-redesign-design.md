# Demo Mode Redesign

## Context

The initial demo mode (PRs #47-52) created a mock Fireflies server and embedded a `captions.html` iframe in a separate panel. Issues:
- Captions iframe was blank (async callback bug, now fixed in #52)
- The dashboard already has a richer "Live Feed" tab with original+translation side-by-side
- No way to see translations without running Ollama/translation service
- Separate demo panel was disconnected from the existing dashboard tabs

## Design

### UX Flow

1. User clicks **"Launch Demo"** button in header
2. Backend starts mock Fireflies server + creates pipeline session
3. Frontend **auto-switches to Live Feed tab**
4. **Auto-selects demo session** in session dropdown, **auto-connects WebSocket**
5. Original transcript + translations start flowing immediately
6. Demo badge + Stop button appear inline in Live Feed tab header
7. Old demo panel + iframe removed entirely

### Demo Content Modes

Dropdown next to Launch Demo button:

| Mode | Behavior |
|------|----------|
| **Live Passthrough** | Full pipeline runs — text passes through translator as-is (proves pipeline works without Ollama) |
| **Pre-translated** | Backend injects captions directly into CaptionBuffer with hardcoded English→Spanish translations, bypassing the translator entirely |

### Pre-translated Implementation

`demo_server.py` gets a `TRANSLATED_TEMPLATES` dict mapping each English conversation template to its Spanish translation. In pretranslated mode, when chunks arrive at the coordinator, the demo manager directly calls `caption_buffer.add_caption()` with both original and translated text — the rolling window translator is bypassed.

Alternatively (simpler): the demo start endpoint, after creating the session, spawns a background task that feeds pre-translated captions directly into the session's CaptionBuffer on a timer, independent of the mock Socket.IO flow. This avoids touching the coordinator at all.

### Files to Change

- `static/fireflies-dashboard.html` — Remove demo panel/iframe, add demo controls to Live Feed tab, auto-switch + auto-connect logic
- `src/services/demo_server.py` — Add `TRANSLATED_TEMPLATES` dict
- `src/services/demo_manager.py` — Add `mode` parameter, add pretranslated caption injection task
- `src/routers/fireflies.py` — Pass mode through demo start endpoint

## Decision Log

- **Live Feed over iframe**: The Live Feed tab already exists with richer UI (speaker names, timestamps, export). No reason to duplicate in an iframe.
- **Direct caption injection for pretranslated**: Simpler than modifying the mock server protocol or coordinator. The CaptionBuffer + WebSocket broadcast path is already proven.
- **Two modes with toggle**: Passthrough proves the pipeline works; pretranslated shows a polished demo to stakeholders.
