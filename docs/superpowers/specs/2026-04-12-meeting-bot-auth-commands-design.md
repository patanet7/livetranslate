# Meeting Bot Service — Full System Design

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Production-ready meeting bot that spawns via Docker, authenticates to Google Meet, captures audio for transcription, displays translated subtitles via virtual webcam, and responds to chat commands.

**Architecture:** Docker-based bot containers spawned by orchestration, persistent browser profile for Google auth, WebSocket for bidirectional communication, virtual webcam for subtitle overlay.

**Tech Stack:** Playwright (TypeScript), FastAPI (Python), Docker, pyvirtualcam, WebSocket

---

## Problem Statement

The meeting-bot-service currently:
- ✅ Builds and runs (TypeScript fixed)
- ✅ Docker image exists (`livetranslate-bot:latest`)
- ❌ Gets blocked by Google's bot detection (redirected to homepage)
- ❌ Uses outdated selectors (Google Meet UI changed)
- ❌ Missing integration with orchestration (audio streaming, commands)

Commercial bots (Fireflies, Recall.ai) solve bot detection by:
1. Signing in with a real Google account once
2. Saving the browser session/cookies  
3. Reusing that authenticated session for all future joins

This spec implements the full bot lifecycle: spawn → auth → join → stream → command → leave.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       meeting-bot-service                             │
├──────────────────────────────────────────────────────────────────────┤
│  HTTP API (:5005)                                                     │
│  ├── POST /api/bot/join         → spawn bot, connect to orchestration│
│  ├── POST /api/auth/setup       → launch headed browser for sign-in  │
│  ├── GET  /api/auth/status      → check if authenticated             │
│  └── GET  /api/bot/screenshot   → debug view                         │
├──────────────────────────────────────────────────────────────────────┤
│  Persistent Auth                                                      │
│  └── /data/chrome-profile/      → cookies, localStorage, session     │
├──────────────────────────────────────────────────────────────────────┤
│  GoogleMeetBot                                                        │
│  ├── Launches with persistent profile (authenticated)                │
│  ├── Updated aria-label selectors (2026)                             │
│  ├── AudioStreamer        → captures audio → sends to orchestration  │
│  ├── ChatPoller           → watches Meet chat for commands           │
│  └── ChatResponder        → posts responses back to Meet chat        │
└───────────────────────┬──────────────────────────────────────────────┘
                        │ WebSocket
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     orchestration-service                             │
├──────────────────────────────────────────────────────────────────────┤
│  WebSocket /ws/bot/:botId                                             │
│  ├── Receives: audio chunks, chat commands, bot status               │
│  ├── Sends: control commands, translation results, config updates    │
│  └── CommandDispatcher → routes commands to handlers                 │
├──────────────────────────────────────────────────────────────────────┤
│  Command Handlers                                                     │
│  ├── /lang <code>      → change source/target language               │
│  ├── /mode <mode>      → switch interpreter/split/subtitle           │
│  ├── /stop             → leave meeting                               │
│  ├── /status           → report current state                        │
│  └── /help             → list available commands                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Full Bot Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BOT LIFECYCLE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. SPAWN          2. AUTH CHECK       3. JOIN            4. ACTIVE     │
│  ┌─────────┐       ┌─────────┐        ┌─────────┐        ┌─────────┐   │
│  │ Docker  │ ───►  │ Profile │ ───►   │ Navigate│ ───►   │ Stream  │   │
│  │ create  │       │ exists? │        │ to Meet │        │ Audio   │   │
│  └─────────┘       └────┬────┘        └─────────┘        │ Captions│   │
│                         │                                 │ Commands│   │
│                    No ──┴── Yes                          └────┬────┘   │
│                    │        │                                 │         │
│                    ▼        ▼                                 ▼         │
│              ┌─────────┐  (skip)                        5. LEAVE       │
│              │ Manual  │                                ┌─────────┐    │
│              │ Sign-in │                                │ Cleanup │    │
│              └─────────┘                                │ Report  │    │
│                                                         └─────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 0. Docker Bot Spawning (Orchestration Side)

The orchestration service spawns bot containers via `DockerBotManager`:

```python
# modules/orchestration-service/src/bot/docker_bot_manager.py

class DockerBotManager:
    async def spawn_bot(self, config: BotConfig) -> str:
        container = docker_client.containers.run(
            image="livetranslate-bot:latest",
            detach=True,
            environment={
                "MEETING_URL": config.meeting_url,
                "BOT_ID": config.connection_id,
                "ORCHESTRATION_WS_URL": config.orchestration_ws_url,
                "CHROME_PROFILE_DIR": "/data/chrome-profile",
            },
            volumes={
                "chrome-profile": {"bind": "/data/chrome-profile", "mode": "rw"},
            },
            network="livetranslate-network",
        )
        return container.id
```

**Key points:**
- Bot container mounts shared `chrome-profile` volume for persistent auth
- Environment variables configure meeting URL and orchestration endpoint
- Container joins `livetranslate-network` for internal communication

---

### 1. Authentication Flow

**One-time setup (manual):**
1. User calls `POST /api/auth/setup`
2. Bot launches HEADED Chromium with persistent profile at `/data/chrome-profile/`
3. Browser opens Google sign-in page
4. User signs in manually, completes 2FA
5. Bot detects successful auth (checks for Google avatar/account menu)
6. Browser closes, profile saved
7. Returns `{authenticated: true, account: "user@gmail.com"}`

**Subsequent joins (automatic):**
1. `POST /api/bot/join {meetingUrl, ...}`
2. Bot launches HEADLESS with same profile directory
3. Already authenticated — no sign-in needed
4. Navigates directly to Meet URL as signed-in user
5. Joins meeting without bot detection redirect

**Session refresh:** Google sessions last ~30 days. When auth expires, `/api/auth/status` returns `{authenticated: false}`, user re-runs setup.

### 2. Updated Selectors (2026)

Using stable `aria-label` patterns from Recall.ai research:

```typescript
const SELECTORS = {
  // Pre-join controls
  micOff: 'button[aria-label*="Turn off microphone"]',
  camOff: 'button[aria-label*="Turn off camera"]',
  joinButton: 'button[aria-label*="Join now"], button[aria-label*="Ask to join"]',
  
  // In-meeting
  leaveButton: 'button[aria-label*="Leave call"], button[aria-label*="Leave meeting"]',
  captionsOn: 'button[aria-label*="Turn on captions"]',
  captionsOff: 'button[aria-label*="Turn off captions"]',
  captionsRegion: '[role="region"][aria-label*="Captions"]',
  
  // Chat
  chatButton: 'button[aria-label*="Chat with everyone"]',
  chatInput: 'textarea[aria-label*="Send a message"]',
  chatMessages: '[data-message-id]',
  
  // Dismissals
  gotIt: 'button:has-text("Got it")',
  dismiss: 'button[aria-label*="Dismiss"], button[aria-label*="Close"]',
  
  // Exit detection
  leftMeeting: 'div[role="heading"]:has-text("You left the meeting")',
};
```

### 3. ChatPoller

Watches Meet chat for slash commands:

```typescript
class ChatPoller {
  private seenMessageIds: Set<string> = new Set();
  private pollInterval: number = 1000; // 1 second
  
  async poll(): Promise<ChatMessage[]> {
    const messages = await page.$$(SELECTORS.chatMessages);
    const newMessages = [];
    
    for (const msg of messages) {
      const id = await msg.getAttribute('data-message-id');
      if (!this.seenMessageIds.has(id)) {
        this.seenMessageIds.add(id);
        const text = await msg.textContent();
        if (text.startsWith('/')) {
          newMessages.push({ id, text, isCommand: true });
        }
      }
    }
    return newMessages;
  }
}
```

### 4. ChatResponder

Posts responses back to Meet chat:

```typescript
class ChatResponder {
  async send(message: string): Promise<void> {
    // Open chat panel if closed
    await page.click(SELECTORS.chatButton);
    await page.fill(SELECTORS.chatInput, message);
    await page.keyboard.press('Enter');
  }
}
```

### 5. CommandDispatcher (Orchestration)

```python
# modules/orchestration-service/src/bot/command_dispatcher.py

class CommandDispatcher:
    handlers = {
        '/lang': LangCommandHandler,
        '/mode': ModeCommandHandler,
        '/stop': StopCommandHandler,
        '/status': StatusCommandHandler,
        '/help': HelpCommandHandler,
    }
    
    async def dispatch(self, command: str, bot_session: BotSession) -> str:
        parts = command.split(maxsplit=1)
        cmd, args = parts[0], parts[1] if len(parts) > 1 else ''
        
        handler = self.handlers.get(cmd)
        if not handler:
            return f"Unknown command: {cmd}. Type /help for available commands."
        
        return await handler.execute(args, bot_session)
```

### 6. Audio Streaming (Bot → Orchestration)

Bot captures meeting audio and streams to orchestration for transcription:

```typescript
// modules/meeting-bot-service/src/audio_streaming.ts

class AudioStreamer {
  private ws: WebSocket;
  private mediaRecorder: MediaRecorder;
  
  async start(orchestrationUrl: string): Promise<void> {
    this.ws = new WebSocket(orchestrationUrl);
    
    // Capture tab audio via getDisplayMedia
    const stream = await navigator.mediaDevices.getDisplayMedia({
      audio: true,
      video: false,
    });
    
    this.mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus',
    });
    
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.ws.send(event.data); // Binary audio chunks
      }
    };
    
    this.mediaRecorder.start(100); // 100ms chunks
  }
}
```

**Data flow:**
```
Meet Audio → getDisplayMedia → MediaRecorder → WebSocket → Orchestration → Transcription
```

---

### 7. Virtual Webcam (Orchestration → Meet)

Translated subtitles rendered back to Meet via virtual webcam:

```python
# modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py

class PILVirtualCamRenderer:
    def __init__(self, width=1280, height=720):
        self.cam = pyvirtualcam.Camera(width, height, fps=30)
        self.caption_buffer = CaptionBuffer()
    
    def render_frame(self) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Render current caption
        caption = self.caption_buffer.get_current()
        if caption:
            # PIL text rendering with theme colors
            draw_text(frame, caption.text, position="bottom", theme=self.theme)
        
        return frame
    
    def update_caption(self, text: str, speaker: str):
        self.caption_buffer.add(CaptionEvent(text=text, speaker=speaker))
```

**Integration:** Bot shares its screen showing virtual webcam output, so participants see translated subtitles overlaid on a background.

---

## Commands

| Command | Arguments | Action | Response |
|---------|-----------|--------|----------|
| `/lang` | `<code>` | Set source language | "✓ Source language: Chinese" |
| `/mode` | `split\|subtitle\|interpreter` | Switch display mode | "✓ Mode: split (dual captions)" |
| `/stop` | — | Leave meeting gracefully | "👋 Leaving meeting..." |
| `/status` | — | Report current state | "🎤 Listening · zh→en · split mode" |
| `/help` | — | List available commands | Shows command list |

---

## Command Flow

```
Meeting Participant                Bot                    Orchestration
       │                            │                            │
       │ types "/lang zh" in chat   │                            │
       │ ─────────────────────────► │                            │
       │                            │ ChatPoller detects command │
       │                            │ ──────────────────────────►│
       │                            │                            │ CommandDispatcher
       │                            │                            │ routes to LangHandler
       │                            │    {cmd: "lang_changed"}   │
       │                            │ ◄────────────────────────  │
       │                            │ ChatResponder posts        │
       │  "✓ Language set to zh"    │                            │
       │ ◄───────────────────────── │                            │
```

---

## File Changes

### meeting-bot-service (TypeScript)

| File | Change |
|------|--------|
| `src/lib/chromium.ts` | Add persistent profile support via `userDataDir` option |
| `src/lib/selectors.ts` | New file — 2026 aria-label selectors for Meet UI |
| `src/chat/chat_poller.ts` | Update to use new selectors, improve reliability |
| `src/chat/chat_responder.ts` | Update to use new selectors |
| `src/bots/GoogleMeetBot.ts` | Use persistent profile, new selectors, integrate ChatPoller, AudioStreamer |
| `src/audio_streaming.ts` | Update to connect to orchestration WebSocket |
| `src/api_server.ts` | Add `/api/auth/setup` and `/api/auth/status` endpoints |
| `Dockerfile` | Volume mount for chrome profile, fix deprecations |
| `docker-compose.yml` | Add chrome-profile volume, network config |

### orchestration-service (Python)

| File | Change |
|------|--------|
| `src/bot/docker_bot_manager.py` | Update container spawn with profile volume, network |
| `src/bot/command_dispatcher.py` | New file — routes chat commands to handlers |
| `src/bot/commands/__init__.py` | New file — command handler exports |
| `src/bot/commands/lang.py` | New file — `/lang` handler |
| `src/bot/commands/mode.py` | New file — `/mode` handler |
| `src/bot/commands/stop.py` | New file — `/stop` handler |
| `src/bot/commands/status.py` | New file — `/status` handler |
| `src/bot/commands/help.py` | New file — `/help` handler |
| `src/routers/bot/bot_websocket.py` | New file — WebSocket endpoint for bot↔orchestration |
| `src/routers/bot/bot_docker_management.py` | Update to use new spawning flow |

---

## Docker Configuration

```yaml
# docker-compose.yml addition
services:
  meeting-bot:
    volumes:
      - chrome-profile:/data/chrome-profile
    environment:
      - CHROME_PROFILE_DIR=/data/chrome-profile

volumes:
  chrome-profile:
```

---

## Testing Strategy

1. **Auth flow test:** Manual — run setup, verify profile created, restart bot, verify no sign-in needed
2. **Selector tests:** Playwright test suite against live Meet (mark as `@e2e`)
3. **ChatPoller test:** Mock page with fake chat messages, verify command detection
4. **CommandDispatcher test:** Unit tests for each handler
5. **Integration test:** Full flow — bot joins, user types command, response appears in chat

---

## Success Criteria

### Core Functionality
- [ ] Bot spawns via DockerBotManager from orchestration service
- [ ] Bot joins Google Meet without being redirected to homepage (auth works)
- [ ] Auth persists across bot restarts (Docker volume mount)
- [ ] Session expiry detected and user prompted to re-auth via `/api/auth/status`

### Audio Pipeline
- [ ] Bot captures meeting audio via getDisplayMedia
- [ ] Audio streams to orchestration via WebSocket
- [ ] Transcription service receives and processes audio

### Subtitle Display
- [ ] Virtual webcam renders translated subtitles
- [ ] Bot shares screen with subtitle overlay visible to participants
- [ ] Captions update in real-time (<500ms latency)

### Chat Commands
- [ ] `/lang`, `/mode`, `/stop`, `/status`, `/help` commands work from Meet chat
- [ ] Bot responds to commands within 2 seconds
- [ ] Commands route through orchestration (not handled locally in bot)

### Operations
- [ ] Bot leaves cleanly on `/stop` or meeting end
- [ ] Container cleanup after bot exits
- [ ] Logs available for debugging

---

## Future Enhancements (Out of Scope)

- OAuth flow via dashboard (per-user accounts without manual browser step)
- Dedicated bot account (`livetranslate-bot@gmail.com`) for cleaner UX
- Google Meet Media API integration (when Developer Preview opens)
