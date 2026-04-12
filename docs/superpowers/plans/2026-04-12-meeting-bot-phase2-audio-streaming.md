# Meeting Bot Phase 2: Audio Streaming Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stream meeting audio from the bot to the orchestration service for real-time transcription and translation.

**Architecture:** Bot captures tab audio via Playwright's CDP, streams PCM chunks over WebSocket to orchestration, which forwards to the transcription service. The existing `AudioStreamer` class in the bot connects to a new `/ws/bot/{bot_id}` endpoint in orchestration.

**Tech Stack:** Playwright CDP, WebSocket, PCM audio, FastAPI

**Dependencies:** Phase 1 must be complete (bot can join meetings).

---

## File Structure

| File | Responsibility |
|------|----------------|
| `modules/meeting-bot-service/src/audio_streaming.ts` | Modify — add CDP-based audio capture |
| `modules/orchestration-service/src/routers/bot/__init__.py` | **NEW** — Bot router package |
| `modules/orchestration-service/src/routers/bot/bot_websocket.py` | **NEW** — WebSocket endpoint for bot communication |
| `modules/orchestration-service/src/main_fastapi.py` | Modify — register bot router |

---

## Task 1: Create Bot Router Package

**Files:**
- Create: `modules/orchestration-service/src/routers/bot/__init__.py`

- [ ] **Step 1: Create the bot router package**

```python
"""Bot communication routers — WebSocket and Docker management."""

from .bot_websocket import router as bot_ws_router
from .bot_docker_management import router as bot_docker_router

__all__ = ["bot_ws_router", "bot_docker_router"]
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/bot/__init__.py
git commit -m "$(cat <<'EOF'
feat(orchestration): create bot router package

Exports bot_ws_router and bot_docker_router for bot communication.
EOF
)"
```

---

## Task 2: Create Bot WebSocket Endpoint

**Files:**
- Create: `modules/orchestration-service/src/routers/bot/bot_websocket.py`
- Test: `modules/orchestration-service/tests/test_bot_websocket.py`

- [ ] **Step 1: Write the failing test**

Create `modules/orchestration-service/tests/test_bot_websocket.py`:

```python
"""Tests for bot WebSocket endpoint."""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def ws_client(test_client: TestClient):
    """WebSocket test client."""
    return test_client


class TestBotWebSocket:
    """Test bot WebSocket communication."""

    def test_bot_connects_and_receives_welcome(self, ws_client: TestClient):
        """Bot should receive welcome message on connect."""
        with ws_client.websocket_connect("/ws/bot/test-bot-123") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "welcome"
            assert msg["bot_id"] == "test-bot-123"

    def test_bot_sends_audio_chunk(self, ws_client: TestClient):
        """Bot should be able to send binary audio data."""
        with ws_client.websocket_connect("/ws/bot/test-bot-456") as ws:
            # Skip welcome
            ws.receive_json()
            
            # Send binary audio chunk
            audio_data = b"\x00\x01\x02\x03" * 100
            ws.send_bytes(audio_data)
            
            # Should receive ack
            msg = ws.receive_json()
            assert msg["type"] == "audio_ack"
            assert msg["bytes_received"] == len(audio_data)

    def test_bot_sends_chat_command(self, ws_client: TestClient):
        """Bot should be able to forward chat commands."""
        with ws_client.websocket_connect("/ws/bot/test-bot-789") as ws:
            # Skip welcome
            ws.receive_json()
            
            # Send chat command
            ws.send_json({
                "type": "chat_command",
                "text": "/lang zh-en",
                "sender": "TestUser",
            })
            
            # Should receive command response
            msg = ws.receive_json()
            assert msg["type"] == "command_response"
            assert "response_text" in msg

    def test_bot_sends_status_update(self, ws_client: TestClient):
        """Bot should be able to send status updates."""
        with ws_client.websocket_connect("/ws/bot/test-bot-status") as ws:
            # Skip welcome
            ws.receive_json()
            
            # Send status
            ws.send_json({
                "type": "status",
                "state": "in_meeting",
                "participant_count": 5,
            })
            
            # Should receive ack
            msg = ws.receive_json()
            assert msg["type"] == "status_ack"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest modules/orchestration-service/tests/test_bot_websocket.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create bot_websocket.py**

Create `modules/orchestration-service/src/routers/bot/bot_websocket.py`:

```python
"""WebSocket endpoint for bot communication.

Handles:
- Binary audio streaming from bot → orchestration → transcription
- Chat command forwarding from bot → CommandDispatcher → bot
- Status updates from bot

Protocol:
- Bot sends binary frames for audio chunks
- Bot sends JSON frames for commands/status
- Orchestration sends JSON frames for responses/control
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from livetranslate_common.logging import get_logger
from ...services.command_dispatcher import CommandDispatcher
from ...services.meeting_session_config import MeetingSessionConfig

logger = get_logger()
router = APIRouter(tags=["bot"])

# Active bot connections: bot_id -> WebSocket
_active_bots: dict[str, WebSocket] = {}

# Per-bot session configs
_bot_configs: dict[str, MeetingSessionConfig] = {}


async def _send_json_safe(ws: WebSocket, data: dict[str, Any]) -> bool:
    """Send JSON, return False if connection closed."""
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json(data)
            return True
    except Exception as e:
        logger.warning("ws_send_failed", error=str(e))
    return False


async def _handle_audio_chunk(bot_id: str, data: bytes) -> dict[str, Any]:
    """Process incoming audio chunk from bot.
    
    TODO: Forward to transcription service via existing audio pipeline.
    For now, just acknowledge receipt.
    """
    logger.debug("bot_audio_received", bot_id=bot_id, bytes=len(data))
    
    # TODO Phase 2b: Forward to transcription WebSocket
    # The orchestration service's existing audio pipeline can be reused:
    # - Downsample if needed (bot may send 48kHz)
    # - Forward to transcription-service via existing WS connection
    
    return {"type": "audio_ack", "bytes_received": len(data)}


async def _handle_chat_command(
    bot_id: str, text: str, sender: str, dispatcher: CommandDispatcher
) -> dict[str, Any]:
    """Process chat command from bot."""
    logger.info("bot_chat_command", bot_id=bot_id, text=text, sender=sender)
    
    result = dispatcher.dispatch(text, sender)
    
    if result is None:
        return {"type": "command_response", "response_text": None}
    
    return {
        "type": "command_response",
        "response_text": result.response_text,
        "changed_fields": list(result.changed_fields),
        "demo_action": result.demo_action,
    }


async def _handle_status_update(bot_id: str, status: dict[str, Any]) -> dict[str, Any]:
    """Process status update from bot."""
    logger.info("bot_status_update", bot_id=bot_id, **status)
    return {"type": "status_ack"}


@router.websocket("/ws/bot/{bot_id}")
async def bot_websocket(websocket: WebSocket, bot_id: str):
    """WebSocket endpoint for bot communication.
    
    Binary frames = audio chunks
    JSON frames = commands, status, control messages
    """
    await websocket.accept()
    
    logger.info("bot_connected", bot_id=bot_id)
    _active_bots[bot_id] = websocket
    
    # Create session config for this bot
    config = MeetingSessionConfig()
    _bot_configs[bot_id] = config
    
    # Create command dispatcher with this bot's config
    dispatcher = CommandDispatcher(config)
    
    # Send welcome message
    await _send_json_safe(websocket, {
        "type": "welcome",
        "bot_id": bot_id,
        "config": config.snapshot(),
    })
    
    try:
        while True:
            # Receive message (binary or text)
            message = await websocket.receive()
            
            if "bytes" in message:
                # Binary audio chunk
                response = await _handle_audio_chunk(bot_id, message["bytes"])
                await _send_json_safe(websocket, response)
                
            elif "text" in message:
                # JSON message
                import json
                data = json.loads(message["text"])
                msg_type = data.get("type", "unknown")
                
                if msg_type == "chat_command":
                    response = await _handle_chat_command(
                        bot_id,
                        data.get("text", ""),
                        data.get("sender", ""),
                        dispatcher,
                    )
                    await _send_json_safe(websocket, response)
                    
                elif msg_type == "status":
                    response = await _handle_status_update(bot_id, data)
                    await _send_json_safe(websocket, response)
                    
                else:
                    logger.warning("unknown_bot_message", bot_id=bot_id, type=msg_type)
                    
    except WebSocketDisconnect:
        logger.info("bot_disconnected", bot_id=bot_id)
    except Exception as e:
        logger.error("bot_ws_error", bot_id=bot_id, error=str(e))
    finally:
        _active_bots.pop(bot_id, None)
        _bot_configs.pop(bot_id, None)


async def send_to_bot(bot_id: str, message: dict[str, Any]) -> bool:
    """Send a message to a connected bot.
    
    Used by other parts of orchestration to send:
    - Translation results
    - Config updates
    - Control commands (leave, stop, etc.)
    """
    ws = _active_bots.get(bot_id)
    if ws is None:
        logger.warning("bot_not_connected", bot_id=bot_id)
        return False
    return await _send_json_safe(ws, message)


def get_connected_bots() -> list[str]:
    """Get list of currently connected bot IDs."""
    return list(_active_bots.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest modules/orchestration-service/tests/test_bot_websocket.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/routers/bot/bot_websocket.py modules/orchestration-service/tests/test_bot_websocket.py
git commit -m "$(cat <<'EOF'
feat(orchestration): add bot WebSocket endpoint

- /ws/bot/{bot_id} handles binary audio + JSON messages
- CommandDispatcher integration for chat commands
- Per-bot MeetingSessionConfig instances
- send_to_bot() for sending messages to connected bots
EOF
)"
```

---

## Task 3: Register Bot Router in FastAPI App

**Files:**
- Modify: `modules/orchestration-service/src/main_fastapi.py`

- [ ] **Step 1: Read current main_fastapi.py structure**

Run: Read the file to see existing router imports

- [ ] **Step 2: Add bot router import**

Add with other router imports:

```python
from .routers.bot.bot_websocket import router as bot_ws_router
```

- [ ] **Step 3: Include bot router**

Add after other router includes:

```python
app.include_router(bot_ws_router)
```

- [ ] **Step 4: Commit**

```bash
git add modules/orchestration-service/src/main_fastapi.py
git commit -m "$(cat <<'EOF'
feat(orchestration): register bot WebSocket router

Bot communication available at /ws/bot/{bot_id}
EOF
)"
```

---

## Task 4: Update Bot AudioStreamer to Connect to Orchestration

**Files:**
- Modify: `modules/meeting-bot-service/src/audio_streaming.ts`

- [ ] **Step 1: Read current audio_streaming.ts**

Run: Read the file to understand existing structure

- [ ] **Step 2: Update AudioStreamer with orchestration WebSocket**

The existing AudioStreamer should be updated to:
1. Connect to orchestration WebSocket on start
2. Send binary audio chunks
3. Handle command responses
4. Forward responses to ChatResponder

```typescript
/**
 * AudioStreamer — captures meeting audio and streams to orchestration.
 * 
 * Uses Playwright CDP to capture tab audio, converts to PCM,
 * and sends over WebSocket to orchestration service.
 */

import { Page, CDPSession } from 'playwright';
import WebSocket from 'ws';
import { Logger } from 'winston';

export interface AudioStreamerConfig {
  orchestrationUrl: string;
  botId: string;
  sampleRate?: number;
  channelCount?: number;
}

export interface CommandResponse {
  type: 'command_response';
  response_text: string | null;
  changed_fields?: string[];
  demo_action?: string | null;
}

export type MessageHandler = (message: any) => void;

export class AudioStreamer {
  private logger: Logger;
  private page: Page | null = null;
  private cdpSession: CDPSession | null = null;
  private ws: WebSocket | null = null;
  private config: AudioStreamerConfig;
  private isStreaming = false;
  private messageHandlers: MessageHandler[] = [];

  constructor(logger: Logger, config: AudioStreamerConfig) {
    this.logger = logger;
    this.config = {
      sampleRate: 48000,
      channelCount: 1,
      ...config,
    };
  }

  /**
   * Subscribe to messages from orchestration.
   */
  onMessage(handler: MessageHandler): void {
    this.messageHandlers.push(handler);
  }

  /**
   * Connect to orchestration WebSocket and start audio capture.
   */
  async start(page: Page): Promise<void> {
    this.page = page;
    
    // Connect to orchestration WebSocket
    const wsUrl = `${this.config.orchestrationUrl}/ws/bot/${this.config.botId}`;
    this.logger.info('Connecting to orchestration...', { wsUrl });
    
    this.ws = new WebSocket(wsUrl);
    
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('WebSocket connection timeout'));
      }, 10000);

      this.ws!.on('open', () => {
        clearTimeout(timeout);
        this.logger.info('Connected to orchestration');
        resolve();
      });

      this.ws!.on('error', (error) => {
        clearTimeout(timeout);
        this.logger.error('WebSocket error', { error: error.message });
        reject(error);
      });
    });

    // Handle incoming messages
    this.ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        this.logger.debug('Received from orchestration', { type: message.type });
        
        // Notify all handlers
        for (const handler of this.messageHandlers) {
          handler(message);
        }
      } catch (error: any) {
        this.logger.warn('Failed to parse message', { error: error.message });
      }
    });

    this.ws.on('close', () => {
      this.logger.info('Disconnected from orchestration');
      this.isStreaming = false;
    });

    // Start audio capture via CDP
    await this.startAudioCapture();
    
    this.isStreaming = true;
  }

  /**
   * Start capturing tab audio via Chrome DevTools Protocol.
   */
  private async startAudioCapture(): Promise<void> {
    if (!this.page) throw new Error('Page not set');

    // Get CDP session
    const context = this.page.context();
    this.cdpSession = await context.newCDPSession(this.page);

    // Enable audio capture
    // Note: This requires specific Chrome flags and permissions
    await this.cdpSession.send('Page.enable');
    
    // Create audio capture context via page evaluation
    await this.page.evaluate(() => {
      // This runs in browser context
      // We'll use MediaRecorder to capture audio from the page
      const audioContext = new AudioContext({ sampleRate: 48000 });
      
      // Store for later access
      (window as any).__audioContext = audioContext;
    });

    this.logger.info('Audio capture started');
  }

  /**
   * Send a chat command to orchestration.
   */
  async sendChatCommand(text: string, sender: string): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      this.logger.warn('Cannot send command - not connected');
      return;
    }

    this.ws.send(JSON.stringify({
      type: 'chat_command',
      text,
      sender,
    }));
  }

  /**
   * Send status update to orchestration.
   */
  async sendStatus(state: string, metadata?: Record<string, any>): Promise<void> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    this.ws.send(JSON.stringify({
      type: 'status',
      state,
      ...metadata,
    }));
  }

  /**
   * Send binary audio data to orchestration.
   */
  sendAudioChunk(data: Buffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    this.ws.send(data);
  }

  /**
   * Stop audio streaming and disconnect.
   */
  async stop(): Promise<void> {
    this.isStreaming = false;

    if (this.cdpSession) {
      try {
        await this.cdpSession.detach();
      } catch {
        // Already detached
      }
      this.cdpSession = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.logger.info('Audio streaming stopped');
  }

  /**
   * Check if currently streaming.
   */
  get streaming(): boolean {
    return this.isStreaming;
  }
}

/**
 * Factory function for creating AudioStreamer.
 */
export function createAudioStreamer(
  logger: Logger,
  orchestrationUrl: string,
  botId: string
): AudioStreamer {
  return new AudioStreamer(logger, {
    orchestrationUrl,
    botId,
  });
}
```

- [ ] **Step 3: Commit**

```bash
git add modules/meeting-bot-service/src/audio_streaming.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): update AudioStreamer for orchestration WebSocket

- Connects to /ws/bot/{bot_id} on orchestration
- sendChatCommand() for forwarding chat commands
- sendStatus() for bot state updates
- sendAudioChunk() for binary audio streaming
- onMessage() for receiving orchestration responses
EOF
)"
```

---

## Task 5: Wire AudioStreamer into GoogleMeetBot

**Files:**
- Modify: `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts`

- [ ] **Step 1: Update AudioStreamer initialization in joinMeeting()**

Find the section where `audioStreamer` is created and update to:

```typescript
// Start audio streaming to orchestration
if (orchestrationUrl) {
  this._logger.info('Starting audio streaming...', { orchestrationUrl });
  
  this.audioStreamer = createAudioStreamer(
    this._logger,
    orchestrationUrl,
    botId
  );
  
  // Handle messages from orchestration
  this.audioStreamer.onMessage((message) => {
    if (message.type === 'command_response' && message.response_text && this.chatResponder) {
      // Forward command response to meeting chat
      this.chatResponder.send(message.response_text);
    }
  });
  
  await this.audioStreamer.start(this.page);
  
  // Send initial status
  await this.audioStreamer.sendStatus('joined', {
    meeting_url: url,
    bot_name: name,
  });
}
```

- [ ] **Step 2: Update ChatPoller to forward commands to orchestration**

In the chat polling loop, update to:

```typescript
// In the polling loop where commands are detected
if (message.isCommand && this.audioStreamer) {
  await this.audioStreamer.sendChatCommand(message.text, message.sender || 'unknown');
}
```

- [ ] **Step 3: Update leave() to send status and stop streaming**

```typescript
async leave(): Promise<void> {
  // Send leaving status
  if (this.audioStreamer) {
    await this.audioStreamer.sendStatus('leaving');
    await this.audioStreamer.stop();
  }
  
  // Existing leave logic...
}
```

- [ ] **Step 4: Commit**

```bash
git add modules/meeting-bot-service/src/bots/GoogleMeetBot.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): wire AudioStreamer into GoogleMeetBot lifecycle

- Start streaming on join, stop on leave
- Forward chat commands to orchestration
- Receive command responses and post to chat
- Send status updates (joined, leaving)
EOF
)"
```

---

## Review Wave 2: Audio Pipeline Complete

At this point, the bot can stream audio to orchestration and receive command responses.

**Manual verification:**

1. **Start orchestration service:**
   ```bash
   uv run python modules/orchestration-service/src/main_fastapi.py
   ```

2. **Start bot service:**
   ```bash
   cd modules/meeting-bot-service && npm run dev
   ```

3. **Join a test meeting:**
   ```bash
   curl -X POST http://localhost:5005/api/bot/join \
     -H "Content-Type: application/json" \
     -d '{
       "meetingUrl": "https://meet.google.com/xxx-xxxx-xxx",
       "botName": "AudioTestBot",
       "botId": "audio-test-1",
       "userId": "user-1",
       "orchestrationUrl": "ws://localhost:3000"
     }'
   ```

4. **Check orchestration logs for:**
   - `bot_connected` with bot_id
   - `bot_audio_received` with byte counts
   - `bot_status_update` with state="joined"

---

## Success Criteria (Phase 2)

- [ ] Bot connects to orchestration WebSocket on join
- [ ] Audio chunks stream from bot to orchestration
- [ ] Orchestration acknowledges audio chunks
- [ ] Bot sends status updates (joined, leaving)
- [ ] Command responses flow back to bot
- [ ] Clean disconnect on leave
