# Meeting Bot Phase 3: Chat Commands Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable meeting participants to control the bot via chat commands (e.g., `/lang zh-en`, `/mode split`, `/help`). Commands are detected by the bot's ChatPoller, routed to orchestration's CommandDispatcher, and responses are posted back to the meeting chat.

**Architecture:** ChatPoller polls Google Meet chat for `/` commands → forwards to orchestration via WebSocket → CommandDispatcher executes → response sent back → ChatResponder posts to meeting chat.

**Tech Stack:** TypeScript (Playwright), Python (FastAPI), WebSocket

**Dependencies:** Phase 1 (auth/selectors) and Phase 2 (audio streaming) must be complete.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `modules/meeting-bot-service/src/chat/chat_poller.ts` | Modify — use new selectors, forward to orchestration |
| `modules/meeting-bot-service/src/chat/chat_responder.ts` | Modify — use new selectors, handle response posting |
| `modules/meeting-bot-service/src/chat/command_parser.ts` | Modify — basic command validation before sending |
| `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts` | Modify — wire ChatPoller + ChatResponder lifecycle |
| `modules/orchestration-service/src/services/command_dispatcher.py` | Exists — add `/stop` handler |
| `modules/orchestration-service/src/routers/bot/bot_websocket.py` | Modify — add stop command handling |

---

## Task 1: Update ChatPoller with New Selectors

**Files:**
- Modify: `modules/meeting-bot-service/src/chat/chat_poller.ts`
- Test: Manual verification (requires running Meet)

- [ ] **Step 1: Read current chat_poller.ts**

Run: Read the file to understand current structure

- [ ] **Step 2: Update imports and selectors**

Replace the imports and selector references:

```typescript
/**
 * ChatPoller — monitors Google Meet chat for slash commands.
 * 
 * Polls the chat messages container for new messages starting with '/'.
 * Uses message IDs to track which messages have been processed.
 */

import { Page, Locator } from 'playwright';
import { Logger } from 'winston';
import {
  findVisible,
  clickFirst,
  CHAT_BUTTON_SELECTORS,
  CHAT_MESSAGES_SELECTORS,
  CHAT_MESSAGE_ITEM,
} from './selectors';

export interface ChatMessage {
  id: string;
  text: string;
  sender: string;
  isCommand: boolean;
  timestamp: Date;
}

export type CommandHandler = (message: ChatMessage) => Promise<void>;

export class ChatPoller {
  private logger: Logger;
  private page: Page;
  private seenMessageIds: Set<string> = new Set();
  private pollInterval: number;
  private pollTimer: NodeJS.Timeout | null = null;
  private commandHandler: CommandHandler | null = null;
  private isPolling = false;

  constructor(logger: Logger, page: Page, pollInterval = 1000) {
    this.logger = logger;
    this.page = page;
    this.pollInterval = pollInterval;
  }

  /**
   * Set handler for detected commands.
   */
  onCommand(handler: CommandHandler): void {
    this.commandHandler = handler;
  }

  /**
   * Start polling for new chat messages.
   */
  async start(): Promise<void> {
    if (this.isPolling) return;
    
    this.logger.info('Starting chat polling...');
    this.isPolling = true;
    
    // Open chat panel if not already open
    await this.ensureChatOpen();
    
    // Start polling loop
    this.pollTimer = setInterval(async () => {
      try {
        await this.poll();
      } catch (error: any) {
        this.logger.warn('Chat poll error', { error: error.message });
      }
    }, this.pollInterval);
  }

  /**
   * Stop polling.
   */
  stop(): void {
    this.isPolling = false;
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.logger.info('Chat polling stopped');
  }

  /**
   * Ensure chat panel is open.
   */
  private async ensureChatOpen(): Promise<void> {
    // Check if chat is already open by looking for messages container
    const chatOpen = await findVisible(this.page, CHAT_MESSAGES_SELECTORS);
    if (chatOpen) return;

    // Open chat panel
    const clicked = await clickFirst(this.page, CHAT_BUTTON_SELECTORS);
    if (clicked) {
      this.logger.info('Opened chat panel');
      await this.page.waitForTimeout(500);
    }
  }

  /**
   * Poll for new messages.
   */
  private async poll(): Promise<void> {
    // Find all message elements
    const messages = await this.page.$$(CHAT_MESSAGE_ITEM);
    
    for (const msgElement of messages) {
      try {
        // Get message ID
        const id = await msgElement.getAttribute('data-message-id');
        if (!id || this.seenMessageIds.has(id)) continue;
        
        // Mark as seen
        this.seenMessageIds.add(id);
        
        // Get message text
        const textElement = await msgElement.$('span');
        const text = await textElement?.textContent() ?? '';
        
        // Get sender name (if available)
        const senderElement = await msgElement.$('[data-sender-name]');
        const sender = await senderElement?.getAttribute('data-sender-name') ?? 
                       await senderElement?.textContent() ?? 'unknown';

        // Check if this is a command
        const trimmedText = text.trim();
        const isCommand = trimmedText.startsWith('/');
        
        const message: ChatMessage = {
          id,
          text: trimmedText,
          sender,
          isCommand,
          timestamp: new Date(),
        };

        this.logger.debug('Chat message detected', { 
          id, 
          text: trimmedText.substring(0, 50),
          isCommand,
          sender,
        });

        // Forward commands to handler
        if (isCommand && this.commandHandler) {
          await this.commandHandler(message);
        }
      } catch (error: any) {
        this.logger.warn('Error processing chat message', { error: error.message });
      }
    }
  }

  /**
   * Get count of seen messages.
   */
  get messageCount(): number {
    return this.seenMessageIds.size;
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add modules/meeting-bot-service/src/chat/chat_poller.ts
git commit -m "$(cat <<'EOF'
refactor(meeting-bot): update ChatPoller with new selectors

- Use selector arrays with fallbacks
- onCommand() handler for command forwarding
- ensureChatOpen() with clickFirst()
- Track sender name for commands
EOF
)"
```

---

## Task 2: Update ChatResponder with New Selectors

**Files:**
- Modify: `modules/meeting-bot-service/src/chat/chat_responder.ts`

- [ ] **Step 1: Read current chat_responder.ts**

Run: Read the file to understand current structure

- [ ] **Step 2: Update with new selectors and error handling**

```typescript
/**
 * ChatResponder — posts messages to Google Meet chat.
 * 
 * Used to send command responses back to meeting participants.
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import {
  findVisible,
  clickFirst,
  CHAT_BUTTON_SELECTORS,
  CHAT_INPUT_SELECTORS,
} from './selectors';

export class ChatResponder {
  private logger: Logger;
  private page: Page;
  private sendQueue: string[] = [];
  private isSending = false;

  constructor(logger: Logger, page: Page) {
    this.logger = logger;
    this.page = page;
  }

  /**
   * Send a message to the meeting chat.
   * 
   * Messages are queued to avoid race conditions.
   */
  async send(message: string): Promise<boolean> {
    this.sendQueue.push(message);
    return this.processQueue();
  }

  /**
   * Process the send queue.
   */
  private async processQueue(): Promise<boolean> {
    if (this.isSending || this.sendQueue.length === 0) return true;
    
    this.isSending = true;
    
    try {
      while (this.sendQueue.length > 0) {
        const message = this.sendQueue.shift()!;
        await this.sendImmediate(message);
        
        // Small delay between messages to avoid rate limiting
        if (this.sendQueue.length > 0) {
          await this.page.waitForTimeout(500);
        }
      }
      return true;
    } catch (error: any) {
      this.logger.error('Failed to send chat message', { error: error.message });
      return false;
    } finally {
      this.isSending = false;
    }
  }

  /**
   * Send a message immediately (internal).
   */
  private async sendImmediate(message: string): Promise<void> {
    this.logger.info('Sending chat message', { message: message.substring(0, 50) });

    // Ensure chat panel is open
    const chatInput = await findVisible(this.page, CHAT_INPUT_SELECTORS);
    if (!chatInput) {
      // Try to open chat panel
      const clicked = await clickFirst(this.page, CHAT_BUTTON_SELECTORS);
      if (!clicked) {
        throw new Error('Could not open chat panel');
      }
      await this.page.waitForTimeout(500);
    }

    // Find and fill the chat input
    const input = await findVisible(this.page, CHAT_INPUT_SELECTORS);
    if (!input) {
      throw new Error('Could not find chat input');
    }

    // Clear existing text and type new message
    await input.click();
    await input.fill(message);
    
    // Press Enter to send
    await this.page.keyboard.press('Enter');
    
    this.logger.debug('Chat message sent', { message: message.substring(0, 50) });
    
    // Wait for message to be sent
    await this.page.waitForTimeout(200);
  }

  /**
   * Send multiple messages in sequence.
   */
  async sendMultiple(messages: string[]): Promise<boolean> {
    for (const message of messages) {
      this.sendQueue.push(message);
    }
    return this.processQueue();
  }
}
```

- [ ] **Step 3: Commit**

```bash
git add modules/meeting-bot-service/src/chat/chat_responder.ts
git commit -m "$(cat <<'EOF'
refactor(meeting-bot): update ChatResponder with new selectors

- Use selector arrays with fallbacks
- Message queue to avoid race conditions
- sendMultiple() for batch messages
EOF
)"
```

---

## Task 3: Add /stop Command Handler in Orchestration

**Files:**
- Modify: `modules/orchestration-service/src/services/command_dispatcher.py`
- Test: `modules/orchestration-service/tests/test_command_dispatcher.py`

- [ ] **Step 1: Write the failing test**

Add to `modules/orchestration-service/tests/test_command_dispatcher.py`:

```python
def test_stop_command_returns_stop_action(mock_config):
    """Test /stop returns a stop_action for the bot to leave."""
    dispatcher = CommandDispatcher(mock_config)
    
    result = dispatcher.dispatch("/stop", sender="TestUser")
    
    assert result is not None
    assert result.response_text == "👋 Leaving meeting..."
    assert result.stop_requested is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest modules/orchestration-service/tests/test_command_dispatcher.py -k "stop" -v`
Expected: FAIL — AttributeError: 'DispatchResult' object has no attribute 'stop_requested'

- [ ] **Step 3: Add stop_requested to DispatchResult**

In `command_dispatcher.py`, update the DispatchResult dataclass:

```python
@dataclass
class DispatchResult:
    """Result of dispatching a chat command."""

    response_text: str
    changed_fields: set[str]
    demo_action: str | None = None
    stop_requested: bool = False  # Add this field
```

- [ ] **Step 4: Add _handle_stop method**

Add to CommandDispatcher class:

```python
def _handle_stop(self) -> DispatchResult:
    """Handle /stop command - signal bot to leave meeting."""
    logger.info("stop_command_received")
    return DispatchResult(
        response_text="👋 Leaving meeting...",
        changed_fields=set(),
        stop_requested=True,
    )
```

- [ ] **Step 5: Add /stop to dispatch routing**

In the dispatch() method, add after the `/help` case:

```python
elif cmd == "/stop":
    return self._handle_stop()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest modules/orchestration-service/tests/test_command_dispatcher.py -k "stop" -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add modules/orchestration-service/src/services/command_dispatcher.py modules/orchestration-service/tests/test_command_dispatcher.py
git commit -m "$(cat <<'EOF'
feat(orchestration): add /stop command to dispatcher

- stop_requested field in DispatchResult
- Bot will leave meeting when stop_requested=True
EOF
)"
```

---

## Task 4: Handle Stop Command in Bot WebSocket

**Files:**
- Modify: `modules/orchestration-service/src/routers/bot/bot_websocket.py`

- [ ] **Step 1: Update _handle_chat_command to include stop_requested**

```python
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
        "stop_requested": result.stop_requested,  # Add this
    }
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/bot/bot_websocket.py
git commit -m "$(cat <<'EOF'
feat(orchestration): forward stop_requested to bot

Bot receives stop_requested=true when /stop command is issued.
EOF
)"
```

---

## Task 5: Wire Chat Commands in GoogleMeetBot

**Files:**
- Modify: `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts`

- [ ] **Step 1: Initialize ChatPoller and ChatResponder in joinMeeting()**

Add after successfully joining the meeting:

```typescript
// Initialize chat components
this._logger.info('Starting chat polling...');

this.chatResponder = new ChatResponder(this._logger, this.page);
this.chatPoller = new ChatPoller(this._logger, this.page);

// Handle detected commands
this.chatPoller.onCommand(async (message) => {
  this._logger.info('Chat command detected', { 
    command: message.text, 
    sender: message.sender 
  });
  
  // Forward to orchestration via AudioStreamer
  if (this.audioStreamer) {
    await this.audioStreamer.sendChatCommand(message.text, message.sender);
  }
});

// Start polling
await this.chatPoller.start();
```

- [ ] **Step 2: Handle stop_requested in orchestration message handler**

Update the AudioStreamer message handler:

```typescript
// Handle messages from orchestration
this.audioStreamer.onMessage(async (message) => {
  if (message.type === 'command_response') {
    // Post response to chat
    if (message.response_text && this.chatResponder) {
      await this.chatResponder.send(message.response_text);
    }
    
    // Handle stop request
    if (message.stop_requested) {
      this._logger.info('Stop requested by orchestration, leaving meeting...');
      await this.leave();
    }
  }
});
```

- [ ] **Step 3: Clean up chat components in leave()**

Update the leave() method:

```typescript
async leave(): Promise<void> {
  // Stop chat polling
  if (this.chatPoller) {
    this.chatPoller.stop();
    this.chatPoller = null;
  }
  
  // Send leaving status
  if (this.audioStreamer) {
    await this.audioStreamer.sendStatus('leaving');
    await this.audioStreamer.stop();
    this.audioStreamer = null;
  }
  
  this.chatResponder = null;
  
  // Existing leave logic (click leave button, cleanup, etc.)
  // ...
}
```

- [ ] **Step 4: Commit**

```bash
git add modules/meeting-bot-service/src/bots/GoogleMeetBot.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): wire ChatPoller and ChatResponder lifecycle

- Start chat polling after joining
- Forward commands to orchestration
- Post command responses to chat
- Handle /stop to leave meeting
- Clean up on leave
EOF
)"
```

---

## Task 6: Add Help Message with Available Commands

**Files:**
- Modify: `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts`

- [ ] **Step 1: Post help message after joining**

Add after successfully joining and starting chat:

```typescript
// Post help message to chat
const helpMessage = "🤖 LiveTranslate bot ready! Commands: /lang <code>, /mode subtitle|split|interpreter, /status, /help, /stop";
await this.chatResponder.send(helpMessage);
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/bots/GoogleMeetBot.ts
git commit -m "$(cat <<'EOF'
feat(meeting-bot): post help message on join

Participants see available commands when bot joins.
EOF
)"
```

---

## Review Wave 3: Chat Commands Complete

At this point, the full bot lifecycle is operational: auth → join → stream audio → chat commands → leave.

**Manual verification:**

1. **@USER: Complete Google sign-in if not done**
   ```bash
   curl -X POST http://localhost:5005/api/auth/setup
   ```

2. **Join a test meeting:**
   ```bash
   curl -X POST http://localhost:5005/api/bot/join \
     -H "Content-Type: application/json" \
     -d '{
       "meetingUrl": "https://meet.google.com/xxx-xxxx-xxx",
       "botName": "CommandTestBot",
       "botId": "cmd-test-1",
       "userId": "user-1",
       "orchestrationUrl": "ws://localhost:3000"
     }'
   ```

3. **Verify in meeting chat:**
   - Bot should post: "🤖 LiveTranslate bot ready! Commands: ..."
   
4. **Test commands in meeting chat:**
   - Type `/help` → Bot responds with command list
   - Type `/status` → Bot responds with current config
   - Type `/lang zh-en` → Bot responds with "✓ Translating: zh → en"
   - Type `/mode split` → Bot responds with "✓ Display mode: split"
   - Type `/stop` → Bot responds with "👋 Leaving meeting..." and leaves

5. **Check orchestration logs for:**
   - `bot_chat_command` with command text
   - `chat_command_received` from CommandDispatcher

---

## Success Criteria (Phase 3)

- [ ] ChatPoller detects `/` commands in meeting chat
- [ ] Commands forwarded to orchestration via WebSocket
- [ ] CommandDispatcher processes all commands (`/lang`, `/mode`, `/status`, `/help`, `/stop`)
- [ ] Responses posted back to meeting chat via ChatResponder
- [ ] `/stop` command triggers bot to leave meeting
- [ ] Help message posted when bot joins
- [ ] No duplicate command processing (message ID tracking works)

---

## Full System Success Criteria (All Phases)

### Phase 1: Authentication
- [ ] Persistent browser profile stores Google auth
- [ ] Bot joins without bot detection redirect

### Phase 2: Audio
- [ ] Audio streams from bot to orchestration
- [ ] Status updates flow correctly

### Phase 3: Commands
- [ ] All chat commands work end-to-end
- [ ] Bot leaves on `/stop`

### Integration
- [ ] Full lifecycle: spawn → auth → join → stream → commands → leave → cleanup
- [ ] Docker volume mounts profile correctly
- [ ] Multiple sequential bot sessions work (profile reused)
