# Meeting Subtitle System — Phase 2: Chat Commands & Live Control

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the TypeScript meeting bot to receive chat commands from Google Meet, dispatch them to the Python orchestration service, and respond in chat — enabling live control of subtitle display.

**Architecture:** TypeScript bot (Playwright) polls Google Meet chat DOM → parses /commands → sends `chat_command` over existing WebSocket → Python orchestration updates `MeetingSessionConfig` → renderers react → Python sends `chat_response` back → bot types confirmation in Meet chat.

**Tech Stack:** TypeScript (meeting-bot-service), Playwright, Python (orchestration-service), WebSocket (existing audio_streaming pattern), Pydantic v2 (ws_messages), Vitest (new TS test framework)

**Spec:** `docs/superpowers/specs/2026-04-10-meeting-subtitle-system-design.md` (Phase 2 section)

**Depends on:** Phase 0+1 complete (merged to main). Specifically: `MeetingSessionConfig`, `CaptionBuffer` multi-subscriber, canonical `theme.py`.

---

## File Structure

### New Files (TypeScript — meeting-bot-service)

| File | Responsibility |
|------|---------------|
| `modules/meeting-bot-service/src/chat/selectors.ts` | Single source of truth for all Google Meet DOM selectors |
| `modules/meeting-bot-service/src/chat/command_parser.ts` | Pure function: parse /slash commands from chat text, return typed results |
| `modules/meeting-bot-service/src/chat/chat_poller.ts` | Opens chat panel, polls DOM every 500ms, detects new messages |
| `modules/meeting-bot-service/src/chat/chat_responder.ts` | Types bot responses into Meet's contenteditable chat input |
| `modules/meeting-bot-service/src/chat/index.ts` | Re-exports for clean imports |
| `modules/meeting-bot-service/tests/chat/command_parser.test.ts` | Unit tests for command parser |
| `modules/meeting-bot-service/vitest.config.ts` | Vitest configuration |

### New Files (Python — orchestration-service)

| File | Responsibility |
|------|---------------|
| `modules/orchestration-service/src/services/command_dispatcher.py` | Receives chat_command, validates, applies to MeetingSessionConfig, returns response text |
| `modules/orchestration-service/tests/test_command_dispatcher.py` | Unit tests for command dispatch |

### Modified Files

| File | Change |
|------|--------|
| `modules/shared/src/livetranslate_common/models/ws_messages.py:293-316` | Add `ChatCommandMessage`, `ChatResponseMessage`, `ConfigChangedMessage` to registries |
| `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts:146-151,384-475` | Skip "Continue without mic/cam" when camera available, integrate chat poller |
| `modules/meeting-bot-service/src/audio_streaming.ts:109-141` | Handle `chat_response` and `config_changed` message types |
| `modules/meeting-bot-service/package.json` | Add vitest dev dependency |
| `modules/orchestration-service/src/routers/audio/websocket_audio.py:821` | Handle `ChatCommandMessage`, send `ChatResponseMessage` |

---

## Task 1: Google Meet Selectors (Single Source of Truth)

**Files:**
- Create: `modules/meeting-bot-service/src/chat/selectors.ts`

- [ ] **Step 1: Create the selectors file**

```typescript
/**
 * Google Meet DOM selectors — single source of truth.
 *
 * Update ONLY this file when Google Meet changes its DOM.
 * Primary strategy: aria-label (survives most restructures).
 * Secondary: button text content (breaks with locale changes).
 */

import { Page, Locator } from 'playwright';

// Chat panel
export const CHAT_BUTTON = '[aria-label="Chat with everyone"]';
export const CHAT_PANEL = '[aria-label="Chat panel"]';
export const CHAT_INPUT = '[aria-label="Send a message to everyone"]';
export const CHAT_SEND_BUTTON = '[aria-label="Send a message"]';
export const CHAT_MESSAGES_CONTAINER = '[aria-label="Chat messages"]';

// Meeting controls
export const LEAVE_CALL_BUTTON = '[aria-label="Leave call"]';
export const PEOPLE_BUTTON_PREFIX = 'button[aria-label^="People"]';
export const CAMERA_BUTTON = '[aria-label="Turn on camera"]';
export const MIC_BUTTON = '[aria-label="Turn on microphone"]';

// Pre-join
export const NAME_INPUT = 'input[type="text"][aria-label="Your name"]';
export const ASK_TO_JOIN_BUTTON = 'button:has-text("Ask to join")';
export const JOIN_NOW_BUTTON = 'button:has-text("Join now")';
export const CONTINUE_WITHOUT_MIC_CAM = 'button:has-text("Continue without microphone and camera")';
export const GOT_IT_BUTTON = 'button:has-text("Got it")';

/**
 * Try multiple selectors in priority order, return first visible one.
 */
export async function findVisible(page: Page, selectors: string[]): Promise<Locator | null> {
  for (const sel of selectors) {
    try {
      const loc = page.locator(sel).first();
      if (await loc.isVisible({ timeout: 1000 })) {
        return loc;
      }
    } catch {
      // Try next selector
    }
  }
  return null;
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/chat/selectors.ts
git commit -m "feat: Google Meet DOM selectors — single source of truth"
```

---

## Task 2: Set Up Vitest for Meeting Bot Service

**Files:**
- Create: `modules/meeting-bot-service/vitest.config.ts`
- Modify: `modules/meeting-bot-service/package.json`

- [ ] **Step 1: Install vitest**

```bash
cd modules/meeting-bot-service && npm install --save-dev vitest
```

- [ ] **Step 2: Create vitest config**

Create `modules/meeting-bot-service/vitest.config.ts`:

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['tests/**/*.test.ts'],
    globals: true,
  },
});
```

- [ ] **Step 3: Add test script to package.json**

In `modules/meeting-bot-service/package.json`, add to `"scripts"`:

```json
"test": "vitest run",
"test:watch": "vitest"
```

- [ ] **Step 4: Verify vitest works**

```bash
cd modules/meeting-bot-service && npx vitest run 2>&1 | tail -5
```

Expected: "No test files found" (nothing to run yet, but vitest itself works)

- [ ] **Step 5: Commit**

```bash
git add modules/meeting-bot-service/vitest.config.ts modules/meeting-bot-service/package.json modules/meeting-bot-service/package-lock.json
git commit -m "chore: add vitest to meeting-bot-service"
```

---

## Task 3: Command Parser (Pure Function, TDD)

**Files:**
- Create: `modules/meeting-bot-service/tests/chat/command_parser.test.ts`
- Create: `modules/meeting-bot-service/src/chat/command_parser.ts`

- [ ] **Step 1: Write the failing tests**

Create `modules/meeting-bot-service/tests/chat/command_parser.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { parseCommand, CommandResult } from '../../src/chat/command_parser';

describe('parseCommand', () => {
  // --- Language commands ---
  it('parses /lang with single target', () => {
    const result = parseCommand('/lang zh');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'auto', target_lang: 'zh' },
    });
  });

  it('parses /lang with explicit pair', () => {
    const result = parseCommand('/lang zh-en');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'zh', target_lang: 'en' },
    });
  });

  // --- Font commands ---
  it('parses /font up', () => {
    const result = parseCommand('/font up');
    expect(result).toEqual({ type: 'adjust_font', delta: 4 });
  });

  it('parses /font down', () => {
    const result = parseCommand('/font down');
    expect(result).toEqual({ type: 'adjust_font', delta: -4 });
  });

  it('parses /font with exact size', () => {
    const result = parseCommand('/font 32');
    expect(result).toEqual({ type: 'set_config', changes: { font_size: 32 } });
  });

  // --- Display mode ---
  it('parses /mode subtitle', () => {
    const result = parseCommand('/mode subtitle');
    expect(result).toEqual({ type: 'set_config', changes: { display_mode: 'subtitle' } });
  });

  it('parses /mode split', () => {
    const result = parseCommand('/mode split');
    expect(result).toEqual({ type: 'set_config', changes: { display_mode: 'split' } });
  });

  it('parses /mode interpreter', () => {
    const result = parseCommand('/mode interpreter');
    expect(result).toEqual({ type: 'set_config', changes: { display_mode: 'interpreter' } });
  });

  // --- Theme ---
  it('parses /theme dark', () => {
    const result = parseCommand('/theme dark');
    expect(result).toEqual({ type: 'set_config', changes: { theme: 'dark' } });
  });

  it('parses /theme contrast as high_contrast', () => {
    const result = parseCommand('/theme contrast');
    expect(result).toEqual({ type: 'set_config', changes: { theme: 'high_contrast' } });
  });

  // --- Toggle commands ---
  it('parses /speakers on', () => {
    const result = parseCommand('/speakers on');
    expect(result).toEqual({ type: 'set_config', changes: { show_speakers: true } });
  });

  it('parses /speakers off', () => {
    const result = parseCommand('/speakers off');
    expect(result).toEqual({ type: 'set_config', changes: { show_speakers: false } });
  });

  it('parses /original on', () => {
    const result = parseCommand('/original on');
    expect(result).toEqual({ type: 'set_config', changes: { show_original: true } });
  });

  it('parses /translate off', () => {
    const result = parseCommand('/translate off');
    expect(result).toEqual({ type: 'set_config', changes: { translation_enabled: false } });
  });

  // --- Source ---
  it('parses /source bot', () => {
    const result = parseCommand('/source bot');
    expect(result).toEqual({ type: 'set_config', changes: { caption_source: 'bot_audio' } });
  });

  it('parses /source fireflies', () => {
    const result = parseCommand('/source fireflies');
    expect(result).toEqual({ type: 'set_config', changes: { caption_source: 'fireflies' } });
  });

  // --- Info commands ---
  it('parses /status', () => {
    const result = parseCommand('/status');
    expect(result).toEqual({ type: 'query', query: 'status' });
  });

  it('parses /help', () => {
    const result = parseCommand('/help');
    expect(result).toEqual({ type: 'query', query: 'help' });
  });

  // --- Non-commands ---
  it('returns null for non-command text', () => {
    expect(parseCommand('hello everyone')).toBeNull();
  });

  it('returns null for empty string', () => {
    expect(parseCommand('')).toBeNull();
  });

  it('returns unknown for unrecognized command', () => {
    const result = parseCommand('/unknown arg');
    expect(result).toEqual({ type: 'unknown', raw: '/unknown arg' });
  });

  // --- Edge cases ---
  it('handles extra whitespace', () => {
    const result = parseCommand('  /lang  zh  ');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'auto', target_lang: 'zh' },
    });
  });

  it('is case insensitive for commands', () => {
    const result = parseCommand('/LANG zh');
    expect(result).toEqual({
      type: 'set_language',
      changes: { source_lang: 'auto', target_lang: 'zh' },
    });
  });

  it('rejects /font with invalid number', () => {
    const result = parseCommand('/font abc');
    expect(result).toEqual({ type: 'unknown', raw: '/font abc' });
  });

  it('rejects /mode with invalid mode', () => {
    const result = parseCommand('/mode invalid');
    expect(result).toEqual({ type: 'unknown', raw: '/mode invalid' });
  });
});
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd modules/meeting-bot-service && npx vitest run tests/chat/command_parser.test.ts 2>&1 | tail -10
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement the command parser**

Create `modules/meeting-bot-service/src/chat/command_parser.ts`:

```typescript
/**
 * Command parser — stateless pure function.
 *
 * Parses /slash commands from Google Meet chat messages.
 * Returns typed CommandResult or null for non-commands.
 */

export type CommandResult =
  | { type: 'set_language'; changes: { source_lang: string; target_lang: string } }
  | { type: 'set_config'; changes: Record<string, string | number | boolean> }
  | { type: 'adjust_font'; delta: number }
  | { type: 'query'; query: 'status' | 'help' }
  | { type: 'unknown'; raw: string };

const VALID_MODES = new Set(['subtitle', 'split', 'interpreter']);
const VALID_THEMES: Record<string, string> = {
  dark: 'dark',
  light: 'light',
  contrast: 'high_contrast',
  high_contrast: 'high_contrast',
  minimal: 'minimal',
  corporate: 'corporate',
};
const VALID_SOURCES: Record<string, string> = {
  bot: 'bot_audio',
  fireflies: 'fireflies',
};

function parseToggle(value: string): boolean | null {
  if (value === 'on' || value === 'true' || value === 'yes') return true;
  if (value === 'off' || value === 'false' || value === 'no') return false;
  return null;
}

export function parseCommand(text: string): CommandResult | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith('/')) return null;

  const parts = trimmed.split(/\s+/);
  const cmd = parts[0].toLowerCase();
  const arg = parts[1]?.toLowerCase() ?? '';

  switch (cmd) {
    case '/lang': {
      if (!arg) return { type: 'unknown', raw: trimmed };
      if (arg.includes('-')) {
        const [source, target] = arg.split('-', 2);
        return { type: 'set_language', changes: { source_lang: source, target_lang: target } };
      }
      return { type: 'set_language', changes: { source_lang: 'auto', target_lang: arg } };
    }

    case '/font': {
      if (arg === 'up') return { type: 'adjust_font', delta: 4 };
      if (arg === 'down') return { type: 'adjust_font', delta: -4 };
      const size = parseInt(arg, 10);
      if (!isNaN(size) && size > 0) return { type: 'set_config', changes: { font_size: size } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/mode': {
      if (VALID_MODES.has(arg)) return { type: 'set_config', changes: { display_mode: arg } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/theme': {
      const theme = VALID_THEMES[arg];
      if (theme) return { type: 'set_config', changes: { theme } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/speakers': {
      const val = parseToggle(arg);
      if (val !== null) return { type: 'set_config', changes: { show_speakers: val } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/original': {
      const val = parseToggle(arg);
      if (val !== null) return { type: 'set_config', changes: { show_original: val } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/translate': {
      const val = parseToggle(arg);
      if (val !== null) return { type: 'set_config', changes: { translation_enabled: val } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/source': {
      const source = VALID_SOURCES[arg];
      if (source) return { type: 'set_config', changes: { caption_source: source } };
      return { type: 'unknown', raw: trimmed };
    }

    case '/status':
      return { type: 'query', query: 'status' };

    case '/help':
      return { type: 'query', query: 'help' };

    default:
      return { type: 'unknown', raw: trimmed };
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd modules/meeting-bot-service && npx vitest run tests/chat/command_parser.test.ts 2>&1 | tail -10
```

Expected: All 26 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/meeting-bot-service/src/chat/command_parser.ts modules/meeting-bot-service/tests/chat/command_parser.test.ts
git commit -m "feat: command parser — stateless pure function for /slash commands"
```

---

## Task 4: WebSocket Message Types (ChatCommand + ChatResponse)

**Files:**
- Modify: `modules/shared/src/livetranslate_common/models/ws_messages.py`
- Test: `modules/shared/tests/test_ws_messages.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to existing ws_messages tests (or create new test file):

```python
"""Tests for chat command/response WebSocket messages."""

import pytest

from livetranslate_common.models.ws_messages import (
    ChatCommandMessage,
    ChatResponseMessage,
    ConfigChangedMessage,
    parse_ws_message,
)


class TestChatMessages:
    def test_parse_chat_command(self):
        msg = parse_ws_message('{"type": "chat_command", "command": "/lang zh", "sender": "Alice"}')
        assert isinstance(msg, ChatCommandMessage)
        assert msg.command == "/lang zh"
        assert msg.sender == "Alice"

    def test_parse_chat_response(self):
        msg = parse_ws_message('{"type": "chat_response", "text": "Language set to zh"}')
        assert isinstance(msg, ChatResponseMessage)
        assert msg.text == "Language set to zh"

    def test_parse_config_changed(self):
        msg = parse_ws_message('{"type": "config_changed", "changes": {"target_lang": "zh"}}')
        assert isinstance(msg, ConfigChangedMessage)
        assert msg.changes == {"target_lang": "zh"}
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run pytest modules/shared/tests/test_chat_messages.py -v
```

Expected: `ImportError` (classes don't exist yet)

- [ ] **Step 3: Add the message types**

In `modules/shared/src/livetranslate_common/models/ws_messages.py`, add before the registries (before line 293):

```python
class ChatCommandMessage(BaseModel):
    """Chat command from meeting participant, forwarded by bot."""

    type: Literal["chat_command"] = "chat_command"
    command: str
    sender: str = ""


class ChatResponseMessage(BaseModel):
    """Response text for bot to type in meeting chat."""

    type: Literal["chat_response"] = "chat_response"
    text: str


class ConfigChangedMessage(BaseModel):
    """Notification that MeetingSessionConfig changed."""

    type: Literal["config_changed"] = "config_changed"
    changes: dict[str, Any] = {}
```

Add to the registries:

```python
# In _CLIENT_MESSAGES (line 293):
"chat_command": ChatCommandMessage,

# In _SERVER_MESSAGES (line 302):
"chat_response": ChatResponseMessage,
"config_changed": ConfigChangedMessage,
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest modules/shared/tests/test_chat_messages.py -v
```

Expected: All 3 PASS

- [ ] **Step 5: Commit**

```bash
git add modules/shared/src/livetranslate_common/models/ws_messages.py modules/shared/tests/test_chat_messages.py
git commit -m "feat: ChatCommand, ChatResponse, ConfigChanged WebSocket message types"
```

---

## Task 5: Command Dispatcher (Python — Config Mutation)

**Files:**
- Create: `modules/orchestration-service/tests/test_command_dispatcher.py`
- Create: `modules/orchestration-service/src/services/command_dispatcher.py`

- [ ] **Step 1: Write the failing tests**

Create `modules/orchestration-service/tests/test_command_dispatcher.py`:

```python
"""Tests for CommandDispatcher — routes chat commands to MeetingSessionConfig."""

import pytest

from services.command_dispatcher import CommandDispatcher
from services.meeting_session_config import MeetingSessionConfig


class TestCommandDispatcher:
    def setup_method(self):
        self.config = MeetingSessionConfig(session_id="test-123")
        self.dispatcher = CommandDispatcher(self.config)

    def test_set_language_single(self):
        result = self.dispatcher.dispatch("/lang zh", sender="Alice")
        assert result.response_text == "✓ Translating: auto-detect → zh"
        assert self.config.target_lang == "zh"
        assert self.config.source_lang == "auto"

    def test_set_language_pair(self):
        result = self.dispatcher.dispatch("/lang zh-en", sender="Alice")
        assert result.response_text == "✓ Translating: zh → en"
        assert self.config.source_lang == "zh"
        assert self.config.target_lang == "en"

    def test_font_up(self):
        result = self.dispatcher.dispatch("/font up", sender="Alice")
        assert self.config.font_size == 28  # 24 + 4
        assert "28" in result.response_text

    def test_font_down(self):
        result = self.dispatcher.dispatch("/font down", sender="Alice")
        assert self.config.font_size == 20  # 24 - 4
        assert "20" in result.response_text

    def test_font_exact(self):
        result = self.dispatcher.dispatch("/font 32", sender="Alice")
        assert self.config.font_size == 32

    def test_mode_change(self):
        result = self.dispatcher.dispatch("/mode split", sender="Alice")
        assert self.config.display_mode == "split"
        assert "split" in result.response_text

    def test_theme_change(self):
        result = self.dispatcher.dispatch("/theme light", sender="Alice")
        assert self.config.theme == "light"

    def test_theme_contrast_alias(self):
        result = self.dispatcher.dispatch("/theme contrast", sender="Alice")
        assert self.config.theme == "high_contrast"

    def test_speakers_toggle(self):
        result = self.dispatcher.dispatch("/speakers off", sender="Alice")
        assert self.config.show_speakers is False

    def test_original_toggle(self):
        result = self.dispatcher.dispatch("/original on", sender="Alice")
        assert self.config.show_original is True

    def test_source_switch(self):
        result = self.dispatcher.dispatch("/source fireflies", sender="Alice")
        assert self.config.caption_source == "fireflies"

    def test_translate_toggle(self):
        result = self.dispatcher.dispatch("/translate off", sender="Alice")
        assert self.config.translation_enabled is False

    def test_status_query(self):
        result = self.dispatcher.dispatch("/status", sender="Alice")
        assert "subtitle" in result.response_text  # default mode
        assert "en" in result.response_text  # default target lang

    def test_help_query(self):
        result = self.dispatcher.dispatch("/help", sender="Alice")
        assert "/lang" in result.response_text
        assert "/font" in result.response_text

    def test_unknown_command(self):
        result = self.dispatcher.dispatch("/unknown blah", sender="Alice")
        assert "unknown" in result.response_text.lower() or "/help" in result.response_text

    def test_non_command_ignored(self):
        result = self.dispatcher.dispatch("hello everyone", sender="Alice")
        assert result is None

    def test_returns_changed_fields(self):
        result = self.dispatcher.dispatch("/lang zh", sender="Alice")
        assert "target_lang" in result.changed_fields
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest modules/orchestration-service/tests/test_command_dispatcher.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement CommandDispatcher**

Create `modules/orchestration-service/src/services/command_dispatcher.py`:

```python
"""CommandDispatcher — routes chat commands to MeetingSessionConfig.

Receives raw command strings from the bot WebSocket, parses them,
applies config changes, and returns response text for the bot to
type in meeting chat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from livetranslate_common.logging import get_logger

logger = get_logger()

VALID_MODES = {"subtitle", "split", "interpreter"}
VALID_THEMES = {
    "dark": "dark",
    "light": "light",
    "contrast": "high_contrast",
    "high_contrast": "high_contrast",
    "minimal": "minimal",
    "corporate": "corporate",
}
VALID_SOURCES = {"bot": "bot_audio", "fireflies": "fireflies"}

HELP_TEXT = (
    "Commands: /lang <code>, /font up|down|<size>, "
    "/mode subtitle|split|interpreter, /theme dark|light|contrast, "
    "/speakers on|off, /original on|off, /source bot|fireflies, "
    "/translate on|off, /status, /help"
)


@dataclass
class DispatchResult:
    """Result of dispatching a chat command."""

    response_text: str
    changed_fields: set[str]


def _parse_toggle(value: str) -> bool | None:
    if value in ("on", "true", "yes"):
        return True
    if value in ("off", "false", "no"):
        return False
    return None


class CommandDispatcher:
    """Routes chat commands to MeetingSessionConfig mutations."""

    def __init__(self, config: Any):
        self._config = config

    def dispatch(self, text: str, sender: str = "") -> DispatchResult | None:
        """Parse and execute a command. Returns None for non-commands."""
        trimmed = text.strip()
        if not trimmed.startswith("/"):
            return None

        parts = trimmed.split()
        cmd = parts[0].lower()
        arg = parts[1].lower() if len(parts) > 1 else ""

        logger.info("chat_command_received", command=cmd, arg=arg, sender=sender)

        if cmd == "/lang":
            return self._handle_lang(arg, trimmed)
        elif cmd == "/font":
            return self._handle_font(arg, trimmed)
        elif cmd == "/mode":
            return self._handle_mode(arg, trimmed)
        elif cmd == "/theme":
            return self._handle_theme(arg, trimmed)
        elif cmd == "/speakers":
            return self._handle_toggle("show_speakers", "Speaker names", arg, trimmed)
        elif cmd == "/original":
            return self._handle_toggle("show_original", "Original text", arg, trimmed)
        elif cmd == "/translate":
            return self._handle_toggle("translation_enabled", "Translation", arg, trimmed)
        elif cmd == "/source":
            return self._handle_source(arg, trimmed)
        elif cmd == "/status":
            return self._handle_status()
        elif cmd == "/help":
            return DispatchResult(response_text=HELP_TEXT, changed_fields=set())
        else:
            return DispatchResult(
                response_text=f"Unknown command: {cmd}. Type /help for commands.",
                changed_fields=set(),
            )

    def _handle_lang(self, arg: str, raw: str) -> DispatchResult:
        if not arg:
            return DispatchResult(response_text="Usage: /lang <code> or /lang <src>-<tgt>", changed_fields=set())
        if "-" in arg:
            source, target = arg.split("-", 1)
            changed = self._config.update(source_lang=source, target_lang=target)
            return DispatchResult(response_text=f"✓ Translating: {source} → {target}", changed_fields=changed)
        changed = self._config.update(source_lang="auto", target_lang=arg)
        return DispatchResult(response_text=f"✓ Translating: auto-detect → {arg}", changed_fields=changed)

    def _handle_font(self, arg: str, raw: str) -> DispatchResult:
        if arg == "up":
            new_size = self._config.font_size + 4
            changed = self._config.update(font_size=new_size)
            return DispatchResult(response_text=f"✓ Font size: {new_size}", changed_fields=changed)
        elif arg == "down":
            new_size = max(8, self._config.font_size - 4)
            changed = self._config.update(font_size=new_size)
            return DispatchResult(response_text=f"✓ Font size: {new_size}", changed_fields=changed)
        else:
            try:
                size = int(arg)
                if size > 0:
                    changed = self._config.update(font_size=size)
                    return DispatchResult(response_text=f"✓ Font size: {size}", changed_fields=changed)
            except ValueError:
                pass
            return DispatchResult(response_text="Usage: /font up|down|<size>", changed_fields=set())

    def _handle_mode(self, arg: str, raw: str) -> DispatchResult:
        if arg in VALID_MODES:
            changed = self._config.update(display_mode=arg)
            return DispatchResult(response_text=f"✓ Display mode: {arg}", changed_fields=changed)
        return DispatchResult(response_text="Usage: /mode subtitle|split|interpreter", changed_fields=set())

    def _handle_theme(self, arg: str, raw: str) -> DispatchResult:
        theme = VALID_THEMES.get(arg)
        if theme:
            changed = self._config.update(theme=theme)
            return DispatchResult(response_text=f"✓ Theme: {theme}", changed_fields=changed)
        return DispatchResult(response_text="Usage: /theme dark|light|contrast|minimal|corporate", changed_fields=set())

    def _handle_toggle(self, field: str, label: str, arg: str, raw: str) -> DispatchResult:
        val = _parse_toggle(arg)
        if val is not None:
            changed = self._config.update(**{field: val})
            state = "on" if val else "off"
            return DispatchResult(response_text=f"✓ {label}: {state}", changed_fields=changed)
        return DispatchResult(response_text=f"Usage: /{raw.split()[0][1:]} on|off", changed_fields=set())

    def _handle_source(self, arg: str, raw: str) -> DispatchResult:
        source = VALID_SOURCES.get(arg)
        if source:
            changed = self._config.update(caption_source=source)
            return DispatchResult(response_text=f"✓ Caption source: {source}", changed_fields=changed)
        return DispatchResult(response_text="Usage: /source bot|fireflies", changed_fields=set())

    def _handle_status(self) -> DispatchResult:
        snap = self._config.snapshot()
        lines = [
            f"Lang: {snap['source_lang']} → {snap['target_lang']}",
            f"Mode: {snap['display_mode']} | Theme: {snap['theme']}",
            f"Font: {snap['font_size']} | Speakers: {'on' if snap['show_speakers'] else 'off'}",
            f"Source: {snap['caption_source']} | Translation: {'on' if snap['translation_enabled'] else 'off'}",
        ]
        return DispatchResult(response_text=" | ".join(lines), changed_fields=set())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest modules/orchestration-service/tests/test_command_dispatcher.py -v
```

Expected: All 18 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/command_dispatcher.py modules/orchestration-service/tests/test_command_dispatcher.py
git commit -m "feat: CommandDispatcher — routes chat commands to MeetingSessionConfig"
```

---

## Task 6: Chat Poller (TypeScript — DOM Polling)

**Files:**
- Create: `modules/meeting-bot-service/src/chat/chat_poller.ts`

- [ ] **Step 1: Implement the chat poller**

```typescript
/**
 * Chat Poller — polls Google Meet chat DOM every 500ms.
 *
 * Opens the chat panel, reads new messages, detects /commands.
 * More reliable than MutationObserver (survives panel open/close,
 * Google Meet DOM restructures).
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import { CHAT_BUTTON, CHAT_MESSAGES_CONTAINER } from './selectors';

export interface ChatMessage {
  sender: string;
  text: string;
  timestamp: number;
}

export type OnCommandCallback = (command: string, sender: string) => void;

export class ChatPoller {
  private page: Page;
  private logger: Logger;
  private onCommand: OnCommandCallback;
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private seenMessages: Set<string> = new Set();
  private isRunning = false;

  constructor(page: Page, logger: Logger, onCommand: OnCommandCallback) {
    this.page = page;
    this.logger = logger;
    this.onCommand = onCommand;
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    // Open chat panel
    try {
      const chatButton = this.page.locator(CHAT_BUTTON).first();
      if (await chatButton.isVisible({ timeout: 3000 })) {
        await chatButton.click();
        this.logger.info('Chat panel opened');
        await this.page.waitForTimeout(500);
      }
    } catch (err) {
      this.logger.warn('Could not open chat panel', { error: (err as Error).message });
    }

    this.isRunning = true;
    this.pollInterval = setInterval(() => this.poll(), 500);
    this.logger.info('Chat poller started');
  }

  stop(): void {
    this.isRunning = false;
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.logger.info('Chat poller stopped');
  }

  private async poll(): Promise<void> {
    if (!this.isRunning) return;

    try {
      const messages = await this.page.evaluate((containerSelector: string) => {
        const container = document.querySelector(containerSelector);
        if (!container) return [];

        const items = container.querySelectorAll('[data-message-text]');
        const results: Array<{ sender: string; text: string; key: string }> = [];

        items.forEach((item) => {
          const text = item.getAttribute('data-message-text') || item.textContent || '';
          // Try to find the sender name from the message structure
          const senderEl = item.closest('[data-sender-name]');
          const sender = senderEl?.getAttribute('data-sender-name') || 'Unknown';
          const key = `${sender}:${text}`;
          results.push({ sender, text, key });
        });

        return results;
      }, CHAT_MESSAGES_CONTAINER);

      for (const msg of messages) {
        if (this.seenMessages.has(msg.key)) continue;
        this.seenMessages.add(msg.key);

        if (msg.text.trim().startsWith('/')) {
          this.logger.info('Chat command detected', { sender: msg.sender, command: msg.text });
          this.onCommand(msg.text.trim(), msg.sender);
        }
      }
    } catch (err) {
      // Chat panel might be closed or DOM changed — log and continue
      this.logger.debug('Chat poll error', { error: (err as Error).message });
    }
  }
}
```

Note: The exact DOM selectors for reading chat messages (`[data-message-text]`, `[data-sender-name]`) may need adjustment based on actual Google Meet DOM. This will be validated during manual testing (Phase 2 manual test protocol, test 3). The `selectors.ts` file is the single place to update when Meet changes.

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/chat/chat_poller.ts
git commit -m "feat: ChatPoller — polls Meet chat DOM every 500ms for /commands"
```

---

## Task 7: Chat Responder (TypeScript — Type in Chat)

**Files:**
- Create: `modules/meeting-bot-service/src/chat/chat_responder.ts`

- [ ] **Step 1: Implement the chat responder**

```typescript
/**
 * Chat Responder — types bot responses into Google Meet chat.
 *
 * Handles the contenteditable div (page.fill doesn't work on these).
 * Uses evaluate() to set text + dispatch InputEvent, then clicks send.
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import { CHAT_INPUT, CHAT_SEND_BUTTON, CHAT_BUTTON } from './selectors';

export class ChatResponder {
  private page: Page;
  private logger: Logger;

  constructor(page: Page, logger: Logger) {
    this.page = page;
    this.logger = logger;
  }

  async sendMessage(text: string): Promise<boolean> {
    try {
      // Ensure chat panel is open
      const chatInput = this.page.locator(CHAT_INPUT).first();
      if (!await chatInput.isVisible({ timeout: 1000 })) {
        // Try opening chat panel
        const chatButton = this.page.locator(CHAT_BUTTON).first();
        if (await chatButton.isVisible({ timeout: 1000 })) {
          await chatButton.click();
          await this.page.waitForTimeout(500);
        }
      }

      // Click the input to focus
      await chatInput.click();

      // Set text via evaluate (page.fill doesn't work on contenteditable)
      await chatInput.evaluate((el: HTMLElement, msg: string) => {
        el.textContent = msg;
        el.dispatchEvent(new InputEvent('input', {
          inputType: 'insertText',
          bubbles: true,
        }));
      }, text);

      // Click send button (more reliable than pressing Enter)
      const sendButton = this.page.locator(CHAT_SEND_BUTTON).first();
      await sendButton.click({ timeout: 2000 });

      this.logger.info('Chat message sent', { text: text.substring(0, 50) });
      return true;
    } catch (err) {
      this.logger.warn('Failed to send chat message', {
        error: (err as Error).message,
        text: text.substring(0, 50),
      });
      return false;
    }
  }

  async sendJoinMessage(): Promise<void> {
    await this.sendMessage(
      'LiveTranslate bot active. Pin my video for subtitles. Type /help for commands.'
    );
  }
}
```

- [ ] **Step 2: Create the index re-export**

Create `modules/meeting-bot-service/src/chat/index.ts`:

```typescript
export { parseCommand } from './command_parser';
export type { CommandResult } from './command_parser';
export { ChatPoller } from './chat_poller';
export type { ChatMessage, OnCommandCallback } from './chat_poller';
export { ChatResponder } from './chat_responder';
export * from './selectors';
```

- [ ] **Step 3: Commit**

```bash
git add modules/meeting-bot-service/src/chat/chat_responder.ts modules/meeting-bot-service/src/chat/index.ts
git commit -m "feat: ChatResponder + chat module index — bot types in Meet chat"
```

---

## Task 8: Wire Chat Commands into AudioStreamer WebSocket

**Files:**
- Modify: `modules/meeting-bot-service/src/audio_streaming.ts:109-141`

- [ ] **Step 1: Add chat_command sending and chat_response/config_changed receiving**

In `modules/meeting-bot-service/src/audio_streaming.ts`, add a public method for sending chat commands:

```typescript
// Add to AudioStreamer class:

sendChatCommand(command: string, sender: string): void {
  this.send({
    type: 'chat_command',
    command,
    sender,
  });
}
```

In the `handleMessage` method (line 109-141), add cases for new message types:

```typescript
// Add to handleMessage switch/if chain:

case 'chat_response':
  if (this.onChatResponse) {
    this.onChatResponse(parsed.text);
  }
  break;

case 'config_changed':
  if (this.onConfigChanged) {
    this.onConfigChanged(parsed.changes);
  }
  break;
```

Add callback properties to the class:

```typescript
// Add to class properties:
public onChatResponse: ((text: string) => void) | null = null;
public onConfigChanged: ((changes: Record<string, any>) => void) | null = null;
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/audio_streaming.ts
git commit -m "feat: AudioStreamer handles chat_command, chat_response, config_changed messages"
```

---

## Task 9: Wire Chat System into GoogleMeetBot

**Files:**
- Modify: `modules/meeting-bot-service/src/bots/GoogleMeetBot.ts`

- [ ] **Step 1: Integrate ChatPoller and ChatResponder into the bot**

In `GoogleMeetBot.ts`, add imports:

```typescript
import { ChatPoller } from '../chat/chat_poller';
import { ChatResponder } from '../chat/chat_responder';
```

Add properties:

```typescript
private chatPoller: ChatPoller | null = null;
private chatResponder: ChatResponder | null = null;
```

After the bot joins the meeting and audio streaming is set up (after line 475), add chat system initialization:

```typescript
// Set up chat system
this.chatResponder = new ChatResponder(this.page, this._logger);
this.chatPoller = new ChatPoller(this.page, this._logger, (command, sender) => {
  // Forward commands to orchestration via WebSocket
  if (this.audioStreamer) {
    this.audioStreamer.sendChatCommand(command, sender);
  }
});

// Wire chat responses from orchestration
if (this.audioStreamer) {
  this.audioStreamer.onChatResponse = async (text: string) => {
    if (this.chatResponder) {
      await this.chatResponder.sendMessage(text);
    }
  };
}

// Start polling chat for commands
await this.chatPoller.start();

// Send join message
await this.chatResponder.sendJoinMessage();
```

In the `leave()` method, add cleanup:

```typescript
// Stop chat poller
if (this.chatPoller) {
  this.chatPoller.stop();
  this.chatPoller = null;
}
this.chatResponder = null;
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/bots/GoogleMeetBot.ts
git commit -m "feat: GoogleMeetBot integrates chat poller, responder, and command forwarding"
```

---

## Task 10: Wire ChatCommandMessage into Orchestration WebSocket Handler

**Files:**
- Modify: `modules/orchestration-service/src/routers/audio/websocket_audio.py`

- [ ] **Step 1: Add ChatCommandMessage handling**

In `websocket_audio.py`, add the import:

```python
from livetranslate_common.models.ws_messages import (
    # ... existing imports ...
    ChatCommandMessage,
    ChatResponseMessage,
    ConfigChangedMessage,
)
from services.command_dispatcher import CommandDispatcher
```

In the message handling section (after the `isinstance(msg, ConfigMessage)` block around line 821), add:

```python
elif isinstance(msg, ChatCommandMessage):
    if command_dispatcher is not None:
        result = command_dispatcher.dispatch(msg.command, sender=msg.sender)
        if result is not None:
            # Send response for bot to type in chat
            await websocket.send_text(
                ChatResponseMessage(text=result.response_text).model_dump_json()
            )
            # Notify all listeners of config changes
            if result.changed_fields:
                await websocket.send_text(
                    ConfigChangedMessage(
                        changes={f: getattr(meeting_config, f) for f in result.changed_fields}
                    ).model_dump_json()
                )
```

Earlier in the handler (where session state is initialized), create the dispatcher:

```python
# After MeetingSessionConfig is created (or create it here):
meeting_config = MeetingSessionConfig(session_id=session_id)
command_dispatcher = CommandDispatcher(meeting_config)
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/audio/websocket_audio.py
git commit -m "feat: orchestration WebSocket handles ChatCommandMessage, dispatches to config"
```

---

## Task 11: Full Test Suite Validation + Build Verification

**Files:** None (validation only)

- [ ] **Step 1: Run TypeScript tests**

```bash
cd modules/meeting-bot-service && npx vitest run 2>&1 | tail -10
```

Expected: All command parser tests pass

- [ ] **Step 2: Run Python tests**

```bash
uv run pytest modules/shared/tests/ -v -q --timeout=30 2>&1 | tail -5
uv run pytest modules/orchestration-service/tests/ -v -q --timeout=30 -m "not e2e and not slow" 2>&1 | tail -10
```

Expected: All pass, no new failures

- [ ] **Step 3: TypeScript build check**

```bash
cd modules/meeting-bot-service && npx tsc --noEmit 2>&1 | tail -10
```

Expected: No type errors

- [ ] **Step 4: Commit any fixes**

---

## Summary

| Task | What | Layer |
|------|------|-------|
| 1 | Google Meet selectors (single source of truth) | TypeScript |
| 2 | Set up Vitest | TypeScript |
| 3 | Command parser (pure function, 26 tests) | TypeScript |
| 4 | WebSocket message types (ChatCommand/Response) | Shared (Python) |
| 5 | CommandDispatcher (config mutation, 18 tests) | Python |
| 6 | Chat poller (DOM polling every 500ms) | TypeScript |
| 7 | Chat responder (contenteditable typing) | TypeScript |
| 8 | AudioStreamer chat command support | TypeScript |
| 9 | GoogleMeetBot chat integration | TypeScript |
| 10 | Orchestration WebSocket chat handling | Python |
| 11 | Full validation | Both |

## Manual Testing After Phase 2

After all 11 tasks, the system needs manual testing from the spec:

1. **Bot join test** — Start bot, join real Google Meet, verify join message in chat
2. **Chat command test** — Type `/lang zh`, verify bot confirms, verify config changes
3. **Font/mode/theme test** — Test each command live
4. **Source switch** — `/source fireflies` then `/source bot`
5. **OBS overlay sync** — Verify overlay updates when chat commands change config

These require a running meeting and cannot be automated without `agent-browser`.
