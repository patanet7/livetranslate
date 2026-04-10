# Meeting Subtitle System — Phase 3: Polish, Source Switching & Overlay Sync

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the meeting subtitle pipeline — live overlay config sync, Fireflies caption source, graceful source switching, CJK text wrapping, error resilience, MeetCaptionsAdapter, and real-data validation fixtures.

**Architecture:** Fill remaining gaps: (1) SvelteKit overlay handles `config_changed` WebSocket events for live switching, (2) `FirefliesCaptionSource` wraps existing `FirefliesChunkAdapter` as a `CaptionSourceAdapter`, (3) source orchestrator manages clean handoff between sources, (4) PIL text renderer gains word/character wrapping for mixed scripts, (5) `MeetCaptionsAdapter` scrapes Google Meet's built-in CC DOM, (6) real meeting fixture data validates the full pipeline end-to-end.

**Tech Stack:** Python (orchestration-service), TypeScript (meeting-bot-service, dashboard-service), SvelteKit + Svelte 5, PIL/Pillow, Playwright, Pydantic v2, Vitest

**Spec:** `docs/superpowers/specs/2026-04-10-meeting-subtitle-system-design.md` (Phase 3 section)

**Depends on:** Phase 0+1 (merged) + Phase 2 (merged). Specifically: `MeetingSessionConfig`, `CaptionSourceAdapter` protocol, `CaptionBuffer` multi-subscriber, `CommandDispatcher`, chat system, `ConfigChangedMessage`.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `modules/orchestration-service/src/services/pipeline/adapters/fireflies_caption_source.py` | `FirefliesCaptionSource` — wraps `FirefliesChunkAdapter` as a `CaptionSourceAdapter` with lifecycle |
| `modules/orchestration-service/tests/test_fireflies_caption_source.py` | Tests for FirefliesCaptionSource |
| `modules/orchestration-service/src/services/source_orchestrator.py` | Manages active source lifecycle: clean stop → start handoff on config change |
| `modules/orchestration-service/tests/test_source_orchestrator.py` | Tests for source switching |
| `modules/orchestration-service/src/bot/text_wrapper.py` | Multi-script text wrapping (Latin word-break, CJK char-break, mixed) |
| `modules/orchestration-service/tests/test_text_wrapper.py` | Tests for text wrapping |
| `modules/meeting-bot-service/src/chat/meet_captions_adapter.ts` | Scrapes Google Meet's built-in CC DOM as a caption source |
| `modules/meeting-bot-service/tests/chat/meet_captions_adapter.test.ts` | Tests for MeetCaptionsAdapter parsing |
| `modules/orchestration-service/tests/fixtures/meeting_data/` | Real meeting fixture directory |
| `modules/orchestration-service/tests/test_real_data_validation.py` | End-to-end pipeline validation with real fixtures |

### Modified Files

| File | Change |
|------|--------|
| `modules/dashboard-service/src/routes/(overlay)/captions/+page.svelte` | Handle `config_changed` WS event, make display config reactive to live changes |
| `modules/dashboard-service/src/lib/types/caption.ts` | Add `ConfigChangedEvent` to CaptionEvent union |
| `modules/orchestration-service/src/bot/virtual_webcam.py` | Use `text_wrapper` for all text rendering |
| `modules/orchestration-service/src/services/pipeline/adapters/source_adapter.py` | Add `FirefliesCaptionSource` import to `__init__` pattern |
| `modules/orchestration-service/src/routers/audio/websocket_audio.py` | Integrate `SourceOrchestrator` for config-driven source switching |
| `modules/meeting-bot-service/src/chat/selectors.ts` | Add Google Meet CC DOM selectors |
| `modules/meeting-bot-service/src/chat/index.ts` | Re-export `MeetCaptionsAdapter` |
| `modules/meeting-bot-service/src/audio_streaming.ts` | Add `sendCaptionEvent` for Meet CC forwarding |

---

## Task 1: Overlay Config Sync — Handle `config_changed` WebSocket Event

**Files:**
- Modify: `modules/dashboard-service/src/routes/(overlay)/captions/+page.svelte`
- Modify: `modules/dashboard-service/src/lib/types/caption.ts`

- [ ] **Step 1: Add `config_changed` to the CaptionEvent type**

In `modules/dashboard-service/src/lib/types/caption.ts`, check what event types exist. Add `config_changed` to the discriminated union so the overlay WS handler can type-narrow it. Read the file first to find the exact type definition.

The overlay page currently has URL-param-only config. We need to add `$state` variables that start from URL params but can be overwritten by `config_changed` WebSocket messages.

- [ ] **Step 2: Make overlay config reactive to WebSocket**

In `modules/dashboard-service/src/routes/(overlay)/captions/+page.svelte`, change the config variables from `$derived` (URL-only) to `$state` (mutable). Initialize from URL params, then update on `config_changed`:

```typescript
// BEFORE (URL-only, immutable):
const fontSize = $derived.by(() => {
  const raw = parseInt($page.url.searchParams.get('fontSize') ?? '18', 10);
  return isNaN(raw) ? 18 : raw;
});

// AFTER (mutable, starts from URL, updated by WS):
let liveShowSpeaker = $state(true);
let liveFontSize = $state(18);
let liveShowOriginal = $state(true);
let liveMode = $state<string>('both');
let liveTheme = $state<string>('dark');

// Initialize from URL params on mount
$effect(() => {
  liveFontSize = parseInt($page.url.searchParams.get('fontSize') ?? '18', 10) || 18;
  liveShowSpeaker = $page.url.searchParams.get('showSpeaker') !== 'false';
  liveShowOriginal = $page.url.searchParams.get('showOriginal') !== 'false';
  liveMode = $page.url.searchParams.get('mode') ?? 'both';
});
```

In the `ws.onMessage` handler, add:

```typescript
case 'config_changed':
  if (msg.changes) {
    if ('font_size' in msg.changes) liveFontSize = msg.changes.font_size;
    if ('show_speakers' in msg.changes) liveShowSpeaker = msg.changes.show_speakers;
    if ('show_original' in msg.changes) liveShowOriginal = msg.changes.show_original;
    if ('display_mode' in msg.changes) liveMode = msg.changes.display_mode;
    if ('theme' in msg.changes) liveTheme = msg.changes.theme;
  }
  break;
```

Replace all template references to the old `$derived` variables with the new `$state` ones.

- [ ] **Step 3: Run dashboard dev to verify no build errors**

```bash
cd modules/dashboard-service && npm run check 2>&1 | tail -10
```

- [ ] **Step 4: Commit**

```bash
git add modules/dashboard-service/src/routes/\(overlay\)/captions/+page.svelte modules/dashboard-service/src/lib/types/caption.ts
git commit -m "feat: overlay handles config_changed WS events for live config switching"
```

---

## Task 2: Text Wrapper — Multi-Script Line Breaking (TDD)

**Files:**
- Create: `modules/orchestration-service/tests/test_text_wrapper.py`
- Create: `modules/orchestration-service/src/bot/text_wrapper.py`

- [ ] **Step 1: Write the failing tests**

Create `modules/orchestration-service/tests/test_text_wrapper.py`:

```python
"""Tests for multi-script text wrapping (Latin word-break, CJK char-break, mixed)."""

import pytest
from bot.text_wrapper import wrap_text


class TestWrapText:
    def test_short_text_no_wrap(self):
        """Text shorter than max_width returns single line."""
        result = wrap_text("Hello world", max_chars=40)
        assert result == ["Hello world"]

    def test_english_word_wrap(self):
        """English text wraps at word boundaries."""
        text = "The quick brown fox jumps over the lazy dog"
        result = wrap_text(text, max_chars=20)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 20

    def test_english_preserves_words(self):
        """Word wrapping doesn't split mid-word."""
        text = "Hello wonderful world"
        result = wrap_text(text, max_chars=12)
        # "Hello" + "wonderful" + "world" — should not split "wonderful"
        assert all(" " not in line.strip() or len(line) <= 12 for line in result)

    def test_cjk_char_wrap(self):
        """CJK text wraps at character boundaries (no spaces)."""
        text = "这是一个很长的中文句子需要换行显示"
        result = wrap_text(text, max_chars=8)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 8

    def test_mixed_script_wrap(self):
        """Mixed CJK + Latin wraps correctly."""
        text = "Hello 你好世界 World 再见"
        result = wrap_text(text, max_chars=10)
        assert len(result) >= 2
        # Rejoin should preserve all text
        assert "".join(result).replace(" ", "") == text.replace(" ", "").replace(" ", "")

    def test_empty_string(self):
        result = wrap_text("", max_chars=40)
        assert result == [""]

    def test_max_lines_truncation(self):
        """Respects max_lines parameter."""
        text = "Line one. Line two. Line three. Line four. Line five."
        result = wrap_text(text, max_chars=12, max_lines=3)
        assert len(result) <= 3

    def test_single_long_word_forced_break(self):
        """A single word longer than max_chars is force-broken."""
        text = "Supercalifragilisticexpialidocious"
        result = wrap_text(text, max_chars=10)
        assert len(result) >= 3
        for line in result:
            assert len(line) <= 10

    def test_japanese_wrap(self):
        """Japanese text (hiragana/katakana) wraps at char boundaries."""
        text = "これはテストの文章です"
        result = wrap_text(text, max_chars=5)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 5

    def test_korean_wrap(self):
        """Korean text wraps at character boundaries."""
        text = "안녕하세요반갑습니다"
        result = wrap_text(text, max_chars=5)
        assert len(result) >= 2
        for line in result:
            assert len(line) <= 5
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest modules/orchestration-service/tests/test_text_wrapper.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the text wrapper**

Create `modules/orchestration-service/src/bot/text_wrapper.py`:

```python
"""Multi-script text wrapping for subtitle rendering.

Handles Latin (word-break), CJK (char-break), and mixed text.
CJK characters are detected via Unicode ranges:
  - CJK Unified Ideographs: U+4E00–U+9FFF
  - Hiragana: U+3040–U+309F
  - Katakana: U+30A0–U+30FF
  - Hangul Syllables: U+AC00–U+D7AF
  - CJK Extension A: U+3400–U+4DBF
  - Full-width punctuation: U+3000–U+303F
"""

from __future__ import annotations


def _is_cjk(ch: str) -> bool:
    """Check if a character is CJK (Chinese, Japanese, Korean)."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF      # CJK Unified Ideographs
        or 0x3040 <= cp <= 0x309F    # Hiragana
        or 0x30A0 <= cp <= 0x30FF    # Katakana
        or 0xAC00 <= cp <= 0xD7AF    # Hangul Syllables
        or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
        or 0x3000 <= cp <= 0x303F    # CJK Punctuation
        or 0xFF00 <= cp <= 0xFFEF    # Full-width forms
    )


def wrap_text(text: str, max_chars: int = 40, max_lines: int = 0) -> list[str]:
    """Wrap text for subtitle display, handling multi-script content.

    Args:
        text: Input text to wrap.
        max_chars: Maximum characters per line.
        max_lines: Maximum number of lines (0 = unlimited).

    Returns:
        List of wrapped lines.
    """
    if not text:
        return [""]

    lines: list[str] = []
    current_line = ""

    i = 0
    while i < len(text):
        ch = text[i]

        # If adding this char would exceed the limit
        if len(current_line) >= max_chars:
            # For CJK: break at current position
            if _is_cjk(ch) or _is_cjk(current_line[-1]) if current_line else False:
                lines.append(current_line)
                current_line = ""
            else:
                # For Latin: find last space to break at word boundary
                last_space = current_line.rfind(" ")
                if last_space > 0:
                    lines.append(current_line[:last_space])
                    current_line = current_line[last_space + 1:]
                else:
                    # Single long word — force break
                    lines.append(current_line)
                    current_line = ""

        current_line += ch
        i += 1

    # Don't forget the last line
    if current_line:
        lines.append(current_line)

    # Trim lines
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]

    if not lines:
        return [""]

    # Apply max_lines truncation
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]

    return lines
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest modules/orchestration-service/tests/test_text_wrapper.py -v 2>&1 | tail -15
```

Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/bot/text_wrapper.py modules/orchestration-service/tests/test_text_wrapper.py
git commit -m "feat: multi-script text wrapper — Latin word-break, CJK char-break, mixed"
```

---

## Task 3: Wire Text Wrapper into VirtualWebcamManager

**Files:**
- Modify: `modules/orchestration-service/src/bot/virtual_webcam.py`

- [ ] **Step 1: Add import and integrate text wrapping**

In `modules/orchestration-service/src/bot/virtual_webcam.py`, add import:

```python
from bot.text_wrapper import wrap_text
```

Find the `_draw_translation_box` method (line ~547). Currently it draws text at a fixed position without wrapping. Modify to use `wrap_text`:

1. Calculate `max_chars` based on box width and approximate font character width
2. Call `wrap_text(text, max_chars=max_chars, max_lines=3)`
3. Draw each line at incrementing y offsets

Apply the same wrapping to `_draw_sidebar_translation`, `_draw_banner_translation`, and `_draw_centered_translation`.

- [ ] **Step 2: Run existing virtual webcam tests**

```bash
uv run pytest modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py -v 2>&1 | tail -10
```

Expected: All existing tests still pass

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/bot/virtual_webcam.py
git commit -m "feat: VirtualWebcamManager uses text_wrapper for multi-script line breaking"
```

---

## Task 4: FirefliesCaptionSource — Wrap ChunkAdapter as CaptionSourceAdapter (TDD)

**Files:**
- Create: `modules/orchestration-service/tests/test_fireflies_caption_source.py`
- Create: `modules/orchestration-service/src/services/pipeline/adapters/fireflies_caption_source.py`

- [ ] **Step 1: Write the failing tests**

Create `modules/orchestration-service/tests/test_fireflies_caption_source.py`:

```python
"""Tests for FirefliesCaptionSource — wraps FirefliesChunkAdapter as CaptionSourceAdapter."""

import asyncio
import pytest
from services.pipeline.adapters.fireflies_caption_source import FirefliesCaptionSource
from services.pipeline.adapters.source_adapter import CaptionEvent, CaptionSourceAdapter


class TestFirefliesCaptionSource:
    def test_implements_protocol(self):
        """FirefliesCaptionSource satisfies CaptionSourceAdapter protocol."""
        source = FirefliesCaptionSource()
        assert isinstance(source, CaptionSourceAdapter)

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        source = FirefliesCaptionSource()
        assert not source.is_running
        await source.start(config=None)
        assert source.is_running
        await source.stop()
        assert not source.is_running

    @pytest.mark.asyncio
    async def test_handle_transcript_emits_caption_event(self):
        source = FirefliesCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        await source.start(config=None)

        await source.handle_chunk({
            "type": "transcript",
            "transcript_id": "t1",
            "chunk_id": "c1",
            "text": "Hello world",
            "speaker_name": "Alice",
            "start_time": 0.0,
            "end_time": 1.25,
        })

        assert len(events) == 1
        assert events[0].text == "Hello world"
        assert events[0].speaker_name == "Alice"
        assert events[0].event_type == "added"

    @pytest.mark.asyncio
    async def test_speaker_colors_assigned(self):
        source = FirefliesCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        await source.start(config=None)

        await source.handle_chunk({"text": "Hi", "speaker_name": "Alice", "chunk_id": "c1", "start_time": 0, "end_time": 1})
        await source.handle_chunk({"text": "Hey", "speaker_name": "Bob", "chunk_id": "c2", "start_time": 1, "end_time": 2})
        await source.handle_chunk({"text": "Again", "speaker_name": "Alice", "chunk_id": "c3", "start_time": 2, "end_time": 3})

        # Alice gets same color both times, Bob gets a different color
        assert events[0].speaker_color == events[2].speaker_color
        assert events[0].speaker_color != events[1].speaker_color

    @pytest.mark.asyncio
    async def test_ignores_events_when_stopped(self):
        source = FirefliesCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        # Don't start

        await source.handle_chunk({"text": "Hi", "speaker_name": "Alice", "chunk_id": "c1", "start_time": 0, "end_time": 1})
        assert len(events) == 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest modules/orchestration-service/tests/test_fireflies_caption_source.py -v 2>&1 | tail -10
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement FirefliesCaptionSource**

Create `modules/orchestration-service/src/services/pipeline/adapters/fireflies_caption_source.py`:

```python
"""FirefliesCaptionSource — wraps FirefliesChunkAdapter as a CaptionSourceAdapter.

Receives Fireflies transcript chunks, converts via the existing ChunkAdapter,
and emits CaptionEvents for the CaptionBuffer/renderers.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Callable

from livetranslate_common.logging import get_logger
from livetranslate_common.theme import SPEAKER_COLORS

from .source_adapter import CaptionEvent

logger = get_logger()


class FirefliesCaptionSource:
    """Lifecycle-managing Fireflies caption source."""

    def __init__(self) -> None:
        self.is_running: bool = False
        self.on_caption: Callable[[CaptionEvent], Any] | None = None
        self._speaker_color_idx: int = 0
        self._speaker_colors: dict[str, str] = {}

    async def start(self, config: Any) -> None:
        self.is_running = True
        logger.info("fireflies_caption_source_started")

    async def stop(self) -> None:
        self.is_running = False
        logger.info("fireflies_caption_source_stopped")

    def _get_speaker_color(self, speaker_name: str) -> str:
        if speaker_name not in self._speaker_colors:
            self._speaker_colors[speaker_name] = SPEAKER_COLORS[
                self._speaker_color_idx % len(SPEAKER_COLORS)
            ]
            self._speaker_color_idx += 1
        return self._speaker_colors[speaker_name]

    async def handle_chunk(self, raw_chunk: dict[str, Any]) -> None:
        """Process a Fireflies transcript chunk into a CaptionEvent."""
        if not self.is_running or not self.on_caption:
            return

        text = raw_chunk.get("text", "")
        speaker_name = raw_chunk.get("speaker_name", "Unknown")
        chunk_id = raw_chunk.get("chunk_id", str(uuid.uuid4()))

        event = CaptionEvent(
            event_type="added",
            caption_id=chunk_id,
            text=text,
            speaker_name=speaker_name,
            speaker_color=self._get_speaker_color(speaker_name),
            source_lang="auto",
            confidence=1.0,
            is_draft=False,
        )

        result = self.on_caption(event)
        if hasattr(result, "__await__"):
            await result
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest modules/orchestration-service/tests/test_fireflies_caption_source.py -v 2>&1 | tail -10
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/pipeline/adapters/fireflies_caption_source.py modules/orchestration-service/tests/test_fireflies_caption_source.py
git commit -m "feat: FirefliesCaptionSource — wraps Fireflies as CaptionSourceAdapter"
```

---

## Task 5: Source Orchestrator — Clean Source Switching (TDD)

**Files:**
- Create: `modules/orchestration-service/tests/test_source_orchestrator.py`
- Create: `modules/orchestration-service/src/services/source_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `modules/orchestration-service/tests/test_source_orchestrator.py`:

```python
"""Tests for SourceOrchestrator — manages clean source switching."""

import asyncio
import pytest
from services.source_orchestrator import SourceOrchestrator
from services.pipeline.adapters.source_adapter import BotAudioCaptionSource, CaptionEvent
from services.pipeline.adapters.fireflies_caption_source import FirefliesCaptionSource
from services.meeting_session_config import MeetingSessionConfig


class TestSourceOrchestrator:
    def setup_method(self):
        self.config = MeetingSessionConfig(session_id="test-123")
        self.events: list[CaptionEvent] = []
        self.orchestrator = SourceOrchestrator(
            config=self.config,
            on_caption=lambda e: self.events.append(e),
        )

    @pytest.mark.asyncio
    async def test_start_with_bot_audio(self):
        await self.orchestrator.start()
        assert self.orchestrator.active_source is not None
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_start_with_fireflies(self):
        self.config.update(caption_source="fireflies")
        await self.orchestrator.start()
        assert isinstance(self.orchestrator.active_source, FirefliesCaptionSource)
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_switch_source_stops_old_starts_new(self):
        await self.orchestrator.start()
        old_source = self.orchestrator.active_source
        assert isinstance(old_source, BotAudioCaptionSource)

        await self.orchestrator.switch_source("fireflies")
        assert not old_source.is_running
        assert isinstance(self.orchestrator.active_source, FirefliesCaptionSource)
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_switch_back_to_bot_audio(self):
        self.config.update(caption_source="fireflies")
        await self.orchestrator.start()
        await self.orchestrator.switch_source("bot_audio")
        assert isinstance(self.orchestrator.active_source, BotAudioCaptionSource)

    @pytest.mark.asyncio
    async def test_switch_to_same_source_is_noop(self):
        await self.orchestrator.start()
        source_before = self.orchestrator.active_source
        await self.orchestrator.switch_source("bot_audio")
        assert self.orchestrator.active_source is source_before

    @pytest.mark.asyncio
    async def test_events_routed_after_switch(self):
        await self.orchestrator.start()
        await self.orchestrator.switch_source("fireflies")

        ff_source = self.orchestrator.active_source
        assert isinstance(ff_source, FirefliesCaptionSource)
        await ff_source.handle_chunk({"text": "test", "speaker_name": "A", "chunk_id": "c1", "start_time": 0, "end_time": 1})

        assert len(self.events) == 1
        assert self.events[0].text == "test"

    @pytest.mark.asyncio
    async def test_stop_stops_active_source(self):
        await self.orchestrator.start()
        source = self.orchestrator.active_source
        await self.orchestrator.stop()
        assert not source.is_running
        assert self.orchestrator.active_source is None

    @pytest.mark.asyncio
    async def test_config_subscriber_triggers_switch(self):
        await self.orchestrator.start()
        assert isinstance(self.orchestrator.active_source, BotAudioCaptionSource)

        # Config change triggers automatic source switch
        self.config.update(caption_source="fireflies")
        # Give async handler time
        await asyncio.sleep(0.05)

        assert isinstance(self.orchestrator.active_source, FirefliesCaptionSource)
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest modules/orchestration-service/tests/test_source_orchestrator.py -v 2>&1 | tail -10
```

- [ ] **Step 3: Implement SourceOrchestrator**

Create `modules/orchestration-service/src/services/source_orchestrator.py`:

```python
"""SourceOrchestrator — manages clean source switching.

Subscribes to MeetingSessionConfig changes. When caption_source changes,
cleanly stops the old source and starts the new one, preserving the
on_caption callback chain.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from livetranslate_common.logging import get_logger

from services.meeting_session_config import MeetingSessionConfig
from services.pipeline.adapters.fireflies_caption_source import FirefliesCaptionSource
from services.pipeline.adapters.source_adapter import (
    BotAudioCaptionSource,
    CaptionEvent,
    CaptionSourceAdapter,
)

logger = get_logger()

_SOURCE_FACTORIES: dict[str, type] = {
    "bot_audio": BotAudioCaptionSource,
    "fireflies": FirefliesCaptionSource,
}


class SourceOrchestrator:
    """Manages active caption source lifecycle and clean switching."""

    def __init__(
        self,
        config: MeetingSessionConfig,
        on_caption: Callable[[CaptionEvent], Any],
    ) -> None:
        self._config = config
        self._on_caption = on_caption
        self.active_source: CaptionSourceAdapter | None = None
        self._config.subscribe(self._on_config_changed)

    async def start(self) -> None:
        """Start the source specified by current config."""
        source_type = self._config.caption_source
        self.active_source = self._create_source(source_type)
        self.active_source.on_caption = self._on_caption
        await self.active_source.start(self._config)
        logger.info("source_orchestrator_started", source=source_type)

    async def stop(self) -> None:
        """Stop the active source."""
        if self.active_source:
            await self.active_source.stop()
            self.active_source = None
        self._config.unsubscribe(self._on_config_changed)
        logger.info("source_orchestrator_stopped")

    async def switch_source(self, new_source_type: str) -> None:
        """Switch to a different source. Clean stop → start."""
        if self.active_source and self._current_source_type() == new_source_type:
            return

        if self.active_source:
            await self.active_source.stop()

        self.active_source = self._create_source(new_source_type)
        self.active_source.on_caption = self._on_caption
        await self.active_source.start(self._config)
        logger.info("source_switched", new_source=new_source_type)

    def _create_source(self, source_type: str) -> Any:
        factory = _SOURCE_FACTORIES.get(source_type, BotAudioCaptionSource)
        return factory()

    def _current_source_type(self) -> str:
        if isinstance(self.active_source, FirefliesCaptionSource):
            return "fireflies"
        return "bot_audio"

    def _on_config_changed(self, changed_fields: set[str]) -> None:
        if "caption_source" in changed_fields:
            new_source = self._config.caption_source
            asyncio.ensure_future(self.switch_source(new_source))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest modules/orchestration-service/tests/test_source_orchestrator.py -v 2>&1 | tail -15
```

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/source_orchestrator.py modules/orchestration-service/tests/test_source_orchestrator.py
git commit -m "feat: SourceOrchestrator — clean source switching with config subscription"
```

---

## Task 6: MeetCaptionsAdapter — Scrape Google Meet Built-in CC (TDD)

**Files:**
- Create: `modules/meeting-bot-service/tests/chat/meet_captions_adapter.test.ts`
- Create: `modules/meeting-bot-service/src/chat/meet_captions_adapter.ts`
- Modify: `modules/meeting-bot-service/src/chat/selectors.ts`
- Modify: `modules/meeting-bot-service/src/chat/index.ts`

- [ ] **Step 1: Add CC selectors to selectors.ts**

In `modules/meeting-bot-service/src/chat/selectors.ts`, add:

```typescript
// Built-in closed captions (Meet CC)
export const CC_BUTTON = '[aria-label="Turn on captions"]';
export const CC_CONTAINER = '[jscontroller="D1tHje"]';  // CC overlay container
export const CC_TEXT_SPAN = '.a4cQT';  // Caption text spans (may change with Meet updates)
```

Note: These selectors are based on observed Google Meet DOM and will need validation during manual testing. The `selectors.ts` file is the single place to update when Meet changes.

- [ ] **Step 2: Write failing tests for caption text parsing**

Create `modules/meeting-bot-service/tests/chat/meet_captions_adapter.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { parseCaptionText, CaptionEntry } from '../../src/chat/meet_captions_adapter';

describe('parseCaptionText', () => {
  it('parses speaker and text from "Speaker: text" format', () => {
    const result = parseCaptionText('Alice: Hello everyone');
    expect(result).toEqual({
      speaker: 'Alice',
      text: 'Hello everyone',
    });
  });

  it('handles text without speaker prefix', () => {
    const result = parseCaptionText('Hello everyone');
    expect(result).toEqual({
      speaker: 'Unknown',
      text: 'Hello everyone',
    });
  });

  it('handles speaker name with colon in text', () => {
    const result = parseCaptionText('Bob: Time is: 3pm');
    expect(result).toEqual({
      speaker: 'Bob',
      text: 'Time is: 3pm',
    });
  });

  it('trims whitespace', () => {
    const result = parseCaptionText('  Alice :  Hello  ');
    expect(result).toEqual({
      speaker: 'Alice',
      text: 'Hello',
    });
  });

  it('returns null for empty string', () => {
    expect(parseCaptionText('')).toBeNull();
  });

  it('returns null for whitespace-only', () => {
    expect(parseCaptionText('   ')).toBeNull();
  });
});
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd modules/meeting-bot-service && npx vitest run tests/chat/meet_captions_adapter.test.ts 2>&1 | tail -10
```

- [ ] **Step 4: Implement MeetCaptionsAdapter**

Create `modules/meeting-bot-service/src/chat/meet_captions_adapter.ts`:

```typescript
/**
 * MeetCaptionsAdapter — scrapes Google Meet's built-in closed captions.
 *
 * Polls the CC overlay DOM, extracts speaker + text, deduplicates,
 * and forwards as caption events via WebSocket.
 *
 * This is a third caption source option alongside bot_audio and fireflies.
 * The CC DOM selectors live in selectors.ts (single source of truth).
 */

import { Page } from 'playwright';
import { Logger } from 'winston';
import { CC_BUTTON, CC_CONTAINER, CC_TEXT_SPAN } from './selectors';

export interface CaptionEntry {
  speaker: string;
  text: string;
}

/**
 * Parse caption text in "Speaker: text" format.
 * Pure function — no DOM dependency, testable.
 */
export function parseCaptionText(raw: string): CaptionEntry | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  const colonIdx = trimmed.indexOf(':');
  if (colonIdx > 0 && colonIdx < 30) {
    // Likely "Speaker: text" format
    const speaker = trimmed.substring(0, colonIdx).trim();
    const text = trimmed.substring(colonIdx + 1).trim();
    if (text) {
      return { speaker, text };
    }
  }

  return { speaker: 'Unknown', text: trimmed };
}

export class MeetCaptionsAdapter {
  private page: Page;
  private logger: Logger;
  private onCaption: ((entry: CaptionEntry) => void) | null = null;
  private pollInterval: ReturnType<typeof setInterval> | null = null;
  private seenTexts: Set<string> = new Set();
  private isRunning = false;

  constructor(page: Page, logger: Logger) {
    this.page = page;
    this.logger = logger;
  }

  setOnCaption(callback: (entry: CaptionEntry) => void): void {
    this.onCaption = callback;
  }

  async start(): Promise<void> {
    if (this.isRunning) return;

    // Enable CC if not already on
    try {
      const ccButton = this.page.locator(CC_BUTTON).first();
      if (await ccButton.isVisible({ timeout: 3000 })) {
        await ccButton.click();
        this.logger.info('Enabled Meet closed captions');
        await this.page.waitForTimeout(500);
      }
    } catch (err) {
      this.logger.warn('Could not enable CC', { error: (err as Error).message });
    }

    this.isRunning = true;
    this.pollInterval = setInterval(() => this.poll(), 300);
    this.logger.info('MeetCaptionsAdapter started');
  }

  stop(): void {
    this.isRunning = false;
    if (this.pollInterval) {
      clearInterval(this.pollInterval);
      this.pollInterval = null;
    }
    this.logger.info('MeetCaptionsAdapter stopped');
  }

  private async poll(): Promise<void> {
    if (!this.isRunning) return;

    try {
      const texts = await this.page.evaluate((containerSel: string, textSel: string) => {
        const container = document.querySelector(containerSel);
        if (!container) return [];

        const spans = container.querySelectorAll(textSel);
        return Array.from(spans).map(s => s.textContent || '');
      }, CC_CONTAINER, CC_TEXT_SPAN);

      for (const rawText of texts) {
        if (!rawText.trim()) continue;
        const key = rawText.trim();
        if (this.seenTexts.has(key)) continue;
        this.seenTexts.add(key);

        // Evict old entries to prevent memory growth
        if (this.seenTexts.size > 500) {
          const firstKey = this.seenTexts.values().next().value;
          if (firstKey) this.seenTexts.delete(firstKey);
        }

        const entry = parseCaptionText(rawText);
        if (entry && this.onCaption) {
          this.onCaption(entry);
        }
      }
    } catch (err) {
      this.logger.debug('CC poll error', { error: (err as Error).message });
    }
  }
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd modules/meeting-bot-service && npx vitest run tests/chat/meet_captions_adapter.test.ts 2>&1 | tail -10
```

Expected: All 6 PASS

- [ ] **Step 6: Update index.ts re-exports**

In `modules/meeting-bot-service/src/chat/index.ts`, add:

```typescript
export { MeetCaptionsAdapter, parseCaptionText } from './meet_captions_adapter';
export type { CaptionEntry } from './meet_captions_adapter';
```

- [ ] **Step 7: Commit**

```bash
git add modules/meeting-bot-service/src/chat/meet_captions_adapter.ts modules/meeting-bot-service/tests/chat/meet_captions_adapter.test.ts modules/meeting-bot-service/src/chat/selectors.ts modules/meeting-bot-service/src/chat/index.ts
git commit -m "feat: MeetCaptionsAdapter — scrape Google Meet built-in CC as caption source"
```

---

## Task 7: Error Resilience — Source Drop Recovery

**Files:**
- Modify: `modules/orchestration-service/src/services/source_orchestrator.py`
- Modify: `modules/orchestration-service/tests/test_source_orchestrator.py`

- [ ] **Step 1: Write failing tests for error resilience**

Add to `modules/orchestration-service/tests/test_source_orchestrator.py`:

```python
class TestSourceOrchestratorResilience:
    def setup_method(self):
        self.config = MeetingSessionConfig(session_id="test-resilience")
        self.events: list[CaptionEvent] = []
        self.orchestrator = SourceOrchestrator(
            config=self.config,
            on_caption=lambda e: self.events.append(e),
        )

    @pytest.mark.asyncio
    async def test_restart_on_source_error(self):
        """If active source crashes, orchestrator restarts it."""
        await self.orchestrator.start()
        source = self.orchestrator.active_source
        assert source.is_running

        # Simulate source crash
        await source.stop()
        assert not source.is_running

        # Orchestrator detects and restarts
        await self.orchestrator.health_check()
        assert self.orchestrator.active_source.is_running

    @pytest.mark.asyncio
    async def test_health_check_noop_when_healthy(self):
        """Health check doesn't restart a healthy source."""
        await self.orchestrator.start()
        source_before = self.orchestrator.active_source
        await self.orchestrator.health_check()
        assert self.orchestrator.active_source is source_before
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest modules/orchestration-service/tests/test_source_orchestrator.py::TestSourceOrchestratorResilience -v 2>&1 | tail -10
```

- [ ] **Step 3: Add health_check to SourceOrchestrator**

In `modules/orchestration-service/src/services/source_orchestrator.py`, add:

```python
async def health_check(self) -> None:
    """Check if active source is healthy. Restart if crashed."""
    if self.active_source and not self.active_source.is_running:
        source_type = self._current_source_type()
        logger.warning("source_crashed_restarting", source=source_type)
        self.active_source = self._create_source(source_type)
        self.active_source.on_caption = self._on_caption
        await self.active_source.start(self._config)
        logger.info("source_restarted", source=source_type)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest modules/orchestration-service/tests/test_source_orchestrator.py -v 2>&1 | tail -15
```

Expected: All tests PASS (original 8 + new 2)

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/source_orchestrator.py modules/orchestration-service/tests/test_source_orchestrator.py
git commit -m "feat: SourceOrchestrator health_check — auto-restart crashed sources"
```

---

## Task 8: Wire SourceOrchestrator into WebSocket Handler

**Files:**
- Modify: `modules/orchestration-service/src/routers/audio/websocket_audio.py`

- [ ] **Step 1: Import and initialize SourceOrchestrator**

In `websocket_audio.py`, add import:

```python
from services.source_orchestrator import SourceOrchestrator
```

In the `StartSessionMessage` handler, after `meeting_config` and `command_dispatcher` initialization, add:

```python
source_orchestrator = SourceOrchestrator(
    config=meeting_config,
    on_caption=lambda event: asyncio.ensure_future(
        safe_send(json.dumps({
            "type": "caption_added",
            "caption": {
                "id": event.caption_id,
                "text": event.text,
                "speaker_name": event.speaker_name,
                "speaker_color": event.speaker_color,
                "source_lang": event.source_lang,
                "is_draft": event.is_draft,
            }
        }))
    ),
)
await source_orchestrator.start()
```

In the cleanup/disconnect section, add:

```python
if source_orchestrator:
    await source_orchestrator.stop()
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/routers/audio/websocket_audio.py
git commit -m "feat: wire SourceOrchestrator into WebSocket handler for config-driven source switching"
```

---

## Task 9: AudioStreamer — Forward Meet CC Events

**Files:**
- Modify: `modules/meeting-bot-service/src/audio_streaming.ts`

- [ ] **Step 1: Add sendCaptionEvent method**

In `modules/meeting-bot-service/src/audio_streaming.ts`, add a method to forward Meet CC captions:

```typescript
/**
 * Forward a caption from Meet's built-in CC to orchestration
 */
sendCaptionEvent(speaker: string, text: string): void {
  this.send({
    type: 'caption_event',
    speaker,
    text,
    source: 'meet_cc',
    timestamp: Date.now(),
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/meeting-bot-service/src/audio_streaming.ts
git commit -m "feat: AudioStreamer.sendCaptionEvent for Meet CC forwarding"
```

---

## Task 10: Real Data Fixtures & Validation Tests

**Files:**
- Create: `modules/orchestration-service/tests/fixtures/meeting_data/README.md`
- Create: `modules/orchestration-service/tests/test_real_data_validation.py`

- [ ] **Step 1: Create fixture directory and README**

Create `modules/orchestration-service/tests/fixtures/meeting_data/README.md`:

```markdown
# Meeting Data Fixtures

Real meeting data for pipeline validation. Each fixture is a JSON file with:

```json
{
  "source": "bot_audio|fireflies|meet_cc",
  "meeting_id": "unique-id",
  "language": "en|zh|ja|ko",
  "segments": [
    {
      "text": "transcribed text",
      "speaker_name": "Alice",
      "start_time": 0.0,
      "end_time": 1.25,
      "is_final": true
    }
  ]
}
```

## Adding fixtures

1. Record from real meetings using `LIVETRANSLATE_RECORD_FIXTURES=1`
2. Export from Fireflies API
3. Export from Google Meet CC scraping

Fixtures should cover: English, Chinese, Japanese, Korean, mixed-language, multi-speaker.
```

- [ ] **Step 2: Create sample fixture for smoke test**

Create `modules/orchestration-service/tests/fixtures/meeting_data/sample_bilingual.json`:

```json
{
  "source": "bot_audio",
  "meeting_id": "fixture-bilingual-001",
  "language": "zh-en",
  "segments": [
    {"text": "大家好，今天我们来讨论新项目", "speaker_name": "Alice", "start_time": 0.0, "end_time": 2.5, "is_final": true},
    {"text": "Hi everyone, let's discuss the new project", "speaker_name": "Bob", "start_time": 3.0, "end_time": 5.0, "is_final": true},
    {"text": "我觉得我们应该先确定时间表", "speaker_name": "Alice", "start_time": 5.5, "end_time": 7.5, "is_final": true},
    {"text": "I agree, timeline is important", "speaker_name": "Bob", "start_time": 8.0, "end_time": 9.5, "is_final": true},
    {"text": "那我们下周一开始吧", "speaker_name": "Alice", "start_time": 10.0, "end_time": 11.5, "is_final": true},
    {"text": "Sounds good, Monday works for me", "speaker_name": "Bob", "start_time": 12.0, "end_time": 13.5, "is_final": true}
  ]
}
```

- [ ] **Step 3: Write validation tests**

Create `modules/orchestration-service/tests/test_real_data_validation.py`:

```python
"""End-to-end pipeline validation with real meeting data fixtures.

These tests load real meeting segments through the caption source adapters,
text wrapper, and verify the full pipeline produces valid output.
"""

import json
from pathlib import Path

import pytest

from bot.text_wrapper import wrap_text
from services.meeting_session_config import MeetingSessionConfig
from services.pipeline.adapters.source_adapter import BotAudioCaptionSource, CaptionEvent

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "meeting_data"


def load_fixture(name: str) -> dict:
    path = FIXTURES_DIR / name
    return json.loads(path.read_text())


class TestRealDataValidation:
    def test_fixture_directory_exists(self):
        assert FIXTURES_DIR.exists()

    def test_sample_bilingual_fixture_loads(self):
        data = load_fixture("sample_bilingual.json")
        assert data["source"] == "bot_audio"
        assert len(data["segments"]) == 6

    @pytest.mark.asyncio
    async def test_bot_audio_source_processes_fixture(self):
        """BotAudioCaptionSource processes all fixture segments."""
        data = load_fixture("sample_bilingual.json")
        source = BotAudioCaptionSource()
        events: list[CaptionEvent] = []
        source.on_caption = lambda e: events.append(e)
        await source.start(config=None)

        for seg in data["segments"]:
            await source.handle_transcription(
                text=seg["text"],
                speaker_name=seg["speaker_name"],
                source_lang="auto",
                is_final=seg["is_final"],
            )

        assert len(events) == 6
        # Verify speaker colors are consistent
        alice_colors = {e.speaker_color for e in events if e.speaker_name == "Alice"}
        bob_colors = {e.speaker_color for e in events if e.speaker_name == "Bob"}
        assert len(alice_colors) == 1  # Alice always same color
        assert len(bob_colors) == 1    # Bob always same color
        assert alice_colors != bob_colors  # Different speakers, different colors

    def test_text_wrapper_handles_cjk_fixture_segments(self):
        """Text wrapper correctly wraps CJK text from fixture."""
        data = load_fixture("sample_bilingual.json")
        for seg in data["segments"]:
            lines = wrap_text(seg["text"], max_chars=20, max_lines=3)
            assert len(lines) >= 1
            for line in lines:
                assert len(line) <= 20
            # Verify no text lost (approximate — wrapping may strip spaces)
            rejoined = "".join(lines)
            assert len(rejoined) >= len(seg["text"]) * 0.8  # Allow some whitespace loss

    def test_meeting_config_snapshot_for_fixture(self):
        """MeetingSessionConfig produces valid snapshot for fixture-based session."""
        data = load_fixture("sample_bilingual.json")
        config = MeetingSessionConfig(
            session_id=data["meeting_id"],
            source_lang="zh",
            target_lang="en",
        )
        snap = config.snapshot()
        assert snap["session_id"] == "fixture-bilingual-001"
        assert snap["source_lang"] == "zh"
        assert snap["target_lang"] == "en"
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest modules/orchestration-service/tests/test_real_data_validation.py -v 2>&1 | tail -15
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/tests/fixtures/meeting_data/ modules/orchestration-service/tests/test_real_data_validation.py
git commit -m "feat: real meeting data fixtures and pipeline validation tests"
```

---

## Task 11: Full Test Suite Validation + Build Verification

**Files:** None (validation only)

- [ ] **Step 1: Run TypeScript tests**

```bash
cd modules/meeting-bot-service && npx vitest run 2>&1 | tail -10
```

Expected: All command parser + MeetCaptionsAdapter tests pass (~31 tests)

- [ ] **Step 2: Run Python tests**

```bash
uv run pytest modules/shared/tests/ -v -q --timeout=30 2>&1 | tail -5
uv run pytest modules/orchestration-service/tests/test_command_dispatcher.py modules/orchestration-service/tests/test_meeting_session_config.py modules/orchestration-service/tests/test_caption_buffer_subscribers.py modules/orchestration-service/tests/test_text_wrapper.py modules/orchestration-service/tests/test_fireflies_caption_source.py modules/orchestration-service/tests/test_source_orchestrator.py modules/orchestration-service/tests/test_real_data_validation.py -v --timeout=30 2>&1 | tail -15
```

Expected: All pass

- [ ] **Step 3: TypeScript build check**

```bash
cd modules/meeting-bot-service && npx tsc --noEmit 2>&1 | grep -E "chat/|audio_streaming" || echo "NO_ERRORS_IN_OUR_FILES"
```

Expected: No errors in our files

- [ ] **Step 4: SvelteKit build check**

```bash
cd modules/dashboard-service && npm run check 2>&1 | tail -10
```

- [ ] **Step 5: Commit any fixes**

---

## Summary

| Task | What | Layer |
|------|------|-------|
| 1 | Overlay config sync — live `config_changed` WS handling | SvelteKit |
| 2 | Text wrapper — CJK/Latin/mixed line breaking (10 tests) | Python |
| 3 | Wire text wrapper into VirtualWebcamManager | Python |
| 4 | FirefliesCaptionSource — wrap adapter as source (5 tests) | Python |
| 5 | SourceOrchestrator — clean source switching (8 tests) | Python |
| 6 | MeetCaptionsAdapter — scrape Meet CC DOM (6 tests) | TypeScript |
| 7 | Error resilience — health check + auto-restart (2 tests) | Python |
| 8 | Wire SourceOrchestrator into WS handler | Python |
| 9 | AudioStreamer — forward Meet CC events | TypeScript |
| 10 | Real data fixtures & validation tests | Both |
| 11 | Full validation | Both |

## Manual Testing After Phase 3

After all 11 tasks, the full manual testing protocol from the spec:

1. **Bot join test** — Video tile, join message, subtitle overlay visible
2. **Audio → subtitles** — Speak, verify ~150ms latency, translation, speaker attribution
3. **Chat commands** — All commands: `/lang`, `/font`, `/mode`, `/speakers`, `/original`, `/source`, `/theme`
4. **Source switch** — `/source fireflies` → back to bot, no gap
5. **Multi-participant** — Interpreter mode, distinct speaker colors
6. **OBS overlay** — Captions in OBS browser source, live config sync from chat
7. **Endurance** — 30+ min meeting, memory/fps monitoring
8. **Agent-browser** — Scripted interactions via agent-browser
9. **Pin test** — Pin bot video, subtitle readable at full size

## Real Data Collection Plan

After Phase 3 code is complete, collect real fixtures:
- **Bot audio**: Record 3+ real meetings with `LIVETRANSLATE_RECORD_FIXTURES=1`
- **Fireflies**: Export 3+ real transcripts via Fireflies API
- **Google Meet CC**: Scrape 3+ real CC sessions via MeetCaptionsAdapter
- **Languages**: Must include English, Chinese, and at least one Japanese or Korean session
- **Edge cases**: Overlapping speakers, code-switching (mid-sentence language switch), silence gaps
