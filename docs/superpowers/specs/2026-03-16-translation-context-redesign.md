# Translation Context & Pipeline Redesign

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

## Goal

Replace the fragile text-level dedup system with structural segment tracking, fix the LLM prompt to use bilingual context pairs, add per-direction context windows for interpreter mode, and standardize draft/final naming across the codebase.

## Architecture

The translation pipeline currently uses `_stable_text_buffer` accumulation and `_last_translated_stable` regex dedup to manage what text gets translated. This causes content loss bugs (confirmed via production logs). The redesign eliminates all text-level dedup in favor of `SegmentStore` (segment_id lifecycle tracking). Context management moves from speaker-keyed windows to `DirectionalContextStore` keyed by `(source_lang, target_lang)`, enabling per-direction context in interpreter mode without clearing on language flip.

## Tech Stack

- Python 3.12+ / Pydantic v2 / FastAPI (orchestration)
- Svelte 5 / TypeScript (dashboard)
- structlog (logging)
- Qwen3.5-7B via Ollama (default) or Qwen3-4B via vLLM-MLX (dev/local)

## Prerequisites (already completed)

These changes were made during the interpreter mode implementation and are already on `main`:

- [x] `detectedLanguage` separated from `sourceLanguage` in loopback store (`loopback.svelte.ts:54`)
- [x] Toolbar `onConfigChange` accepts `language: string | null` for auto-detect reset (`Toolbar.svelte:17`)
- [x] Source language effect sends `null` to server for auto-detect (`Toolbar.svelte:113`)
- [x] `transcription_client.send_config()` uses `_UNCHANGED` sentinel — `language=None` explicitly sends `"language": null` (`transcription_client.py:112`)
- [x] `draft_max_tokens` bumped from 160 to 256 (`config.py:24`)
- [x] `InterpreterView` component with cross-panel rendering
- [x] `drainAndDisconnect()` for graceful stop-capture
- [x] `translationPhase` simplified (any final = paragraph complete)

---

## Draft/Final Segment Protocol

These semantics are the foundation of the entire design. All code, comments, and documentation must use these definitions consistently.

### Definitions

| Flag | Meaning | Set by |
|------|---------|--------|
| `is_draft=True` | **First-pass VAC snapshot.** Non-destructive audio buffer read at `stride_s/2`. Same GPU model as final, less audio. | `api.py:621` |
| `is_draft=False` | **Second-pass VAC consume.** Destructive audio buffer read at full `stride_s`. Same `segment_id` as draft, usually more/better text. | `api.py:596` |
| `is_final=True` | **Sentence boundary.** Text ends with punctuation (`.!?`). Independent from `is_draft`. | `vac_online_processor.py:476` |
| `is_final=False` | **Incomplete sentence.** No terminal punctuation detected yet. | `vac_online_processor.py:483` |

### Valid Combinations

| `is_draft` | `is_final` | Example | Translation action |
|---|---|---|---|
| `true` | `false` | `"Okay let's try this"` | Draft translation (fast, no context) |
| `true` | `true` | `"Okay let's try this."` | Draft translation (fast, no context) |
| `false` | `false` | `"Okay let's try this little thing out and"` | Accumulate — don't translate yet |
| `false` | `true` | `"Okay let's try this little thing out and see what happens."` | Final translation (streaming, with context) |

### Segment Lifecycle

```
audio → VAC buffer → stride/2 reached → draft (segment_id=N, is_draft=True)
                   → full stride reached → final (segment_id=N, is_draft=False)
                                          → if is_final: translate with context
                                          → if !is_final: wait for sentence boundary
```

Draft and final share the same `segment_id`. The final replaces the draft on the frontend.

---

## Phase 0: Docfix + Prompt Quick Wins

**Goal:** Fix the highest-impact translation quality issue (bilingual context) and correct misleading documentation. Zero regression risk.

### Task 0.1: Fix `is_final` docstring

**File:** `modules/shared/src/livetranslate_common/models/ws_messages.py:149`

**Current (wrong):**
```python
is_final: True when the segment will not be updated further.
```

**Corrected:**
```python
is_final: bool
"""True when the segment text ends at a sentence boundary (punctuation).

WARNING: Does NOT mean "last segment" or "will not be updated."
A segment with is_final=False can still be the definitive transcription
for its audio window. See ARCHITECTURE.md § Draft/Final Protocol.
"""
```

Also fix in `modules/shared/src/livetranslate_common/models/transcription.py` if present.

**Acceptance:** Docstring matches the sentence-boundary semantics. CLAUDE.md gotcha is consistent.

### Task 0.2: Fix `is_draft` docstring

**File:** `modules/shared/src/livetranslate_common/models/ws_messages.py:150`

**Current (incomplete):**
```python
is_draft: True for fast first-pass captions that will be refined by a later final.
```

**Corrected:**
```python
is_draft: bool = False
"""True for first-pass VAC snapshot (non-destructive, stride/2 audio).

Draft and final segments share the same segment_id. The final is a
second-pass with the full audio stride — same model, more audio,
usually longer/more accurate text. The frontend replaces the draft
in-place when the final arrives.
"""
```

### Task 0.3: Mirror docstrings in TypeScript

**File:** `modules/dashboard-service/src/lib/types/ws-messages.ts:62-63`

Add JSDoc comments on `is_final` and `is_draft` matching the Python docstrings.

### Task 0.4: Bilingual context in LLM prompt

**File:** `modules/orchestration-service/src/translation/llm_client.py`

**Current** (`_build_messages`, line 140-144):
```python
if context:
    user_parts.append("[Context — previous conversation, do NOT translate:]")
    for ctx in context:
        user_parts.append(f"- {ctx.translation}")
    user_parts.append("")
```

**Redesigned:**
```python
if context:
    user_parts.append("[Prior:]")
    for ctx in context:
        src = ctx.text.replace("\n", " ")
        tgt = ctx.translation.replace("\n", " ")
        user_parts.append(f"[{source_language}] {src}")
        user_parts.append(f"[{target_language}] {tgt}")
    user_parts.append("")
```

**Example rendered prompt (en→es, 2 context pairs):**
```
[Prior:]
[en] Hello, how are you?
[es] Hola, cmo ests?
[en] I'm doing great, thanks.
[es] Estoy muy bien, gracias.

[New:]
That's wonderful to hear. /nothink
```

### Task 0.5: Simplify system prompt

**File:** `modules/orchestration-service/src/translation/llm_client.py`

**Current:**
```python
system_prompt = (
    f"Translate spoken {src_name} to natural {tgt_name}.{extra} "
    f"Output ONLY the {tgt_name} translation. "
    f"Never output {src_name} or any other language."
)
```

**Redesigned — two variants:**

```python
# Final path (has context — needs "Never repeat context" guard)
system_prompt = (
    f"Translate {src_name} speech to {tgt_name}.{extra} "
    f"Output ONLY the {tgt_name} translation. "
    f"Never repeat context."
)

# Draft path (no context — shorter prompt, lower latency)
system_prompt = (
    f"Translate {src_name} speech to {tgt_name}.{extra} "
    f"Output ONLY the {tgt_name} translation."
)
```

Add `is_draft: bool = False` parameter to `_build_messages` to select variant.

### Task 0.6: Compact user message for drafts

**Current:** Both draft and final use `[Translate this:]` label.

**Redesigned:**
```python
if has_context:
    # Final path
    user_parts.append("[New:]")
    user_parts.append(text)
else:
    # Draft path — no section labels needed
    user_parts.append(f"Translate: {text}")
```

**Acceptance criteria (Phase 0):**
- [ ] `is_final` docstring says "sentence boundary", not "will not be updated"
- [ ] `is_draft` docstring explains shared segment_id and two-pass audio pipeline
- [ ] TypeScript mirrors have JSDoc comments
- [ ] LLM prompt shows `[lang] source` + `[lang] translation` pairs
- [ ] System prompt is shorter, draft-specific variant exists
- [ ] `uv run pytest modules/shared/tests/ -v` passes
- [ ] `npx svelte-check --threshold error` passes
- [ ] Manual test: capture speech, verify context pairs visible in orchestration debug logs

---

## Phase 1: DirectionalContextStore

**Goal:** Replace speaker-keyed context with direction-keyed context. Interpreter mode direction flip becomes a no-op — each direction has its own window.

### Task 1.1: Create DirectionalContextStore

**New file:** `modules/orchestration-service/src/translation/context_store.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field

from translation.context import RollingContextWindow


@dataclass
class DirectionalContextStore:
    """Per-(source_lang, target_lang) rolling context windows.

    Interpreter mode uses two directions simultaneously (zh->en, en->zh).
    Standard mode uses one. Direction flip is a no-op — separate keys.
    """

    max_entries: int = 5
    max_tokens: int = 800
    cross_direction_max_tokens: int = 200
    _windows: dict[tuple[str, str], RollingContextWindow] = field(
        default_factory=dict, repr=False,
    )

    def _key(self, source: str, target: str) -> tuple[str, str]:
        return (source.lower(), target.lower())

    def get(self, source: str, target: str) -> list:
        key = self._key(source, target)
        if key not in self._windows:
            return []
        return self._windows[key].get_context()

    def get_cross_direction(self, source: str, target: str) -> list:
        """Get 1-2 entries from the opposite direction for referent tracking."""
        opposite = self._key(target, source)
        if opposite not in self._windows:
            return []
        entries = self._windows[opposite].get_context()
        # Return last 2 entries, bounded by cross_direction_max_tokens
        return entries[-2:] if entries else []

    def add(
        self, source: str, target: str, source_text: str, translation: str,
    ) -> None:
        key = self._key(source, target)
        if key not in self._windows:
            self._windows[key] = RollingContextWindow(
                max_entries=self.max_entries,
                max_tokens=self.max_tokens,
            )
        self._windows[key].add(source_text, translation)

    def clear_direction(self, source: str, target: str) -> None:
        key = self._key(source, target)
        if key in self._windows:
            self._windows[key].clear()

    def clear_all(self) -> None:
        self._windows.clear()
```

### Task 1.2: Update TranslationService to use DirectionalContextStore

**File:** `modules/orchestration-service/src/translation/service.py`

Replace `_contexts: dict[str | None, RollingContextWindow]` with injected `DirectionalContextStore`. Update `get_context()`, `clear_context()`, and the `_get_context_window()` method to pass `(source_lang, target_lang)` instead of `speaker_name`.

### Task 1.3: Update websocket_audio.py call sites

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

- Create `DirectionalContextStore` at session scope (alongside pipeline, transcription_client, etc.)
- Pass `(msg.language, effective_target)` to context operations
- In `_translate_and_send` final path: fetch context via `context_store.get(source_lang, target_lang)`
- In interpreter mode: also fetch `context_store.get_cross_direction(source_lang, target_lang)` and include in prompt under a `[Recent context (other speaker):]` header

### Task 1.4: Add cross_direction_max_tokens to config

**File:** `modules/orchestration-service/src/translation/config.py`

```python
max_context_tokens: int = 800          # was 500
cross_direction_max_tokens: int = 200  # new
```

### Task 1.5: Wire cross-direction context into prompt

**File:** `modules/orchestration-service/src/translation/llm_client.py`

Add optional `cross_context: list[TranslationContext] | None = None` parameter to `_build_messages`. When present:

```python
if cross_context:
    user_parts.append("[Recent context (other speaker):]")
    for ctx in cross_context:
        src = ctx.text.replace("\n", " ")
        tgt = ctx.translation.replace("\n", " ")
        user_parts.append(f"[{source_language}] {src}")
        user_parts.append(f"[{target_language}] {tgt}")
    user_parts.append("")
```

### Task 1.6: Update ConfigMessage handler context-clearing

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

When `target_language` changes via ConfigMessage (current line ~678), update from `translation_service.clear_context()` to `context_store.clear_direction(source_language, previous_target)` — only clear the affected direction, not all windows.

When `interpreter_languages` is received, no longer call `translation_service.clear_context()` — the directional store handles isolation automatically.

**Acceptance criteria (Phase 1):**
- [ ] Context keyed by `(source_lang, target_lang)`, not speaker name
- [ ] Interpreter mode direction flip does NOT clear context
- [ ] Cross-direction entries appear in interpreter mode prompts
- [ ] Standard (non-interpreter) mode works unchanged
- [ ] ConfigMessage target_language change only clears the affected direction
- [ ] Unit test: `DirectionalContextStore` with two directions, verify independent windows
- [ ] `uv run pytest modules/orchestration-service/tests/ -v` passes
- [ ] Manual test: interpreter mode, speak Language A, then Language B — verify both directions have independent context windows in logs

---

## Phase 2: SegmentStore Replaces Text Dedup

**Goal:** Delete `_stable_text_buffer`, `_last_translated_stable`, and all regex dedup. Replace with structural segment_id tracking.

### Task 2.1: Create SegmentRecord and SegmentPhase

**New file:** `modules/orchestration-service/src/translation/segment_record.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentPhase(str, Enum):
    DRAFT_RECEIVED = "draft_received"
    DRAFT_TRANSLATED = "draft_translated"
    FINAL_RECEIVED = "final_received"
    FINAL_TRANSLATED = "final_translated"


@dataclass
class SegmentRecord:
    segment_id: int
    source_text: str
    source_lang: str
    target_lang: str
    phase: SegmentPhase = SegmentPhase.DRAFT_RECEIVED
    draft_translation: str | None = None
    final_translation: str | None = None
```

### Task 2.2: Create SegmentStore

**New file:** `modules/orchestration-service/src/translation/segment_store.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field

from translation.segment_record import SegmentPhase, SegmentRecord


@dataclass
class SegmentStore:
    """Per-session segment lifecycle tracker.

    Tracks each segment_id through draft/final phases.
    Replaces _stable_text_buffer and _last_translated_stable.

    Sentence accumulation: non-final finals (is_draft=False, is_final=False)
    append their text to _pending_sentence. When an is_final=True segment
    arrives, the accumulated sentence is returned for translation. This
    replaces the old _stable_text_buffer without any text-level dedup.
    """

    _records: dict[int, SegmentRecord] = field(default_factory=dict)
    _pending_sentence: str = ""
    _pending_segment_ids: list[int] = field(default_factory=list)

    def on_draft_received(
        self, segment_id: int, text: str, source_lang: str, target_lang: str,
    ) -> SegmentRecord:
        rec = SegmentRecord(
            segment_id=segment_id,
            source_text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            phase=SegmentPhase.DRAFT_RECEIVED,
        )
        self._records[segment_id] = rec
        return rec

    def on_draft_translated(
        self, segment_id: int, translation: str,
    ) -> SegmentRecord | None:
        rec = self._records.get(segment_id)
        if rec is None:
            return None
        rec.draft_translation = translation
        rec.phase = SegmentPhase.DRAFT_TRANSLATED
        return rec

    def on_final_received(
        self, segment_id: int, text: str, is_final: bool,
        source_lang: str, target_lang: str,
    ) -> tuple[SegmentRecord, str]:
        """Register a non-draft segment. Returns (record, translate_text).

        If is_final=False: accumulates text into _pending_sentence,
        returns empty translate_text (don't translate yet).

        If is_final=True: flushes _pending_sentence + this segment's text,
        returns the full accumulated sentence for translation.
        """
        rec = self._records.get(segment_id)
        if rec is None:
            rec = SegmentRecord(
                segment_id=segment_id,
                source_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                phase=SegmentPhase.FINAL_RECEIVED,
            )
            self._records[segment_id] = rec
        else:
            rec.source_text = text
            rec.phase = SegmentPhase.FINAL_RECEIVED

        if not is_final:
            # Accumulate — sentence not complete yet
            self._pending_sentence += (" " if self._pending_sentence else "") + text.strip()
            self._pending_segment_ids.append(segment_id)
            return rec, ""

        # Sentence boundary: flush accumulated text + this segment
        accumulated = self._pending_sentence
        if accumulated:
            translate_text = (accumulated + " " + text.strip()).strip()
        else:
            translate_text = text.strip()

        self._pending_sentence = ""
        self._pending_segment_ids.clear()
        return rec, translate_text

    def on_final_translated(
        self, segment_id: int, translation: str,
    ) -> SegmentRecord | None:
        rec = self._records.get(segment_id)
        if rec is None:
            return None
        rec.final_translation = translation
        rec.phase = SegmentPhase.FINAL_TRANSLATED
        return rec

    def is_final_translated(self, segment_id: int) -> bool:
        rec = self._records.get(segment_id)
        return rec is not None and rec.phase == SegmentPhase.FINAL_TRANSLATED

    def get(self, segment_id: int) -> SegmentRecord | None:
        return self._records.get(segment_id)

    def flush_pending(self) -> str:
        """Force-flush any pending accumulated text (e.g., on session end)."""
        text = self._pending_sentence
        self._pending_sentence = ""
        self._pending_segment_ids.clear()
        return text

    def reset(self) -> None:
        self._records.clear()
        self._pending_sentence = ""
        self._pending_segment_ids.clear()

    def evict_old(self, keep_last: int = 50) -> None:
        if len(self._records) <= keep_last:
            return
        sorted_ids = sorted(self._records)
        for sid in sorted_ids[:-keep_last]:
            del self._records[sid]
```

### Task 2.3: Rewrite handle_transcription_segment

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

**Delete:** `_stable_text_buffer`, `_last_translated_stable`, `_recent_segment_texts` variables and all regex dedup logic (lines 386-453 current).

**Replace with:**

```python
# Session-scoped state (replaces bare nonlocal variables)
segment_store = SegmentStore()

# --- Draft path ---
if msg.is_draft:
    if _draft_lock.locked():
        return
    segment_store.on_draft_received(msg.segment_id, msg.text, msg.language, effective_target)
    task = asyncio.create_task(...)  # draft translate as before
    return

# --- Final path ---
rec, translate_text = segment_store.on_final_received(
    msg.segment_id, msg.text, msg.is_final, msg.language, effective_target,
)

# on_final_received returns empty string for non-final segments (accumulated)
# and the full sentence for is_final=True segments (flushed)
if not translate_text:
    return  # accumulated into pending_sentence, waiting for sentence boundary

# Guard: don't re-translate
if segment_store.is_final_translated(msg.segment_id):
    return

translate_text = translate_text.strip()
if not translate_text:
    return

context = context_store.get(msg.language, effective_target)
task = asyncio.create_task(
    _translate_and_send(
        safe_send, translation_service,
        segment_id=msg.segment_id,
        text=translate_text,
        source_lang=msg.language,
        target_lang=effective_target,
        speaker_name=msg.speaker_id,
        pipeline=pipeline,
        is_draft=False,
    )
)
```

**Key change:** `SegmentStore.on_final_received()` handles accumulation internally. Non-final segments (`is_final=False`) accumulate text in `_pending_sentence`. When `is_final=True` arrives, the full accumulated sentence is returned for translation. No text-level dedup — accumulation is purely structural (append). The context window provides cross-sentence continuity via `[Prior:]` pairs.

### Task 2.4: Update _translate_and_send to use stores

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

Update the final streaming path (current line ~852) to fetch context from `context_store` instead of `translation_service.get_context(speaker_name)`:

```python
# Final path context — from DirectionalContextStore, not speaker-keyed
context = context_store.get(source_lang, target_lang)
```

After a successful final translation:
```python
# Write to context (finals only)
context_store.add(source_lang, target_lang, text, translation)
segment_store.on_final_translated(segment_id, translation)
```

After a successful draft translation:
```python
segment_store.on_draft_translated(segment_id, translation)
# Do NOT write to context_store — drafts never pollute context
```

### Task 2.5: Clean up direction flip in interpreter mode

**Delete** the `_last_translation_direction` tracking and `clear_context()` call on direction flip. With `DirectionalContextStore`, this is a no-op — each direction already has its own window.

The interpreter `effective_target` computation stays (it's correct). Only the context-clearing logic goes.

### Task 2.6: Flush pending text on session end

In the `EndSessionMessage` handler and the `finally` cleanup block, flush any accumulated text:

```python
# On end_session or disconnect — translate any pending accumulated text
pending = segment_store.flush_pending()
if pending and translation_service and effective_target:
    task = asyncio.create_task(
        _translate_and_send(..., text=pending, is_draft=False)
    )
```

This ensures the last partial sentence before stop-capture still gets translated.

**Acceptance criteria (Phase 2):**
- [ ] `_stable_text_buffer` deleted — grep confirms zero references
- [ ] `_last_translated_stable` deleted — grep confirms zero references
- [ ] `_recent_segment_texts` deleted — grep confirms zero references
- [ ] Non-final final segments accumulate in `SegmentStore._pending_sentence`
- [ ] `is_final=True` flushes accumulated sentence for translation
- [ ] Session end flushes any remaining pending text
- [ ] Context window receives (source_text, translation) after each final translation
- [ ] Streaming final path fetches context from `context_store.get()`, not `translation_service.get_context()`
- [ ] No text-level regex dedup anywhere in websocket_audio.py
- [ ] `uv run pytest modules/orchestration-service/tests/ -v` passes
- [ ] Manual test: long conversation, verify no content loss in translations
- [ ] Manual test: multi-segment sentence (no punctuation until end) — verify full sentence translated

---

## Phase 3: Draft Routing + Provisional Context

**Goal:** Route draft translations through TranslationService properly. Give drafts read-only access to the context window for better quality.

### Task 3.1: Route drafts through TranslationService

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

**Current (bypasses encapsulation):**
```python
translation = await translation_service._client.translate(
    text=text, source_language=source_lang, target_language=target_lang,
    max_tokens=translation_service.config.draft_max_tokens, max_retries=0,
)
```

**Redesigned:** Add a `translate_draft()` method to `TranslationService` that accepts explicit context (read-only from `DirectionalContextStore`) and uses draft-specific settings:

```python
# In TranslationService:
async def translate_draft(
    self, text: str, source_lang: str, target_lang: str,
    context: list[TranslationContext] | None = None,
) -> str:
    """Draft translation: fast, fail-fast, no context write-back."""
    return await self._client.translate(
        text=text, source_language=source_lang, target_language=target_lang,
        context=context, max_tokens=self.config.draft_max_tokens, max_retries=0,
    )
```

This preserves encapsulation (no `._client` access from websocket handler) and makes draft-specific behavior explicit. The caller passes context from `context_store.get()` — the service never reads its own internal context for drafts.

### Task 3.2: Provisional draft context

Give drafts read-only access to the final context window. The draft reads existing finals as context but never writes back.

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

In the draft path of `_translate_and_send`:
```python
if is_draft:
    # Read finals from context for better quality, but don't write back
    context = context_store.get(source_lang, target_lang)
    translation = await translation_service.translate_draft(
        text=text, source_lang=source_lang, target_lang=target_lang,
        context=context[-3:] if context else None,  # last 3 finals only
        max_retries=0,
    )
```

Don't label drafts as provisional in the prompt — the model doesn't need to know.

**Acceptance criteria (Phase 3):**
- [ ] Draft translations go through `TranslationService`, not `._client` directly
- [ ] Drafts receive finals-only context (read, no write)
- [ ] Draft translations do NOT appear in context window
- [ ] `skip_context=True` prevents context write-back
- [ ] Manual test: draft arrives faster with context, translation quality improves

---

## Phase 4: Interpreter Mode + Backend Switched

**Goal:** Wire cross-direction context for interpreter mode. Handle backend switches properly.

### Task 4.1: Interpreter cross-direction context in prompt

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

When `interpreter_languages` is set, pass cross-direction context to `_translate_and_send`:

```python
cross_context = context_store.get_cross_direction(msg.language, effective_target)
```

Pass through to `_build_messages` as the `cross_context` parameter (from Phase 1, Task 1.5).

### Task 4.2: Handle backend_switched message

**File:** `modules/orchestration-service/src/routers/audio/websocket_audio.py`

Currently `backend_switched` is a no-op. Add:

```python
async def handle_backend_switched(data: dict) -> None:
    """Reset segment store on backend switch — word overlap patterns change."""
    segment_store.reset()
    # context_store intentionally NOT reset — prior translations remain valid
    await safe_send(
        BackendSwitchedMessage(
            backend=data.get("backend", ""),
            model=data.get("model", ""),
            language=data.get("language", ""),
        ).model_dump_json()
    )
```

Register: `transcription_client.on_backend_switched(handle_backend_switched)`

**Note:** `WebSocketTranscriptionClient` already has `on_backend_switched()` (line 87 of `transcription_client.py`), but it is not currently wired in the WebSocket handler. Add the registration alongside the existing `on_segment`, `on_language_detected`, and `on_error` callbacks at lines 585-587.

### Task 4.3: Unify frontend translation state

**Files to update:**
- `modules/dashboard-service/src/lib/stores/loopback.svelte.ts` — `CaptionEntry` interface + `addSegment`, `appendTranslationChunk`, `addTranslation`
- `modules/dashboard-service/src/lib/components/loopback/paragraph-helpers.ts` — `hasPendingTranslation`, `translationPhase`
- `modules/dashboard-service/src/lib/components/loopback/SplitView.svelte` — reads `translationPhase`
- `modules/dashboard-service/src/lib/components/loopback/SubtitleView.svelte` — reads `translationPhase`
- `modules/dashboard-service/src/lib/components/loopback/TranscriptView.svelte` — reads `translationPhase`
- `modules/dashboard-service/src/lib/components/loopback/InterpreterView.svelte` — reads `translationPhase`

Replace `translationIsDraft: boolean` + `translationComplete: boolean` in `CaptionEntry` with:

```typescript
translationState: 'pending' | 'draft' | 'streaming' | 'complete';
```

**State transitions:**
- `'pending'` — no translation yet (initial state)
- `'draft'` — draft `TranslationMessage` received (`is_draft=true`)
- `'streaming'` — `TranslationChunkMessage` received, final not yet complete
- `'complete'` — final `TranslationMessage` received (`is_draft=false`)

Update `addTranslation`:
```typescript
caption.translationState = isDraft ? 'draft' : 'complete';
```

Update `appendTranslationChunk`:
```typescript
if (caption.translationState === 'complete') return;  // ignore chunks after final
if (caption.translationState === 'draft') caption.translation = '';  // clear draft
caption.translationState = 'streaming';
```

Update `translationPhase` in `paragraph-helpers.ts`:
```typescript
export function translationPhase(captions: CaptionEntry[]): TranslationState {
    const anyComplete = captions.some(c => c.translationState === 'complete');
    const anyDraft = captions.some(c => c.translationState === 'draft');
    const anyStreaming = captions.some(c => c.translationState === 'streaming');
    if (anyComplete) return 'complete';
    if (anyStreaming) return 'streaming';
    if (anyDraft) return 'draft';
    return 'pending';
}
```

### Task 4.4: Remove segment_id collision workaround

**File:** `modules/dashboard-service/src/lib/stores/loopback.svelte.ts`

Remove the `segment_id + 100000` synthetic offset (line 107). With `SegmentStore` tracking lifecycle, the orchestration layer won't send duplicate finals.

**Acceptance criteria (Phase 4):**
- [ ] Interpreter mode shows cross-direction context in prompt (visible in logs)
- [ ] Backend switch resets segment store but preserves translation context
- [ ] Frontend uses single `translationState` field — no invalid state combinations
- [ ] `npx svelte-check --threshold error` passes
- [ ] Manual test: interpreter mode with cross-direction referent ("I agree with that" — "that" resolved correctly)

---

## Phase 5: Documentation

**Goal:** Single authoritative reference for the draft/final protocol.

### Task 5.1: Add Draft/Final Protocol section to ARCHITECTURE.md

Document the full lifecycle:
1. Audio accumulates in VAC buffer
2. At stride/2: non-destructive snapshot → draft inference → `SegmentMessage(segment_id=N, is_draft=True)`
3. At full stride: destructive consume → final inference → `SegmentMessage(segment_id=N, is_draft=False)`
4. Draft segment → fast non-streaming translation (no context write, provisional context read)
5. Final segment + `is_final=True` → streaming translation with full context
6. Context window updated with (source_text, translation) after final translation
7. Frontend: draft replaced in-place when final arrives (same segment_id)

### Task 5.2: Add per-direction context documentation

Document in ARCHITECTURE.md:
- `DirectionalContextStore` keyed by `(source_lang, target_lang)`
- Interpreter mode: each direction has independent window
- Cross-direction entries: last 1-2 from opposite direction for referent tracking
- Token budget: 800 matching + 200 cross-direction

### Task 5.3: Segment eviction for long sessions

**File:** `modules/orchestration-service/src/translation/segment_store.py`

Call `segment_store.evict_old(keep_last=50)` periodically in `handle_transcription_segment` when store grows beyond threshold.

### Task 5.4: Update CLAUDE.md

Update the project CLAUDE.md and orchestration-service CLAUDE.md with:
- New file locations (`context_store.py`, `segment_record.py`, `segment_store.py`)
- Updated translation config defaults (`max_context_tokens: 800`, `cross_direction_max_tokens: 200`, `draft_max_tokens: 256`)
- Remove references to `_stable_text_buffer` and dedup logic

**Acceptance criteria (Phase 5):**
- [ ] ARCHITECTURE.md has Draft/Final Protocol section
- [ ] ARCHITECTURE.md has Per-Direction Context section
- [ ] CLAUDE.md files updated with new file locations and config
- [ ] Long session eviction works (> 50 segments, oldest are evicted)

---

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `modules/orchestration-service/src/translation/context_store.py` | `DirectionalContextStore` — per-direction context windows |
| `modules/orchestration-service/src/translation/segment_record.py` | `SegmentRecord` + `SegmentPhase` enum |
| `modules/orchestration-service/src/translation/segment_store.py` | `SegmentStore` — segment lifecycle tracking |

### Modified Files
| File | Changes |
|------|---------|
| `modules/orchestration-service/src/translation/llm_client.py` | Bilingual context pairs, compact labels, draft/final prompt variants, cross-direction context |
| `modules/orchestration-service/src/translation/service.py` | Use `DirectionalContextStore`, accept direction params |
| `modules/orchestration-service/src/translation/config.py` | `max_context_tokens: 800`, `cross_direction_max_tokens: 200` |
| `modules/orchestration-service/src/routers/audio/websocket_audio.py` | Delete dedup, use SegmentStore + DirectionalContextStore, handle backend_switched |
| `modules/shared/src/livetranslate_common/models/ws_messages.py` | Fix is_final/is_draft docstrings |
| `modules/dashboard-service/src/lib/types/ws-messages.ts` | Mirror docstrings |
| `modules/dashboard-service/src/lib/stores/loopback.svelte.ts` | Unify translationState, remove segment_id collision hack |
| `modules/dashboard-service/src/lib/components/loopback/paragraph-helpers.ts` | Update for new translationState field |
| `ARCHITECTURE.md` | Draft/Final Protocol + Per-Direction Context sections |
| `CLAUDE.md` | Updated file locations, config defaults |

### Deleted Code
| What | Where | Lines |
|------|-------|-------|
| `_stable_text_buffer` | `websocket_audio.py` | ~200 |
| `_last_translated_stable` | `websocket_audio.py` | ~200 |
| `_recent_segment_texts` deque | `websocket_audio.py` | ~205 |
| `_last_translation_direction` | `websocket_audio.py` | ~198 |
| Word-level prefix dedup regex block | `websocket_audio.py` | ~386-453 |
| 50% safety net fallback | `websocket_audio.py` | ~415-427 |
| `segment_id + 100000` collision hack | `loopback.svelte.ts` | ~107 |

---

## Critical Path

```
Phase 0 (docfix + prompt) ──→ Phase 1 (DirectionalContextStore)
                                        │
                                        ▼
                               Phase 2 (SegmentStore) ──→ Phase 3 (draft routing)
                                                                    │
                                                                    ▼
                                                          Phase 4 (interpreter + cleanup)
                                                                    │
                                                                    ▼
                                                          Phase 5 (documentation)
```

Phase 0 and Phase 1 can be parallelized. All other phases are sequential.

---

## Verification

After all phases:

```bash
# Shared contracts
uv run pytest modules/shared/tests/ -v

# Orchestration
uv run pytest modules/orchestration-service/tests/ -v

# Dashboard type check
cd modules/dashboard-service && npx svelte-check --threshold error

# Ruff lint
uv run ruff check modules/orchestration-service/src/ modules/shared/src/

# Manual test checklist:
# 1. Split mode: speech → transcription → translation with context (verify bilingual pairs in logs)
# 2. Interpreter mode: alternate languages → verify per-direction context preserved
# 3. Interpreter mode: cross-direction referent tracking ("I agree with that")
# 4. Long session (> 50 segments): verify eviction, no memory growth
# 5. Stop capture: verify drain completes, translations finish
# 6. Backend switch: verify segment store resets, context preserved
```
