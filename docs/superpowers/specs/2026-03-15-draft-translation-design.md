# Draft Translation Design

## Problem

Translation latency is 200-500ms per LLM call. The transcription service already sends two versions of each segment — a fast draft (~600ms) and a refined final (~1.2-6s) with the same segment_id. Currently only finals are translated. Users wait 1.2-6s before seeing any translation. Translating drafts gives rough feedback in ~800ms.

## Architecture

```
Draft segment (is_draft=true, ~600ms):
  → bypass _stable_text_buffer entirely
  → translate msg.text directly (full draft text)
  → non-streaming translate(max_tokens=160, max_retries=0, timeout=4s)
  → skip context window write
  → send TranslationMessage(is_draft=true)
  → Semaphore(1), non-blocking drop when busy

Final segment (is_draft=false, ~1.2-6s):
  → existing stable_text accumulation + sentence boundary detection
  → streaming translate_stream(max_tokens=512)
  → write to context window on completion
  → send translation_chunks + TranslationMessage(is_draft=false)
  → no semaphore (finals are canonical, never dropped)
```

Expected latency: ~800ms to rough translation, ~5s to refined final.

## Design Decisions

### Backend (orchestration)

**Two separate translation paths in `handle_transcription_segment`:**

The draft path and final path are completely separate code branches. This is the most important architectural decision — mixing them through the same stable_text accumulation path would cause the draft's text to be recorded as "already translated," silently deduping the final.

- Draft path: translates `msg.text` directly, non-streaming `translate(max_tokens=160, max_retries=0)`, no context window write, no persistence, `Semaphore(1)` with non-blocking acquire (drop when busy), `draft_timeout_s` timeout (default 4s)
- Final path: existing stable_text accumulation with sentence boundary detection, streaming `translate_stream(max_tokens=512)`, context window write, DB persistence in meeting mode

**`_translate_and_send` changes:**

Add `is_draft: bool = False` parameter that controls:
- Streaming vs non-streaming (finals stream, drafts don't)
- max_tokens (draft_max_tokens for drafts, max_tokens for finals — both configurable)
- max_retries (0 for drafts — fail fast, don't retry disposable translations)
- Context window write (finals only)
- DB persistence (finals only)
- `is_draft` flag on outgoing `TranslationMessage`
- Logging: all translation log events include `is_draft` field for observability

**LLM Client changes:**

`translate()` gains a `max_tokens: int | None = None` and `max_retries: int = 1` kwarg, forwarded through `_call_llm()` to `_request_body()`. When `max_tokens` is None, uses the config default.

`_request_body()` gains a `max_tokens: int = 512` kwarg (no longer hardcoded).

**TranslationConfig additions:**

```python
max_tokens: int = 512             # LLM_MAX_TOKENS — finals
draft_max_tokens: int = 160       # LLM_DRAFT_MAX_TOKENS — drafts (160 safe for CJK pairs)
draft_timeout_s: int = 4          # LLM_DRAFT_TIMEOUT_S — short timeout for disposable drafts
```

Note: `draft_max_tokens` defaults to 160 (not 128) because CJK→English translations can exceed 128 tokens for longer draft segments. 160 is the safe floor for all language pairs.

**Backpressure:**

`asyncio.Semaphore(1)` for draft translations with non-blocking acquire. If a draft can't acquire (another draft in flight), it's dropped with a `draft_translation_dropped` log event (not silent — observable). Finals are never gated by this semaphore.

**Draft cancellation:** Not implemented. When a final arrives while a draft translation is in-flight, the draft completes but its result is discarded by the frontend last-writer-wins guard. The wasted compute is <200ms at max_tokens=160 — not worth the complexity of threading cancellation events through httpx.

**Draft timeout:** Drafts use `asyncio.wait_for(..., timeout=config.draft_timeout_s)` (default 4s). If a draft hangs, the semaphore is released quickly so subsequent drafts aren't starved. The final's 30s timeout is unchanged.

**Context window safety:** Draft path calls `LLMClient.translate()` directly, NOT `TranslationService.translate()` (which unconditionally writes to context). Additionally, `TranslationService.translate()` gains a `skip_context: bool = False` parameter as a safety net — if `skip_context=True`, the context window write is skipped.

**Observability:**

All translation log events include `is_draft=True/False`:
- `translation_trigger` — includes `is_draft`
- `translation_complete` / `translation_failed` — includes `is_draft`
- `draft_translation_dropped` — new event when semaphore busy
- `draft_translation_timeout` — new event when draft exceeds `draft_timeout_s`

### Protocol (wire messages)

Add `is_draft: bool = False` to `TranslationMessage` and `TranslationChunkMessage` in both Python (`ws_messages.py`) and TypeScript (`ws-messages.ts`).

- No new message types
- No protocol version bump (field defaults to `false`, backward-compatible)
- No sequence numbers (`transcript_id` + `is_draft` sufficient for demuxing)
- `translation_chunk` messages only sent for finals (drafts are non-streaming)
- `is_draft` on `TranslationChunkMessage` exists for forward-compatibility
- **Rolling deployment safety:** TypeScript consumers MUST treat missing `is_draft` as `false` (use `msg.is_draft ?? false` or `!!msg.is_draft`). During rolling deployments, the backend may send messages without this field.
- Draft translation messages may interleave with streaming chunks for a different segment on the WebSocket. The frontend demuxes by `transcript_id`, so this is safe.

### Frontend (store + components)

**Store (`loopback.svelte.ts`):**

Add `translationIsDraft: boolean` to `CaptionEntry`. Keep existing `translationComplete: boolean`.

Visual state derived from existing fields (no enum):

| translationIsDraft | translation | translationComplete | Visual |
|---|---|---|---|
| false | null | false | Waiting — pulsing dot |
| true | non-null | false | Draft shown — dimmed (65% opacity), solid dot |
| false | non-null | false | Final streaming — chunks arriving, fast pulse with glow |
| false | non-null | true | Complete — full opacity, no indicator |

**Store mutation guards (last-writer-wins):**

```
addTranslation(msg):
  1. If caption.translationComplete === true → ignore
  2. If msg.is_draft && !caption.translationIsDraft → ignore (don't downgrade)
  3. Otherwise → accept, set translation and flags
     - caption.translationIsDraft = !!msg.is_draft
     - caption.translationComplete = !msg.is_draft  (drafts are not complete)

appendTranslationChunk(msg):
  1. If caption.translationComplete === true → ignore
  2. If caption.translationIsDraft === true → CLEAR translation first, set
     translationIsDraft = false (first final chunk replaces draft text)
  3. Append delta to caption.translation
```

**Critical: chunk-on-draft guard (Issue #6).** When the first final streaming chunk arrives for a segment that has a draft translation, the chunk handler must CLEAR the existing draft text before appending. Otherwise the final chunks concatenate onto the draft text, producing garbage like `"rough draftHello world"`. The guard in step 2 detects this by checking `translationIsDraft === true` and resets before appending.

Finals can overwrite drafts. Drafts can never overwrite finals.

**Draft→final segment replacement in `addSegment`:**

When a final segment replaces a draft segment (same segment_id), preserve the existing draft translation so the user sees something while the final translation generates.

**Shared `TranslationText.svelte` component:**

Props: `isDraft`, `text`, `isComplete`. Used by SplitView, SubtitleView, TranscriptView.

- Opacity: 0.65 (draft) → 0.85 (streaming) → 1.0 (complete), CSS transition 0.3s
- Indicator: pulsing dot (waiting), solid dot at 50% (draft received), fast pulse with glow (streaming), hidden (complete)
- Same visual language as transcription draft/final opacity pattern

**Extract `paragraph-helpers.ts`:**

Shared `paragraphTranslation()`, `hasPendingTranslation()` — eliminates duplication between SplitView and TranscriptView.

### Context Window

Drafts never enter the rolling context window. Only finals call `translation_service._get_context_window(speaker_name).add(text, translation)`.

Draft translations use whatever context exists from previous finals. This is acceptable because drafts are rough placeholders — speed over accuracy.

`TranslationService.translate()` gains a `skip_context: bool = False` safety parameter. If the draft path ever accidentally routes through `TranslationService.translate()` instead of `LLMClient.translate()`, passing `skip_context=True` prevents context pollution.

### DB Persistence

Only final translations persist via `save_translation()` in meeting mode. Draft translations are ephemeral.

## Critical Invariants

1. **Draft path never touches `_stable_text_buffer` or `_last_translated_stable`** — bypass entirely, translate `msg.text` directly.
2. **Draft translations never enter the rolling context window** — only finals.
3. **Frontend last-writer-wins** — `translationComplete === true` blocks all further updates for that segment_id.
4. **Draft semaphore is non-blocking** — drop when busy, never queue.
5. **First final chunk clears draft text** — `appendTranslationChunk` must clear `translation` when `translationIsDraft === true` before appending.
6. **Draft retries = 0** — fail fast, don't retry disposable translations.

## Files Changed

| File | Change |
|------|--------|
| `orchestration-service/src/routers/audio/websocket_audio.py` | Draft translation branch, Semaphore(1), is_draft param on _translate_and_send, draft timeout |
| `orchestration-service/src/translation/llm_client.py` | max_tokens + max_retries params on translate(), _call_llm(), _request_body() |
| `orchestration-service/src/translation/config.py` | Add max_tokens, draft_max_tokens, draft_timeout_s fields |
| `orchestration-service/src/translation/service.py` | Add skip_context param to translate() |
| `shared/src/.../models/ws_messages.py` | is_draft field on TranslationMessage + TranslationChunkMessage |
| `dashboard-service/src/lib/types/ws-messages.ts` | is_draft field on both interfaces |
| `dashboard-service/src/lib/stores/loopback.svelte.ts` | translationIsDraft field, mutation guards, chunk-on-draft clear |
| `dashboard-service/src/lib/components/loopback/TranslationText.svelte` | NEW — shared component |
| `dashboard-service/src/lib/components/loopback/paragraph-helpers.ts` | NEW — shared helpers |
| `dashboard-service/src/lib/components/loopback/SplitView.svelte` | Use TranslationText, import helpers |
| `dashboard-service/src/lib/components/loopback/SubtitleView.svelte` | Use TranslationText |
| `dashboard-service/src/lib/components/loopback/TranscriptView.svelte` | Use TranslationText, import helpers |

## Testing (TDD)

All tests written before implementation:

### Backend
- Think filter: existing 11 tests pass
- `_translate_and_send` with `is_draft=True`: non-streaming, max_tokens=160, no context write, max_retries=0
- `_translate_and_send` with `is_draft=False`: streaming, max_tokens=512, context write
- Draft semaphore: drop when busy (with log event), finals bypass
- Draft timeout: `asyncio.wait_for` at `draft_timeout_s`, semaphore released on timeout
- Draft path bypasses `_stable_text_buffer` (regression test for stable_text dedup bug)
- `TranslationService.translate(skip_context=True)` skips context window write

### Frontend
- Store: draft→final translation lifecycle (addTranslation with is_draft=true, then is_draft=false)
- Store: last-writer-wins guard (final already complete, ignore late draft)
- Store: appendTranslationChunk guard (ignore chunks after translationComplete)
- Store: **appendTranslationChunk clears draft text on first final chunk** (regression test for concat bug)
- Store: addSegment draft→final preserves draft translation
- Component: TranslationText renders correct opacity per phase
- Component: indicator varies by phase

### Integration
- Translation persistence: save_translation only for finals (existing tests, needs Docker)

## Verification

```bash
uv run pytest modules/orchestration-service/tests/unit/ -v -k "not Integration"
uv run pytest modules/shared/tests/ -v
cd modules/dashboard-service && npx svelte-check --threshold error
cd modules/dashboard-service && npx vitest run

# Manual: just dev-debug, capture audio, verify:
# 1. Draft translation appears dimmed at ~800ms
# 2. Final translation replaces at full opacity at ~5s
# 3. Pulsing dot behavior matches phase table
# 4. No duplicate translations
# 5. Context window contains only final translations
# 6. Rapid speech: drafts dropped gracefully (semaphore log event)
# 7. Draft timeout: draft semaphore released within 4s on hang
```

## Expert Reviews

Design validated by:
- **LLM Architect**: throughput sustainable, non-streaming drafts correct, context isolation critical. Review: raised max_tokens configurability, param threading, retry behavior → all addressed.
- **Media Streaming**: protocol design (is_draft flag), backpressure (Semaphore + drop), latency targets (draft <1.5s, final <3s). Review: caught chunk-on-draft concat bug (high severity), rolling deployment safety → all addressed.
- **MLOps Engineer**: KV cache fine at Semaphore(1), max_tokens=128 marginal for CJK (raised to 160), draft timeout needed (added 4s), observability gaps (added is_draft to all log events). Review: all issues addressed.
- **Svelte Developer**: shared TranslationText component, paragraph helpers, store testing patterns.
- **Architect Reviewer**: simplified generation counter → frontend guards, caught stable_text dedup bug, validated Semaphore(1) over Semaphore(2).
