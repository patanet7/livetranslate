# Draft Translation Design

## Problem

Translation latency is 200-500ms per LLM call. The transcription service already sends two versions of each segment — a fast draft (~600ms) and a refined final (~1.2-6s) with the same segment_id. Currently only finals are translated. Users wait 1.2-6s before seeing any translation. Translating drafts gives rough feedback in ~800ms.

## Architecture

```
Draft segment (is_draft=true, ~600ms):
  → bypass _stable_text_buffer entirely
  → translate msg.text directly (full draft text)
  → non-streaming translate(max_tokens=128)
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

- Draft path: translates `msg.text` directly, non-streaming `translate(max_tokens=128)`, no context window write, no persistence, `Semaphore(1)` with non-blocking acquire (drop when busy)
- Final path: existing stable_text accumulation with sentence boundary detection, streaming `translate_stream(max_tokens=512)`, context window write, DB persistence in meeting mode

**`_translate_and_send` changes:**

Add `is_draft: bool = False` parameter that controls:
- Streaming vs non-streaming (finals stream, drafts don't)
- max_tokens (128 for drafts, 512 for finals)
- Context window write (finals only)
- DB persistence (finals only)
- `is_draft` flag on outgoing `TranslationMessage`

**LLM Client change:**

`_request_body()` gets a `max_tokens` parameter (default 512). No other LLM client changes — drafts use existing `translate()`, finals use existing `translate_stream()`.

**Backpressure:**

`asyncio.Semaphore(1)` for draft translations with non-blocking acquire. If a draft can't acquire (another draft in flight), it's silently dropped. The final will arrive soon anyway. Finals are never gated by this semaphore.

### Protocol (wire messages)

Add `is_draft: bool = False` to `TranslationMessage` and `TranslationChunkMessage` in both Python (`ws_messages.py`) and TypeScript (`ws-messages.ts`).

- No new message types
- No protocol version bump (field defaults to `false`, backward-compatible)
- No sequence numbers (`transcript_id` + `is_draft` sufficient for demuxing)
- `translation_chunk` messages only sent for finals (drafts are non-streaming)
- `is_draft` on `TranslationChunkMessage` exists for forward-compatibility

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

appendTranslationChunk(msg):
  1. If caption.translationComplete === true → ignore
  2. Otherwise → append delta
```

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

### DB Persistence

Only final translations persist via `save_translation()` in meeting mode. Draft translations are ephemeral.

## Critical Invariants

1. **Draft path never touches `_stable_text_buffer` or `_last_translated_stable`** — bypass entirely, translate `msg.text` directly.
2. **Draft translations never enter the rolling context window** — only finals.
3. **Frontend last-writer-wins** — `translationComplete === true` blocks all further updates for that segment_id.
4. **Draft semaphore is non-blocking** — drop when busy, never queue.

## Files Changed

| File | Change |
|------|--------|
| `orchestration-service/src/routers/audio/websocket_audio.py` | Draft translation branch, Semaphore(1), is_draft param on _translate_and_send |
| `orchestration-service/src/translation/llm_client.py` | max_tokens param on _request_body() |
| `shared/src/.../models/ws_messages.py` | is_draft field on TranslationMessage + TranslationChunkMessage |
| `dashboard-service/src/lib/types/ws-messages.ts` | is_draft field on both interfaces |
| `dashboard-service/src/lib/stores/loopback.svelte.ts` | translationIsDraft field, mutation guards |
| `dashboard-service/src/lib/components/loopback/TranslationText.svelte` | NEW — shared component |
| `dashboard-service/src/lib/components/loopback/paragraph-helpers.ts` | NEW — shared helpers |
| `dashboard-service/src/lib/components/loopback/SplitView.svelte` | Use TranslationText, import helpers |
| `dashboard-service/src/lib/components/loopback/SubtitleView.svelte` | Use TranslationText |
| `dashboard-service/src/lib/components/loopback/TranscriptView.svelte` | Use TranslationText, import helpers |

## Testing (TDD)

All tests written before implementation:

### Backend
- Think filter: existing 11 tests pass
- `_translate_and_send` with `is_draft=True`: non-streaming, max_tokens=128, no context write
- `_translate_and_send` with `is_draft=False`: streaming, max_tokens=512, context write
- Draft semaphore: drop when busy, finals bypass
- Draft path bypasses `_stable_text_buffer` (regression test for architect-reviewer's critical bug)

### Frontend
- Store: draft→final translation lifecycle (addTranslation with is_draft=true, then is_draft=false)
- Store: last-writer-wins guard (final already complete, ignore late draft)
- Store: appendTranslationChunk guard (ignore chunks after translationComplete)
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
# 6. Rapid speech: drafts dropped gracefully (semaphore)
```

## Expert Reviews

Design validated by:
- **LLM Architect**: throughput sustainable, non-streaming drafts correct, context isolation critical
- **Media Streaming**: protocol design (is_draft flag), backpressure (Semaphore + cancel-on-supersede), latency targets (draft <1.5s, final <3s)
- **Svelte Developer**: shared TranslationText component, paragraph helpers, store testing patterns
- **Architect Reviewer**: simplified generation counter → frontend guards, caught stable_text dedup bug, validated Semaphore(1) over Semaphore(2)
