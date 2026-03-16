# Draft Translation — Implementation Plan

Spec: `docs/superpowers/specs/2026-03-15-draft-translation-design.md`

TDD throughout. Tests first, then implementation, then verify.

---

## Step 1: Config + LLM Client params

**Files:** `translation/config.py`, `translation/llm_client.py`, `translation/service.py`

- Add `max_tokens`, `draft_max_tokens`, `draft_timeout_s` to `TranslationConfig`
- Add `max_tokens` kwarg to `_request_body()`, `_call_llm()`, `translate()`, `translate_stream()`
- Add `max_retries` kwarg to `translate()` → `_call_llm()`
- Add `skip_context` kwarg to `TranslationService.translate()`

**Tests:** `_request_body` respects max_tokens param, `translate(max_retries=0)` doesn't retry.

## Step 2: Protocol — `is_draft` on wire messages

**Files:** `ws_messages.py` (Python), `ws-messages.ts` (TypeScript)

- Add `is_draft: bool = False` to `TranslationMessage` and `TranslationChunkMessage`
- Run shared alignment tests

**Tests:** Existing `test_ts_python_alignment.py` catches field mismatches.

## Step 3: Backend — Draft translation path

**Files:** `websocket_audio.py`

- Remove `if msg.is_draft: return` guard
- Add draft branch: translate `msg.text` directly, non-streaming, `Semaphore(1)` non-blocking, `draft_timeout_s`, no context write
- Final branch unchanged (existing streaming path)
- Add `is_draft` to all translation log events
- Log `draft_translation_dropped` and `draft_translation_timeout`
- Pass `is_draft` and `pipeline` to `_translate_and_send`
- `_translate_and_send`: branch on `is_draft` for streaming vs non-streaming, max_tokens, retries, context, persistence

**Tests:** Draft bypasses stable_text buffer, draft uses non-streaming, semaphore drops on busy, timeout releases semaphore.

## Step 4: Frontend — Store mutations

**Files:** `loopback.svelte.ts`

- Add `translationIsDraft: boolean` to `CaptionEntry`
- `addTranslation`: last-writer-wins guards, set `translationIsDraft` from `msg.is_draft ?? false`, only set `translationComplete` on non-draft
- `appendTranslationChunk`: clear draft text on first final chunk (`translationIsDraft === true` → reset before append), ignore if `translationComplete`
- `addSegment` draft→final: preserve draft translation

**Tests (vitest):** Draft→final lifecycle, last-writer-wins guard, chunk-on-draft clear, segment replacement preserves translation.

## Step 5: Frontend — Shared component + views

**Files:** NEW `TranslationText.svelte`, NEW `paragraph-helpers.ts`, update SplitView/SubtitleView/TranscriptView

- `TranslationText.svelte`: opacity by phase (0.65/0.85/1.0), indicator by phase, CSS transitions
- `paragraph-helpers.ts`: extract shared `paragraphTranslation()`, `hasPendingTranslation()`
- Wire into all three views

**Tests:** svelte-check, component renders correct CSS classes per phase.

## Step 6: Page wiring + manual verification

**Files:** `loopback/+page.svelte`

- Handle `is_draft` in `addTranslation` call (already uses `msg` passthrough)
- Restart services, test live with `just dev-debug`
- Verify: draft at ~800ms dimmed, final at ~5s full opacity, no garbled text, semaphore drops logged

---

## Verification

```bash
uv run pytest modules/orchestration-service/tests/unit/ -v -k "not Integration"
uv run pytest modules/shared/tests/ -v
cd modules/dashboard-service && npx svelte-check --threshold error
cd modules/dashboard-service && npx vitest run
```
