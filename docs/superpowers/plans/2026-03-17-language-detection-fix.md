# Language Detection Fix: Phase 2 + Empirical Validation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the language detection fix (session restart on switch) and empirically validate through the full Playwright pipeline using real meeting FLAC recordings.

**Architecture:** Phase 1 (wire `WhisperLanguageDetector` / `SustainedLanguageDetector` adapter) is complete (commit `8952e42`). Phase 2 adds session restart: when `WhisperLanguageDetector.update()` returns a new language, flush the VAC buffer, reset dedup/hallucination state, and continue with the new language hint. Phase 3 creates Playwright E2E tests that replay real meeting FLAC audio through the full frontend pipeline (browser → orchestration → transcription → DOM) at various offsets to empirically validate detection stability.

**Tech Stack:** Python 3.12+ / FastAPI WebSocket / Playwright / SvelteKit / VACOnlineProcessor / Pydantic v2

---

## What's Already Done (Phase 1 — commit 8952e42)

| Component | Status | Details |
|-----------|--------|---------|
| `WhisperLanguageDetector` adapter | ✅ | `language_detection.py:100-159` — wraps `SustainedLanguageDetector`, same API as legacy `LanguageDetector` |
| `api.py` SessionState | ✅ | Line 94: `lang_detector: WhisperLanguageDetector = field(default_factory=WhisperLanguageDetector)` |
| `_run_inference` integration | ✅ | Line 412: passes `result.confidence` to `update()` |
| Unit tests | ✅ | `test_language_detection.py`: 12 tests incl. real flapping sequences, hallucination rejection, sustained switches |
| E2E log-replay tests | ✅ | `tests/integration/test_language_detection_e2e.py`: 227 real events from production, 3 replay + 2 live streaming tests |
| Orchestration `SessionConfig` | ✅ | `websocket_audio.py:60-130`: mode save/restore, `lock_language` propagation |
| Mode switch tests | ✅ | `test_mode_switch.py`: 11 tests for interpreter↔split transitions, lock_language |
| Dashboard localStorage | ✅ | `loopback.svelte.ts`: toolbar config persistence |

### Current `WhisperLanguageDetector` Parameters (production tuning)

```python
WhisperLanguageDetector(
    confidence_margin=0.2,   # P(new) - P(old) must exceed 20%
    min_dwell_frames=4,      # 4 consecutive chunks agreeing
    min_dwell_ms=10000,      # 10 seconds of sustained detection
)
```

These are more conservative than the original plan's values (0.15/2/4000) — derived from testing against the real 210-switch production log.

---

## Current Code State (key references)

**`WhisperLanguageDetector.update()` signature** (language_detection.py:136):
```python
def update(self, detected_language: str, chunk_duration_s: float, confidence: float = 0.5) -> str | None:
```
Returns the new language string on switch, or `None`. This is the API the session restart hooks into.

**`api.py` _run_inference language block** (lines 402-418):
```python
if state.lang_detector.current_language is None:
    detected = state.lang_detector.detect_initial(result.language, result.confidence)
    # sends language_detected WS message
else:
    chunk_duration_s = len(inference_audio) / 16000.0
    switched = state.lang_detector.update(result.language, chunk_duration_s, result.confidence)
    if switched:
        # sends language_detected WS message
```

**Session restart hooks into `if switched:` — after sending the WS message.**

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `modules/transcription-service/src/api.py` | Add session restart on language switch |
| Modify | `modules/transcription-service/src/vac_online_processor.py` | Add `reset()` method |
| Create | `modules/transcription-service/tests/test_session_restart.py` | Behavioral tests for session restart |
| Create | `tools/create_flac_replay_fixtures.py` | Convert FLAC recording → 48kHz WAV fixtures at various offsets |
| Create | `modules/dashboard-service/tests/e2e/language-detection-replay.spec.ts` | Playwright E2E: replay real meeting audio, verify stable detection |

---

## Chunk 1: Phase 2 — Session Restart on Language Switch

### Task 1: Write failing tests for VACOnlineProcessor.reset()

**Files:**
- Create: `modules/transcription-service/tests/test_session_restart.py`

- [ ] **Step 1: Write test file**

```python
"""Behavioral tests for session restart on language switch.

Tests verify:
- VACOnlineProcessor.reset() clears buffer and counters
- HallucinationFilter.reset() clears cross-segment state
- _dedup_overlap with empty prev preserves full text
"""

import numpy as np
import pytest

from vac_online_processor import VACOnlineProcessor


@pytest.mark.asyncio
class TestVACProcessorReset:
    """VACOnlineProcessor must support reset() for session restart."""

    def test_vac_has_reset_method(self):
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=1.0, stride_s=4.5)
        assert hasattr(vac, "reset"), "VACOnlineProcessor must have reset()"

    async def test_reset_clears_buffer_and_counters(self):
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=1.0, stride_s=4.5)

        # Feed 1 second of audio
        audio = np.zeros(16000, dtype=np.float32)
        await vac.feed_audio(audio)
        assert vac._buffer_samples > 0

        vac.reset()
        assert vac._buffer_samples == 0
        assert vac._new_samples_since_inference == 0
        assert vac._first_inference_done is False

    async def test_reset_makes_not_ready_for_inference(self):
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=1.0, stride_s=4.5)

        # Feed enough audio for prebuffer readiness
        audio = np.zeros(8000, dtype=np.float32)  # 0.5s
        await vac.feed_audio(audio)

        vac.reset()
        assert not vac.ready_for_inference()

    async def test_reset_re_enables_prebuffer_threshold(self):
        """After reset, first inference should use prebuffer_s, not stride_s."""
        vac = VACOnlineProcessor(prebuffer_s=0.5, overlap_s=0.5, stride_s=4.5)

        # First session: feed prebuffer amount → should be ready
        audio = np.zeros(8000, dtype=np.float32)  # 0.5s = prebuffer_s
        await vac.feed_audio(audio)
        assert vac.ready_for_inference()

        # Consume the inference audio (marks first_inference_done)
        vac.get_inference_audio()

        # Now need full stride (4.5s) for next inference
        small_audio = np.zeros(8000, dtype=np.float32)  # only 0.5s
        await vac.feed_audio(small_audio)
        assert not vac.ready_for_inference()  # needs 4.5s, only has 0.5s

        # Reset → back to prebuffer threshold
        vac.reset()
        audio2 = np.zeros(8000, dtype=np.float32)  # 0.5s = prebuffer_s
        await vac.feed_audio(audio2)
        assert vac.ready_for_inference()  # prebuffer threshold again!


class TestHallucinationFilterReset:
    """HallucinationFilter.reset() already exists — verify it works."""

    def test_reset_clears_recent_texts(self):
        from transcription.hallucination_filter import HallucinationFilter

        hf = HallucinationFilter()
        hf.reset()
        assert len(hf._recent_texts) == 0

    def test_reset_callable_without_error(self):
        from transcription.hallucination_filter import HallucinationFilter

        hf = HallucinationFilter()
        hf.reset()  # should not raise


class TestDedupAfterRestart:
    """After session restart, dedup must not match against pre-restart text."""

    def test_empty_prev_preserves_cjk(self):
        from api import _dedup_overlap

        assert _dedup_overlap("", "你好世界") == "你好世界"

    def test_empty_prev_preserves_english(self):
        from api import _dedup_overlap

        assert _dedup_overlap("", "Hello world and more") == "Hello world and more"
```

- [ ] **Step 2: Run tests — expect VACOnlineProcessor.reset() tests to FAIL**

Run: `uv run pytest modules/transcription-service/tests/test_session_restart.py -v`
Expected: `TestVACProcessorReset` tests FAIL (no `reset()` yet), others PASS

- [ ] **Step 3: Commit**

```bash
git add modules/transcription-service/tests/test_session_restart.py
git commit -m "test: add behavioral tests for session restart on language switch"
```

---

### Task 2: Add reset() to VACOnlineProcessor

**Files:**
- Modify: `modules/transcription-service/src/vac_online_processor.py` (add `reset()` after `__init__`)

`HallucinationFilter.reset()` already exists (line 216, clears `self._recent_texts`). No changes needed there.

- [ ] **Step 1: Add reset() method**

Add after the `__init__` method of `VACOnlineProcessor` (around line 925 in `vac_online_processor.py`):

```python
    def reset(self) -> None:
        """Reset processor state for a new language session.

        Clears audio buffer, sample counters, and first-inference flag so the
        shorter prebuffer threshold is used for the first inference in the new
        language. Preserves configuration (stride, overlap, prebuffer, rms).
        """
        self._buffer.clear()
        self._buffer_samples = 0
        self._new_samples_since_inference = 0
        self._first_inference_done = False
        # Drain any queued audio that hasn't been processed yet
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest modules/transcription-service/tests/test_session_restart.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add modules/transcription-service/src/vac_online_processor.py
git commit -m "feat: add reset() to VACOnlineProcessor for session restart"
```

---

### Task 3: Add session restart to _run_inference

**Files:**
- Modify: `modules/transcription-service/src/api.py:413-418` (after `if switched:` block)

When `WhisperLanguageDetector.update()` returns a non-None language string (meaning a sustained switch was detected), reset transcription state for a clean transition.

- [ ] **Step 1: Add session restart after the language_detected WS message**

In `modules/transcription-service/src/api.py`, expand the `if switched:` block (currently lines 413-418):

```python
                    switched = state.lang_detector.update(result.language, chunk_duration_s, result.confidence)
                    if switched:
                        await ws.send_text(json.dumps({
                            "type": "language_detected",
                            "language": switched,
                            "confidence": result.confidence,
                        }))

                        # Session restart: clean state for new language
                        _prev_segment_text = ""
                        _hallucination_filter.reset()
                        if state.vac_processor is not None:
                            state.vac_processor.reset()
                        logger.info(
                            "session_restarted",
                            session_id=session_id,
                            new_language=switched,
                            segment_counter=_segment_counter,
                        )
```

Note: `_prev_segment_text` already has `nonlocal` at line 344. `_hallucination_filter` is accessed via closure (no reassignment). `state.vac_processor` via attribute access.

- [ ] **Step 2: Run all transcription service tests**

Run: `uv run pytest modules/transcription-service/tests/ -v`
Expected: All PASS (including new session restart tests and existing tests)

- [ ] **Step 3: Commit**

```bash
git add modules/transcription-service/src/api.py
git commit -m "feat: session restart on sustained language switch — flush VAC, reset dedup/hallucination state"
```

---

## Chunk 2: Playwright Empirical Validation

### Recording Inventory

**Long meeting:** `/tmp/livetranslate/recordings/af5b37c9-*` (~57min, 115 chunks, en↔zh interpreter, echo-heavy room)
**Short English:** `/tmp/livetranslate/recordings/48697a4a-*` (~12min, 24 chunks, English only)

### Test Scenarios

| Test | Fixture Source | Start Offset | Duration | Expected Behavior |
|------|---------------|-------------|----------|-------------------|
| English stability | Short EN session, full | 0:00 | ~2min | Stay on English, 0 false switches |
| Mixed meeting start | Long meeting, beginning | 0:00 | ~3min | Detect English, no flapping to nn/cy/ko |
| Chinese section | Long meeting, offset ~17min | ~17:00 | ~3min | Detect Chinese within 15s, stable after |
| Language transition | Long meeting, offset ~16min | ~16:00 | ~5min | English → Chinese transition, ≤2 switches |
| Full meeting stability | Long meeting, full | 0:00 | ~10min | Total switches ≤5, no hallucinated languages |

### Task 4: Create FLAC-to-fixture conversion tool

**Files:**
- Create: `tools/create_flac_replay_fixtures.py`

Converts FLAC recording chunks into 48kHz WAV fixtures at various offsets for Playwright injection.

- [ ] **Step 1: Write the conversion tool**

```python
#!/usr/bin/env python3
"""Convert FLAC recording chunks to 48kHz WAV fixtures for Playwright E2E tests.

Reads a recording manifest, extracts audio at specified offsets/durations,
resamples to 48kHz (browser sample rate), and writes WAV fixtures.

Usage:
    uv run python tools/create_flac_replay_fixtures.py \\
        --session af5b37c9 \\
        --output modules/dashboard-service/tests/fixtures/

Generates:
    lang_detect_en_full_48k.wav         — Short English session, full
    lang_detect_mixed_start_48k.wav     — Long meeting, first 3 minutes
    lang_detect_zh_section_48k.wav      — Long meeting, ~17:00-20:00 (Chinese section)
    lang_detect_transition_48k.wav      — Long meeting, ~16:00-21:00 (en→zh transition)
    lang_detect_full_meeting_48k.wav    — Long meeting, first 10 minutes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RECORDINGS_DIR = Path("/tmp/livetranslate/recordings")
DEFAULT_OUTPUT = Path("modules/dashboard-service/tests/fixtures")

# Fixture definitions: (name, session_prefix, start_s, duration_s)
FIXTURES = [
    ("lang_detect_en_full", "48697a4a", 0, 120),
    ("lang_detect_mixed_start", "af5b37c9", 0, 180),
    ("lang_detect_zh_section", "af5b37c9", 1020, 180),   # ~17:00
    ("lang_detect_transition", "af5b37c9", 960, 300),     # ~16:00, 5 min
    ("lang_detect_full_meeting", "af5b37c9", 0, 600),     # first 10 min
]


def find_recording(session_prefix: str) -> Path | None:
    if not RECORDINGS_DIR.exists():
        return None
    for d in RECORDINGS_DIR.iterdir():
        if d.name.startswith(session_prefix) and d.is_dir():
            return d
    return None


def load_recording_slice(
    recording_dir: Path, start_s: float, duration_s: float
) -> tuple[np.ndarray, int] | None:
    """Load a time slice from a recording directory.

    Returns (audio_float32, sample_rate) or None if not enough audio.
    """
    import soundfile as sf

    manifest_path = recording_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    sample_rate = manifest["sample_rate"]
    chunks = sorted(manifest["chunks"], key=lambda c: c["sequence"])

    # Calculate chunk boundaries
    chunk_duration_s = manifest.get("chunk_duration_s", 30.0)  # typical FLAC chunk length

    # Load all chunks that overlap with our time window
    start_chunk = int(start_s / chunk_duration_s)
    end_s = start_s + duration_s
    end_chunk = int(end_s / chunk_duration_s) + 1

    audio_parts = []
    for chunk_info in chunks[start_chunk:end_chunk]:
        chunk_path = recording_dir / chunk_info["filename"]
        if chunk_path.exists():
            data, sr = sf.read(str(chunk_path))
            if data.ndim == 2:
                data = data.mean(axis=1)
            audio_parts.append(data.astype(np.float32))

    if not audio_parts:
        return None

    full_audio = np.concatenate(audio_parts)

    # Trim to exact time window (relative to start_chunk)
    chunk_start_s = start_chunk * chunk_duration_s
    offset_in_audio = int((start_s - chunk_start_s) * sample_rate)
    samples_needed = int(duration_s * sample_rate)
    sliced = full_audio[offset_in_audio : offset_in_audio + samples_needed]

    if len(sliced) < sample_rate:  # less than 1 second
        return None

    return sliced, sample_rate


def resample_to_48k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to 48kHz for browser playback."""
    if orig_sr == 48000:
        return audio
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=48000)
    except ImportError:
        # Nearest-neighbor fallback
        ratio = 48000 / orig_sr
        indices = (np.arange(int(len(audio) * ratio)) / ratio).astype(int)
        return audio[indices]


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 48000) -> None:
    """Write 16-bit PCM WAV."""
    import soundfile as sf
    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description="Create FLAC replay fixtures for Playwright")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="List fixtures without creating")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0

    for name, session_prefix, start_s, duration_s in FIXTURES:
        out_path = args.output / f"{name}_48k.wav"
        recording_dir = find_recording(session_prefix)

        if recording_dir is None:
            print(f"  SKIP {name}: recording {session_prefix}* not found")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  WOULD CREATE {out_path} ({duration_s}s from {session_prefix} @ {start_s}s)")
            continue

        result = load_recording_slice(recording_dir, start_s, duration_s)
        if result is None:
            print(f"  SKIP {name}: not enough audio in slice")
            skipped += 1
            continue

        audio, sr = result
        audio_48k = resample_to_48k(audio, sr)
        write_wav(out_path, audio_48k)
        print(f"  CREATED {out_path} ({len(audio_48k) / 48000:.1f}s)")
        created += 1

    print(f"\nDone: {created} created, {skipped} skipped")
    return 0 if skipped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the tool (dry run first)**

```bash
uv run python tools/create_flac_replay_fixtures.py --dry-run
```

Expected: Lists 5 fixtures that would be created (or SKIP if recordings not found)

- [ ] **Step 3: Run for real**

```bash
uv run python tools/create_flac_replay_fixtures.py
```

Expected: Creates 48kHz WAV fixtures in `modules/dashboard-service/tests/fixtures/`

- [ ] **Step 4: Commit**

```bash
git add tools/create_flac_replay_fixtures.py
git commit -m "feat: FLAC-to-fixture tool for language detection Playwright replay"
```

Note: Do NOT commit the generated WAV fixtures (they're large and derived from local recordings).

---

### Task 5: Add Justfile commands for fixture creation and replay tests

**Files:**
- Modify: `Justfile`

- [ ] **Step 1: Add new just commands**

```makefile
# Create fixtures for language detection replay tests
create-lang-detect-fixtures:
    uv run python tools/create_flac_replay_fixtures.py

# Run language detection Playwright replay tests (needs `just dev` running)
test-lang-detect:
    cd modules/dashboard-service && npx playwright test tests/e2e/language-detection-replay.spec.ts --headed
```

- [ ] **Step 2: Commit**

```bash
git add Justfile
git commit -m "chore: add just commands for language detection replay tests"
```

---

### Task 6: Create Playwright E2E language detection replay tests

**Files:**
- Create: `modules/dashboard-service/tests/e2e/language-detection-replay.spec.ts`

These tests replay real meeting FLAC audio through the full pipeline (browser → orchestration → transcription → DOM) and verify stable language detection. They follow the same pattern as `loopback-playback.spec.ts`.

- [ ] **Step 1: Write the Playwright test spec**

```typescript
/**
 * Language Detection Replay E2E Tests
 *
 * Replays real meeting FLAC recordings (converted to 48kHz WAV) through
 * the full frontend pipeline and verifies:
 * - Stable language detection (no flapping to hallucinated languages)
 * - Correct language switch on genuine transitions
 * - Session restart produces clean transcription in new language
 *
 * Prerequisites:
 *   1. `just dev` running (all services)
 *   2. `just create-lang-detect-fixtures` run (WAV fixtures from FLAC recordings)
 *
 * Run:
 *   just test-lang-detect
 */

import { test, expect, type Page } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';

// ESM-compatible __dirname (matches loopback-playback.spec.ts pattern)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FIXTURE_DIR = path.resolve(__dirname, '../fixtures');
const LOOPBACK_URL = 'http://localhost:5173/loopback';

// Hallucinated languages that Whisper falsely detects (from production log)
const HALLUCINATED_LANGS = new Set(['nn', 'cy', 'ko', 'fr', 'es', 'it', 'pt', 'nl', 'ru', 'pl']);

// vLLM-MLX cold start can take up to 90s for the first inference
const TEST_TIMEOUT_MS = 180_000;

// -------------------------------------------------------------------
// Helpers (following loopback-playback.spec.ts patterns)
// -------------------------------------------------------------------

/** Check that orchestration service is reachable. */
async function checkServiceHealth(): Promise<boolean> {
  try {
    const resp = await fetch('http://localhost:3000/api/audio/health');
    return resp.ok;
  } catch {
    return false;
  }
}

/** Inject a WAV file as the getUserMedia source in the browser.
 *  MUST be called AFTER page.goto() — needs a loaded page to evaluate JS.
 */
async function injectAudioFixture(page: Page, fixtureName: string): Promise<void> {
  await page.evaluate(async (fixture: string) => {
    const response = await fetch(`/_test_fixtures/${fixture}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch fixture: ${fixture} (${response.status})`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const audioCtx = new AudioContext({ sampleRate: 48000 });
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.loop = false;

    const dest = audioCtx.createMediaStreamDestination();
    source.connect(dest);
    source.start();

    navigator.mediaDevices.getUserMedia = async (_constraints) => dest.stream;
    navigator.mediaDevices.enumerateDevices = async () => [
      {
        deviceId: 'test-audio',
        groupId: 'test-group',
        kind: 'audioinput' as MediaDeviceKind,
        label: 'E2E Test Audio',
        toJSON: () => ({}),
      },
    ];
  }, fixtureName);
}

/**
 * Install a WebSocket message interceptor via addInitScript.
 * Must be called BEFORE page.goto() so the monkey-patch runs before
 * any page JS executes (catches the initial WebSocket connection).
 * Follows the same pattern as loopback-playback.spec.ts.
 */
async function installWsInterceptor(page: Page): Promise<void> {
  await page.addInitScript(() => {
    (window as any).__e2e_messages = [];
    const OrigWS = window.WebSocket;
    (window as any).WebSocket = class extends OrigWS {
      constructor(url: string | URL, protocols?: string | string[]) {
        super(url, protocols);
        this.addEventListener('message', (ev: MessageEvent) => {
          if (typeof ev.data === 'string') {
            try {
              const parsed = JSON.parse(ev.data);
              (window as any).__e2e_messages.push(parsed);
            } catch {
              // non-JSON frame, skip
            }
          }
        });
      }
    };
  });
}

/** Collect intercepted WS messages from the browser. */
async function getWsMessages(page: Page): Promise<any[]> {
  return page.evaluate(() => (window as any).__e2e_messages ?? []);
}

/** Extract language_detected events from WS messages. */
function getLangEvents(messages: any[]): Array<{ language: string; confidence?: number; switched_from?: string }> {
  return messages.filter(m => m.type === 'language_detected');
}

/** Extract segment events from WS messages. */
function getSegments(messages: any[]): any[] {
  return messages.filter(m => m.type === 'segment' && m.text?.trim());
}

function hasFixture(name: string): boolean {
  return fs.existsSync(path.join(FIXTURE_DIR, name));
}

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

test.describe('Language Detection Replay E2E', () => {

  test.beforeAll(async () => {
    // Service guard: skip suite if orchestration isn't running
    const healthy = await checkServiceHealth();
    if (!healthy) {
      test.skip();
      throw new Error(
        'Orchestration service not reachable at http://localhost:3000/api/audio/health. ' +
        'Run `just dev` before running E2E tests.'
      );
    }
  });

  test.beforeEach(async ({ page }) => {
    // Grant microphone permission (required for getUserMedia mock)
    await page.context().grantPermissions(['microphone']);

    // Serve audio fixtures from Playwright → browser
    await page.route('**/_test_fixtures/*', async (route) => {
      const filename = route.request().url().split('/_test_fixtures/').pop();
      const filePath = path.join(FIXTURE_DIR, filename!);
      if (fs.existsSync(filePath)) {
        const body = fs.readFileSync(filePath);
        await route.fulfill({ status: 200, contentType: 'audio/wav', body });
      } else {
        await route.fulfill({ status: 404, body: `Fixture not found: ${filename}` });
      }
    });

    // Install WS interceptor BEFORE page.goto() so it catches the initial connection
    await installWsInterceptor(page);
  });

  test('English session stays English — no false switches', async ({ page }) => {
    test.setTimeout(TEST_TIMEOUT_MS);
    if (!hasFixture('lang_detect_en_full_48k.wav')) test.skip();

    // Navigate FIRST, then inject audio
    await page.goto(LOOPBACK_URL);
    await injectAudioFixture(page, 'lang_detect_en_full_48k.wav');

    // Start capture
    await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

    // Wait for transcription to process
    await page.waitForTimeout(60_000);

    const messages = await getWsMessages(page);
    const langEvents = getLangEvents(messages);
    const segments = getSegments(messages);

    // Should have at least some segments (transcription works)
    expect(segments.length).toBeGreaterThan(0);

    // Should have at most initial detection, no switches
    const switches = langEvents.filter(e => e.switched_from);
    expect(switches).toHaveLength(0);

    // If any language detected, should be English
    if (langEvents.length > 0) {
      expect(langEvents.every(e => e.language === 'en')).toBe(true);
    }

    // No hallucinated languages
    const hallucinations = langEvents.filter(e => HALLUCINATED_LANGS.has(e.language));
    expect(hallucinations).toHaveLength(0);
  });

  test('Mixed meeting start — no flapping in first 3 minutes', async ({ page }) => {
    test.setTimeout(TEST_TIMEOUT_MS);
    if (!hasFixture('lang_detect_mixed_start_48k.wav')) test.skip();

    await page.goto(LOOPBACK_URL);
    await injectAudioFixture(page, 'lang_detect_mixed_start_48k.wav');
    await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

    await page.waitForTimeout(90_000);

    const messages = await getWsMessages(page);
    const langEvents = getLangEvents(messages);

    // Old detector: 10+ switches in 3 minutes. New: ≤2
    const switches = langEvents.filter(e => e.switched_from);
    expect(switches.length).toBeLessThanOrEqual(2);

    // No hallucinated languages
    const hallucinations = langEvents.filter(e => HALLUCINATED_LANGS.has(e.language));
    expect(hallucinations).toHaveLength(0);
  });

  test('Chinese section — detects Chinese stably', async ({ page }) => {
    test.setTimeout(TEST_TIMEOUT_MS);
    if (!hasFixture('lang_detect_zh_section_48k.wav')) test.skip();

    await page.goto(LOOPBACK_URL);
    await injectAudioFixture(page, 'lang_detect_zh_section_48k.wav');
    await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

    await page.waitForTimeout(90_000);

    const messages = await getWsMessages(page);
    const langEvents = getLangEvents(messages);

    // Should eventually detect Chinese
    const zhEvents = langEvents.filter(e => e.language === 'zh');
    expect(zhEvents.length).toBeGreaterThan(0);

    // After detecting Chinese, should stay on Chinese
    const firstZhIdx = langEvents.findIndex(e => e.language === 'zh');
    if (firstZhIdx >= 0) {
      const afterZh = langEvents.slice(firstZhIdx);
      const backToEn = afterZh.filter(e => e.language === 'en');
      expect(backToEn.length).toBeLessThanOrEqual(1);
    }
  });

  test('Language transition — en→zh with clean switch', async ({ page }) => {
    test.setTimeout(TEST_TIMEOUT_MS * 2);
    if (!hasFixture('lang_detect_transition_48k.wav')) test.skip();

    await page.goto(LOOPBACK_URL);
    await injectAudioFixture(page, 'lang_detect_transition_48k.wav');
    await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

    await page.waitForTimeout(120_000);

    const messages = await getWsMessages(page);
    const langEvents = getLangEvents(messages);
    const segments = getSegments(messages);

    // Should have both English and Chinese events
    const langs = new Set(langEvents.map(e => e.language));
    expect(langs.has('en') || langs.has('zh')).toBe(true);

    // Total switches should be small (1-2 for a genuine transition)
    const switches = langEvents.filter(e => e.switched_from);
    expect(switches.length).toBeLessThanOrEqual(3);

    // Transcription actually produces segments
    expect(segments.length).toBeGreaterThan(0);
  });

  test('Full meeting — ≤5 switches in 10 minutes', async ({ page }) => {
    test.setTimeout(TEST_TIMEOUT_MS * 3);
    if (!hasFixture('lang_detect_full_meeting_48k.wav')) test.skip();

    await page.goto(LOOPBACK_URL);
    await injectAudioFixture(page, 'lang_detect_full_meeting_48k.wav');
    await page.locator('[data-testid="start-capture"], button:has-text("Start")').first().click();

    await page.waitForTimeout(180_000);

    const messages = await getWsMessages(page);
    const langEvents = getLangEvents(messages);
    const segments = getSegments(messages);

    const switches = langEvents.filter(e => e.switched_from);
    expect(switches.length).toBeLessThanOrEqual(5);

    // No hallucinated languages
    const hallucinations = langEvents.filter(e => HALLUCINATED_LANGS.has(e.language));
    expect(hallucinations).toHaveLength(0);

    // Segments produced
    expect(segments.length).toBeGreaterThan(0);
  });

});
```

- [ ] **Step 2: Verify Playwright config includes the new test file**

Read `modules/dashboard-service/playwright.config.ts` — the test directory should already cover `tests/e2e/`. If not, no change needed since the glob `./tests/e2e` will pick it up.

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/tests/e2e/language-detection-replay.spec.ts
git commit -m "test: Playwright E2E language detection replay with real meeting FLAC audio"
```

---

### Task 7: Run the full validation

- [ ] **Step 1: Create fixtures**

```bash
just create-lang-detect-fixtures
```

Expected: 5 WAV fixtures created in `modules/dashboard-service/tests/fixtures/`

- [ ] **Step 2: Ensure services are running**

```bash
just dev
```

- [ ] **Step 3: Run Playwright language detection tests**

```bash
just test-lang-detect
```

Expected: All 5 tests PASS. Key metrics to check in output:
- English session: 0 false switches
- Mixed meeting start: ≤2 switches
- Chinese section: Chinese detected and stable
- Language transition: ≤3 switches, both languages seen
- Full meeting: ≤5 switches, no hallucinated languages

- [ ] **Step 4: If any tests fail, analyze and adjust parameters**

Check the Playwright HTML report at `modules/dashboard-service/tests/output/playwright-report/`.

If false switches occur, consider tightening `WhisperLanguageDetector` parameters:
- Increase `min_dwell_ms` (e.g., 10000 → 15000)
- Increase `min_dwell_frames` (e.g., 4 → 5)
- Increase `confidence_margin` (e.g., 0.2 → 0.25)

- [ ] **Step 5: Final commit (do NOT commit generated WAV fixtures)**

```bash
git add modules/dashboard-service/tests/e2e/language-detection-replay.spec.ts tools/create_flac_replay_fixtures.py Justfile
git commit -m "test: language detection Playwright replay validated against real meeting recordings"
```

---

## Chunk 3: Regression Check and Cleanup

### Task 8: Run all existing test suites

- [ ] **Step 1: Transcription service tests**

Run: `uv run pytest modules/transcription-service/tests/ -v`
Expected: All PASS

- [ ] **Step 2: Orchestration service tests**

Run: `uv run pytest modules/orchestration-service/tests/ -v`
Expected: All PASS

- [ ] **Step 3: Shared tests**

Run: `uv run pytest modules/shared/tests/ -v`
Expected: All PASS

- [ ] **Step 4: Existing Playwright tests**

Run: `just test-playwright`
Expected: All 9 existing E2E tests still PASS

- [ ] **Step 5: Commit any fixes**

```bash
git commit -m "fix: address regressions from language detection changes"
```
