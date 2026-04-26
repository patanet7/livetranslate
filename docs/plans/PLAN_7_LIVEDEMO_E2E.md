# Plan 7: `livedemo` — End-to-End Live Translation + Subtitle Bot Pipeline

**Status:** Implemented (Phases 0–7, 9.1–9.7, 6.5 ✅) — Phase 8 (merge) and Phase 6.75 (live Meet smoke) are manual gates
**Owner:** TBD
**Last updated:** 2026-04-25
**Related plans:** PLAN_6_STREAMING_TUNING.md (translation streaming)
**Related ADRs:** ADR 0003 (`docs/adr/0003-canvas-ws-vs-pyvirtualcam.md`) — dual-sink decision

## 0. Implementation status

| Phase | Status | Notes |
|---|---|---|
| 0. Foundation | ✅ | `LiveDemoConfig` (Pydantic Settings, env > yaml > defaults) + preflight registry |
| 1. Bot harness + TS runner | ✅ | Python WS server + `bot_runner/runner.ts` with init script, B5/B6 verified in real Chromium |
| 2. Sinks | ✅ | `PngSink`, `CanvasWsSink`, plus `PyVirtualCamSink` (added in 9.1) |
| 3. File source + smoke | ✅ | `FileSource`, `WSRecorder`, B7+B8 round-trip lossless replay |
| 4. Fireflies source | ✅ | DI-friendly with gated real-API E2E |
| 5. Mic source | ✅ | Production WS protocol, gated real-orchestration E2E |
| 6. CLI + console-script | ✅ | `livedemo` registered in `[project.scripts]`, `doctor`/`run`/`smoke`/`replay` |
| 6.5. Pipeline E2E | ✅ | file → canvas_ws → harness → fake bot, full record-replay round-trip |
| 7. add_translation shim | ✅ | Single edit in `virtual_webcam.py:284-301` makes all 9 callers paired-block aware |
| **8. Merge bot-auth** | ⏸ | Manual gate — needs user authorization for the merge |
| **6.75. Live Meet smoke** | ⏸ | Manual gate — needs user's meeting URL + admit-the-bot flow |
| 9.1. pyvirtualcam sink + ADR | ✅ | `sinks/pyvirtualcam.py` + ADR 0003 |
| 9.2. MeetAudioSource | ✅ | Bot-page audio capture (init_script Web Audio tap) → harness → orchestration |
| 9.3. Centralised selectors | ✅ | `bot_runner/src/lib/selectors.ts` (will consolidate post-merge with meeting-bot-service) |
| 9.4. structlog | ✅ | `run_started`/`run_completed`/`frame_rendered` events; no `import logging` lint test |
| 9.5. Mic-off + cam-on | ✅ | `muteMic` + `ensureCameraOn` helpers run pre-`clickJoin` |
| 9.6. Join message + `/stop` | ✅ | Canonical chat message on join; `leave_request` WS handler |
| 9.7. MeetingSessionConfig live | ✅ | All three sinks subscribe; `apply_meeting_config_snapshot` per frame |

---

## 1. Purpose

Build a single, DRY, config-driven CLI (`livedemo`) that exercises the **complete LiveTranslate stack end-to-end** by:

1. Starting a Google Meet bot that joins a real meeting
2. Capturing audio either from a local mic OR replaying a Fireflies transcript
3. Routing audio/text through the **production** orchestration-service WebSocket → transcription-service → `TranslationService` (the same path the real frontend uses)
4. Rendering paired (original + translation) captions onto a canvas-backed `MediaStream` that becomes the bot's webcam, so other meeting participants see live subtitles on the bot's video tile
5. Verifying the run via recorded WebSocket message logs that double as fixtures for regression tests

The system MUST hit the real components — no mocked transcription, no in-process translation shortcut for the mic path. If a service is down, **preflight** fails the run before the bot joins the meeting.

## 2. Why now

- Three throwaway test scripts (`test_subtitle_canvas.ts`, `test_subtitle_stream.ts`, `/tmp/render_test_subtitles.py`) currently demonstrate parts of this in isolation, none of them DRY, none truly end-to-end.
- We have no automated way to reproduce a "live demo" — the chain breaks silently when any of postgres/ollama/transcription/orchestration/chrome-profile is misconfigured.
- The bot's webcam-subtitle path was just rebuilt (`add_caption()` API, paired-block rendering, diarization-tag suppression) and lacks an integration harness that proves it works against the real translation stream.
- Fireflies replay gives us deterministic, repeatable end-to-end runs without needing two humans on a Meet call.

## 3. Goals & Non-goals

### Goals
- One CLI entry point, one config schema, one bot harness — eliminate duplicated test code.
- **All sources go through the orchestration-service WebSocket** so we exercise production code paths (WebSocket framing, draft/final routing, context store, lock-language propagation).
- Preflight every dependency before joining a Meet; print actionable hints.
- Recorded WS message logs per run, replayable for regression tests.
- Deterministic smoke test runnable in CI (no network, no meeting).
- Migrate production callers from legacy `add_translation()` to `add_caption()` so they share the paired-block rendering path.

### Non-goals
- Replacing the production frontend (the React UI keeps its own audio capture).
- Building new translation features — translation logic is already in `src/translation/`.
- Multi-bot scenarios.
- Cross-platform mic capture (macOS-first; Linux later if needed).
- Headless rendering optimisation — 10fps canvas writes are fine.

## 4. Scope

### In scope
- New Python package `modules/orchestration-service/src/livedemo/` (CLI, config, preflight, harness, sources, sinks, recorder).
- Thin TypeScript bot runner that the harness spawns as a Playwright subprocess.
- Migration of existing callers (`bot_integration.py`, `pil_virtual_cam_renderer.py`, `tests/test_bot_lifecycle.py`) to `add_caption()`.
- Deletion of replaced one-off scripts.
- New tests asserting paired-block layout and an end-to-end smoke run.

### Out of scope
- Touching the production React frontend.
- Adding new LLM providers.
- Schema/migration changes (DB stays as-is; `alembic_head` only checked in preflight).

## 5. Architecture overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                     livedemo CLI (Python)                              │
│                                                                        │
│  ┌──────────┐  ┌────────────┐  ┌─────────────────────────────────┐     │
│  │ config   │→ │ preflight  │→ │  Source ──► WS recorder ──► Sink│     │
│  │ (Pydantic│  │  (doctor)  │  │                                 │     │
│  │  + YAML) │  └────────────┘  └─────────────────────────────────┘     │
│  └──────────┘                                                          │
└─────┬───────────────────────────────────────────────────────┬──────────┘
      │ spawns                                                │ canvas WS
      ▼                                                       ▼
┌──────────────────┐                               ┌───────────────────┐
│ Playwright Node  │  ── bot's webcam ──►          │  Google Meet      │
│ subprocess       │     (canvas captureStream)    │  (real meeting)   │
└──────────────────┘                               └───────────────────┘
      ▲                                                       ▲
      │ getUserMedia override                                 │ user admits bot
      │
      └── frame WS  ◄── PNG-encoded frames from VirtualWebcamManager

Source variants:
  ┌────────────┐         ┌──────────────┐    ┌────────────┐
  │ mic source │ ──audio─►│ orchestration│──►│transcription│──► segments
  └────────────┘         │   /ws/audio  │    └────────────┘    │
                         │   (prod path)│                      ▼
                         └──────────────┘                ┌─────────────┐
                                ▲                       │TranslationS.│
                                │                       └─────────────┘
                                │                              │
  ┌────────────┐  text+timing   │                              │
  │fireflies   │ ───────────────┘                              ▼
  │ source     │                                       ┌─────────────────┐
  └────────────┘                                       │ frontend WS msgs│
                                                       │ (translation_*) │
                                                       └────────┬────────┘
  ┌────────────┐                                                │
  │ file source│ ───────── replay raw recorded WS ──────────────┤
  │ (recorded) │                                                ▼
  └────────────┘                                       VirtualWebcamManager
                                                       .add_caption()
                                                          │
                                                          ▼ PNG frame
                                                       canvas WS sink
```

### Key design decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | Python orchestrator, TS bot subprocess | Translation + Fireflies + render are Python; bot is Playwright/TS already. Python owns the lifecycle. |
| 2 | Mic source goes through orchestration WS, NOT in-process | Tests the real WebSocket framing, draft/final routing, context store. If demo passes, prod frontend works. |
| 3 | Fireflies source calls `TranslationService` directly | Skipping transcription is correct — Fireflies *is* the transcription. Still uses prod LLM client + context store. |
| 4 | Frames over WS, not file polling | 1280×720 RGB at 10fps = 26 MB/s — file polling adds disk thrash; WS streams without IO contention. |
| 5 | WS recorder per run | Every WS message (audio frames, transcripts, translations, frame pushes) is captured to `runs/<timestamp>/messages.jsonl`. Replayable as a `file` source. |
| 6 | Single Pydantic config | `LiveDemoConfig` is the only place env vars / YAML / CLI flags merge. Eliminates the "where does this setting live" problem. |
| 7 | Canvas appended to DOM | Fixes the captureStream stall observed in the prior `test_subtitle_canvas.ts` run. |
| 8 | Stronger in-call signal | People-panel button or self-tile presence — `Leave` button is visible in the lobby too, which caused the prior misfire. |

## 6. Component layout

```
modules/orchestration-service/src/livedemo/
├── __init__.py
├── cli.py                       # `livedemo run|doctor|smoke|replay` (typer)
├── config.py                    # LiveDemoConfig (Pydantic Settings + from_yaml)
├── preflight.py                 # check_all() returns CheckResult[]
├── recorder.py                  # WSRecorder: dump every msg to JSONL
├── pipeline.py                  # Wires source → recorder → sink, owns loop
├── bot_harness.py               # Async ctx mgr: spawns TS bot, opens WS to it
├── sources/
│   ├── __init__.py
│   ├── base.py                  # SubtitleSource ABC, CaptionEvent dataclass
│   ├── mic.py                   # sounddevice → orchestration /ws/audio
│   ├── fireflies.py             # GraphQL → TranslationService → events
│   └── file.py                  # Replay recorder JSONL
├── sinks/
│   ├── __init__.py
│   ├── base.py                  # CaptionSink ABC
│   ├── canvas_ws.py             # Push frames to bot harness
│   └── png.py                   # Write PNGs to disk (CI / fast iteration)
└── bot_runner/                  # TS subprocess
    ├── package.json
    ├── tsconfig.json
    └── src/
        └── runner.ts            # Init script, getUserMedia override, frame WS

modules/orchestration-service/tests/
├── test_livedemo_config.py
├── test_livedemo_preflight.py
├── test_livedemo_smoke.py       # file → png, fully offline
└── fixtures/livedemo/
    └── short-dialog.jsonl       # Recorded run, 6 captions, deterministic

docs/plans/PLAN_7_LIVEDEMO_E2E.md    # this doc
```

### Files DELETED by this plan
- `.worktrees/meeting-bot-auth/modules/meeting-bot-service/test_subtitle_canvas.ts`
- `.worktrees/meeting-bot-auth/modules/meeting-bot-service/test_subtitle_stream.ts`
- `/tmp/render_test_subtitles.py` (move into `sinks/png.py` + a fixture)
- `test_google_meet_join.ts` modifications reverted (harness covers it)

## 7. Configuration

Single source of truth: `LiveDemoConfig` in `livedemo/config.py`.

Loaded from (in precedence order, highest wins):
1. CLI flags (`--meeting-url`)
2. Environment vars (`LIVEDEMO_MEETING_URL`)
3. YAML file (`--config livedemo.yaml`)
4. Pydantic defaults

```python
class LiveDemoConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LIVEDEMO_",
        env_file=".env",
        extra="ignore",
    )

    # ── Bot ───────────────────────────────────────────────
    meeting_url: HttpUrl
    chrome_profile_dir: Path = Path.home() / ".config/livetranslate/chrome-profile"
    bot_display_mode: DisplayMode = DisplayMode.SUBTITLE
    bot_show_diarization_ids: bool = False

    # ── Bridge ────────────────────────────────────────────
    canvas_ws_port: int = 7081
    frame_fps: int = 10

    # ── Source selection ──────────────────────────────────
    source: Literal["mic", "fireflies", "file"]
    sink: Literal["canvas", "png"] = "canvas"

    # ── Source: mic ───────────────────────────────────────
    mic_device: str | None = None
    orchestration_ws_url: str = "ws://localhost:8000/ws/audio"

    # ── Source: fireflies ─────────────────────────────────
    fireflies_meeting_id: str | None = None
    fireflies_replay_speed: float = 1.0  # 0 = instant

    # ── Source: file ──────────────────────────────────────
    replay_jsonl: Path | None = None

    # ── Translation ───────────────────────────────────────
    source_language: str = "auto"
    target_language: str = "en"

    # ── Recording ─────────────────────────────────────────
    runs_dir: Path = Path("runs/livedemo")
    record_messages: bool = True
```

### Example YAML

```yaml
# configs/livedemo/mic-demo.yaml
meeting_url: https://meet.google.com/qzw-gpwm-ttq
source: mic
sink: canvas
target_language: zh
mic_device: "MacBook Pro Microphone"
bot_display_mode: SUBTITLE
```

```yaml
# configs/livedemo/fireflies-replay.yaml
meeting_url: https://meet.google.com/qzw-gpwm-ttq
source: fireflies
sink: canvas
fireflies_meeting_id: ABC123XYZ
fireflies_replay_speed: 2.0
target_language: en
```

## 8. Preflight (`livedemo doctor`)

Runs **before every `run`** unless `--skip-doctor`. Each check is a small function returning `(name, ok, hint)` — wholly DRY, easy to add new ones.

| Check | Always | Mic | Fireflies | File |
|---|:-:|:-:|:-:|:-:|
| `chrome_profile` exists, populated | ✓ | ✓ | ✓ | |
| `playwright_chromium` installed | ✓ | ✓ | ✓ | |
| `canvas_ws_port` free | ✓ | ✓ | ✓ | |
| `orchestration_ws` reachable | | ✓ | | |
| `transcription_service` healthy | | ✓ | | |
| `ollama` up & model present | ✓ | ✓ | ✓ | ✓ |
| `postgres` reachable | ✓ | ✓ | ✓ | |
| `alembic_head` matches | ✓ | ✓ | ✓ | |
| `fireflies_api` reachable + key valid | | | ✓ | |
| `mic_device` enumerable | | ✓ | | |
| `replay_jsonl` exists & parseable | | | | ✓ |

Output:

```
$ uv run livedemo doctor --source=mic
livedemo doctor — source=mic, sink=canvas
─────────────────────────────────────────
✓ chrome_profile               (~156 MB at /Users/.../chrome-profile)
✓ playwright_chromium          (chromium 120.x installed)
✓ canvas_ws_port               (port 7081 free)
✓ orchestration_ws             (ws://localhost:8000 — handshake ok in 12ms)
✓ transcription_service        (http://localhost:5001/health — 200 in 8ms)
✗ ollama                       (model 'qwen2.5:14b' not pulled)
                                hint: ollama pull qwen2.5:14b
✓ postgres                     (5432 — connected as livetranslate)
✓ alembic_head                 (009_meeting_retry_cols)
✓ mic_device                   ("MacBook Pro Microphone" enumerated)
─────────────────────────────────────────
1 check failed. Fix the issue above and re-run.
```

Exit code = number of failed checks (capped at 1).

## 9. Bot harness (DRY core)

`bot_harness.py` is the **only** place Playwright + canvas + getUserMedia override lives.

```python
class BotHarness:
    def __init__(self, config: LiveDemoConfig, recorder: WSRecorder | None = None):
        ...

    async def __aenter__(self) -> "BotHarness":
        self._proc = await self._spawn_node_runner()
        self._ws = await self._await_handshake()
        await self._wait_in_call()  # uses People-panel signal, not Leave button
        return self

    async def push_frame(self, rgb: np.ndarray, *, ts: float | None = None) -> None:
        png = _encode_png(rgb)
        await self._ws.send_bytes(_pack_frame(png, ts))
        if self.recorder:
            self.recorder.record("frame", {"size": len(png), "ts": ts})

    async def push_caption(self, c: CaptionEvent) -> None:
        # Convenience: caller doesn't need a WebcamManager — harness owns one.
        self._webcam.add_caption(...)
        self._webcam._generate_frame()
        await self.push_frame(self._webcam.current_frame)

    async def __aexit__(self, *exc) -> None:
        await self._leave_meeting()
        self._proc.terminate()
```

The TS subprocess `bot_runner/src/runner.ts` is intentionally thin:
- Loads init script (canvas appended to `document.body` with `display:none`, getUserMedia override)
- Joins meeting (handles "Switch here", modal-dismiss)
- Connects to Python over WS on `canvas_ws_port`
- On `frame` message: `Image.onload → ctx.drawImage(img, 0, 0)`
- On `leave` message: clicks Leave button

Production callers (`bot_integration.py`) will eventually move to this same harness, deleting their bespoke Playwright code. That's a Phase-6 follow-up, not blocking.

## 10. Sources

All sources implement `SubtitleSource`:

```python
class CaptionEvent(BaseModel):
    speaker_name: str | None
    speaker_id: str | None       # diarization id, may be ignored by harness
    src_lang: str
    tgt_lang: str
    original: str
    translation: str
    ts: float                    # event-time seconds, monotonic
    is_final: bool = True
    is_draft: bool = False
    pair_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class SubtitleSource(ABC):
    @abstractmethod
    async def stream(self) -> AsyncIterator[CaptionEvent]: ...
```

### `mic` source — true E2E

1. `sounddevice.InputStream(samplerate=16000, channels=1, blocksize=320)` — 20ms frames
2. Connects to `orchestration_ws_url` (`ws://localhost:3000/api/audio/stream`)
3. Sends `start_session` message + `config(target_language)`
4. Pumps PCM binary frames
5. Receives back `segment`, `translation_chunk`, `translation` events
6. Yields `CaptionEvent` on every `translation` (`is_draft=False`)

This is the **canonical end-to-end test**: if mic mode renders subtitles correctly, the production WebSocket protocol, draft/final routing, context store, and LLM client all work.

### `meet_audio` source (added Phase 9.2) — bot-page audio E2E

Composes the running `BotHarness` with the same `MicSource` WS protocol. Architecture:

```
Meet `<audio>` element
       │  (init_script `__livedemoAudio.startCapture(stream)` — Web Audio tap)
       ▼
bot_runner page (resamples 48k→16k, packs int16 PCM, queues 20ms chunks)
       │  (binary WS frames over canvas WS, ~20ms intervals via runner audioPump)
       ▼
BotHarness._audio_queue   (drops oldest if downstream slow)
       │  (audio_chunks() async iter)
       ▼
MicSource(audio_provider=harness.audio_chunks)
       │  (same start_session + binary frames protocol as the mic source)
       ▼
orchestration → transcription → translation → CaptionEvent
```

Operationally identical to mic mode from orchestration's perspective; the bot is just a different audio source. Source returns the same `CaptionEvent` shape.

### `fireflies` source — replay

1. GraphQL `transcript(id: $id) { sentences { text speaker_name start_time } }` (works on free tier per orchestration `CLAUDE.md` line 67-69)
2. Compute deltas between sentences; sleep `delta / replay_speed` seconds
3. Each sentence → `await TranslationService.translate_text(text, src=auto, tgt=tgt_lang, is_final=True, ...)`
4. Yield `CaptionEvent`

Fireflies path skips transcription (Fireflies *is* the transcript) but still hits LLM + context store.

### `file` source — recorded replay

Reads a `messages.jsonl` produced by a previous run's `WSRecorder` and re-emits the same events at the same timing. Used for:
- Deterministic CI smoke tests (`livedemo smoke`)
- Reproducing a bug from a recorded run
- Theme/layout iteration without spinning up the stack

## 11. Sinks

```python
class CaptionSink(ABC):
    async def __aenter__(self): ...
    async def consume(self, c: CaptionEvent) -> None: ...
    async def __aexit__(self, *exc): ...
```

### `canvas` sink
Wraps `BotHarness`. On each `CaptionEvent`, calls `harness.push_caption(c)` which renders via `VirtualWebcamManager` → encodes PNG → pushes over WS → JS draws to canvas.

### `png` sink
Writes per-event PNG to `<runs_dir>/frames/NNNN.png`. No bot. No Playwright. Used for fast layout iteration and the CI smoke test.

### `pyvirtualcam` sink (added Phase 9.1)
Production-spec OS-level virtual camera. Wraps `pyvirtualcam.Camera` (DI-friendly so tests pass a fake camera). Macs need OBS Virtual Camera; Linux needs v4l2loopback. See ADR 0003 for the canvas-vs-pyvirtualcam decision.

### Live `MeetingSessionConfig` updates (added Phase 9.7)
All three sinks accept an optional `meeting_config: MeetingSessionConfig`. They subscribe to its updates and snapshot the config per frame via `apply_meeting_config_snapshot()`, so `/mode split` or `/font 28` from chat propagates to the next-rendered frame without a restart. Subscribers are unsubscribed cleanly on `__aexit__`.

## 12. WS recorder & replay

`recorder.py`:

```python
class WSRecorder:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "messages.jsonl"
        self._fh = self.path.open("a", buffering=1)

    def record(self, kind: str, payload: dict) -> None:
        self._fh.write(json.dumps({
            "ts": time.monotonic(),
            "wall": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
        }) + "\n")
```

Wired into:
- Mic source: records every WS frame sent to orchestration + every event received
- Fireflies source: records every TranslationService input/output
- Bot harness: records every frame push (size + ts; not full PNG bytes by default — `--record-frames` flag to include them)

`runs/livedemo/<timestamp>/`:
```
config.snapshot.yaml      # Full resolved config (env + flags + YAML)
preflight.json            # All check results
messages.jsonl            # WS recorder output
frames/0001.png …         # Only if --save-frames
bot.stdout.log
bot.stderr.log
summary.md                # Counts + duration + final status
```

Replay:
```bash
uv run livedemo run --source=file --replay-jsonl=runs/livedemo/2026-04-25T10:00:00/messages.jsonl --sink=png
```

This makes every demo run reproducible. A bug reported as "the bot dropped captions at 02:14" can be replayed offline.

## 13. CLI surface

```bash
# Health
uv run livedemo doctor [--source=mic|fireflies|file]

# Live mic demo (most common)
uv run livedemo run --config configs/livedemo/mic-demo.yaml

# Fireflies replay
uv run livedemo run --config configs/livedemo/fireflies-replay.yaml

# Layout iteration (no bot, no network)
uv run livedemo run --source=file --replay-jsonl=fixtures/dialog.jsonl --sink=png

# CI smoke test — fully offline, deterministic
uv run livedemo smoke

# Replay a previous run
uv run livedemo replay runs/livedemo/2026-04-25T10:00:00
```

`run` always invokes `doctor` first; `--skip-doctor` for power users.

## 14. Behavioural contracts

These are the system invariants the implementation MUST uphold. They become assertions in tests.

### B1 — Preflight blocks bad runs
`run` MUST NOT join a Meet if any required preflight check fails. The bot only spawns after all required checks return `ok=True`. Verifies via: `test_livedemo_preflight.py` — patch ollama check to fail, assert harness never spawned.

### B2 — Mic source uses production WebSocket
Mic source MUST connect to `orchestration_ws_url`. It MUST NOT call `TranslationService` in-process. Verifies via: integration test asserting recorder JSONL contains `kind=ws_send` and `kind=ws_recv` entries.

### B3 — Paired captions render together
Every `CaptionEvent` with both `original` and `translation` MUST produce a frame containing both, drawn as paired blocks (original secondary, translation primary). Verifies via: `test_virtual_webcam_subtitles.py::test_paired_block_layout`.

### B4 — Diarization tags hidden by default
`bot_show_diarization_ids=False` (default) MUST NOT render `(SPEAKER_XX)` in speaker labels even if `speaker_id` is set. Verifies via: same test, asserts label string lacks regex `\(SPEAKER_\d+\)`.

### B5 — Bot in-call signal is robust
Harness MUST NOT report "in call" while still in lobby. Uses People-panel button OR self-tile element OR absence of "Still trying to get in..." — any one is sufficient, all-of nothing required. Verifies via: contract test on the runner script with mocked Meet DOM.

### B6 — Canvas captureStream stays live
Canvas MUST be appended to `document.body` (hidden) before `captureStream()` is called. Verifies via: harness self-test in init script that asserts `canvas.parentElement === document.body`.

### B7 — Every run is recorded
`record_messages=True` (default) MUST produce a non-empty `messages.jsonl` with at least one `kind=caption` entry per yielded `CaptionEvent`. Verifies via: `test_livedemo_smoke.py` asserts file exists, line count == event count.

### B8 — File source is byte-exact replay
`file` source replaying a recorder JSONL MUST emit the same CaptionEvent sequence (same `pair_id`, same text) at the same relative timing (within ±50ms). Verifies via: round-trip test — record → replay → record → diff JSONL.

### B9 — Fireflies replay routes through TranslationService
Fireflies source MUST emit translations from `TranslationService.translate_text()` — not a stub. Verifies via: integration test pinning `TranslationService` and asserting it received `translate_text` calls equal to the sentence count.

### B10 — Configuration is resolved deterministically
`config.snapshot.yaml` in the run dir MUST contain the **fully resolved** config (defaults + YAML + env + flags). Verifies via: snapshot test on a known fixture.

## 15. Implementation phases

| Phase | Deliverable | Rough effort | Blocking |
|---|---|---|---|
| **0. Foundation** | `config.py`, `preflight.py` skeleton, `runs/` directory layout | 1.5h | — |
| **1. Bot harness** | `bot_harness.py` + `bot_runner/runner.ts`, contract tests | 3h | Phase 0 |
| **2. Sinks** | `png.py`, `canvas_ws.py` | 1.5h | Phase 1 |
| **3. File source + smoke** | `file.py`, `test_livedemo_smoke.py`, fixture JSONL | 1.5h | Phase 2 |
| **4. Fireflies source** | `fireflies.py` + integration test | 2h | Phase 2 |
| **5. Mic source** | `mic.py` + integration test against running stack | 2.5h | Phase 2 |
| **6. CLI + recorder** | `cli.py`, `recorder.py`, `pipeline.py` | 2h | Phases 3–5 |
| **7. Migrations** | `add_translation()` callers → `add_caption()`, delete dead scripts | 1.5h | Phase 6 |
| **8. Merge bot-auth** | Merge `feature/meeting-bot-auth-commands` to main, cherry-pick screenshot endpoint | 1h | Phase 7 |

**Total ≈ 14.5h.** Phase 3 yields the first runnable artifact (PNG sink + file source) without any external deps. Phase 5 is the true E2E. Phases 7–8 are cleanup.

### Parallelisation opportunities
- Phase 4 (Fireflies) and Phase 5 (Mic) are independent after Phase 2.
- Phase 7 migrations are file-by-file independent of each other.

## 16. Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Meet DOM changes break in-call signal | Medium | Demo fails | B5 uses three signals; any one passes. Add screenshot dump on failure. |
| Canvas captureStream stalls | Low (after fix) | No subtitles visible | B6 asserts canvas in DOM; framerate watchdog logs WARN if no frames sent in 1s. |
| `getUserMedia` override race vs. Meet's bundled JS | Low | Bot's webcam = real cam, not canvas | Init script runs before any page JS via `addInitScript`; verified in B5 via console log marker. |
| Orchestration service drops messages under load | Low | Mic source loses captions | Recorder captures both ws_send and ws_recv; failed translations visible in JSONL. |
| Fireflies rate limit (50/day on free tier) | Medium | Demo hits 429 | Use single-meeting `transcript(id)` query; cache responses in `runs/<>/fireflies.cache.json`. |
| Chrome profile lock contention if user has Chrome open | Medium | Bot fails to launch | Preflight check: try-acquire profile lock; hint user to close Chrome. |
| Demo joins meeting but waits in lobby forever | Medium | User confusion | Harness emits "WAITING_FOR_ADMIT" log line; CLI prints prominent "Admit the bot from the meeting now" prompt with timeout. |

## 17. Open questions — resolutions

1. **`add_translation()` deletion or shim?** RESOLVED — kept as a thin shim over `add_caption()` (`virtual_webcam.py:284-301`). Single edit gives all 9 callers paired-block rendering + B4 diarization-tag suppression for free, with no call-site churn. Future: delete during a separate refactor pass once callers are clearly migrated.
2. **Long-term bot harness home?** Still Phase 8+ work — once `feature/meeting-bot-auth-commands` merges, fold `bot_integration.py`'s Playwright code into `BotHarness` so production and demo share one bot. Out of this plan's scope.
3. **`livedemo` as console-script?** RESOLVED — landed in Phase 6, entry `livedemo = "src.livedemo.cli:main"` in `pyproject.toml`. `uv run livedemo --help` works.
4. **Frame compression**: still PNG. At 10fps × 200-800KB = 2-8 MB/s on localhost is fine. WebP swap deferred to when remote bot becomes a goal.
5. **canvas-WS vs pyvirtualcam**: RESOLVED via ADR 0003. Both ship behind `cfg.sink ∈ {canvas, png, pyvirtualcam}`. Demo defaults to canvas (zero install); production deploys with pyvirtualcam.

## 17b. Spec conformance audit (added Phase 9)

Cross-referenced against `docs/superpowers/specs/2026-04-10-meeting-subtitle-system-design.md` and `docs/superpowers/specs/2026-04-12-meeting-bot-auth-commands-design.md`. Capabilities map across three layers:

| Capability | Owner | Status in livedemo |
|---|---|---|
| Persistent profile auth | `meeting-bot-service` (canonical) + `livedemo` (consumes via `chrome_profile_dir`) | ✅ |
| Stealth args | `livedemo/bot_runner/runner.ts:78` | ✅ |
| In-call signal (B5: 3-way OR) | `runner.ts` waitForInCall | ✅ |
| Mic-off + cam-on pre-join | `runner.ts` muteMic + ensureCameraOn (Phase 9.5) | ✅ |
| Canvas getUserMedia override | `init_script.ts` | ✅ |
| Bot-page audio capture | `init_script.ts:__livedemoAudio` (Phase 9.2) | ✅ |
| pyvirtualcam virtual-camera | `livedemo/sinks/pyvirtualcam.py` (Phase 9.1) | ✅ |
| Canonical Meet selectors | `bot_runner/src/lib/selectors.ts` (Phase 9.3) | ✅ — duplicate-then-consolidate post Phase-8 merge |
| MeetingSessionConfig live updates | All three sinks (Phase 9.7) | ✅ |
| Join chat message | `runner.ts` posts canonical string after waitForInCall | ✅ |
| `/stop` → leave | Harness `send_leave_request()` → runner clicks Leave | ✅ — orchestration-side `/stop` parsing lands with Phase 8 merge |
| `ChatPoller` (read incoming chat commands) | `meeting-bot-service` (Phase 8 merge) | 🔁 deferred to canonical bot |
| `CommandDispatcher` (parse `/lang` etc.) | `meeting-bot-service` (Phase 8 merge) | 🔁 deferred to canonical bot |
| `BotAudioCaptionSource` / `FirefliesCaptionSource` adapters | `services/pipeline/adapters/source_adapter.py` | 🔁 livedemo's `MicSource`/`FirefliesSource`/`MeetAudioSource` are demo-shaped peers — production uses the canonical adapters |
| `MeetCaptionsAdapter` (scrape Meet CC DOM) | future Phase-3 of subtitle system | ❌ not in livedemo |
| `SourceOrchestrator` live-switching | future | ❌ livedemo is single-source per run |
| Docker bot container, HTTP API endpoints | `meeting-bot-service` | 🔁 production-bot territory, livedemo runs Node directly |
| structlog observability | `livedemo/pipeline.py`, `livedemo/sinks/*.py` (Phase 9.4) | ✅ |

## 18. Out of scope (deferred)

- Multi-bot or multi-meeting demo orchestration.
- Cloud-hosted demo runs (currently localhost-only).
- Demo dashboard / Grafana panel showing live metrics.
- Auto-admit (the human in the meeting must admit the bot — that's by design for Meet's anti-abuse posture).
- Replacing JS-side canvas drawing with full GPU compositing.

## 19. Appendix: relevant prior context

- `modules/orchestration-service/CLAUDE.md` — service overview, Alembic rules, Translation module map, Fireflies API constraints.
- `modules/orchestration-service/src/bot/virtual_webcam.py` — `VirtualWebcamManager`, `add_caption()`, paired-block rendering.
- `modules/orchestration-service/src/translation/service.py` — `TranslationService`.
- `modules/orchestration-service/src/routers/audio/websocket_audio.py` — `/ws/audio` handler that mic source connects to.
- `.worktrees/meeting-bot-auth/modules/meeting-bot-service/test_subtitle_canvas.ts` — current canvas approach; lessons folded into `bot_runner/runner.ts`.
- `.worktrees/meeting-bot-auth/modules/meeting-bot-service/src/lib/chromium.ts` — `createBrowserContext`; harness will reuse.
