# ADR 0003: Canvas-WS vs pyvirtualcam — dual-sink design for the bot

**Status:** Accepted
**Date:** 2026-04-25
**Related:** PLAN_7_LIVEDEMO_E2E.md, docs/superpowers/specs/2026-04-10-meeting-subtitle-system-design.md

## Context

The Google Meet bot needs to expose a virtual webcam that displays paired
(original + translation) captions to other meeting participants. The canonical
design spec (`2026-04-10-meeting-subtitle-system-design.md`) selects
**pyvirtualcam** as the production renderer, requiring:

- macOS: OBS Virtual Camera installed + manually activated by the user
- Linux: v4l2loopback kernel module loaded (`/dev/videoN` passthrough)

This is the right production choice — it produces a *real* OS-level virtual
camera device that any conferencing app sees alongside physical webcams.

**However** during PLAN_7 (livedemo E2E pipeline) we hit two friction points:

1. The `livedemo` smoke run is intended to work on a fresh laptop with **zero**
   pre-install steps beyond `uv sync`. Requiring users to install OBS Virtual
   Camera before running a demo defeats that goal.
2. Several runs in the prior session showed pyvirtualcam fail silently (the
   manual one-time OBS activation step is easy to forget after a Mac reboot),
   making demo failures look like bot bugs.

We considered three options:

- **A.** Stick with pyvirtualcam everywhere; add a clear preflight error
  message when OBS isn't running.
- **B.** Replace pyvirtualcam with a canvas-backed `getUserMedia()` override
  inside the bot's Chromium page. No system-level camera needed.
- **C.** Both — a `CaptionSink` ABC with `pyvirtualcam` and `canvas_ws`
  implementations selectable by config (`cfg.sink`).

## Decision

**Adopt option C.** Both sinks coexist behind `livedemo.sinks.base.CaptionSink`.
The CLI selects via `cfg.sink ∈ {"canvas", "png", "pyvirtualcam"}`. Both
sinks share the same `VirtualWebcamManager` rendering pipeline, so identical
pixels regardless of output path.

| Sink | Use case | Pre-install | Output path |
|---|---|---|---|
| `pyvirtualcam` | Production deployments, spec-conformant | OBS Virtual Camera (macOS) / v4l2loopback (Linux) | OS-level virtual webcam → any conferencing app |
| `canvas` | Demos, fresh-laptop runs, CI | None | Canvas in Chromium → `getUserMedia()` override → Meet's `<video>` element |
| `png` | Layout iteration, deterministic CI smoke | None | Numbered PNGs on disk |

## Trade-offs

### canvas-ws sink

**Pros:**
- Zero install dependencies — works on any laptop with Chromium
- Same pixels as pyvirtualcam (shared `VirtualWebcamManager`)
- Behavioral E2E verified: real Chromium → real `captureStream` → real
  consumer `<video>` element produces frames (see
  `bot_runner/tests/frame_pipeline.behavior.test.ts`)
- Explicit recorder of the WS protocol — every frame push captured to
  `messages.jsonl` for replay (B7/B8 in PLAN_7)

**Cons:**
- Only works inside the bot's own Chromium page. Cannot be used as a webcam
  by other applications.
- Tied to Meet's `getUserMedia` call surface. If Google ships a sandbox
  change that breaks our override, both sinks need separate fixes.

### pyvirtualcam sink

**Pros:**
- True OS-level camera — works for any current and future conferencing app
  (Zoom, Teams, Webex, etc.) without per-app code
- Spec-conformant per the canonical subtitle system design
- Production-tested via existing `PILVirtualCamRenderer` in
  `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py`

**Cons:**
- Requires OS-level setup (OBS Virtual Camera or v4l2loopback)
- macOS users must manually activate OBS Virtual Camera after each reboot
  if not configured to launch on login — an easy step to forget
- Not available in CI / fresh-laptop demo flow

## Implementation

Both sinks implement the single-method `CaptionSink` ABC:

```python
class CaptionSink(ABC):
    async def __aenter__(self) -> "CaptionSink": ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    @abstractmethod
    async def consume(self, caption: CaptionEvent) -> None: ...
```

Both:

1. Use `VirtualWebcamManager.add_caption()` to render the paired-block frame
2. Call `VirtualWebcamManager._generate_frame()` to produce the RGB array
3. Hand the array off to their respective output (PIL→PNG, base64→WS,
   raw→pyvirtualcam.send())

This means a layout/theme/font fix lands once in `VirtualWebcamManager` and
flows to every sink.

The `livedemo.sinks.make_sink(kind, **kwargs)` factory selects by string —
the CLI never imports a specific sink. Optional sinks (`pyvirtualcam`)
fail-import lazily so the package works without those deps installed.

## Consequences

- **Two sinks to maintain.** Mitigated by the shared `VirtualWebcamManager`
  pipeline — divergence is only possible in transport, not rendering.
- **Two sets of preflight checks.** `pyvirtualcam` needs an "OBS available"
  probe; `canvas` needs the existing `playwright_chromium` + `chrome_profile`
  checks.
- **Test coverage doubles for the rendering path.** Acceptable because the
  shared `VirtualWebcamManager` already has dedicated tests
  (`tests/integration/test_virtual_webcam_subtitles.py` +
  `tests/livedemo_tests/test_sinks.py`).
- **Production deployments choose one.** Demos use `canvas`; live
  deployments use `pyvirtualcam`. The choice is a single config line, not a
  fork in the codebase.

## References

- `modules/orchestration-service/src/livedemo/sinks/canvas_ws.py`
- `modules/orchestration-service/src/livedemo/sinks/pyvirtualcam.py`
- `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py`
  (canonical pyvirtualcam wrapper, used by the production bot)
- `modules/orchestration-service/src/livedemo/bot_runner/src/init_script.ts`
  (canvas + getUserMedia override)
- PLAN_7_LIVEDEMO_E2E.md §17 "Open questions" — supersedes the unresolved
  pyvirtualcam-vs-canvas question with this ADR
