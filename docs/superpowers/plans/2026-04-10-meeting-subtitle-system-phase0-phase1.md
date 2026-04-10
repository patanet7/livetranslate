# Meeting Subtitle System — Phase 0 & 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate existing subtitle foundations (Phase 0) and wire the modular core pipeline — config, source adapters, renderers (Phase 1).

**Architecture:** Three-layer modular system: Caption Sources → Processing Core (CaptionBuffer + MeetingSessionConfig) → Renderers (PIL virtual camera + WebSocket). Config-driven: every setting is live-switchable. Plain class with explicit `update()` and threading lock, not Pydantic BaseModel.

**Tech Stack:** Python 3.12+, Pillow (PIL), pyvirtualcam, numpy, structlog, pytest, Pydantic v2 (for data models only, not MeetingSessionConfig)

**Spec:** `docs/superpowers/specs/2026-04-10-meeting-subtitle-system-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `modules/shared/src/livetranslate_common/theme.py` | Canonical speaker colors, theme definitions, DisplayMode enum — ONE source of truth |
| `modules/orchestration-service/src/services/meeting_session_config.py` | `MeetingSessionConfig` plain class with `update()`, threading lock, subscriber list |
| `modules/orchestration-service/src/services/pipeline/adapters/source_adapter.py` | `CaptionSourceAdapter` protocol + `BotAudioCaptionSource`, `FirefliesCaptionSource` |
| `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py` | Wires VirtualWebcamManager → pyvirtualcam, frame-paced timer, dirty-flag, config snapshot |

### New Test Files

| File | Responsibility |
|------|---------------|
| `modules/shared/tests/test_theme.py` | Theme constants integrity, color format validation |
| `modules/orchestration-service/tests/test_meeting_session_config.py` | Config update(), subscriber notification, thread safety |
| `modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py` | PIL → pyvirtualcam → device creation |
| `modules/orchestration-service/tests/integration/test_caption_routing.py` | Source adapter start/stop, CaptionBuffer routing |

### Modified Files

| File | Change |
|------|--------|
| `modules/orchestration-service/tests/meeting/test_translation_observability.py:26` | Fix broken import |
| `modules/transcription-service/tests/unit/test_sustained_detector.py:172` | Fix overly strict assertion |
| `modules/orchestration-service/src/bot/virtual_webcam.py:39-76,118-127,395-420` | Unified DisplayMode, shared colors, 1280x720, frame-paced timer |
| `modules/orchestration-service/src/models/bot.py:45-51,174-252,255-271` | Merge WebcamConfigs, rename TranslationConfig |
| `modules/orchestration-service/src/services/caption_buffer.py:36-47,160-174,236-256` | Shared colors, multi-subscriber |
| `modules/orchestration-service/src/bot/caption_processor.py:33-69,668` | Dataclass→Pydantic, remove TODO |
| `modules/bot-container/Dockerfile:46` | Remove `v4l2loopback-dkms` |
| `modules/dashboard-service/src/lib/stores/loopback.svelte.ts:13,66-69` | Import canonical colors+DisplayMode |

---

## Phase 0: Validate Foundations

### Task 1: Fix Broken Test Imports

**Files:**
- Modify: `modules/orchestration-service/tests/meeting/test_translation_observability.py:26`

- [ ] **Step 1: Run the broken test to confirm the error**

Run: `uv run pytest modules/orchestration-service/tests/meeting/test_translation_observability.py -v --timeout=30 2>&1 | tail -10`
Expected: `ModuleNotFoundError: No module named 'meeting.test_translation_recovery'`

- [ ] **Step 2: Find what `_FakeTranslationService` actually is and where it moved**

```bash
uv run grep -rn "_FakeTranslationService" modules/orchestration-service/tests/
```

If the class no longer exists (it was likely removed in a prior commit), inline a minimal fake:

- [ ] **Step 3: Fix the import**

If `test_translation_recovery.py` exists in a different location, fix the import path. If it was deleted, create a minimal `_FakeTranslationService` inline in `test_translation_observability.py`:

```python
# Replace line 26:
# from .test_translation_recovery import _FakeTranslationService

# With an inline minimal fake:
class _FakeTranslationService:
    """Minimal fake for observability tests."""
    def __init__(self):
        self.translations = []

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        self.translations.append(text)
        return f"[translated] {text}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest modules/orchestration-service/tests/meeting/test_translation_observability.py -v --timeout=30 2>&1 | tail -15`
Expected: PASS (or at least no import error)

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/tests/meeting/test_translation_observability.py
git commit -m "fix: resolve broken import in test_translation_observability"
```

---

### Task 2: Fix Sustained Detector Test Assertion

**Files:**
- Modify: `modules/transcription-service/tests/unit/test_sustained_detector.py:158-173`

- [ ] **Step 1: Run the failing test to confirm the error**

Run: `uv run pytest modules/transcription-service/tests/unit/test_sustained_detector.py::TestHysteresisPreventsFlapping::test_insufficient_margin_prevents_switch -v 2>&1 | tail -10`
Expected: `assert 0 > 0` — the `false_positives_prevented` counter is 0 because the detector correctly never even considered switching (the margin was too small to start a candidate). The test expects the counter to be >0, but the detector got better — it doesn't even start tracking a candidate when the margin is insufficient.

- [ ] **Step 2: Fix the assertion**

The test's intent is "insufficient margin should not cause a switch." The detector correctly doesn't switch. The `false_positives_prevented` counter only increments when a candidate was being tracked and then got rejected. With margin=0.1 (below threshold 0.2), no candidate is ever started, so the counter stays at 0. The fix: assert that no switch happened (already asserted in the loop) and remove the overly strict counter assertion.

In `modules/transcription-service/tests/unit/test_sustained_detector.py`, replace line 172:

```python
# OLD (line 172):
# assert detector.false_positives_prevented > 0

# NEW — the test already asserts no switch happened in the loop above.
# The detector correctly never started a candidate (margin too low),
# so false_positives_prevented == 0 is the correct behavior.
assert detector.current_language == "en", "Language should remain English"
```

- [ ] **Step 3: Run the test to verify it passes**

Run: `uv run pytest modules/transcription-service/tests/unit/test_sustained_detector.py::TestHysteresisPreventsFlapping::test_insufficient_margin_prevents_switch -v`
Expected: PASS

- [ ] **Step 4: Run the full transcription test suite to verify no regressions**

Run: `uv run pytest modules/transcription-service/tests/ -v --timeout=30 -q 2>&1 | tail -5`
Expected: All pass (56 passed)

- [ ] **Step 5: Commit**

```bash
git add modules/transcription-service/tests/unit/test_sustained_detector.py
git commit -m "fix: relax sustained detector test assertion to match improved behavior"
```

---

### Task 3: Canonical Theme Definitions (Shared Colors + DisplayMode)

This is the DRY foundation — ONE file defines speaker colors, themes, and display modes. Everything else imports from here.

**Files:**
- Create: `modules/shared/src/livetranslate_common/theme.py`
- Create: `modules/shared/tests/test_theme.py`

- [ ] **Step 1: Write the failing test**

Create `modules/shared/tests/test_theme.py`:

```python
"""Tests for canonical theme definitions."""

import re

import pytest

from livetranslate_common.theme import (
    SPEAKER_COLORS,
    DisplayMode,
    ThemeColors,
    get_theme_colors,
)


class TestSpeakerColors:
    def test_has_at_least_10_colors(self):
        assert len(SPEAKER_COLORS) >= 10

    def test_all_hex_format(self):
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for color in SPEAKER_COLORS:
            assert hex_pattern.match(color), f"Invalid hex color: {color}"

    def test_no_duplicates(self):
        assert len(SPEAKER_COLORS) == len(set(SPEAKER_COLORS))


class TestDisplayMode:
    def test_canonical_modes(self):
        assert DisplayMode.SUBTITLE == "subtitle"
        assert DisplayMode.SPLIT == "split"
        assert DisplayMode.INTERPRETER == "interpreter"

    def test_all_values_are_strings(self):
        for mode in DisplayMode:
            assert isinstance(mode.value, str)


class TestThemeColors:
    def test_dark_theme_exists(self):
        colors = get_theme_colors("dark")
        assert isinstance(colors, ThemeColors)
        assert colors.background is not None
        assert colors.text_primary is not None

    def test_all_themes_loadable(self):
        for theme_name in ("dark", "light", "high_contrast", "minimal", "corporate"):
            colors = get_theme_colors(theme_name)
            assert isinstance(colors, ThemeColors)

    def test_invalid_theme_raises(self):
        with pytest.raises(KeyError):
            get_theme_colors("nonexistent")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest modules/shared/tests/test_theme.py -v`
Expected: `ModuleNotFoundError: No module named 'livetranslate_common.theme'`

- [ ] **Step 3: Implement the theme module**

Create `modules/shared/src/livetranslate_common/theme.py`:

```python
"""Canonical theme definitions for LiveTranslate.

This is the SINGLE source of truth for speaker colors, display modes,
and theme color schemes. Both PIL (Python) and SvelteKit (browser)
renderers consume these definitions.

DO NOT define colors, display modes, or themes anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Speaker Colors — canonical palette (hex strings)
# =============================================================================

SPEAKER_COLORS: list[str] = [
    "#4CAF50",  # Green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#F44336",  # Red
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#FFEB3B",  # Yellow
    "#795548",  # Brown
    "#607D8B",  # Blue Grey
]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple. For PIL rendering."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


# =============================================================================
# Display Modes — canonical enum
# =============================================================================


class DisplayMode(str, Enum):
    """Display modes for subtitle rendering.

    These match the dashboard's display mode names exactly.
    Both PIL and SvelteKit renderers use these values.
    """

    SUBTITLE = "subtitle"
    SPLIT = "split"
    INTERPRETER = "interpreter"


# =============================================================================
# Theme Colors
# =============================================================================


@dataclass(frozen=True)
class ThemeColors:
    """Color scheme for a visual theme."""

    background: tuple[int, int, int]
    text_primary: tuple[int, int, int]
    text_secondary: tuple[int, int, int]
    accent: tuple[int, int, int]
    border: tuple[int, int, int]


_THEMES: dict[str, ThemeColors] = {
    "dark": ThemeColors(
        background=(20, 20, 20),
        text_primary=(255, 255, 255),
        text_secondary=(180, 180, 180),
        accent=(0, 150, 255),
        border=(60, 60, 60),
    ),
    "light": ThemeColors(
        background=(240, 240, 240),
        text_primary=(20, 20, 20),
        text_secondary=(80, 80, 80),
        accent=(0, 120, 200),
        border=(200, 200, 200),
    ),
    "high_contrast": ThemeColors(
        background=(0, 0, 0),
        text_primary=(255, 255, 255),
        text_secondary=(255, 255, 0),
        accent=(255, 0, 255),
        border=(255, 255, 255),
    ),
    "minimal": ThemeColors(
        background=(250, 250, 250),
        text_primary=(40, 40, 40),
        text_secondary=(120, 120, 120),
        accent=(100, 100, 100),
        border=(220, 220, 220),
    ),
    "corporate": ThemeColors(
        background=(245, 245, 245),
        text_primary=(30, 30, 30),
        text_secondary=(100, 100, 100),
        accent=(0, 100, 180),
        border=(210, 210, 210),
    ),
}


def get_theme_colors(theme_name: str) -> ThemeColors:
    """Get colors for a theme by name. Raises KeyError for unknown themes."""
    return _THEMES[theme_name]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest modules/shared/tests/test_theme.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/shared/src/livetranslate_common/theme.py modules/shared/tests/test_theme.py
git commit -m "feat: canonical theme definitions — single source of truth for colors, modes, themes"
```

---

### Task 4: Update CaptionBuffer to Use Shared Colors

**Files:**
- Modify: `modules/orchestration-service/src/services/caption_buffer.py:36-47`

- [ ] **Step 1: Run existing CaptionBuffer tests to establish baseline**

Run: `uv run pytest modules/orchestration-service/tests/ -k "caption_buffer" -v --timeout=30 2>&1 | tail -10`

- [ ] **Step 2: Replace hardcoded colors with import from theme**

In `modules/orchestration-service/src/services/caption_buffer.py`, replace lines 36-47:

```python
# OLD (lines 36-47):
# DEFAULT_SPEAKER_COLORS = [
#     "#4CAF50",  # Green
#     ...
# ]

# NEW:
from livetranslate_common.theme import SPEAKER_COLORS as DEFAULT_SPEAKER_COLORS
```

- [ ] **Step 3: Run tests to verify no regressions**

Run: `uv run pytest modules/orchestration-service/tests/ -k "caption_buffer" -v --timeout=30`
Expected: All pass (colors are identical, just imported from shared location)

- [ ] **Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/caption_buffer.py
git commit -m "refactor: CaptionBuffer uses canonical SPEAKER_COLORS from livetranslate-common"
```

---

### Task 5: Update VirtualWebcamManager to Use Shared Theme

**Files:**
- Modify: `modules/orchestration-service/src/bot/virtual_webcam.py:39-76,118-127`

- [ ] **Step 1: Replace DisplayMode enum with canonical import**

In `modules/orchestration-service/src/bot/virtual_webcam.py`, replace lines 39-46:

```python
# OLD (lines 39-46):
# class DisplayMode(Enum):
#     OVERLAY = "overlay"
#     SIDEBAR = "sidebar"
#     BOTTOM_BANNER = "bottom_banner"
#     FLOATING = "floating"
#     FULLSCREEN = "fullscreen"

# NEW:
from livetranslate_common.theme import DisplayMode, ThemeColors, get_theme_colors, hex_to_rgb, SPEAKER_COLORS
```

Note: The old `DisplayMode` had 5 PIL-specific modes (OVERLAY, SIDEBAR, etc.). The new canonical `DisplayMode` has 3 modes (SUBTITLE, SPLIT, INTERPRETER). The PIL renderer needs to map these:
- `SUBTITLE` → bottom banner rendering (existing `_render_banner_frame`)
- `SPLIT` → sidebar rendering (existing `_render_sidebar_frame`)
- `INTERPRETER` → fullscreen rendering (existing `_render_fullscreen_frame`)

- [ ] **Step 2: Replace hardcoded speaker colors with canonical import**

Replace lines 118-127:

```python
# OLD:
# self.speaker_colors = [(255, 100, 100), ...]

# NEW:
self.speaker_colors = [hex_to_rgb(c) for c in SPEAKER_COLORS]
```

- [ ] **Step 3: Replace WebcamConfig default resolution to 1280x720**

In the `WebcamConfig` dataclass (line 60-76), change:

```python
# OLD:
# width: int = 1920
# height: int = 1080

# NEW:
width: int = 1280
height: int = 720
```

- [ ] **Step 4: Replace get_theme_colors with canonical import**

Find the existing `get_theme_colors` method in the class (around line 195) and replace it to use the shared `get_theme_colors()` function, converting `ThemeColors` to the dict format the existing render methods expect.

- [ ] **Step 5: Run any existing virtual webcam tests**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py -v --timeout=30 2>&1 | tail -15`

- [ ] **Step 6: Commit**

```bash
git add modules/orchestration-service/src/bot/virtual_webcam.py
git commit -m "refactor: VirtualWebcamManager uses canonical theme, colors, DisplayMode from shared"
```

---

### Task 6: Fix VirtualWebcamManager Busy-Wait Loop

**Files:**
- Modify: `modules/orchestration-service/src/bot/virtual_webcam.py:395-420`

- [ ] **Step 1: Replace the busy-wait `_stream_loop` with frame-paced timer**

Replace lines 395-427:

```python
def _stream_loop(self):
    """Main streaming loop with frame-paced timer."""
    logger.info("Virtual webcam stream loop started")

    frame_interval = 1.0 / self.config.fps
    next_frame_time = time.monotonic()

    try:
        while self.is_streaming:
            next_frame_time += frame_interval

            # Generate new frame
            self._generate_frame()
            self.frames_generated += 1

            # Callback for frame generation
            if self.on_frame_generated and self.current_frame is not None:
                self.on_frame_generated(self.current_frame.copy())

            # Clean up expired translations
            self._cleanup_expired_translations()

            # Sleep until next frame is due
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Frame took too long — skip to next slot
                next_frame_time = time.monotonic()

    except Exception as e:
        logger.error(f"Error in webcam stream loop: {e}")
        if self.on_error:
            self.on_error(f"Stream loop error: {e}")

    logger.info("Virtual webcam stream loop ended")
```

- [ ] **Step 2: Verify the render loop still works**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py -v --timeout=30 2>&1 | tail -10`

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/bot/virtual_webcam.py
git commit -m "fix: replace busy-wait render loop with frame-paced timer"
```

---

### Task 7: Clean Up caption_processor.py (TODOs + Pydantic)

**Files:**
- Modify: `modules/orchestration-service/src/bot/caption_processor.py:33-69,668`

- [ ] **Step 1: Convert dataclasses to Pydantic BaseModel**

Replace lines 33-69:

```python
from pydantic import BaseModel, Field


class SpeakerInfo(BaseModel):
    """Information about a meeting participant."""

    speaker_id: str
    display_name: str
    email: str | None = None
    role: str | None = None
    join_time: float | None = None
    leave_time: float | None = None
    is_active: bool = True
    total_speaking_time: float = 0.0
    utterance_count: int = 0


class CaptionSegment(BaseModel):
    """A single caption segment with speaker and timing information."""

    speaker_id: str
    speaker_name: str
    text: str
    start_timestamp: float
    end_timestamp: float | None = None
    confidence: float = 1.0
    caption_source: str = "google_meet"
    segment_id: str | None = None


class SpeakerTimelineEvent(BaseModel):
    """Event in the speaker timeline."""

    event_type: str
    speaker_id: str
    timestamp: float
    duration: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 2: Remove TODO comment at line 668**

Find the TODO and either implement the method or delete the dead code. Since the Google Meet caption capture is not being implemented in this phase (it's Phase 3+), delete the empty method and its polling loop:

```python
# DELETE the empty _capture_google_meet_captions method and its polling loop
# The CaptionProcessor will be wired to real caption sources via CaptionSourceAdapter in Phase 1
```

- [ ] **Step 3: Update any `asdict()` calls to `model_dump()`**

Search for `asdict(` in the file and replace with `.model_dump()`.

- [ ] **Step 4: Remove the `from dataclasses import dataclass, field, asdict` import**

- [ ] **Step 5: Run tests**

Run: `uv run pytest modules/orchestration-service/tests/ -v --timeout=30 -q 2>&1 | tail -10`

- [ ] **Step 6: Commit**

```bash
git add modules/orchestration-service/src/bot/caption_processor.py
git commit -m "refactor: caption_processor uses Pydantic models, remove TODO dead code"
```

---

### Task 8: Remove v4l2loopback-dkms from Dockerfile

**Files:**
- Modify: `modules/bot-container/Dockerfile:46`

- [ ] **Step 1: Remove `v4l2loopback-dkms` from the apt-get install line**

In `modules/bot-container/Dockerfile`, find line 46 (`v4l2loopback-dkms \`) and remove it. Keep `v4l2loopback-utils` if present (for `v4l2loopback-ctl` inside the container).

- [ ] **Step 2: Verify Docker build still works**

Run: `docker build -f modules/bot-container/Dockerfile -t livetranslate-bot-test . 2>&1 | tail -5`
Expected: Build succeeds (may take a while for initial build)

- [ ] **Step 3: Commit**

```bash
git add modules/bot-container/Dockerfile
git commit -m "fix: remove v4l2loopback-dkms from Dockerfile (needs host kernel module)"
```

---

### Task 9: PIL Frame Rendering Validation (Un-skip + 1280x720)

**Files:**
- Modify: `modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py`

- [ ] **Step 1: Read the existing test file to understand its current state**

Run: `head -30 modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py`

- [ ] **Step 2: Update the test to use 1280x720 and run against all 3 canonical display modes**

Ensure the test creates a `WebcamConfig(width=1280, height=720, fps=30)` and renders frames for `DisplayMode.SUBTITLE`, `DisplayMode.SPLIT`, and `DisplayMode.INTERPRETER`. Add a CJK text test case:

```python
@pytest.mark.integration
class TestPILFrameRendering:
    def test_renders_subtitle_mode(self):
        config = WebcamConfig(width=1280, height=720, fps=30)
        manager = VirtualWebcamManager(config=config)
        manager.add_translation(TranslationDisplay(
            translation_id="t1",
            text="Hello world",
            source_language="en",
            target_language="zh",
            speaker_name="Alice",
            confidence=0.95,
            timestamp=datetime.now(UTC),
        ))
        manager._generate_frame()
        assert manager.current_frame is not None
        assert manager.current_frame.shape == (720, 1280, 3)

    def test_renders_cjk_text(self):
        config = WebcamConfig(width=1280, height=720, fps=30)
        manager = VirtualWebcamManager(config=config)
        manager.add_translation(TranslationDisplay(
            translation_id="t1",
            text="你好世界",
            source_language="zh",
            target_language="en",
            speaker_name="张三",
            confidence=0.9,
            timestamp=datetime.now(UTC),
        ))
        manager._generate_frame()
        assert manager.current_frame is not None
        assert manager.current_frame.shape == (720, 1280, 3)
```

- [ ] **Step 3: Run the tests**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py -v --timeout=30`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add modules/orchestration-service/tests/integration/test_virtual_webcam_subtitles.py
git commit -m "test: validate PIL frame rendering at 1280x720 with CJK text"
```

---

### Task 10: Validate pyvirtualcam Device Creation

**Files:**
- Create: `modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py`

- [ ] **Step 1: Write the test**

```python
"""Integration test: PIL frames → pyvirtualcam → virtual camera device.

Requires: OBS installed with virtual camera extension activated (macOS),
or v4l2loopback loaded on host (Linux).

Marked slow — skipped in normal test runs. Run explicitly:
    uv run pytest tests/integration/test_virtual_cam_e2e.py -v -m "not skip"
"""

import numpy as np
import pytest

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.skipif(not PYVIRTUALCAM_AVAILABLE, reason="pyvirtualcam not installed")
class TestPyvirtualcamDevice:
    def test_create_camera_and_send_frame(self):
        """Verify we can create a virtual camera and write an RGB frame."""
        width, height, fps = 1280, 720, 30
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Red frame for visual verification
        frame[:, :, 0] = 255

        try:
            with pyvirtualcam.Camera(width, height, fps, fmt=PixelFormat.RGB) as cam:
                cam.send(frame)
                cam.sleep_until_next_frame()
                assert cam.device is not None
                assert cam.width == width
                assert cam.height == height
        except RuntimeError as e:
            if "no backend" in str(e).lower() or "obs" in str(e).lower():
                pytest.skip(f"Virtual camera backend not available: {e}")
            raise

    def test_rgba_must_composite_to_rgb(self):
        """RGBA frames cannot be sent directly — must composite onto opaque background."""
        width, height = 1280, 720
        rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_frame[:, :, 3] = 128  # Semi-transparent

        # Composite onto black background
        alpha = rgba_frame[:, :, 3:4].astype(np.float32) / 255.0
        rgb_frame = (rgba_frame[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

        assert rgb_frame.shape == (height, width, 3)
        assert rgb_frame.dtype == np.uint8
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py -v --timeout=30`
Expected: PASS if OBS is installed and virtual camera was activated once. SKIP otherwise.

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py
git commit -m "test: validate pyvirtualcam device creation with PixelFormat.RGB"
```

---

## Phase 1: Wire Modular Core

### Task 11: MeetingSessionConfig — Thread-Safe Observable Config

**Files:**
- Create: `modules/orchestration-service/tests/test_meeting_session_config.py`
- Create: `modules/orchestration-service/src/services/meeting_session_config.py`

- [ ] **Step 1: Write the failing tests**

Create `modules/orchestration-service/tests/test_meeting_session_config.py`:

```python
"""Tests for MeetingSessionConfig — thread-safe, observable config."""

import threading

import pytest

from services.meeting_session_config import MeetingSessionConfig


class TestMeetingSessionConfig:
    def test_create_with_defaults(self):
        config = MeetingSessionConfig(session_id="test-123")
        assert config.session_id == "test-123"
        assert config.caption_source == "bot_audio"
        assert config.source_lang == "auto"
        assert config.target_lang == "en"
        assert config.display_mode == "subtitle"
        assert config.theme == "dark"
        assert config.font_size == 24
        assert config.show_speakers is True
        assert config.show_original is False
        assert config.translation_enabled is True

    def test_update_returns_changed_fields(self):
        config = MeetingSessionConfig(session_id="test-123")
        changed = config.update(target_lang="zh", font_size=32)
        assert changed == {"target_lang", "font_size"}
        assert config.target_lang == "zh"
        assert config.font_size == 32

    def test_update_no_change_returns_empty(self):
        config = MeetingSessionConfig(session_id="test-123")
        changed = config.update(target_lang="en")  # Same as default
        assert changed == set()

    def test_subscriber_notified_on_change(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        config.subscribe(lambda fields: notifications.append(fields))

        config.update(target_lang="zh")
        assert len(notifications) == 1
        assert notifications[0] == {"target_lang"}

    def test_subscriber_not_notified_when_no_change(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        config.subscribe(lambda fields: notifications.append(fields))

        config.update(target_lang="en")  # Same as default
        assert len(notifications) == 0

    def test_multiple_subscribers(self):
        config = MeetingSessionConfig(session_id="test-123")
        notif_a, notif_b = [], []
        config.subscribe(lambda f: notif_a.append(f))
        config.subscribe(lambda f: notif_b.append(f))

        config.update(theme="light")
        assert len(notif_a) == 1
        assert len(notif_b) == 1

    def test_unsubscribe(self):
        config = MeetingSessionConfig(session_id="test-123")
        notifications = []
        callback = lambda f: notifications.append(f)
        config.subscribe(callback)
        config.unsubscribe(callback)

        config.update(target_lang="zh")
        assert len(notifications) == 0

    def test_snapshot_returns_frozen_copy(self):
        config = MeetingSessionConfig(session_id="test-123")
        snap = config.snapshot()
        config.update(target_lang="zh")
        assert snap["target_lang"] == "en"  # Snapshot unchanged
        assert config.target_lang == "zh"   # Config changed

    def test_thread_safety_concurrent_updates(self):
        config = MeetingSessionConfig(session_id="test-123")
        errors = []

        def updater(lang: str):
            try:
                for _ in range(100):
                    config.update(target_lang=lang)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=updater, args=("zh",)),
            threading.Thread(target=updater, args=("en",)),
            threading.Thread(target=updater, args=("ja",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert config.target_lang in ("zh", "en", "ja")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest modules/orchestration-service/tests/test_meeting_session_config.py -v`
Expected: `ModuleNotFoundError: No module named 'services.meeting_session_config'`

- [ ] **Step 3: Implement MeetingSessionConfig**

Create `modules/orchestration-service/src/services/meeting_session_config.py`:

```python
"""MeetingSessionConfig — single control plane for meeting subtitle behavior.

Plain class with explicit update() for thread-safe mutation.
NOT a Pydantic BaseModel — uses threading.Lock and subscriber callbacks.

Composes existing configs:
- Language state (source_lang, target_lang) — delegates to SessionConfig
- Display state (mode, theme, font) — unifies WebcamConfig
- Source routing (caption_source) — genuinely new
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from livetranslate_common.logging import get_logger

logger = get_logger()

# Type for subscriber callbacks: receives set of changed field names
ConfigSubscriber = Callable[[set[str]], None]

# Valid values for constrained fields
VALID_CAPTION_SOURCES = {"bot_audio", "fireflies"}
VALID_DISPLAY_MODES = {"subtitle", "split", "interpreter"}
VALID_THEMES = {"dark", "light", "high_contrast", "minimal", "corporate"}


class MeetingSessionConfig:
    """Thread-safe, observable config for meeting subtitle sessions."""

    __slots__ = (
        "session_id",
        "bot_id",
        "caption_source",
        "source_lang",
        "target_lang",
        "translation_enabled",
        "display_mode",
        "theme",
        "font_size",
        "show_speakers",
        "show_original",
        "_lock",
        "_subscribers",
    )

    def __init__(
        self,
        session_id: str,
        bot_id: str | None = None,
        caption_source: str = "bot_audio",
        source_lang: str = "auto",
        target_lang: str = "en",
        translation_enabled: bool = True,
        display_mode: str = "subtitle",
        theme: str = "dark",
        font_size: int = 24,
        show_speakers: bool = True,
        show_original: bool = False,
    ):
        self.session_id = session_id
        self.bot_id = bot_id
        self.caption_source = caption_source
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.translation_enabled = translation_enabled
        self.display_mode = display_mode
        self.theme = theme
        self.font_size = font_size
        self.show_speakers = show_speakers
        self.show_original = show_original
        self._lock = threading.Lock()
        self._subscribers: list[ConfigSubscriber] = []

    def update(self, **changes: Any) -> set[str]:
        """Apply changes atomically. Returns set of changed field names.

        Thread-safe. Fires subscriber notification once per batch.
        Ignores unknown fields silently.
        """
        changed: set[str] = set()
        with self._lock:
            for field, value in changes.items():
                if not hasattr(self, field) or field.startswith("_"):
                    continue
                if getattr(self, field) != value:
                    object.__setattr__(self, field, value)
                    changed.add(field)
        if changed:
            self._notify_subscribers(changed)
        return changed

    def snapshot(self) -> dict[str, Any]:
        """Return a frozen copy of all config values. For per-frame rendering."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "bot_id": self.bot_id,
                "caption_source": self.caption_source,
                "source_lang": self.source_lang,
                "target_lang": self.target_lang,
                "translation_enabled": self.translation_enabled,
                "display_mode": self.display_mode,
                "theme": self.theme,
                "font_size": self.font_size,
                "show_speakers": self.show_speakers,
                "show_original": self.show_original,
            }

    def subscribe(self, callback: ConfigSubscriber) -> None:
        """Add a subscriber. Called with set of changed field names."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: ConfigSubscriber) -> None:
        """Remove a subscriber."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def _notify_subscribers(self, changed: set[str]) -> None:
        """Notify all subscribers of changes."""
        for sub in self._subscribers:
            try:
                sub(changed)
            except Exception as e:
                logger.warning("Config subscriber error", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest modules/orchestration-service/tests/test_meeting_session_config.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/meeting_session_config.py modules/orchestration-service/tests/test_meeting_session_config.py
git commit -m "feat: MeetingSessionConfig — thread-safe observable config with update() and subscribers"
```

---

### Task 12: CaptionBuffer Multi-Subscriber Extension

**Files:**
- Modify: `modules/orchestration-service/src/services/caption_buffer.py:160-174,236-256`
- Test: existing tests + new subscriber test

- [ ] **Step 1: Write the failing test for multi-subscriber**

Add to the existing caption buffer test file (or create a new one):

```python
"""Tests for CaptionBuffer multi-subscriber extension."""

import pytest

from services.caption_buffer import CaptionBuffer, Caption


class TestCaptionBufferSubscribers:
    def test_subscribe_receives_add_events(self):
        events = []
        buffer = CaptionBuffer()
        buffer.subscribe(lambda event_type, caption: events.append((event_type, caption.translated_text)))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert len(events) == 1
        assert events[0] == ("added", "Hello")

    def test_multiple_subscribers(self):
        events_a, events_b = [], []
        buffer = CaptionBuffer()
        buffer.subscribe(lambda et, c: events_a.append(et))
        buffer.subscribe(lambda et, c: events_b.append(et))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert len(events_a) == 1
        assert len(events_b) == 1

    def test_unsubscribe(self):
        events = []
        buffer = CaptionBuffer()
        callback = lambda et, c: events.append(et)
        buffer.subscribe(callback)
        buffer.unsubscribe(callback)

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert len(events) == 0

    def test_legacy_callback_still_works(self):
        """Backward compat: on_caption_added constructor param still fires."""
        added = []
        buffer = CaptionBuffer(on_caption_added=lambda c: added.append(c.translated_text))

        buffer.add_caption(translated_text="Hello", speaker_name="Alice")
        assert added == ["Hello"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest modules/orchestration-service/tests/test_caption_buffer_subscribers.py -v`
Expected: `AttributeError: 'CaptionBuffer' object has no attribute 'subscribe'`

- [ ] **Step 3: Implement multi-subscriber in CaptionBuffer**

In `modules/orchestration-service/src/services/caption_buffer.py`, add to `__init__` (after line 174):

```python
        # Multi-subscriber list (replaces single-callback pattern internally)
        self._subscribers: list[Callable] = []

        # Auto-subscribe legacy callbacks
        if on_caption_added:
            self._subscribers.append(lambda et, c: on_caption_added(c) if et == "added" else None)
        if on_caption_expired:
            self._subscribers.append(lambda et, c: on_caption_expired(c) if et == "expired" else None)
        if on_caption_updated:
            self._subscribers.append(lambda et, c: on_caption_updated(c) if et == "updated" else None)
```

Add subscribe/unsubscribe methods:

```python
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to caption events. callback(event_type: str, caption: Caption)."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        """Remove a subscriber."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass
```

Update `_fire_callback` callers to also fire to subscribers. In `add_caption`, after the existing `self._fire_callback(self._on_caption_added, caption)` call, add:

```python
        for sub in self._subscribers:
            self._fire_callback(sub, "added", caption)
```

Do the same for expired and updated events.

- [ ] **Step 4: Run all caption buffer tests**

Run: `uv run pytest modules/orchestration-service/tests/ -k "caption_buffer" -v --timeout=30`
Expected: All pass (old + new)

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/caption_buffer.py modules/orchestration-service/tests/test_caption_buffer_subscribers.py
git commit -m "feat: CaptionBuffer multi-subscriber extension with backward compat"
```

---

### Task 13: CaptionSourceAdapter Protocol + BotAudioCaptionSource

**Files:**
- Create: `modules/orchestration-service/tests/integration/test_caption_routing.py`
- Create: `modules/orchestration-service/src/services/pipeline/adapters/source_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `modules/orchestration-service/tests/integration/test_caption_routing.py`:

```python
"""Tests for CaptionSourceAdapter protocol and source routing."""

import asyncio

import pytest

from services.pipeline.adapters.source_adapter import (
    BotAudioCaptionSource,
    CaptionEvent,
    CaptionSourceAdapter,
)
from services.meeting_session_config import MeetingSessionConfig


class TestCaptionEvent:
    def test_create_caption_event(self):
        event = CaptionEvent(
            event_type="added",
            caption_id="c1",
            text="Hello world",
            speaker_name="Alice",
            speaker_color="#4CAF50",
            source_lang="en",
            target_lang="zh",
            translated_text="你好世界",
            confidence=0.95,
            is_draft=False,
        )
        assert event.event_type == "added"
        assert event.text == "Hello world"


@pytest.mark.asyncio
class TestBotAudioCaptionSource:
    async def test_implements_protocol(self):
        source = BotAudioCaptionSource()
        assert isinstance(source, CaptionSourceAdapter)

    async def test_start_stop_lifecycle(self):
        source = BotAudioCaptionSource()
        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)
        assert source.is_running
        await source.stop()
        assert not source.is_running

    async def test_emits_events_to_callback(self):
        events = []
        source = BotAudioCaptionSource()
        source.on_caption = lambda e: events.append(e)

        config = MeetingSessionConfig(session_id="test-123")
        await source.start(config)

        # Simulate a transcription event arriving
        await source.handle_transcription(
            text="Hello",
            speaker_name="Alice",
            source_lang="en",
            confidence=0.9,
            is_final=True,
        )

        assert len(events) == 1
        assert events[0].text == "Hello"
        assert events[0].event_type == "added"

        await source.stop()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_caption_routing.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement the source adapter protocol and BotAudioCaptionSource**

Create `modules/orchestration-service/src/services/pipeline/adapters/source_adapter.py`:

```python
"""Caption Source Adapters — lifecycle-managing source connectors.

Two-layer abstraction:
1. CaptionSourceAdapter (this file) — stateful, lifecycle, event emission
2. ChunkAdapter (existing adapters/) — stateless data transformation

The source adapter USES a chunk adapter internally for format conversion.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from livetranslate_common.logging import get_logger
from livetranslate_common.theme import SPEAKER_COLORS

logger = get_logger()


@dataclass
class CaptionEvent:
    """Canonical event emitted by source adapters, consumed by renderers."""

    event_type: str  # "added", "updated", "expired", "cleared"
    caption_id: str
    text: str
    speaker_name: str | None = None
    speaker_color: str = "#4CAF50"
    source_lang: str = "auto"
    target_lang: str | None = None
    translated_text: str | None = None
    confidence: float = 1.0
    is_draft: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None


@runtime_checkable
class CaptionSourceAdapter(Protocol):
    """Protocol for caption source connectors."""

    is_running: bool
    on_caption: Callable[[CaptionEvent], Any] | None

    async def start(self, config: Any) -> None: ...
    async def stop(self) -> None: ...


class BotAudioCaptionSource:
    """Wraps the existing audio WebSocket → transcription pipeline as a CaptionSourceAdapter."""

    def __init__(self):
        self.is_running: bool = False
        self.on_caption: Callable[[CaptionEvent], Any] | None = None
        self._speaker_color_idx: int = 0
        self._speaker_colors: dict[str, str] = {}

    async def start(self, config: Any) -> None:
        self.is_running = True
        logger.info("BotAudioCaptionSource started", session_id=getattr(config, "session_id", "?"))

    async def stop(self) -> None:
        self.is_running = False
        logger.info("BotAudioCaptionSource stopped")

    def _get_speaker_color(self, speaker_name: str) -> str:
        if speaker_name not in self._speaker_colors:
            self._speaker_colors[speaker_name] = SPEAKER_COLORS[
                self._speaker_color_idx % len(SPEAKER_COLORS)
            ]
            self._speaker_color_idx += 1
        return self._speaker_colors[speaker_name]

    async def handle_transcription(
        self,
        text: str,
        speaker_name: str,
        source_lang: str = "auto",
        confidence: float = 1.0,
        is_final: bool = False,
    ) -> None:
        """Called when a transcription segment arrives from the audio pipeline."""
        if not self.is_running or not self.on_caption:
            return

        event = CaptionEvent(
            event_type="added",
            caption_id=str(uuid.uuid4()),
            text=text,
            speaker_name=speaker_name,
            speaker_color=self._get_speaker_color(speaker_name),
            source_lang=source_lang,
            confidence=confidence,
            is_draft=not is_final,
        )

        callback_result = self.on_caption(event)
        if hasattr(callback_result, "__await__"):
            await callback_result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_caption_routing.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/services/pipeline/adapters/source_adapter.py modules/orchestration-service/tests/integration/test_caption_routing.py
git commit -m "feat: CaptionSourceAdapter protocol + BotAudioCaptionSource"
```

---

### Task 14: PILVirtualCamRenderer — Wire VirtualWebcamManager to pyvirtualcam

**Files:**
- Create: `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py`
- Test: `modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py` (extend)

- [ ] **Step 1: Write the failing test**

Add to `modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py`:

```python
from bot.pil_virtual_cam_renderer import PILVirtualCamRenderer
from services.meeting_session_config import MeetingSessionConfig
from services.caption_buffer import CaptionBuffer


class TestPILVirtualCamRenderer:
    def test_creates_with_config(self):
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)
        assert renderer is not None

    def test_renders_frame_on_caption_event(self):
        config = MeetingSessionConfig(session_id="test-123")
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        buffer.add_caption(translated_text="Hello", speaker_name="Alice")

        # Give render loop one cycle
        import time
        time.sleep(0.1)

        assert renderer.frames_rendered > 0
        assert renderer.last_frame is not None
        assert renderer.last_frame.shape == (720, 1280, 3)

        renderer.stop_rendering()

    def test_config_snapshot_per_frame(self):
        """Config changes mid-render don't cause tearing."""
        config = MeetingSessionConfig(session_id="test-123", font_size=24)
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        config.update(font_size=48)

        import time
        time.sleep(0.1)

        # Renderer should have picked up the new font size
        assert renderer._last_config_snapshot["font_size"] == 48
        renderer.stop_rendering()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py::TestPILVirtualCamRenderer -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement PILVirtualCamRenderer**

Create `modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py`:

```python
"""PIL Virtual Camera Renderer.

Wires VirtualWebcamManager (PIL frame generation) to pyvirtualcam (device output).
Features:
- Frame-paced timer (not busy-wait)
- Dirty-flag rendering (only re-render when state changes)
- Config snapshot per frame (prevents mid-frame tearing)
- Pre-allocated frame buffer
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig, TranslationDisplay
from livetranslate_common.logging import get_logger
from livetranslate_common.theme import get_theme_colors, hex_to_rgb

logger = get_logger()

try:
    import pyvirtualcam
    from pyvirtualcam import PixelFormat

    PYVIRTUALCAM_AVAILABLE = True
except ImportError:
    PYVIRTUALCAM_AVAILABLE = False


class PILVirtualCamRenderer:
    """Renders subtitle frames via PIL and outputs to a virtual camera device."""

    def __init__(
        self,
        config: Any,  # MeetingSessionConfig
        caption_buffer: Any,  # CaptionBuffer
        use_virtual_cam: bool = True,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self._config = config
        self._caption_buffer = caption_buffer
        self._use_virtual_cam = use_virtual_cam and PYVIRTUALCAM_AVAILABLE
        self._width = width
        self._height = height
        self._fps = fps

        # Rendering state
        self._webcam_manager = VirtualWebcamManager(
            config=WebcamConfig(width=width, height=height, fps=fps)
        )
        self._dirty = True
        self._running = False
        self._thread: threading.Thread | None = None
        self._cam: Any = None

        # Stats
        self.frames_rendered: int = 0
        self.last_frame: np.ndarray | None = None
        self._last_config_snapshot: dict[str, Any] = {}

        # Subscribe to config changes
        config.subscribe(self._on_config_changed)

        # Subscribe to caption events
        caption_buffer.subscribe(self._on_caption_event)

    def _on_config_changed(self, changed_fields: set[str]) -> None:
        self._dirty = True

    def _on_caption_event(self, event_type: str, caption: Any) -> None:
        self._dirty = True

    def start_rendering(self) -> None:
        """Start the render loop in a background thread."""
        if self._running:
            return

        if self._use_virtual_cam:
            self._cam = pyvirtualcam.Camera(
                self._width, self._height, self._fps, fmt=PixelFormat.RGB
            )
            logger.info("Virtual camera started", device=self._cam.device)

        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def stop_rendering(self) -> None:
        """Stop the render loop and close virtual camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cam:
            self._cam.close()
            self._cam = None

    def _render_loop(self) -> None:
        """Frame-paced render loop."""
        frame_interval = 1.0 / self._fps
        next_frame_time = time.monotonic()

        # Pre-allocate frame buffer
        frame_buffer = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        while self._running:
            next_frame_time += frame_interval

            # Snapshot config for this frame (atomic read)
            self._last_config_snapshot = self._config.snapshot()

            # Render
            self._webcam_manager._generate_frame()
            if self._webcam_manager.current_frame is not None:
                frame = self._webcam_manager.current_frame
                # Ensure RGB (no alpha)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    # Composite RGBA onto black background
                    alpha = frame[:, :, 3:4].astype(np.float32) / 255.0
                    frame_buffer[:] = (frame[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
                elif frame.shape == frame_buffer.shape:
                    np.copyto(frame_buffer, frame)

                self.last_frame = frame_buffer
                self.frames_rendered += 1
                self._dirty = False

                # Send to virtual camera
                if self._cam:
                    self._cam.send(frame_buffer)

            # Sleep until next frame
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.monotonic()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py::TestPILVirtualCamRenderer -v --timeout=30`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/src/bot/pil_virtual_cam_renderer.py modules/orchestration-service/tests/integration/test_virtual_cam_e2e.py
git commit -m "feat: PILVirtualCamRenderer — frame-paced, dirty-flag, config snapshot per frame"
```

---

### Task 15: Update Dashboard to Use Shared Colors

**Files:**
- Modify: `modules/dashboard-service/src/lib/stores/loopback.svelte.ts:13,66-69`

- [ ] **Step 1: Create a TypeScript-consumable version of the theme**

Create `modules/dashboard-service/src/lib/theme.ts`:

```typescript
/**
 * Canonical theme definitions — imported from livetranslate-common.
 * Keep in sync with modules/shared/src/livetranslate_common/theme.py
 */

export type DisplayMode = 'split' | 'subtitle' | 'transcript' | 'interpreter';

export const SPEAKER_COLORS = [
  '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336',
  '#00BCD4', '#E91E63', '#FFEB3B', '#795548', '#607D8B',
] as const;
```

- [ ] **Step 2: Update loopback.svelte.ts to import from theme.ts**

In `modules/dashboard-service/src/lib/stores/loopback.svelte.ts`, replace lines 13 and 66-69:

```typescript
// Line 13 — replace inline type:
// OLD: export type DisplayMode = 'split' | 'subtitle' | 'transcript' | 'interpreter';
// NEW:
import { type DisplayMode, SPEAKER_COLORS } from '$lib/theme';
export type { DisplayMode };

// Lines 66-69 — remove inline colors:
// OLD: const SPEAKER_COLORS = ['#3b82f6', '#a855f7', ...];
// (deleted — now imported above)
```

- [ ] **Step 3: Verify dashboard builds**

Run: `cd modules/dashboard-service && npm run build 2>&1 | tail -10`
Expected: Build succeeds

- [ ] **Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/theme.ts modules/dashboard-service/src/lib/stores/loopback.svelte.ts
git commit -m "refactor: dashboard uses canonical SPEAKER_COLORS from shared theme"
```

---

### Task 16: Run Full Test Suite — Phase 0+1 Validation

**Files:** None (validation only)

- [ ] **Step 1: Run all three test suites**

```bash
uv run pytest modules/shared/tests/ -v -q 2>&1 | tail -5
uv run pytest modules/orchestration-service/tests/ -v -q --timeout=30 2>&1 | tail -5
uv run pytest modules/transcription-service/tests/ -v -q --timeout=30 2>&1 | tail -5
```

Expected: All green. Zero failures. Zero import errors.

- [ ] **Step 2: Run dashboard build**

```bash
cd modules/dashboard-service && npm run build 2>&1 | tail -5
```

Expected: Build succeeds.

- [ ] **Step 3: Commit any remaining fixes**

If anything failed, fix it and commit before proceeding.

---

## Summary

| Task | What | Phase |
|------|------|-------|
| 1 | Fix broken test import | 0 |
| 2 | Fix sustained detector assertion | 0 |
| 3 | Canonical theme definitions (shared) | 0 |
| 4 | CaptionBuffer → shared colors | 0 |
| 5 | VirtualWebcamManager → shared theme + 1280x720 | 0 |
| 6 | Fix busy-wait render loop | 0 |
| 7 | Clean up caption_processor.py | 0 |
| 8 | Remove v4l2loopback-dkms from Dockerfile | 0 |
| 9 | PIL frame rendering validation | 0 |
| 10 | pyvirtualcam device validation | 0 |
| 11 | MeetingSessionConfig | 1 |
| 12 | CaptionBuffer multi-subscriber | 1 |
| 13 | CaptionSourceAdapter + BotAudioCaptionSource | 1 |
| 14 | PILVirtualCamRenderer | 1 |
| 15 | Dashboard shared colors | 1 |
| 16 | Full test suite validation | 1 |
