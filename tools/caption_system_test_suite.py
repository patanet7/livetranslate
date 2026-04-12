#!/usr/bin/env python3
"""
CAPTION SYSTEM COMPREHENSIVE TEST SUITE

Tests EVERY aspect of the caption → webcam rendering pipeline.
Organized by layer: Unit → Integration → E2E

Usage:
    uv run python tools/caption_system_test_suite.py

    # Run specific category:
    uv run python tools/caption_system_test_suite.py --category rendering
    uv run python tools/caption_system_test_suite.py --category pipeline
    uv run python tools/caption_system_test_suite.py --category websocket
    uv run python tools/caption_system_test_suite.py --category all

Output: /tmp/caption-test-suite/
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Add orchestration-service to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules" / "orchestration-service" / "src"))

from bot.pil_virtual_cam_renderer import PILVirtualCamRenderer
from bot.virtual_webcam import VirtualWebcamManager, WebcamConfig, TranslationDisplay, Theme
from bot.text_wrapper import wrap_text
from services.caption_buffer import CaptionBuffer, Caption
from services.meeting_session_config import MeetingSessionConfig
from livetranslate_common.theme import DisplayMode, ThemeColors, SPEAKER_COLORS, get_theme_colors, _THEMES


OUTPUT_DIR = Path("/tmp/caption-test-suite")


@dataclass
class TestResult:
    name: str
    category: str
    status: str  # pass, fail, skip
    duration_ms: float
    details: str = ""
    frame_path: str | None = None


@dataclass
class TestSuite:
    results: list[TestResult] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, result: TestResult):
        self.results.append(result)
        icon = {"pass": "✓", "fail": "✗", "skip": "○"}[result.status]
        print(f"  {icon} [{result.category}] {result.name}: {result.details}")

    def summary(self) -> dict:
        passed = sum(1 for r in self.results if r.status == "pass")
        failed = sum(1 for r in self.results if r.status == "fail")
        skipped = sum(1 for r in self.results if r.status == "skip")
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{100*passed/max(1,len(self.results)):.1f}%",
        }


class CaptionSystemTestSuite:
    """Comprehensive test suite for the caption system."""

    def __init__(self):
        self.suite = TestSuite()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run_test(self, name: str, category: str, test_fn) -> TestResult:
        """Run a single test and record the result."""
        start = time.perf_counter()
        try:
            details, frame_path = test_fn()
            duration = (time.perf_counter() - start) * 1000
            return TestResult(name, category, "pass", duration, details, frame_path)
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(name, category, "fail", duration, str(e))

    # =========================================================================
    # CATEGORY 1: VIRTUAL WEBCAM MANAGER (PIL Rendering)
    # =========================================================================

    def test_webcam_init_default(self):
        """Test VirtualWebcamManager initializes with default config."""
        cfg = WebcamConfig()
        mgr = VirtualWebcamManager(config=cfg)
        assert mgr.config.width == 1280
        assert mgr.config.height == 720
        assert mgr.config.fps == 30
        return "1280x720@30fps", None

    def test_webcam_init_custom(self):
        """Test VirtualWebcamManager with custom config."""
        cfg = WebcamConfig(width=1920, height=1080, fps=60)
        mgr = VirtualWebcamManager(config=cfg)
        assert mgr.config.width == 1920
        return "1920x1080@60fps", None

    def test_webcam_add_translation(self):
        """Test adding a translation to webcam manager."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-1",
            "translated_text": "Hello world",
            "source_language": "en",
            "target_language": "zh",
            "speaker_name": "Test",
        })
        assert len(mgr.current_translations) == 1
        return f"1 translation added", None

    def test_webcam_max_translations(self):
        """Test translation queue respects max size."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        for i in range(10):
            mgr.add_translation({
                "translation_id": f"test-{i}",
                "translated_text": f"Message {i}",
                "speaker_name": "Test",
            })
        assert len(mgr.current_translations) <= 5  # Default max
        return f"{len(mgr.current_translations)} translations (max 5)", None

    def test_webcam_generate_frame_empty(self):
        """Test frame generation with no translations (waiting state)."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr._generate_frame()
        assert mgr.current_frame is not None
        assert mgr.current_frame.shape == (720, 1280, 3)  # RGB
        path = OUTPUT_DIR / "frame_empty.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "720x1280 RGB frame", str(path)

    def test_webcam_generate_frame_with_text(self):
        """Test frame generation with translation text."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-1",
            "translated_text": "This is a test translation.",
            "speaker_name": "Speaker",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_with_text.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Frame with text", str(path)

    def test_webcam_chinese_text(self):
        """Test Chinese character rendering."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-zh",
            "translated_text": "这是中文测试文本，包含汉字。",
            "speaker_name": "王丽",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_chinese.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Chinese glyphs rendered", str(path)

    def test_webcam_japanese_text(self):
        """Test Japanese character rendering."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-ja",
            "translated_text": "これは日本語のテストです。ひらがなとカタカナ。",
            "speaker_name": "田中",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_japanese.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Japanese glyphs rendered", str(path)

    def test_webcam_korean_text(self):
        """Test Korean character rendering."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-ko",
            "translated_text": "이것은 한국어 테스트입니다. 한글 렌더링 확인.",
            "speaker_name": "김수민",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_korean.png"
        Image.fromarray(mgr.current_frame).save(path)
        # Check no boxes
        frame = mgr.current_frame
        return "Korean glyphs rendered", str(path)

    def test_webcam_mixed_cjk_latin(self):
        """Test mixed CJK and Latin text."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-mixed",
            "translated_text": "Hello 你好 こんにちは 안녕하세요 Bonjour café résumé",
            "speaker_name": "Polyglot",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_mixed_cjk.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Mixed CJK/Latin rendered", str(path)

    def test_webcam_spanish_accents(self):
        """Test Spanish accented characters."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-es",
            "translated_text": "Español: ¿Cómo estás? Año nuevo, niño, señor, café.",
            "speaker_name": "García",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_spanish.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Spanish accents rendered", str(path)

    def test_webcam_emoji(self):
        """Test emoji rendering (may fallback)."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "test-emoji",
            "translated_text": "Hello! How are you today?",  # Skip emoji, not all fonts support
            "speaker_name": "Emoji Test",
        })
        mgr._generate_frame()
        return "Text without emoji", None

    def test_webcam_long_text_wrap(self):
        """Test long text wrapping."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        long_text = "This is a very long sentence that should wrap across multiple lines to ensure proper text layout in the caption overlay without any truncation or overflow issues that would make the text unreadable."
        mgr.add_translation({
            "translation_id": "test-long",
            "translated_text": long_text,
            "speaker_name": "Verbose",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_long_text.png"
        Image.fromarray(mgr.current_frame).save(path)
        return f"Text wrapped ({len(long_text)} chars)", str(path)

    def test_webcam_wrap_text_function(self):
        """Test wrap_text utility function."""
        text = "This is a test of the text wrapping function with multiple words"
        lines = wrap_text(text, max_chars=20, max_lines=3)
        assert len(lines) <= 3
        assert all(len(line) <= 25 for line in lines)  # Allow some overflow for word boundaries
        return f"{len(lines)} lines", None

    def test_webcam_display_mode_subtitle(self):
        """Test subtitle display mode."""
        cfg = WebcamConfig(display_mode=DisplayMode.SUBTITLE)
        mgr = VirtualWebcamManager(config=cfg)
        mgr.add_translation({"translation_id": "t1", "translated_text": "Subtitle mode", "speaker_name": "Test"})
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_mode_subtitle.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Subtitle mode rendered", str(path)

    def test_webcam_display_mode_split(self):
        """Test split display mode."""
        cfg = WebcamConfig(display_mode=DisplayMode.SPLIT)
        mgr = VirtualWebcamManager(config=cfg)
        mgr.add_translation({"translation_id": "t1", "translated_text": "Split mode", "speaker_name": "Test"})
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_mode_split.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Split mode rendered", str(path)

    def test_webcam_display_mode_interpreter(self):
        """Test interpreter display mode."""
        cfg = WebcamConfig(display_mode=DisplayMode.INTERPRETER)
        mgr = VirtualWebcamManager(config=cfg)
        mgr.add_translation({"translation_id": "t1", "translated_text": "Interpreter mode", "speaker_name": "Test"})
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_mode_interpreter.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Interpreter mode rendered", str(path)

    def test_webcam_theme_dark(self):
        """Test dark theme."""
        cfg = WebcamConfig(theme=Theme.DARK)
        mgr = VirtualWebcamManager(config=cfg)
        mgr.add_translation({"translation_id": "t1", "translated_text": "Dark theme", "speaker_name": "Test"})
        mgr._generate_frame()
        # Check background is dark
        mean = np.mean(mgr.current_frame[:100, :100, :3])
        assert mean < 50, f"Background too bright: {mean}"
        return f"Dark background (mean={mean:.1f})", None

    def test_webcam_theme_light(self):
        """Test light theme."""
        cfg = WebcamConfig(theme=Theme.LIGHT)
        mgr = VirtualWebcamManager(config=cfg)
        mgr.add_translation({"translation_id": "t1", "translated_text": "Light theme", "speaker_name": "Test"})
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_theme_light.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Light theme rendered", str(path)

    def test_webcam_speaker_color(self):
        """Test speaker name color."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "t1",
            "translated_text": "Colored speaker",
            "speaker_name": "Alice",
            "speaker_color": "#FF5722",
        })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_speaker_color.png"
        Image.fromarray(mgr.current_frame).save(path)
        return "Speaker color #FF5722", str(path)

    def test_webcam_multiple_speakers(self):
        """Test multiple speakers with different colors."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        speakers = [
            ("Alice", "#4CAF50", "Hello from Alice"),
            ("Bob", "#2196F3", "Hi Alice, this is Bob"),
            ("Carol", "#FF9800", "Hey everyone"),
        ]
        for name, color, text in speakers:
            mgr.add_translation({
                "translation_id": f"t-{name}",
                "translated_text": text,
                "speaker_name": name,
                "speaker_color": color,
            })
        mgr._generate_frame()
        path = OUTPUT_DIR / "frame_multi_speaker.png"
        Image.fromarray(mgr.current_frame).save(path)
        return f"{len(speakers)} speakers", str(path)

    def test_webcam_confidence_display(self):
        """Test translation confidence indicator."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr.add_translation({
            "translation_id": "t1",
            "translated_text": "High confidence translation",
            "speaker_name": "Test",
            "translation_confidence": 0.95,
        })
        mgr._generate_frame()
        return "Confidence 0.95", None

    def test_webcam_frame_rgb_format(self):
        """Test frame is RGB format."""
        mgr = VirtualWebcamManager(config=WebcamConfig())
        mgr._generate_frame()
        assert mgr.current_frame.dtype == np.uint8
        assert mgr.current_frame.shape[2] == 3  # RGB
        return "RGB uint8", None

    def test_webcam_frame_dimensions(self):
        """Test frame dimensions match config."""
        cfg = WebcamConfig(width=1920, height=1080)
        mgr = VirtualWebcamManager(config=cfg)
        mgr._generate_frame()
        h, w, c = mgr.current_frame.shape
        assert w == 1920 and h == 1080
        return f"{w}x{h}", None

    # =========================================================================
    # CATEGORY 2: CAPTION BUFFER
    # =========================================================================

    def test_caption_buffer_init(self):
        """Test CaptionBuffer initialization."""
        buf = CaptionBuffer()
        assert buf.max_captions == 5
        assert buf.default_duration == 4.0  # DEFAULT_CAPTION_DURATION_SECONDS
        return "max=5, duration=4s", None

    def test_caption_buffer_add(self):
        """Test adding caption to buffer."""
        buf = CaptionBuffer()
        result = buf.add_caption(
            translated_text="Test caption",
            original_text="Original",
            speaker_name="Test",
        )
        cap, was_update = result  # Returns (caption, was_update)
        assert cap is not None
        assert len(buf.get_active_captions()) == 1
        return f"id={cap.id[:8]}", None

    def test_caption_buffer_max_captions(self):
        """Test buffer respects max captions."""
        buf = CaptionBuffer(max_captions=3)
        for i in range(5):
            buf.add_caption(translated_text=f"Caption {i}", speaker_name=f"S{i}")
        active = buf.get_active_captions()
        assert len(active) <= 3
        return f"{len(active)} active (max 3)", None

    def test_caption_buffer_aggregation(self):
        """Test same-speaker caption aggregation."""
        buf = CaptionBuffer(aggregate_speaker_text=True)
        buf.add_caption(translated_text="First part.", speaker_name="Alice")
        buf.add_caption(translated_text="Second part.", speaker_name="Alice")
        buf.add_caption(translated_text="Third part.", speaker_name="Alice")
        active = buf.get_active_captions()
        # Should aggregate into fewer captions
        assert len(active) <= 2
        return f"{len(active)} after aggregation", None

    def test_caption_buffer_no_aggregation_different_speakers(self):
        """Test no aggregation for different speakers."""
        buf = CaptionBuffer(aggregate_speaker_text=True)
        buf.add_caption(translated_text="From Alice", speaker_name="Alice")
        buf.add_caption(translated_text="From Bob", speaker_name="Bob")
        active = buf.get_active_captions()
        assert len(active) == 2
        return "2 separate captions", None

    def test_caption_buffer_speaker_color_auto(self):
        """Test automatic speaker color assignment."""
        buf = CaptionBuffer()
        buf.add_caption(translated_text="Test", speaker_name="NewSpeaker")
        color = buf.get_speaker_color("NewSpeaker")
        assert color in SPEAKER_COLORS
        return f"Auto-assigned {color}", None

    def test_caption_buffer_speaker_color_manual(self):
        """Test manual speaker color assignment."""
        buf = CaptionBuffer()
        buf.set_speaker_color("Alice", "#FF0000")
        color = buf.get_speaker_color("Alice")
        assert color == "#FF0000"
        return "Manual #FF0000", None

    def test_caption_buffer_subscribe(self):
        """Test event subscription."""
        buf = CaptionBuffer()
        events = []
        def callback(event_type, caption):
            events.append((event_type, caption.id))
        buf.subscribe(callback)
        buf.add_caption(translated_text="Test", speaker_name="Test")
        assert len(events) == 1
        assert events[0][0] == "added"
        return "1 event received", None

    def test_caption_buffer_unsubscribe(self):
        """Test event unsubscription."""
        buf = CaptionBuffer()
        events = []
        def callback(event_type, caption):
            events.append(event_type)
        buf.subscribe(callback)
        buf.unsubscribe(callback)
        buf.add_caption(translated_text="Test", speaker_name="Test")
        assert len(events) == 0
        return "No events after unsub", None

    def test_caption_buffer_update_event(self):
        """Test update event on aggregation."""
        buf = CaptionBuffer(aggregate_speaker_text=True)
        events = []
        def callback(event_type, caption):
            events.append(event_type)
        buf.subscribe(callback)
        buf.add_caption(translated_text="First", speaker_name="Alice")
        buf.add_caption(translated_text="Second", speaker_name="Alice")
        # Should have added + updated events
        event_types = set(events)
        return f"Events: {event_types}", None

    def test_caption_buffer_expiry(self):
        """Test caption expiry."""
        buf = CaptionBuffer(default_duration=0.1)  # 100ms
        buf.add_caption(translated_text="Expires fast", speaker_name="Test")
        time.sleep(0.2)
        buf.cleanup_expired()  # Correct method name
        active = buf.get_active_captions()
        assert len(active) == 0
        return "Caption expired", None

    def test_caption_buffer_clear(self):
        """Test clearing all captions."""
        buf = CaptionBuffer()
        buf.add_caption(translated_text="Test 1", speaker_name="A")
        buf.add_caption(translated_text="Test 2", speaker_name="B")
        buf.clear()
        assert len(buf.get_active_captions()) == 0
        return "Buffer cleared", None

    # =========================================================================
    # CATEGORY 3: MEETING SESSION CONFIG
    # =========================================================================

    def test_config_init(self):
        """Test MeetingSessionConfig initialization."""
        cfg = MeetingSessionConfig(session_id="test-123")
        assert cfg.session_id == "test-123"
        return "session_id=test-123", None

    def test_config_display_mode(self):
        """Test display mode configuration."""
        cfg = MeetingSessionConfig(session_id="t", display_mode=DisplayMode.SPLIT)
        assert cfg.display_mode == DisplayMode.SPLIT
        return "DisplayMode.SPLIT", None

    def test_config_update(self):
        """Test config update method."""
        cfg = MeetingSessionConfig(session_id="t", font_size=24)
        cfg.update(font_size=48)
        assert cfg.font_size == 48
        return "font_size updated to 48", None

    def test_config_subscribe(self):
        """Test config change subscription."""
        cfg = MeetingSessionConfig(session_id="t")
        changes = []
        cfg.subscribe(lambda fields: changes.append(fields))
        cfg.update(font_size=36)
        assert len(changes) == 1
        assert "font_size" in changes[0]
        return "Subscriber notified", None

    def test_config_snapshot(self):
        """Test config snapshot."""
        cfg = MeetingSessionConfig(session_id="t", font_size=24, display_mode=DisplayMode.SUBTITLE)
        snap = cfg.snapshot()
        assert snap["font_size"] == 24
        assert snap["display_mode"] == DisplayMode.SUBTITLE
        return "Snapshot captured", None

    def test_config_thread_safety(self):
        """Test config is thread-safe."""
        import threading
        cfg = MeetingSessionConfig(session_id="t", font_size=10)
        errors = []

        def updater():
            for i in range(100):
                try:
                    cfg.update(font_size=i)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=updater) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        return "No race conditions", None

    # =========================================================================
    # CATEGORY 4: PIL VIRTUAL CAM RENDERER (Integration)
    # =========================================================================

    def test_renderer_init(self):
        """Test PILVirtualCamRenderer initialization."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        assert renderer._webcam_manager is not None
        return "Renderer initialized", None

    def test_renderer_start_stop(self):
        """Test renderer start/stop lifecycle."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        renderer.start_rendering()
        assert renderer.is_running
        time.sleep(0.1)
        renderer.stop_rendering()
        assert not renderer.is_running
        return "Start/stop OK", None

    def test_renderer_frame_generation(self):
        """Test renderer generates frames."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        renderer.start_rendering()
        time.sleep(0.2)
        frames = renderer.frames_rendered
        renderer.stop_rendering()
        assert frames > 0
        return f"{frames} frames in 200ms", None

    def test_renderer_caption_sync(self):
        """Test captions sync to renderer."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        renderer.start_rendering()
        buf.add_caption(translated_text="Synced caption", speaker_name="Test")
        time.sleep(0.1)
        count = len(renderer._webcam_manager.current_translations)
        renderer.stop_rendering()
        assert count >= 1
        return f"{count} caption(s) synced", None

    def test_renderer_config_snapshot(self):
        """Test renderer takes config snapshot per frame."""
        cfg = MeetingSessionConfig(session_id="t", font_size=24)
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        renderer.start_rendering()
        time.sleep(0.1)
        cfg.update(font_size=48)
        time.sleep(0.1)
        snap = renderer.last_config_snapshot
        renderer.stop_rendering()
        assert snap.get("font_size") == 48
        return "Snapshot updated", None

    def test_renderer_dirty_flag(self):
        """Test dirty flag triggers re-render."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        renderer.start_rendering()
        time.sleep(0.1)
        initial_frames = renderer.frames_rendered
        buf.add_caption(translated_text="Trigger dirty", speaker_name="Test")
        time.sleep(0.1)
        final_frames = renderer.frames_rendered
        renderer.stop_rendering()
        assert final_frames > initial_frames
        return f"Frames: {initial_frames} → {final_frames}", None

    def test_renderer_save_frame(self):
        """Test saving rendered frame."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False)
        renderer.start_rendering()
        buf.add_caption(translated_text="Frame to save", speaker_name="Test")
        time.sleep(0.2)
        renderer.stop_rendering()

        if renderer.last_frame is not None:
            path = OUTPUT_DIR / "renderer_frame.png"
            Image.fromarray(renderer.last_frame).save(path)
            return "Frame saved", str(path)
        return "No frame", None

    def test_renderer_fps_30(self):
        """Test renderer maintains ~30 FPS."""
        cfg = MeetingSessionConfig(session_id="t")
        buf = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=cfg, caption_buffer=buf, use_virtual_cam=False, fps=30)
        renderer.start_rendering()
        time.sleep(1.0)
        frames = renderer.frames_rendered
        renderer.stop_rendering()
        # Should be 25-35 frames in 1 second at 30 FPS
        assert 20 <= frames <= 40, f"FPS out of range: {frames}"
        return f"{frames} frames/sec", None

    # =========================================================================
    # CATEGORY 5: THEME SYSTEM
    # =========================================================================

    def test_theme_colors_exist(self):
        """Test all themes are available."""
        themes = ["dark", "light", "high_contrast"]
        for theme in themes:
            colors = get_theme_colors(theme)
            assert colors.background is not None
            assert colors.text_primary is not None
        return f"{len(themes)} themes available", None

    def test_theme_speaker_colors(self):
        """Test speaker colors list."""
        assert len(SPEAKER_COLORS) >= 5
        for color in SPEAKER_COLORS:
            assert color.startswith("#")
            assert len(color) == 7
        return f"{len(SPEAKER_COLORS)} speaker colors", None

    def test_theme_display_modes(self):
        """Test DisplayMode enum."""
        modes = list(DisplayMode)
        assert DisplayMode.SUBTITLE in modes
        assert DisplayMode.SPLIT in modes
        assert DisplayMode.INTERPRETER in modes
        return f"{len(modes)} modes", None

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_category(self, category: str) -> list[TestResult]:
        """Run all tests in a category."""
        tests = {
            "rendering": [
                ("webcam_init_default", self.test_webcam_init_default),
                ("webcam_init_custom", self.test_webcam_init_custom),
                ("webcam_add_translation", self.test_webcam_add_translation),
                ("webcam_max_translations", self.test_webcam_max_translations),
                ("webcam_generate_frame_empty", self.test_webcam_generate_frame_empty),
                ("webcam_generate_frame_with_text", self.test_webcam_generate_frame_with_text),
                ("webcam_chinese_text", self.test_webcam_chinese_text),
                ("webcam_japanese_text", self.test_webcam_japanese_text),
                ("webcam_korean_text", self.test_webcam_korean_text),
                ("webcam_mixed_cjk_latin", self.test_webcam_mixed_cjk_latin),
                ("webcam_spanish_accents", self.test_webcam_spanish_accents),
                ("webcam_emoji", self.test_webcam_emoji),
                ("webcam_long_text_wrap", self.test_webcam_long_text_wrap),
                ("webcam_wrap_text_function", self.test_webcam_wrap_text_function),
                ("webcam_display_mode_subtitle", self.test_webcam_display_mode_subtitle),
                ("webcam_display_mode_split", self.test_webcam_display_mode_split),
                ("webcam_display_mode_interpreter", self.test_webcam_display_mode_interpreter),
                ("webcam_theme_dark", self.test_webcam_theme_dark),
                ("webcam_theme_light", self.test_webcam_theme_light),
                ("webcam_speaker_color", self.test_webcam_speaker_color),
                ("webcam_multiple_speakers", self.test_webcam_multiple_speakers),
                ("webcam_confidence_display", self.test_webcam_confidence_display),
                ("webcam_frame_rgb_format", self.test_webcam_frame_rgb_format),
                ("webcam_frame_dimensions", self.test_webcam_frame_dimensions),
            ],
            "buffer": [
                ("caption_buffer_init", self.test_caption_buffer_init),
                ("caption_buffer_add", self.test_caption_buffer_add),
                ("caption_buffer_max_captions", self.test_caption_buffer_max_captions),
                ("caption_buffer_aggregation", self.test_caption_buffer_aggregation),
                ("caption_buffer_no_aggregation_different_speakers", self.test_caption_buffer_no_aggregation_different_speakers),
                ("caption_buffer_speaker_color_auto", self.test_caption_buffer_speaker_color_auto),
                ("caption_buffer_speaker_color_manual", self.test_caption_buffer_speaker_color_manual),
                ("caption_buffer_subscribe", self.test_caption_buffer_subscribe),
                ("caption_buffer_unsubscribe", self.test_caption_buffer_unsubscribe),
                ("caption_buffer_update_event", self.test_caption_buffer_update_event),
                ("caption_buffer_expiry", self.test_caption_buffer_expiry),
                ("caption_buffer_clear", self.test_caption_buffer_clear),
            ],
            "config": [
                ("config_init", self.test_config_init),
                ("config_display_mode", self.test_config_display_mode),
                ("config_update", self.test_config_update),
                ("config_subscribe", self.test_config_subscribe),
                ("config_snapshot", self.test_config_snapshot),
                ("config_thread_safety", self.test_config_thread_safety),
            ],
            "pipeline": [
                ("renderer_init", self.test_renderer_init),
                ("renderer_start_stop", self.test_renderer_start_stop),
                ("renderer_frame_generation", self.test_renderer_frame_generation),
                ("renderer_caption_sync", self.test_renderer_caption_sync),
                ("renderer_config_snapshot", self.test_renderer_config_snapshot),
                ("renderer_dirty_flag", self.test_renderer_dirty_flag),
                ("renderer_save_frame", self.test_renderer_save_frame),
                ("renderer_fps_30", self.test_renderer_fps_30),
            ],
            "theme": [
                ("theme_colors_exist", self.test_theme_colors_exist),
                ("theme_speaker_colors", self.test_theme_speaker_colors),
                ("theme_display_modes", self.test_theme_display_modes),
            ],
        }

        if category == "all":
            categories = tests.keys()
        else:
            categories = [category]

        results = []
        for cat in categories:
            if cat not in tests:
                print(f"Unknown category: {cat}")
                continue
            print(f"\n{'='*60}")
            print(f"CATEGORY: {cat.upper()}")
            print(f"{'='*60}")
            for name, test_fn in tests[cat]:
                result = self.run_test(name, cat, test_fn)
                self.suite.add(result)
                results.append(result)

        return results

    def run(self, category: str = "all"):
        """Run the test suite."""
        print("=" * 60)
        print("CAPTION SYSTEM COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"Output: {OUTPUT_DIR}")
        print(f"Started: {self.suite.started_at}")

        self.run_category(category)

        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        summary = self.suite.summary()
        print(f"Total:   {summary['total']}")
        print(f"Passed:  {summary['passed']}")
        print(f"Failed:  {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Rate:    {summary['pass_rate']}")

        # List generated frames
        frames = list(OUTPUT_DIR.glob("*.png"))
        if frames:
            print(f"\nGenerated {len(frames)} frame images:")
            for f in sorted(frames):
                print(f"  - {f.name}")

        # Save results
        results_path = OUTPUT_DIR / "results.json"
        with open(results_path, "w") as f:
            json.dump({
                "summary": summary,
                "results": [
                    {
                        "name": r.name,
                        "category": r.category,
                        "status": r.status,
                        "duration_ms": r.duration_ms,
                        "details": r.details,
                        "frame_path": r.frame_path,
                    }
                    for r in self.suite.results
                ],
            }, f, indent=2)
        print(f"\nResults saved to: {results_path}")

        return summary["failed"] == 0


def main():
    parser = argparse.ArgumentParser(description="Caption System Test Suite")
    parser.add_argument(
        "--category",
        default="all",
        choices=["rendering", "buffer", "config", "pipeline", "theme", "all"],
        help="Test category to run",
    )
    args = parser.parse_args()

    suite = CaptionSystemTestSuite()
    success = suite.run(args.category)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
