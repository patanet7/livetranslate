#!/usr/bin/env python3
"""
Renderer Integration Test — CaptionBuffer → PILVirtualCamRenderer

Tests the rendering pipeline integration WITHOUT transcription:
1. Creates MeetingSessionConfig (like orchestration does)
2. Creates CaptionBuffer with subscribers
3. Creates PILVirtualCamRenderer
4. Pumps captions through the system
5. Verifies frames are generated with correct content

NO SERVICES REQUIRED — runs entirely in-process.

Usage:
    uv run python tools/renderer_integration_test.py

Output: /tmp/renderer-integration-test/*.png
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add orchestration-service to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules" / "orchestration-service" / "src"))

import numpy as np
from PIL import Image

from bot.pil_virtual_cam_renderer import PILVirtualCamRenderer
from services.caption_buffer import CaptionBuffer
from services.meeting_session_config import MeetingSessionConfig
from livetranslate_common.theme import DisplayMode


OUTPUT_DIR = Path("/tmp/renderer-integration-test")


class RendererIntegrationTest:
    """Tests the full rendering pipeline without external services."""

    def __init__(self):
        self.results = {
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "frames_saved": [],
        }

    def log(self, step: str, status: str, details: str = ""):
        entry = {"step": step, "status": status, "details": details}
        self.results["steps"].append(entry)
        icon = "✓" if status == "pass" else "✗" if status == "fail" else "○"
        print(f"  {icon} {step}: {details}" if details else f"  {icon} {step}")

    def save_frame(self, renderer: PILVirtualCamRenderer, filename: str) -> bool:
        """Save current frame and verify it has content."""
        if renderer.last_frame is None:
            self.log(f"Save {filename}", "fail", "No frame generated")
            return False

        frame = renderer.last_frame
        filepath = OUTPUT_DIR / filename

        # Save the frame
        img = Image.fromarray(frame)
        img.save(filepath)
        self.results["frames_saved"].append(filename)

        # Verify frame has content (not all black)
        mean_pixel = np.mean(frame)
        if mean_pixel < 5:  # Nearly black
            self.log(f"Save {filename}", "fail", f"Frame is blank (mean={mean_pixel:.1f})")
            return False

        self.log(f"Save {filename}", "pass", f"{frame.shape[1]}x{frame.shape[0]}, mean={mean_pixel:.1f}")
        return True

    def run(self) -> bool:
        """Run the integration test."""
        print("=" * 60)
        print("RENDERER INTEGRATION TEST — Caption → Frame Pipeline")
        print("=" * 60)
        print(f"Output: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        all_passed = True

        # ========================================
        # TEST 1: Basic Pipeline Wiring
        # ========================================
        print("\n[1/5] Testing Basic Pipeline Wiring...")

        config = MeetingSessionConfig(
            session_id="integration-test-1",
            display_mode=DisplayMode.SUBTITLE,
        )
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(
            config=config,
            caption_buffer=buffer,
            use_virtual_cam=False,  # Don't need actual device
        )

        # Start rendering
        renderer.start_rendering()
        self.log("Renderer started", "pass", f"FPS={renderer._fps}")

        # Add a caption
        buffer.add_caption(
            translated_text="This is a test caption for integration testing.",
            original_text="这是集成测试的测试字幕。",
            speaker_name="Integration Test",
        )

        # Wait for frame
        time.sleep(0.3)

        if not self.save_frame(renderer, "01-basic-wiring.png"):
            all_passed = False

        # Verify frames_rendered increased
        if renderer.frames_rendered > 0:
            self.log("Frames rendered", "pass", f"{renderer.frames_rendered} frames")
        else:
            self.log("Frames rendered", "fail", "No frames rendered")
            all_passed = False

        renderer.stop_rendering()

        # ========================================
        # TEST 2: Caption Event Flow
        # ========================================
        print("\n[2/5] Testing Caption Event Flow...")

        config2 = MeetingSessionConfig(
            session_id="integration-test-2",
            display_mode=DisplayMode.SUBTITLE,
        )
        buffer2 = CaptionBuffer()
        renderer2 = PILVirtualCamRenderer(
            config=config2,
            caption_buffer=buffer2,
            use_virtual_cam=False,
        )

        renderer2.start_rendering()

        # Add multiple captions rapidly
        for i in range(3):
            buffer2.add_caption(
                translated_text=f"Caption {i+1}: Testing rapid caption addition.",
                original_text=f"字幕 {i+1}",
                speaker_name=f"Speaker {chr(65+i)}",
            )
            time.sleep(0.1)

        time.sleep(0.2)

        # Check that captions made it to webcam manager
        translation_count = len(renderer2._webcam_manager.current_translations)
        if translation_count >= 1:
            self.log("Captions synced", "pass", f"{translation_count} in renderer")
        else:
            self.log("Captions synced", "fail", "No captions reached renderer")
            all_passed = False

        if not self.save_frame(renderer2, "02-event-flow.png"):
            all_passed = False

        renderer2.stop_rendering()

        # ========================================
        # TEST 3: Config Snapshot Consistency
        # ========================================
        print("\n[3/5] Testing Config Snapshot...")

        config3 = MeetingSessionConfig(
            session_id="integration-test-3",
            display_mode=DisplayMode.SUBTITLE,
            font_size=24,
        )
        buffer3 = CaptionBuffer()
        renderer3 = PILVirtualCamRenderer(
            config=config3,
            caption_buffer=buffer3,
            use_virtual_cam=False,
        )

        renderer3.start_rendering()

        buffer3.add_caption(
            translated_text="Before config change",
            original_text="配置更改前",
            speaker_name="Config Test",
        )
        time.sleep(0.2)
        self.save_frame(renderer3, "03a-before-config.png")

        # Change config mid-render
        config3.update(font_size=48)
        time.sleep(0.2)

        # Verify snapshot was taken
        snapshot = renderer3.last_config_snapshot
        if snapshot.get("font_size") == 48:
            self.log("Config snapshot", "pass", f"font_size={snapshot.get('font_size')}")
        else:
            self.log("Config snapshot", "fail", f"Expected 48, got {snapshot.get('font_size')}")
            all_passed = False

        self.save_frame(renderer3, "03b-after-config.png")
        renderer3.stop_rendering()

        # ========================================
        # TEST 4: Speaker Color Assignment
        # ========================================
        print("\n[4/5] Testing Speaker Colors...")

        config4 = MeetingSessionConfig(
            session_id="integration-test-4",
            display_mode=DisplayMode.SUBTITLE,
        )
        buffer4 = CaptionBuffer()
        renderer4 = PILVirtualCamRenderer(
            config=config4,
            caption_buffer=buffer4,
            use_virtual_cam=False,
        )

        renderer4.start_rendering()

        # Assign specific colors
        buffer4.set_speaker_color("Alice", "#FF5722")
        buffer4.set_speaker_color("Bob", "#2196F3")

        buffer4.add_caption(
            translated_text="Hello from Alice!",
            original_text="爱丽丝说你好！",
            speaker_name="Alice",
        )
        buffer4.add_caption(
            translated_text="Hi Alice, this is Bob!",
            original_text="嗨爱丽丝，我是鲍勃！",
            speaker_name="Bob",
        )

        time.sleep(0.2)

        # Verify colors were assigned
        alice_color = buffer4.get_speaker_color("Alice")
        bob_color = buffer4.get_speaker_color("Bob")

        if alice_color == "#FF5722" and bob_color == "#2196F3":
            self.log("Speaker colors", "pass", f"Alice={alice_color}, Bob={bob_color}")
        else:
            self.log("Speaker colors", "fail", f"Alice={alice_color}, Bob={bob_color}")
            all_passed = False

        self.save_frame(renderer4, "04-speaker-colors.png")
        renderer4.stop_rendering()

        # ========================================
        # TEST 5: Display Mode Switching
        # ========================================
        print("\n[5/5] Testing Display Modes...")

        for mode in [DisplayMode.SUBTITLE, DisplayMode.SPLIT, DisplayMode.INTERPRETER]:
            config5 = MeetingSessionConfig(
                session_id=f"integration-test-5-{mode.value}",
                display_mode=mode,
            )
            buffer5 = CaptionBuffer()
            renderer5 = PILVirtualCamRenderer(
                config=config5,
                caption_buffer=buffer5,
                use_virtual_cam=False,
            )

            renderer5.start_rendering()

            buffer5.add_caption(
                translated_text=f"Testing {mode.value} display mode.",
                original_text="测试显示模式",
                speaker_name="Mode Test",
            )

            time.sleep(0.2)

            if not self.save_frame(renderer5, f"05-mode-{mode.value}.png"):
                all_passed = False

            renderer5.stop_rendering()
            self.log(f"Mode {mode.value}", "pass", "Frame generated")

        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for s in self.results["steps"] if s["status"] == "pass")
        failed = sum(1 for s in self.results["steps"] if s["status"] == "fail")

        print(f"Steps: {passed} passed, {failed} failed")
        print(f"Frames saved: {len(self.results['frames_saved'])}")
        print(f"Output: {OUTPUT_DIR}")

        # List frames
        print("\nGenerated frames:")
        for f in sorted(OUTPUT_DIR.glob("*.png")):
            print(f"  - {f.name} ({f.stat().st_size:,} bytes)")

        return all_passed


def main():
    test = RendererIntegrationTest()
    success = test.run()
    print(f"\n{'✓ All tests passed' if success else '✗ Some tests failed'}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
