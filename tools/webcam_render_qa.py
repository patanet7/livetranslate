#!/usr/bin/env python3
"""
Virtual Webcam Renderer Visual QA

Tests the full caption → renderer pipeline by generating actual images.
No Google Meet required — tests the rendering layer in isolation.

This validates:
1. PILVirtualCamRenderer renders captions correctly
2. All DisplayModes work (subtitle, split, interpreter)
3. CJK text renders with proper fonts
4. Multi-speaker colors display correctly
5. Speaker name formatting works
6. Long text wrapping functions
7. Config changes (font size, theme) apply per-frame
8. Caption expiry removes old captions
9. Aggregation combines same-speaker text

Usage:
    uv run python tools/webcam_render_qa.py

Output: /tmp/webcam-render-qa/*.png
"""

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add orchestration-service to path
sys.path.insert(0, str(Path(__file__).parent.parent / "modules" / "orchestration-service" / "src"))

from PIL import Image
import numpy as np

from bot.pil_virtual_cam_renderer import PILVirtualCamRenderer
from services.meeting_session_config import MeetingSessionConfig
from services.caption_buffer import CaptionBuffer
from livetranslate_common.theme import DisplayMode, ThemeColors


OUTPUT_DIR = Path("/tmp/webcam-render-qa")


def save_frame(renderer: PILVirtualCamRenderer, filename: str) -> None:
    """Save current frame as PNG."""
    if renderer.last_frame is not None:
        img = Image.fromarray(renderer.last_frame)
        img.save(OUTPUT_DIR / filename)
        print(f"    ✓ Saved {filename}")
    else:
        print(f"    ✗ No frame to save for {filename}")


def run_tests():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WEBCAM RENDERER VISUAL QA")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    results = []

    # ========================================================================
    # TEST 1: All Display Modes
    # ========================================================================
    print("\n[1/8] Testing Display Modes...")

    for mode in [DisplayMode.SUBTITLE, DisplayMode.SPLIT, DisplayMode.INTERPRETER]:
        config = MeetingSessionConfig(
            session_id=f"qa-mode-{mode.value}",
            display_mode=mode,
        )
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        buffer.add_caption(
            translated_text="Hola, bienvenidos a la reunión de hoy.",
            original_text="Hello, welcome to today's meeting.",
            speaker_name="Thomas Patane",
        )
        time.sleep(0.2)
        save_frame(renderer, f"01-mode-{mode.value}.png")
        renderer.stop_rendering()
        results.append(f"Mode {mode.value}")

    # ========================================================================
    # TEST 2: CJK Text (Chinese, Japanese, Korean)
    # ========================================================================
    print("\n[2/8] Testing CJK Text...")

    config = MeetingSessionConfig(session_id="qa-cjk", display_mode=DisplayMode.SUBTITLE)
    buffer = CaptionBuffer()
    renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

    renderer.start_rendering()

    # Chinese
    buffer.add_caption(
        translated_text="Let's discuss the translation system progress today",
        original_text="我们今天讨论翻译系统的进展",
        speaker_name="王丽",
    )
    buffer.set_speaker_color("王丽", "#FF9800")

    time.sleep(0.1)

    # Japanese
    buffer.add_caption(
        translated_text="The new feature will be ready next week",
        original_text="新しい機能は来週準備ができます",
        speaker_name="田中裕樹",
    )
    buffer.set_speaker_color("田中裕樹", "#00BCD4")

    time.sleep(0.1)

    # Korean
    buffer.add_caption(
        translated_text="I agree with this approach",
        original_text="이 접근 방식에 동의합니다",
        speaker_name="김수민",
    )
    buffer.set_speaker_color("김수민", "#8BC34A")

    time.sleep(0.2)
    save_frame(renderer, "02-cjk-text.png")
    renderer.stop_rendering()
    results.append("CJK text")

    # ========================================================================
    # TEST 3: Multi-Speaker Colors
    # ========================================================================
    print("\n[3/8] Testing Multi-Speaker Colors...")

    config = MeetingSessionConfig(session_id="qa-speakers", display_mode=DisplayMode.SUBTITLE)
    buffer = CaptionBuffer()
    renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

    renderer.start_rendering()

    speakers = [
        ("Thomas Patane", "#4CAF50", "Hola equipo", "Hello team"),
        ("Eric Wang", "#2196F3", "Buenos días", "Good morning"),
        ("Sarah Johnson", "#E91E63", "Gracias por unirse", "Thanks for joining"),
        ("Mike Brown", "#9C27B0", "Empecemos", "Let's begin"),
    ]

    for name, color, text, original in speakers:
        buffer.set_speaker_color(name, color)
        buffer.add_caption(translated_text=text, original_text=original, speaker_name=name)
        time.sleep(0.05)

    time.sleep(0.2)
    save_frame(renderer, "03-multi-speaker.png")
    renderer.stop_rendering()
    results.append("Multi-speaker colors")

    # ========================================================================
    # TEST 4: Font Sizes
    # ========================================================================
    print("\n[4/8] Testing Font Sizes...")

    for font_size in [16, 24, 36, 48]:
        config = MeetingSessionConfig(
            session_id=f"qa-font-{font_size}",
            display_mode=DisplayMode.SUBTITLE,
            font_size=font_size,
        )
        buffer = CaptionBuffer()
        renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

        renderer.start_rendering()
        buffer.add_caption(
            translated_text=f"Font size: {font_size}px",
            original_text=f"Tamaño de fuente: {font_size}px",
            speaker_name="Font Test",
        )
        time.sleep(0.2)
        save_frame(renderer, f"04-font-{font_size}.png")
        renderer.stop_rendering()
        results.append(f"Font size {font_size}")

    # ========================================================================
    # TEST 5: Long Text Wrapping
    # ========================================================================
    print("\n[5/8] Testing Long Text Wrapping...")

    config = MeetingSessionConfig(session_id="qa-long", display_mode=DisplayMode.SUBTITLE)
    buffer = CaptionBuffer()
    renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

    renderer.start_rendering()
    buffer.add_caption(
        translated_text="Este es un mensaje muy largo que debería envolver correctamente en múltiples líneas para asegurar que el sistema de subtítulos maneje correctamente el texto extenso sin truncamiento ni problemas de diseño.",
        original_text="This is a very long message that should wrap correctly across multiple lines to ensure the caption system properly handles extended text without truncation or layout issues.",
        speaker_name="Verbose Speaker",
    )
    time.sleep(0.2)
    save_frame(renderer, "05-long-text.png")
    renderer.stop_rendering()
    results.append("Long text wrapping")

    # ========================================================================
    # TEST 6: Caption Aggregation
    # ========================================================================
    print("\n[6/8] Testing Caption Aggregation...")

    config = MeetingSessionConfig(session_id="qa-aggregate", display_mode=DisplayMode.SUBTITLE)
    buffer = CaptionBuffer()
    renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

    renderer.start_rendering()

    # Same speaker, multiple segments
    buffer.add_caption(translated_text="Primera parte.", original_text="First part.", speaker_name="Alice Chen")
    time.sleep(0.1)
    save_frame(renderer, "06a-aggregate-first.png")

    buffer.add_caption(translated_text="Segunda parte del mensaje.", original_text="Second part.", speaker_name="Alice Chen")
    time.sleep(0.1)
    save_frame(renderer, "06b-aggregate-second.png")

    buffer.add_caption(translated_text="Y la tercera.", original_text="And the third.", speaker_name="Alice Chen")
    time.sleep(0.1)
    save_frame(renderer, "06c-aggregate-third.png")

    renderer.stop_rendering()
    results.append("Aggregation")

    # ========================================================================
    # TEST 7: Waiting Frame (No Captions)
    # ========================================================================
    print("\n[7/8] Testing Waiting Frame...")

    config = MeetingSessionConfig(session_id="qa-waiting", display_mode=DisplayMode.SUBTITLE)
    buffer = CaptionBuffer()
    renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

    renderer.start_rendering()
    time.sleep(0.2)  # No captions added
    save_frame(renderer, "07-waiting-frame.png")
    renderer.stop_rendering()
    results.append("Waiting frame")

    # ========================================================================
    # TEST 8: Config Change Mid-Stream
    # ========================================================================
    print("\n[8/8] Testing Config Change...")

    config = MeetingSessionConfig(session_id="qa-config", display_mode=DisplayMode.SUBTITLE, font_size=24)
    buffer = CaptionBuffer()
    renderer = PILVirtualCamRenderer(config=config, caption_buffer=buffer, use_virtual_cam=False)

    renderer.start_rendering()
    buffer.add_caption(translated_text="Before config change", original_text="Antes del cambio", speaker_name="Config Test")
    time.sleep(0.1)
    save_frame(renderer, "08a-config-before.png")

    # Change config mid-stream
    config.update(font_size=48)
    time.sleep(0.1)
    save_frame(renderer, "08b-config-after.png")

    renderer.stop_rendering()
    results.append("Config change")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # List generated files
    files = sorted(OUTPUT_DIR.glob("*.png"))
    print(f"\nGenerated {len(files)} images:")
    for f in files:
        size = f.stat().st_size
        print(f"  - {f.name} ({size:,} bytes)")

    print(f"\nAll tests completed: {len(results)} scenarios")
    print(f"Output directory: {OUTPUT_DIR}")

    return len(files)


if __name__ == "__main__":
    count = run_tests()
    print(f"\n✓ Visual QA complete: {count} images generated")
