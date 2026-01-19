#!/usr/bin/env python3
"""
Virtual Webcam Live Demo - Real-time Subtitle Display

A simple standalone demo that shows the virtual webcam system displaying
both original transcriptions AND translations in real-time.

This demonstrates:
- Real-time subtitle generation
- Bilingual display (transcription + translation)
- Speaker attribution
- Professional overlay rendering

No database or external services required - pure demonstration.

Usage:
    python demo_virtual_webcam_live.py

Output:
    - Real-time console logging
    - Saved frames in test_output/virtual_webcam_demo/
    - Can be converted to video with ffmpeg
"""

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bot.virtual_webcam import (
    DisplayMode,
    Theme,
    VirtualWebcamManager,
    WebcamConfig,
)

# Configure logging with colors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class VirtualWebcamLiveDemo:
    """
    Live demonstration of virtual webcam with real-time subtitles.
    """

    def __init__(self):
        self.output_dir = Path(__file__).parent / "test_output" / "virtual_webcam_demo"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.webcam_manager = None
        self.session_id = f"demo_{int(time.time())}"
        self.frames_saved = []

    def print_banner(self, text: str):
        """Print a banner for visual clarity."""
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80 + "\n")

    async def setup(self):
        """Initialize virtual webcam."""
        self.print_banner("🎥 LIVETRANSLATE VIRTUAL WEBCAM - LIVE DEMO")

        print("📋 Configuration:")
        print("   Display Mode: OVERLAY (transparent background with floating boxes)")
        print("   Theme: DARK")
        print("   Resolution: 1920x1080 @ 30fps")
        print("   Max Subtitles: 5 simultaneous")
        print("   Duration: 10 seconds per subtitle")
        print(f"   Output: {self.output_dir}\n")

        # Create webcam config
        webcam_config = WebcamConfig(
            width=1920,
            height=1080,
            fps=30,
            display_mode=DisplayMode.OVERLAY,
            theme=Theme.DARK,
            max_translations_displayed=5,
            translation_duration_seconds=10.0,
            font_size=32,
            show_speaker_names=True,
            show_confidence=True,
            show_timestamps=False,
        )

        # Initialize virtual webcam
        self.webcam_manager = VirtualWebcamManager(webcam_config)

        # Set up frame callback
        self.webcam_manager.on_frame_generated = self._on_frame_generated

        print("✅ Virtual webcam initialized\n")

    def _on_frame_generated(self, frame: np.ndarray):
        """Save periodic frames."""
        frame_count = len(self.frames_saved)

        # Save every 30th frame (1 per second)
        if frame_count % 30 == 0:
            frame_path = self.output_dir / f"frame_{frame_count:04d}.png"

            # Convert and save
            if frame.shape[2] == 4:  # RGBA
                img = Image.fromarray(frame, "RGBA")
            else:  # RGB
                img = Image.fromarray(frame, "RGB")

            img.save(frame_path)
            self.frames_saved.append(frame_path)

    async def run_demo(self):
        """Run the live demonstration."""

        # Start webcam stream
        await self.webcam_manager.start_stream(self.session_id)
        print("🎬 Webcam stream started!\n")

        # === SCENE 1: English Meeting Opening ===
        self.print_banner("SCENE 1: Meeting Opening (English → Spanish)")

        await self._add_subtitle_pair(
            transcription="Good morning everyone! Welcome to today's quarterly review.",
            translation="¡Buenos días a todos! Bienvenidos a la revisión trimestral de hoy.",
            speaker_id="SPEAKER_00",
            speaker_name="Sarah Chen",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(2.0)

        await self._add_subtitle_pair(
            transcription="Let's start by reviewing our key achievements this quarter.",
            translation="Comencemos revisando nuestros logros clave este trimestre.",
            speaker_id="SPEAKER_00",
            speaker_name="Sarah Chen",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(2.5)

        # === SCENE 2: Bilingual Discussion ===
        self.print_banner("SCENE 2: Bilingual Discussion (English ↔ French)")

        await self._add_subtitle_pair(
            transcription="Our revenue increased by 35% compared to last quarter.",
            translation="Notre chiffre d'affaires a augmenté de 35% par rapport au trimestre dernier.",
            speaker_id="SPEAKER_01",
            speaker_name="Michael Rodriguez",
            source_lang="en",
            target_lang="fr",
        )

        await asyncio.sleep(2.0)

        await self._add_subtitle_pair(
            transcription="Excellent! What about our European markets?",
            translation="Excellent! Qu'en est-il de nos marchés européens?",
            speaker_id="SPEAKER_02",
            speaker_name="Emma Wilson",
            source_lang="en",
            target_lang="fr",
        )

        await asyncio.sleep(2.5)

        # === SCENE 3: Multilingual Team ===
        self.print_banner("SCENE 3: International Team (Multiple Languages)")

        await self._add_subtitle_pair(
            transcription="Our Paris office had outstanding results this quarter.",
            translation="Notre bureau de Paris a obtenu des résultats exceptionnels ce trimestre.",
            speaker_id="SPEAKER_03",
            speaker_name="Jean Dupont",
            source_lang="en",
            target_lang="fr",
        )

        await asyncio.sleep(1.5)

        await self._add_subtitle_pair(
            transcription="The Madrid team exceeded all targets.",
            translation="El equipo de Madrid superó todos los objetivos.",
            speaker_id="SPEAKER_04",
            speaker_name="Carlos Martínez",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(1.5)

        await self._add_subtitle_pair(
            transcription="And Tokyo launched three major client projects.",
            translation="そして東京は3つの大規模クライアントプロジェクトを立ち上げました。",
            speaker_id="SPEAKER_05",
            speaker_name="Yuki Tanaka",
            source_lang="en",
            target_lang="ja",
        )

        await asyncio.sleep(2.5)

        # === SCENE 4: Rapid Conversation ===
        self.print_banner("SCENE 4: Rapid Team Discussion")

        await self._add_subtitle_pair(
            transcription="That's incredible progress team!",
            translation="¡Eso es un progreso increíble equipo!",
            speaker_id="SPEAKER_00",
            speaker_name="Sarah Chen",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(1.0)

        await self._add_subtitle_pair(
            transcription="What's our focus for next quarter?",
            translation="¿Cuál es nuestro enfoque para el próximo trimestre?",
            speaker_id="SPEAKER_01",
            speaker_name="Michael Rodriguez",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(1.0)

        await self._add_subtitle_pair(
            transcription="We're expanding into three new markets.",
            translation="Nos estamos expandiendo a tres nuevos mercados.",
            speaker_id="SPEAKER_00",
            speaker_name="Sarah Chen",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(1.0)

        await self._add_subtitle_pair(
            transcription="Exciting! Let's discuss the strategy.",
            translation="¡Emocionante! Discutamos la estrategia.",
            speaker_id="SPEAKER_02",
            speaker_name="Emma Wilson",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(3.0)

        # === SCENE 5: Closing ===
        self.print_banner("SCENE 5: Meeting Closing")

        await self._add_subtitle_pair(
            transcription="Thank you all for your incredible work this quarter.",
            translation="Gracias a todos por su increíble trabajo este trimestre.",
            speaker_id="SPEAKER_00",
            speaker_name="Sarah Chen",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(2.0)

        await self._add_subtitle_pair(
            transcription="Looking forward to an even better next quarter!",
            translation="¡Esperamos un trimestre aún mejor!",
            speaker_id="SPEAKER_00",
            speaker_name="Sarah Chen",
            source_lang="en",
            target_lang="es",
        )

        await asyncio.sleep(3.0)

        # Let final subtitles display
        print("\n⏳ Waiting for final subtitles to display...")
        await asyncio.sleep(5.0)

    async def _add_subtitle_pair(
        self,
        transcription: str,
        translation: str,
        speaker_id: str,
        speaker_name: str,
        source_lang: str,
        target_lang: str,
    ):
        """Add a transcription/translation pair to display."""

        # Add original transcription
        transcription_data = {
            "translated_text": transcription,
            "source_language": source_lang,
            "target_language": source_lang,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "translation_confidence": 0.95,
            "is_original_transcription": True,
            "timestamp": datetime.now(),
        }

        print(f"🎤 [{speaker_name}] ({source_lang.upper()})")
        print(f"   {transcription}")

        self.webcam_manager.add_translation(transcription_data)

        # Small delay
        await asyncio.sleep(0.3)

        # Add translation
        translation_data = {
            "translated_text": translation,
            "source_language": source_lang,
            "target_language": target_lang,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "translation_confidence": 0.88,
            "is_original_transcription": False,
            "timestamp": datetime.now(),
        }

        print(f"🌐 [{speaker_name}] ({source_lang.upper()} → {target_lang.upper()})")
        print(f"   {translation}\n")

        self.webcam_manager.add_translation(translation_data)

    async def cleanup(self):
        """Stop stream and cleanup."""
        print("\n⏹️  Stopping webcam stream...")

        if self.webcam_manager and self.webcam_manager.is_streaming:
            await self.webcam_manager.stop_stream()

        # Wait for final frames
        await asyncio.sleep(1.0)

    def generate_summary(self):
        """Print summary and next steps."""
        self.print_banner("✅ DEMO COMPLETE")

        print("📊 Results:")
        print(f"   Frames saved: {len(self.frames_saved)}")
        print(f"   Output directory: {self.output_dir}")

        if self.webcam_manager:
            duration = (
                time.time() - self.webcam_manager.start_time
                if self.webcam_manager.start_time
                else 0
            )
            fps = self.webcam_manager.frames_generated / duration if duration > 0 else 0
            print(f"   Total frames generated: {self.webcam_manager.frames_generated}")
            print(f"   Duration: {duration:.1f}s")
            print(f"   Average FPS: {fps:.1f}")

        print("\n📁 Saved Frames:")
        for i, frame_path in enumerate(self.frames_saved[:10], 1):
            print(f"   {i}. {frame_path.name}")

        if len(self.frames_saved) > 10:
            print(f"   ... and {len(self.frames_saved) - 10} more frames")

        print("\n🎬 Create Video:")
        print(f"   cd {self.output_dir}")
        print(
            "   ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p -vf 'scale=1920:1080' demo_output.mp4"
        )

        print("\n💡 What You'll See:")
        print("   - Original transcriptions (🎤) in source language")
        print("   - Real-time translations (🌐) in target language")
        print("   - Speaker names with attribution")
        print("   - Confidence scores")
        print("   - Professional dark theme overlay")
        print("   - Multiple speakers with color coding")
        print("   - Smooth fade-in/fade-out (10 second display duration)")

        print("\n" + "=" * 80)
        print("  Thank you for using LiveTranslate Virtual Webcam!")
        print("=" * 80 + "\n")


async def main():
    """Run the live demo."""
    demo = VirtualWebcamLiveDemo()

    try:
        # Setup
        await demo.setup()

        # Run demonstration
        await demo.run_demo()

        # Cleanup
        await demo.cleanup()

        # Show summary
        demo.generate_summary()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        await demo.cleanup()
        return 1

    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("\n🚀 Starting LiveTranslate Virtual Webcam Demo...\n")
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
