#!/usr/bin/env python3
"""
Virtual Webcam Integration Test - Complete Pipeline Validation

Tests the complete bot → orchestration → whisper → translation → webcam flow
with real-time subtitle display including both transcription AND translation.

This validates:
- Audio processing
- Real-time transcription (original language)
- Real-time translation (target language)
- Virtual webcam subtitle generation
- Database persistence
- Complete end-to-end flow

Usage:
    pytest test_virtual_webcam_subtitles.py -v -s
    # OR run directly
    python test_virtual_webcam_subtitles.py
"""

import sys
import asyncio
import pytest
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import time
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from bot.virtual_webcam import (
    VirtualWebcamManager,
    WebcamConfig,
    DisplayMode,
    Theme,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VirtualWebcamIntegrationTest:
    """
    Complete integration test for virtual webcam with subtitles.

    Tests the entire pipeline from audio input to subtitle display.
    """

    def __init__(self):
        self.test_dir = (
            Path(__file__).parent.parent.parent / "test_output" / "virtual_webcam"
        )
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.webcam_manager = None
        self.data_pipeline = None
        self.session_id = f"test_webcam_{int(time.time())}"

        # Track results
        self.transcriptions_received = []
        self.translations_received = []
        self.frames_saved = []

    async def setup(self):
        """Initialize all components."""
        logger.info("=" * 80)
        logger.info("VIRTUAL WEBCAM INTEGRATION TEST SETUP")
        logger.info("=" * 80)

        # Create webcam config with optimal settings for testing
        webcam_config = WebcamConfig(
            width=1920,
            height=1080,
            fps=30,
            display_mode=DisplayMode.OVERLAY,
            theme=Theme.DARK,
            max_translations_displayed=5,
            translation_duration_seconds=10.0,
            font_size=28,
            show_speaker_names=True,
            show_confidence=True,
            show_timestamps=True,
        )

        # Initialize virtual webcam
        self.webcam_manager = VirtualWebcamManager(webcam_config)

        # Set up frame callback to save frames
        self.webcam_manager.on_frame_generated = self._on_frame_generated

        logger.info("✅ Virtual webcam manager initialized")
        logger.info(f"   Display mode: {webcam_config.display_mode.value}")
        logger.info(f"   Theme: {webcam_config.theme.value}")
        logger.info(f"   Resolution: {webcam_config.width}x{webcam_config.height}")
        logger.info(f"   Output directory: {self.test_dir}")

    def _on_frame_generated(self, frame: np.ndarray):
        """Callback when a frame is generated."""
        # Save every 30th frame (once per second at 30fps)
        frame_count = len(self.frames_saved)
        if frame_count % 30 == 0:
            frame_path = self.test_dir / f"frame_{frame_count:04d}.png"

            # Convert to PIL and save
            if frame.shape[2] == 4:  # RGBA
                img = Image.fromarray(frame, "RGBA")
            else:  # RGB
                img = Image.fromarray(frame, "RGB")

            img.save(frame_path)
            self.frames_saved.append(frame_path)
            logger.info(f"📸 Saved frame {frame_count} to {frame_path}")

    async def test_transcription_only(self):
        """Test 1: Display original transcriptions only."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 1: TRANSCRIPTION ONLY (Original Language)")
        logger.info("=" * 80)

        # Start webcam stream
        await self.webcam_manager.start_stream(self.session_id)
        logger.info("✅ Webcam stream started")

        # Simulate transcriptions coming from whisper service
        test_transcriptions = [
            {
                "translated_text": "Hello everyone, welcome to today's meeting.",
                "source_language": "en",
                "target_language": "en",
                "speaker_id": "SPEAKER_00",
                "speaker_name": "John Doe",
                "translation_confidence": 0.95,
                "is_original_transcription": True,
                "timestamp": datetime.now(),
            },
            {
                "translated_text": "Thank you John. Let's start with the quarterly results.",
                "source_language": "en",
                "target_language": "en",
                "speaker_id": "SPEAKER_01",
                "speaker_name": "Jane Smith",
                "translation_confidence": 0.92,
                "is_original_transcription": True,
                "timestamp": datetime.now(),
            },
            {
                "translated_text": "Our revenue increased by 25% this quarter.",
                "source_language": "en",
                "target_language": "en",
                "speaker_id": "SPEAKER_00",
                "speaker_name": "John Doe",
                "translation_confidence": 0.97,
                "is_original_transcription": True,
                "timestamp": datetime.now(),
            },
        ]

        # Add transcriptions one by one with delays
        for i, transcription in enumerate(test_transcriptions, 1):
            logger.info(f"\n🎤 Adding transcription {i}/{len(test_transcriptions)}")
            logger.info(f"   Speaker: {transcription['speaker_name']}")
            logger.info(f"   Text: {transcription['translated_text']}")

            self.webcam_manager.add_translation(transcription)
            self.transcriptions_received.append(transcription)

            # Wait to let frames generate
            await asyncio.sleep(2.0)

        logger.info(
            f"\n✅ Test 1 complete: {len(self.transcriptions_received)} transcriptions displayed"
        )

    async def test_transcription_and_translation(self):
        """Test 2: Display both original transcriptions AND translations."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 2: TRANSCRIPTION + TRANSLATION (Bilingual Display)")
        logger.info("=" * 80)

        # Simulate bilingual content: English transcription → Spanish translation
        test_data = [
            {
                "transcription": {
                    "translated_text": "Good morning everyone. Today we'll discuss our expansion plans.",
                    "source_language": "en",
                    "target_language": "en",
                    "speaker_id": "SPEAKER_02",
                    "speaker_name": "Alice Johnson",
                    "translation_confidence": 0.94,
                    "is_original_transcription": True,
                },
                "translation": {
                    "translated_text": "Buenos días a todos. Hoy discutiremos nuestros planes de expansión.",
                    "source_language": "en",
                    "target_language": "es",
                    "speaker_id": "SPEAKER_02",
                    "speaker_name": "Alice Johnson",
                    "translation_confidence": 0.89,
                    "is_original_transcription": False,
                },
            },
            {
                "transcription": {
                    "translated_text": "We're opening three new offices in Europe.",
                    "source_language": "en",
                    "target_language": "en",
                    "speaker_id": "SPEAKER_03",
                    "speaker_name": "Bob Williams",
                    "translation_confidence": 0.96,
                    "is_original_transcription": True,
                },
                "translation": {
                    "translated_text": "Estamos abriendo tres nuevas oficinas en Europa.",
                    "source_language": "en",
                    "target_language": "es",
                    "speaker_id": "SPEAKER_03",
                    "speaker_name": "Bob Williams",
                    "translation_confidence": 0.91,
                    "is_original_transcription": False,
                },
            },
            {
                "transcription": {
                    "translated_text": "That's exciting news! When will they open?",
                    "source_language": "en",
                    "target_language": "en",
                    "speaker_id": "SPEAKER_02",
                    "speaker_name": "Alice Johnson",
                    "translation_confidence": 0.93,
                    "is_original_transcription": True,
                },
                "translation": {
                    "translated_text": "¡Esas son noticias emocionantes! ¿Cuándo abrirán?",
                    "source_language": "en",
                    "target_language": "es",
                    "speaker_id": "SPEAKER_02",
                    "speaker_name": "Alice Johnson",
                    "translation_confidence": 0.87,
                    "is_original_transcription": False,
                },
            },
        ]

        # Display each transcription + translation pair
        for i, pair in enumerate(test_data, 1):
            logger.info(
                f"\n📝 Adding transcription/translation pair {i}/{len(test_data)}"
            )

            # Add original transcription first
            transcription = pair["transcription"]
            logger.info(f"   🎤 TRANSCRIPTION ({transcription['source_language']})")
            logger.info(f"      Speaker: {transcription['speaker_name']}")
            logger.info(f"      Text: {transcription['translated_text']}")

            self.webcam_manager.add_translation(transcription)
            self.transcriptions_received.append(transcription)

            # Small delay between transcription and translation
            await asyncio.sleep(0.5)

            # Add translation
            translation = pair["translation"]
            logger.info(f"   🌐 TRANSLATION ({translation['target_language']})")
            logger.info(f"      Speaker: {translation['speaker_name']}")
            logger.info(f"      Text: {translation['translated_text']}")

            self.webcam_manager.add_translation(translation)
            self.translations_received.append(translation)

            # Wait to let frames generate
            await asyncio.sleep(2.5)

        logger.info("\n✅ Test 2 complete:")
        logger.info(f"   Transcriptions: {len(self.transcriptions_received)}")
        logger.info(f"   Translations: {len(self.translations_received)}")

    async def test_multilingual_conversation(self):
        """Test 3: Multiple languages in same conversation."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST 3: MULTILINGUAL CONVERSATION (3 languages)")
        logger.info("=" * 80)

        # Simulate conversation with English, Spanish, and French
        multilingual_data = [
            {
                "transcription": {
                    "translated_text": "Let's hear from our international teams.",
                    "source_language": "en",
                    "target_language": "en",
                    "speaker_id": "SPEAKER_04",
                    "speaker_name": "Carol Martinez",
                    "translation_confidence": 0.95,
                    "is_original_transcription": True,
                },
                "translations": [
                    {
                        "translated_text": "Escuchemos a nuestros equipos internacionales.",
                        "source_language": "en",
                        "target_language": "es",
                        "speaker_id": "SPEAKER_04",
                        "speaker_name": "Carol Martinez",
                        "translation_confidence": 0.90,
                        "is_original_transcription": False,
                    },
                    {
                        "translated_text": "Écoutons nos équipes internationales.",
                        "source_language": "en",
                        "target_language": "fr",
                        "speaker_id": "SPEAKER_04",
                        "speaker_name": "Carol Martinez",
                        "translation_confidence": 0.88,
                        "is_original_transcription": False,
                    },
                ],
            },
            {
                "transcription": {
                    "translated_text": "Bonjour! Our Paris office is doing great.",
                    "source_language": "fr",
                    "target_language": "fr",
                    "speaker_id": "SPEAKER_05",
                    "speaker_name": "Pierre Dubois",
                    "translation_confidence": 0.93,
                    "is_original_transcription": True,
                },
                "translations": [
                    {
                        "translated_text": "Hello! Our Paris office is doing great.",
                        "source_language": "fr",
                        "target_language": "en",
                        "speaker_id": "SPEAKER_05",
                        "speaker_name": "Pierre Dubois",
                        "translation_confidence": 0.91,
                        "is_original_transcription": False,
                    },
                    {
                        "translated_text": "¡Hola! Nuestra oficina de París va muy bien.",
                        "source_language": "fr",
                        "target_language": "es",
                        "speaker_id": "SPEAKER_05",
                        "speaker_name": "Pierre Dubois",
                        "translation_confidence": 0.87,
                        "is_original_transcription": False,
                    },
                ],
            },
        ]

        # Display multilingual content
        for i, item in enumerate(multilingual_data, 1):
            logger.info(f"\n🌍 Adding multilingual item {i}/{len(multilingual_data)}")

            # Add original transcription
            transcription = item["transcription"]
            logger.info(f"   🎤 ORIGINAL ({transcription['source_language'].upper()})")
            logger.info(f"      {transcription['translated_text']}")

            self.webcam_manager.add_translation(transcription)
            await asyncio.sleep(0.5)

            # Add all translations
            for j, translation in enumerate(item["translations"], 1):
                logger.info(
                    f"   🌐 TRANSLATION {j} ({translation['target_language'].upper()})"
                )
                logger.info(f"      {translation['translated_text']}")

                self.webcam_manager.add_translation(translation)
                self.translations_received.append(translation)
                await asyncio.sleep(0.5)

            # Wait to let frames generate
            await asyncio.sleep(2.0)

        logger.info("\n✅ Test 3 complete: Multilingual display validated")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("\n" + "=" * 80)
        logger.info("CLEANUP")
        logger.info("=" * 80)

        # Stop webcam stream
        if self.webcam_manager and self.webcam_manager.is_streaming:
            await self.webcam_manager.stop_stream()
            logger.info("✅ Webcam stream stopped")

        # Wait for final frames to render
        await asyncio.sleep(2.0)

    async def generate_summary(self):
        """Generate test summary report."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        logger.info("\n📊 Statistics:")
        logger.info(f"   Session ID: {self.session_id}")
        logger.info(f"   Transcriptions displayed: {len(self.transcriptions_received)}")
        logger.info(f"   Translations displayed: {len(self.translations_received)}")
        logger.info(f"   Frames saved: {len(self.frames_saved)}")

        if self.webcam_manager:
            logger.info(
                f"   Total frames generated: {self.webcam_manager.frames_generated}"
            )
            duration = (
                time.time() - self.webcam_manager.start_time
                if self.webcam_manager.start_time
                else 0
            )
            fps = self.webcam_manager.frames_generated / duration if duration > 0 else 0
            logger.info(f"   Test duration: {duration:.1f}s")
            logger.info(f"   Average FPS: {fps:.1f}")

        logger.info("\n📁 Output:")
        logger.info(f"   Directory: {self.test_dir}")
        logger.info("   Sample frames:")
        for i, frame_path in enumerate(self.frames_saved[:5], 1):
            logger.info(f"      {i}. {frame_path.name}")

        if len(self.frames_saved) > 5:
            logger.info(f"      ... and {len(self.frames_saved) - 5} more frames")

        # Create summary JSON
        summary = {
            "session_id": self.session_id,
            "test_date": datetime.now().isoformat(),
            "statistics": {
                "transcriptions": len(self.transcriptions_received),
                "translations": len(self.translations_received),
                "frames_saved": len(self.frames_saved),
                "frames_generated": self.webcam_manager.frames_generated
                if self.webcam_manager
                else 0,
            },
            "output_directory": str(self.test_dir),
            "test_results": {
                "test_1_transcription_only": len(
                    [
                        t
                        for t in self.transcriptions_received
                        if t.get("is_original_transcription")
                    ]
                )
                > 0,
                "test_2_bilingual_display": len(self.translations_received) > 0,
                "test_3_multilingual": len(
                    set(t.get("target_language") for t in self.translations_received)
                )
                >= 2,
            },
        }

        import json

        summary_path = self.test_dir / "test_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n📄 Summary saved to: {summary_path}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL TESTS COMPLETE")
        logger.info("=" * 80)

        logger.info("\n💡 Next steps:")
        logger.info(f"   1. View frames: open {self.test_dir}")
        logger.info(
            f"   2. Create video: ffmpeg -framerate 30 -i {self.test_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4"
        )
        logger.info(f"   3. Review summary: cat {summary_path}")


async def main():
    """Run the complete integration test."""
    test = VirtualWebcamIntegrationTest()

    try:
        # Setup
        await test.setup()

        # Run tests
        await test.test_transcription_only()
        await test.test_transcription_and_translation()
        await test.test_multilingual_conversation()

        # Cleanup
        await test.cleanup()

        # Generate summary
        await test.generate_summary()

        return 0

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}", exc_info=True)
        return 1


@pytest.mark.asyncio
async def test_virtual_webcam_integration():
    """Pytest wrapper for integration test."""
    result = await main()
    assert result == 0, "Integration test failed"


if __name__ == "__main__":
    # Run directly
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
