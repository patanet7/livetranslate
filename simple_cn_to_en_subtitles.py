#!/usr/bin/env python3
"""
CHINESE → ENGLISH SUBTITLE DISPLAY

Captures Chinese audio (loopback or mic) and displays English subtitles
in a clean, consistent window.

Flow:
  Chinese Audio → Whisper (transcribe) → Translation (CN→EN) → Display EN subtitles

Usage:
    python simple_cn_to_en_subtitles.py

Requirements:
    pip install pyaudio httpx opencv-python pillow numpy
"""

import asyncio
import io
import json
import logging
import sys
import time
import wave
from pathlib import Path

import cv2
import httpx
import pyaudio

# Add orchestration service to path for virtual webcam
sys.path.insert(0, str(Path(__file__).parent / "modules" / "orchestration-service" / "src"))

from bot.virtual_webcam import DisplayMode, Theme, VirtualWebcamManager, WebcamConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Config
ORCHESTRATION_URL = "http://localhost:3000"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5
FORMAT = pyaudio.paInt16


class LoopbackCapture:
    """Loopback audio capture"""

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.device_index = None

    def find_loopback(self):
        """Find loopback device"""
        logger.info("🔍 Searching for loopback device...")

        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name = info["name"].lower()

            if any(kw in name for kw in ["blackhole", "soundflower", "loopback"]):
                if info["maxInputChannels"] > 0:
                    logger.info(f"✅ Found: {info['name']}")
                    self.device_index = i
                    return True

        logger.warning("⚠️  No loopback - using default mic")
        return False

    def start(self, callback):
        """Start capture"""
        self.find_loopback()

        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=int(SAMPLE_RATE * CHUNK_DURATION),
            stream_callback=callback,
        )
        self.stream.start_stream()

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class SubtitleDisplay:
    """Simple subtitle display"""

    def __init__(self):
        self.config = WebcamConfig(
            width=1920,
            height=1080,
            fps=30,
            display_mode=DisplayMode.OVERLAY,
            theme=Theme.DARK,
            max_translations_displayed=3,  # Show CN original + EN translation
            translation_duration_seconds=20.0,
            font_size=36,
            show_speaker_names=False,
            show_confidence=True,
            show_timestamps=True,
        )

        self.webcam = VirtualWebcamManager(self.config)
        self.session_id = f"cn_en_{int(time.time())}"
        self.window_name = "Chinese → English Subtitles"
        self.display_running = False

    async def start(self):
        """Start display"""
        logger.info("🎥 Starting subtitle display...")

        await self.webcam.start_stream(self.session_id)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        logger.info(f"✅ Display ready: {self.window_name}")
        self.display_running = True

    def update_display(self):
        """Update display"""
        if not self.display_running:
            return

        frame = self.webcam.current_frame
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.window_name, frame_bgr)
            cv2.waitKey(1)

    def add_translation(self, chinese_text: str, english_text: str, confidence: float = 0.9):
        """Add Chinese original + English translation"""

        # Display Chinese original
        self.webcam.add_translation(
            translation_id=f"cn_{int(time.time() * 1000)}",
            text=f"🇨🇳 {chinese_text}",
            source_language="zh",
            target_language="zh",
            speaker_name=None,
            confidence=1.0,
        )

        # Display English translation
        self.webcam.add_translation(
            translation_id=f"en_{int(time.time() * 1000)}",
            text=f"🇺🇸 {english_text}",
            source_language="zh",
            target_language="en",
            speaker_name=None,
            confidence=confidence,
        )

        self.update_display()

    async def stop(self):
        self.display_running = False
        await self.webcam.stop_stream()
        cv2.destroyAllWindows()


class ChineseEnglishSession:
    """CN→EN translation session"""

    def __init__(self):
        self.session_id = f"cn_en_{int(time.time())}"
        self.chunk_count = 0
        self.display = SubtitleDisplay()
        # Background task tracking (prevents fire-and-forget)
        self._background_tasks: set = set()

    async def check_services(self):
        """Check orchestration"""
        logger.info("🔍 Checking services...")

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{ORCHESTRATION_URL}/api/health")
                if resp.status_code == 200:
                    logger.info("✅ Orchestration: READY")
                    return True
                else:
                    logger.error(f"❌ Orchestration: HTTP {resp.status_code}")
                    return False
        except Exception:
            logger.error("❌ Orchestration not running")
            logger.info("💡 Start: cd modules/orchestration-service && python src/main.py")
            return False

    async def process_chunk(self, audio_data: bytes):
        """Process Chinese audio → English subtitles"""
        self.chunk_count += 1
        chunk_id = f"chunk_{self.chunk_count:04d}"

        # Convert to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        wav_buffer.seek(0)

        logger.info(f"📤 Processing chunk {self.chunk_count}...")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                files = {"file": ("audio.wav", wav_buffer.read(), "audio/wav")}
                data = {
                    "chunk_id": chunk_id,
                    "session_id": self.session_id,
                    "target_languages": json.dumps(["en"]),  # Only English
                    "enable_transcription": "true",
                    "enable_translation": "true",
                    "enable_diarization": "false",
                    "whisper_model": "whisper-base",
                }

                resp = await client.post(
                    f"{ORCHESTRATION_URL}/api/audio/upload", files=files, data=data
                )

                if resp.status_code == 200:
                    result = resp.json()

                    # Get Chinese transcription
                    transcription = result.get("transcription", {})
                    chinese_text = transcription.get("text", "").strip()
                    transcription.get("language", "unknown")

                    if chinese_text:
                        logger.info(f"✅ Chinese: {chinese_text[:50]}...")

                        # Get English translation
                        translations = result.get("translations", {})
                        english_data = translations.get("en", {})

                        if isinstance(english_data, dict):
                            english_text = english_data.get(
                                "translated_text", english_data.get("text", "")
                            )
                            confidence = english_data.get("confidence", 0.0)
                        else:
                            english_text = str(english_data)
                            confidence = 0.0

                        if english_text:
                            logger.info(f"✅ English: {english_text[:50]}...")

                            # Display both
                            self.display.add_translation(chinese_text, english_text, confidence)
                        else:
                            logger.warning("⚠️  No English translation received")
                    else:
                        logger.info("⚠️  No speech detected")

                else:
                    logger.error(f"❌ Failed: HTTP {resp.status_code}")

        except Exception as e:
            logger.error(f"❌ Error: {e}")

        self.display.update_display()

    async def run(self):
        """Run session"""
        print("\n" + "=" * 80)
        print("  🎤 CHINESE → ENGLISH SUBTITLES")
        print("  Real-time translation display")
        print("=" * 80 + "\n")

        if not await self.check_services():
            return 1

        await self.display.start()

        capture = LoopbackCapture()
        audio_queue = asyncio.Queue()

        def audio_callback(in_data, _frame_count, _time_info, _status):
            task = asyncio.create_task(audio_queue.put(in_data))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return (in_data, pyaudio.paContinue)

        try:
            capture.start(audio_callback)

            print("🎙️  LISTENING FOR CHINESE AUDIO...")
            print(f"   Session: {self.session_id}")
            print("   Translation: Chinese → English")
            print(f"   Chunk size: {CHUNK_DURATION}s")
            print("\n   💡 Play Chinese audio/video to see English subtitles!")
            print("   💡 Press 'q' in window or Ctrl+C to stop\n")

            while True:
                if cv2.getWindowProperty(self.display.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed")
                    break

                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    await self.process_chunk(audio_data)
                except TimeoutError:
                    self.display.update_display()
                    continue

        except KeyboardInterrupt:
            print("\n\n⚠️  Stopping...")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            capture.stop()
            await self.display.stop()

            print("\n" + "=" * 80)
            print(f"  📊 STATS: {self.chunk_count} chunks processed")
            print("=" * 80 + "\n")

        return 0


async def main():
    session = ChineseEnglishSession()
    return await session.run()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
