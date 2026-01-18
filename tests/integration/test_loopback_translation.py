#!/usr/bin/env python3
"""
SIMPLE LOOPBACK AUDIO → TRANSLATION TEST

This script:
1. Captures system audio (loopback)
2. Sends to Whisper for transcription
3. Sends to Translation service
4. Prints results in real-time

NO BACKEND COMPLEXITY - Just pure translation flow testing.

Requirements:
- macOS with BlackHole or similar loopback device
- Translation service running (Ollama recommended)
- Whisper service running (any model)

Usage:
    # Start translation service first:
    cd modules/translation-service
    python src/api_server_fastapi.py

    # Start whisper service:
    cd modules/whisper-service
    python src/main.py

    # Then run this test:
    python test_loopback_translation.py
"""

import asyncio
import io
import logging
import sys
import time
import wave
from datetime import datetime

import httpx
import pyaudio

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

WHISPER_URL = "http://localhost:5001"
TRANSLATION_URL = "http://localhost:5003"

# Audio settings (16kHz for Whisper)
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5  # seconds
FORMAT = pyaudio.paInt16

# Translation languages
TARGET_LANGUAGES = ["es", "fr", "de"]  # Spanish, French, German

# ============================================================================
# Loopback Audio Capture
# ============================================================================


class LoopbackAudioCapture:
    """Captures system audio using loopback device."""

    def __init__(self, sample_rate=16000, chunk_duration=5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.loopback_device_index = None

    def find_loopback_device(self):
        """Find loopback audio device (BlackHole, Soundflower, etc.)"""
        logger.info("🔍 Searching for loopback audio device...")

        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            name = info["name"].lower()

            # Look for common loopback device names
            if any(
                keyword in name for keyword in ["blackhole", "soundflower", "loopback", "virtual"]
            ) and info["maxInputChannels"] > 0:
                logger.info(f"✅ Found loopback device: {info['name']} (index {i})")
                self.loopback_device_index = i
                return True

        logger.error("❌ No loopback device found!")
        logger.info("   Install BlackHole: brew install blackhole-2ch")
        logger.info("   Or use default input device (will capture mic)")
        return False

    def list_audio_devices(self):
        """List all audio devices for debugging."""
        print("\n📋 Available Audio Devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"   [{i}] {info['name']}")
            print(f"       Input channels: {info['maxInputChannels']}")
            print(f"       Output channels: {info['maxOutputChannels']}")
            print(f"       Sample rate: {int(info['defaultSampleRate'])}Hz")
            print()

    def start_capture(self, callback):
        """Start capturing audio."""
        if not self.loopback_device_index and not self.find_loopback_device():
            # Fallback to default input device
            logger.warning("⚠️  Using default input device (microphone)")
            self.loopback_device_index = None

        logger.info("🎙️  Starting audio capture...")
        logger.info(f"   Sample rate: {self.sample_rate}Hz")
        logger.info(f"   Chunk duration: {self.chunk_duration}s")

        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.loopback_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=callback,
            )

            self.stream.start_stream()
            logger.info("✅ Audio capture started")

        except Exception as e:
            logger.error(f"❌ Failed to start audio capture: {e}")
            raise

    def stop_capture(self):
        """Stop capturing audio."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        logger.info("🛑 Audio capture stopped")


# ============================================================================
# Translation Pipeline
# ============================================================================


class SimpleTranslationPipeline:
    """Simple pipeline: Audio → Whisper → Translation → Print"""

    def __init__(self):
        self.chunk_count = 0
        self.total_translations = 0
        self.start_time = time.time()

    async def check_services(self):
        """Check if services are running."""
        logger.info("🔍 Checking services...")

        async with httpx.AsyncClient(timeout=3.0) as client:
            # Check Whisper
            try:
                response = await client.get(f"{WHISPER_URL}/health")
                if response.status_code == 200:
                    logger.info("✅ Whisper service: READY")
                else:
                    logger.error(f"❌ Whisper service: HTTP {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ Whisper service: NOT RUNNING ({e})")
                return False

            # Check Translation
            try:
                response = await client.get(f"{TRANSLATION_URL}/api/health")
                if response.status_code == 200:
                    data = response.json()
                    logger.info(
                        f"✅ Translation service: READY (backend: {data.get('backend', 'unknown')})"
                    )
                else:
                    logger.error(f"❌ Translation service: HTTP {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ Translation service: NOT RUNNING ({e})")
                return False

        return True

    async def transcribe_audio(self, audio_bytes: bytes) -> dict:
        """Transcribe audio using Whisper service."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                data = {"language": "auto", "enable_diarization": "false"}

                response = await client.post(f"{WHISPER_URL}/transcribe", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result.get("text", ""),
                        "language": result.get("language", "unknown"),
                        "success": True,
                    }
                else:
                    logger.error(f"Whisper transcription failed: {response.status_code}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {"success": False, "error": str(e)}

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> dict:
        """Translate text using translation service."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                request_data = {
                    "text": text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "model": "ollama",
                    "quality": "balanced",
                }

                response = await client.post(f"{TRANSLATION_URL}/api/translate", json=request_data)

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result.get("translated_text", ""),
                        "confidence": result.get("confidence", 0.0),
                        "processing_time": result.get("processing_time", 0.0),
                        "success": True,
                    }
                else:
                    logger.error(f"Translation failed: {response.status_code}")
                    return {"success": False, "error": f"HTTP {response.status_code}"}

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return {"success": False, "error": str(e)}

    async def process_audio_chunk(self, audio_data: bytes):
        """Process one audio chunk: transcribe + translate."""
        self.chunk_count += 1

        # Convert raw audio to WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data)

        wav_buffer.seek(0)
        wav_bytes = wav_buffer.read()

        print("\n" + "=" * 80)
        print(f"🎵 CHUNK {self.chunk_count} | {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)

        # Step 1: Transcribe
        print("📝 Transcribing...")
        transcription = await self.transcribe_audio(wav_bytes)

        if not transcription.get("success"):
            print(f"❌ Transcription failed: {transcription.get('error')}")
            return

        text = transcription["text"].strip()
        language = transcription["language"]

        if not text:
            print("⚠️  No speech detected (silent audio)")
            return

        print(f'✅ Transcribed ({language}): "{text}"')

        # Step 2: Translate to all target languages
        print(f"\n🌐 Translating to {len(TARGET_LANGUAGES)} languages...")

        for target_lang in TARGET_LANGUAGES:
            if target_lang == language:
                print(f"   [{target_lang}] Skipped (same as source)")
                continue

            translation = await self.translate_text(text, language, target_lang)

            if translation.get("success"):
                translated_text = translation["text"]
                confidence = translation["confidence"]
                proc_time = translation["processing_time"]

                print(f'   [{target_lang}] "{translated_text}"')
                print(f"           (confidence: {confidence:.2f}, time: {proc_time:.2f}s)")

                self.total_translations += 1
            else:
                print(f"   [{target_lang}] ❌ Failed: {translation.get('error')}")

        # Stats
        elapsed = time.time() - self.start_time
        print(
            f"\n📊 Stats: {self.chunk_count} chunks, {self.total_translations} translations, {elapsed:.1f}s"
        )


# ============================================================================
# Main
# ============================================================================


async def main():
    print("\n" + "=" * 80)
    print("  🎤 LOOPBACK AUDIO → TRANSLATION TEST")
    print("  Real-time system audio capture with translation")
    print("=" * 80 + "\n")

    # Initialize pipeline
    pipeline = SimpleTranslationPipeline()

    # Check services
    if not await pipeline.check_services():
        print("\n❌ Services not ready. Please start them first:")
        print(
            "   1. Translation service: cd modules/translation-service && python src/api_server_fastapi.py"
        )
        print("   2. Whisper service: cd modules/whisper-service && python src/main.py")
        return 1

    # Initialize audio capture
    capture = LoopbackAudioCapture(SAMPLE_RATE, CHUNK_DURATION)

    print("\n" + "=" * 80)
    capture.list_audio_devices()
    print("=" * 80 + "\n")

    # Audio callback
    audio_queue = asyncio.Queue()

    def audio_callback(in_data, _frame_count, _time_info, _status):
        asyncio.create_task(audio_queue.put(in_data))
        return (in_data, pyaudio.paContinue)

    # Start capture
    try:
        capture.start_capture(audio_callback)

        print("\n🎙️  LISTENING FOR AUDIO... (Press Ctrl+C to stop)")
        print(f"   Target languages: {', '.join(TARGET_LANGUAGES)}")
        print(f"   Chunk duration: {CHUNK_DURATION}s")
        print("\n   💡 Play some audio on your system to test!\n")

        # Process audio chunks
        while True:
            audio_data = await audio_queue.get()
            await pipeline.process_audio_chunk(audio_data)

    except KeyboardInterrupt:
        print("\n\n⚠️  Stopping capture...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        capture.stop_capture()

        print("\n" + "=" * 80)
        print("  📊 FINAL STATS")
        print(f"     Chunks processed: {pipeline.chunk_count}")
        print(f"     Translations: {pipeline.total_translations}")
        print(f"     Duration: {time.time() - pipeline.start_time:.1f}s")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
