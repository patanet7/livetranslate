#!/usr/bin/env python3
"""
FULL-STACK LOOPBACK TRANSLATION TEST
Goes through the complete orchestration pipeline (not direct service calls)

Flow:
  Loopback Audio → Orchestration /api/audio/upload → Whisper → Translation → Display

This mimics exactly what the frontend does - uses the REAL orchestration API.
"""

import asyncio
import io
import json
import logging
import sys
import time
import wave
from datetime import datetime

import httpx
import pyaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# URLs
ORCHESTRATION_URL = "http://localhost:3000"
WHISPER_URL = "http://localhost:5001"
TRANSLATION_URL = "http://localhost:5003"

# Audio config
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 5
FORMAT = pyaudio.paInt16

# Target languages
TARGET_LANGUAGES = ["es", "fr", "de"]


class LoopbackCapture:
    """Simple loopback audio capture"""

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
                    logger.info(f"✅ Found: {info['name']} (index {i})")
                    self.device_index = i
                    return True

        logger.warning("⚠️  No loopback device - using default mic")
        return False

    def start(self, callback):
        """Start capture"""
        self.find_loopback()

        logger.info(f"🎙️  Starting capture ({CHUNK_DURATION}s chunks @ {SAMPLE_RATE}Hz)")

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
        """Stop capture"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


class FullStackPipeline:
    """Full-stack pipeline through orchestration"""

    def __init__(self):
        self.session_id = f"loopback_test_{int(time.time())}"
        self.chunk_count = 0
        self.total_translations = 0
        self.start_time = time.time()

    async def check_services(self):
        """Check if all services are ready"""
        logger.info("🔍 Checking services...")

        all_ready = True
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Check orchestration
            try:
                resp = await client.get(f"{ORCHESTRATION_URL}/api/health")
                if resp.status_code == 200:
                    logger.info("✅ Orchestration: READY")
                else:
                    logger.error(f"❌ Orchestration: HTTP {resp.status_code}")
                    all_ready = False
            except Exception as e:
                logger.error(f"❌ Orchestration: {e}")
                all_ready = False

            # Check whisper
            try:
                resp = await client.get(f"{WHISPER_URL}/health")
                if resp.status_code == 200:
                    logger.info("✅ Whisper: READY")
                else:
                    logger.error(f"❌ Whisper: HTTP {resp.status_code}")
                    all_ready = False
            except Exception as e:
                logger.error(f"❌ Whisper: {e}")
                all_ready = False

            # Check translation
            try:
                resp = await client.get(f"{TRANSLATION_URL}/api/health")
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        f"✅ Translation: READY (backend: {data.get('backend', 'unknown')})"
                    )
                else:
                    logger.error(f"❌ Translation: HTTP {resp.status_code}")
                    all_ready = False
            except Exception as e:
                logger.error(f"❌ Translation: {e}")
                all_ready = False

        return all_ready

    async def process_chunk(self, audio_data: bytes):
        """Send audio through orchestration pipeline"""
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

        print("\n" + "=" * 80)
        print(f"🎵 CHUNK {self.chunk_count} | {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Send via orchestration API (like frontend does)
                files = {"file": ("audio.wav", wav_buffer.read(), "audio/wav")}
                data = {
                    "chunk_id": chunk_id,
                    "session_id": self.session_id,
                    "target_languages": json.dumps(TARGET_LANGUAGES),
                    "enable_transcription": "true",
                    "enable_translation": "true",
                    "enable_diarization": "false",
                    "whisper_model": "whisper-base",
                }

                print("📤 Sending to orchestration: /api/audio/upload")
                resp = await client.post(
                    f"{ORCHESTRATION_URL}/api/audio/upload", files=files, data=data
                )

                if resp.status_code == 200:
                    result = resp.json()

                    # Display transcription
                    transcription = result.get("transcription", {})
                    text = transcription.get("text", "").strip()
                    lang = transcription.get("language", "unknown")

                    if text:
                        print(f'✅ Transcribed ({lang}): "{text}"')

                        # Display translations
                        translations = result.get("translations", {})
                        if translations:
                            print("\n🌐 Translations:")
                            for target_lang, trans_data in translations.items():
                                if isinstance(trans_data, dict):
                                    trans_text = trans_data.get(
                                        "translated_text", trans_data.get("text", "")
                                    )
                                    confidence = trans_data.get("confidence", 0.0)
                                    print(f'   [{target_lang}] "{trans_text}"')
                                    print(f"           (confidence: {confidence:.2f})")
                                    self.total_translations += 1
                                else:
                                    print(f"   [{target_lang}] {trans_data}")
                        else:
                            print("⚠️  No translations received")
                    else:
                        print("⚠️  No speech detected")

                else:
                    print(f"❌ Orchestration returned {resp.status_code}")
                    print(f"   Response: {resp.text[:200]}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback

            traceback.print_exc()

        # Stats
        elapsed = time.time() - self.start_time
        print(
            f"\n📊 Stats: {self.chunk_count} chunks, {self.total_translations} translations, {elapsed:.1f}s"
        )


async def main():
    print("\n" + "=" * 80)
    print("  🎤 FULL-STACK LOOPBACK TRANSLATION TEST")
    print("  Loopback → Orchestration → Whisper → Translation")
    print("=" * 80 + "\n")

    # Setup
    pipeline = FullStackPipeline()

    # Check services
    if not await pipeline.check_services():
        print("\n❌ Not all services ready!")
        print("\nStart services:")
        print("  Terminal 1: cd modules/orchestration-service && python src/main.py")
        print("  Terminal 2: cd modules/whisper-service && python src/main.py")
        print("  Terminal 3: cd modules/translation-service && python src/api_server_fastapi.py")
        return 1

    # Audio capture
    capture = LoopbackCapture()
    audio_queue = asyncio.Queue()
    background_tasks: set = set()

    def audio_callback(in_data, _frame_count, _time_info, _status):
        task = asyncio.create_task(audio_queue.put(in_data))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return (in_data, pyaudio.paContinue)

    try:
        capture.start(audio_callback)

        print("\n🎙️  LISTENING... (Ctrl+C to stop)")
        print(f"   Session: {pipeline.session_id}")
        print(f"   Languages: {', '.join(TARGET_LANGUAGES)}")
        print(f"   Chunk size: {CHUNK_DURATION}s\n")

        # Process chunks
        while True:
            audio_data = await audio_queue.get()
            await pipeline.process_chunk(audio_data)

    except KeyboardInterrupt:
        print("\n\n⚠️  Stopping...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        capture.stop()

        print("\n" + "=" * 80)
        print("  📊 FINAL STATS")
        print(f"     Chunks: {pipeline.chunk_count}")
        print(f"     Translations: {pipeline.total_translations}")
        print(f"     Duration: {time.time() - pipeline.start_time:.1f}s")
        print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
