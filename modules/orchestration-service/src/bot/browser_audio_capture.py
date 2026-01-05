"""
Browser Audio Capture for Google Meet

This module provides specialized audio capture from Google Meet browser sessions,
ensuring the audio source matches the meeting being processed.
"""

import asyncio
import logging
import time
import json
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np
import io
import wave
import subprocess
import os

try:
    import sounddevice as sd

    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    from selenium import webdriver

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BrowserAudioConfig:
    """Configuration for browser audio capture"""

    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 2.0  # seconds
    audio_format: str = "float32"
    quality_threshold: float = 0.01

    # Browser audio specific
    browser_process_name: str = "chrome"
    virtual_audio_device: Optional[str] = None
    loopback_device: Optional[str] = None

    # Audio processing
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    audio_gain: float = 1.0


class BrowserAudioCapture:
    """
    Captures audio specifically from Google Meet browser sessions using
    various methods including virtual audio devices and loopback capture.
    """

    def __init__(
        self,
        config: BrowserAudioConfig,
        orchestration_url: str = "http://localhost:3000",
    ):
        self.config = config
        self.orchestration_url = orchestration_url

        # Audio capture state
        self.is_capturing = False
        self.audio_stream = None
        self.capture_task: Optional[asyncio.Task] = None

        # Meeting context
        self.meeting_id: Optional[str] = None
        self.session_id: Optional[str] = None
        self.browser_driver: Optional[webdriver.Chrome] = None

        # Audio data buffers
        self.audio_buffer = []
        self.chunk_size = int(self.config.sample_rate * self.config.chunk_duration)

        # Callbacks
        self.on_audio_chunk: Optional[Callable[[bytes, Dict[str, Any]], None]] = None
        self.on_audio_error: Optional[Callable[[str], None]] = None

        # Statistics
        self.total_chunks_sent = 0
        self.total_audio_duration = 0.0
        self.last_audio_time = 0

        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice is required for audio capture")

    async def initialize(
        self,
        meeting_id: str,
        session_id: str,
        browser_driver: Optional[webdriver.Chrome] = None,
    ):
        """Initialize browser audio capture for a specific meeting"""
        try:
            self.meeting_id = meeting_id
            self.session_id = session_id
            self.browser_driver = browser_driver

            logger.info(f"Initializing browser audio capture for meeting: {meeting_id}")

            # Detect audio capture method
            audio_method = await self._detect_audio_capture_method()
            logger.info(f"Using audio capture method: {audio_method}")

            # Configure audio capture based on available method
            if audio_method == "virtual_device":
                await self._setup_virtual_audio_device()
            elif audio_method == "loopback":
                await self._setup_loopback_capture()
            elif audio_method == "system_default":
                await self._setup_system_audio_capture()
            else:
                raise RuntimeError("No suitable audio capture method available")

            logger.info("Browser audio capture initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize browser audio capture: {e}")
            raise

    async def start_capture(self) -> bool:
        """Start capturing audio from the browser"""
        try:
            if self.is_capturing:
                logger.warning("Audio capture already running")
                return True

            logger.info("Starting browser audio capture")

            # Configure audio stream
            stream_config = {
                "callback": self._audio_callback,
                "channels": self.config.channels,
                "samplerate": self.config.sample_rate,
                "dtype": self.config.audio_format,
                "blocksize": self.chunk_size,
            }

            # Add device configuration if available
            if hasattr(self, "_audio_device_id"):
                stream_config["device"] = self._audio_device_id

            # Start audio stream
            self.audio_stream = sd.InputStream(**stream_config)
            self.audio_stream.start()

            self.is_capturing = True

            # Start processing task
            self.capture_task = asyncio.create_task(self._process_audio_chunks())

            logger.info("Browser audio capture started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            if self.on_audio_error:
                await self._safe_callback(
                    self.on_audio_error, f"Audio capture start failed: {e}"
                )
            return False

    async def stop_capture(self):
        """Stop audio capture"""
        try:
            if not self.is_capturing:
                return

            logger.info("Stopping browser audio capture")

            self.is_capturing = False

            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            # Cancel processing task
            if self.capture_task:
                self.capture_task.cancel()
                try:
                    await self.capture_task
                except asyncio.CancelledError:
                    pass

            # Process any remaining audio
            if self.audio_buffer:
                await self._process_audio_buffer()

            logger.info("Browser audio capture stopped")

        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        try:
            if status:
                logger.warning(f"Audio callback status: {status}")

            if self.is_capturing and len(indata) > 0:
                # Add to buffer
                self.audio_buffer.extend(indata.flatten())
                self.last_audio_time = time.time()

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    async def _process_audio_chunks(self):
        """Process audio chunks and send to orchestration service"""
        try:
            while self.is_capturing:
                try:
                    # Check if we have enough audio for a chunk
                    if len(self.audio_buffer) >= self.chunk_size:
                        await self._process_audio_buffer()

                    # Sleep briefly to avoid busy waiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Audio processing task cancelled")
        except Exception as e:
            logger.error(f"Audio processing task failed: {e}")

    async def _process_audio_buffer(self):
        """Process the current audio buffer"""
        try:
            if len(self.audio_buffer) < self.chunk_size:
                return

            # Extract chunk
            chunk_data = np.array(
                self.audio_buffer[: self.chunk_size], dtype=np.float32
            )
            self.audio_buffer = self.audio_buffer[self.chunk_size :]

            # Validate audio quality
            if not self._validate_audio_quality(chunk_data):
                logger.debug("Audio chunk quality too low, skipping")
                return

            # Convert to bytes
            audio_bytes = self._audio_to_bytes(chunk_data)

            # Create chunk metadata
            chunk_metadata = {
                "chunk_id": f"{self.session_id}_{self.total_chunks_sent}",
                "session_id": self.session_id,
                "meeting_id": self.meeting_id,
                "timestamp": time.time(),
                "duration": self.config.chunk_duration,
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "format": self.config.audio_format,
                "source": "google_meet_browser",
                "chunk_index": self.total_chunks_sent,
            }

            # Send to orchestration service
            await self._send_audio_to_orchestration(audio_bytes, chunk_metadata)

            # Update statistics
            self.total_chunks_sent += 1
            self.total_audio_duration += self.config.chunk_duration

            # Trigger callback if set
            if self.on_audio_chunk:
                await self._safe_callback(
                    self.on_audio_chunk, audio_bytes, chunk_metadata
                )

        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}")

    async def _send_audio_to_orchestration(
        self, audio_bytes: bytes, metadata: Dict[str, Any]
    ):
        """Send audio chunk to orchestration service for processing"""
        try:
            import httpx

            # Prepare multipart form data
            files = {
                "file": ("audio_chunk.wav", audio_bytes, "audio/wav"),
            }

            data = {
                "chunk_id": metadata["chunk_id"],
                "session_id": metadata["session_id"],
                "target_languages": json.dumps(["es", "fr", "de"]),  # Default languages
                "enable_transcription": "true",
                "enable_translation": "true",
                "enable_diarization": "true",
                "whisper_model": "whisper-base",
                "metadata": json.dumps(metadata),
            }

            # Send to orchestration service
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.orchestration_url}/api/audio/upload", files=files, data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.debug(
                        f"Audio chunk processed successfully: {metadata['chunk_id']}"
                    )
                    return result
                else:
                    logger.error(
                        f"Failed to send audio chunk: {response.status_code} - {response.text}"
                    )
                    return None

        except Exception as e:
            logger.error(f"Error sending audio to orchestration service: {e}")
            if self.on_audio_error:
                await self._safe_callback(
                    self.on_audio_error, f"Failed to send audio: {e}"
                )
            return None

    def _validate_audio_quality(self, audio_data: np.ndarray) -> bool:
        """Validate audio quality before processing"""
        try:
            # Check for silence
            rms_level = np.sqrt(np.mean(audio_data**2))
            if rms_level < self.config.quality_threshold:
                return False

            # Check for clipping
            peak_level = np.max(np.abs(audio_data))
            if peak_level > 0.98:  # Near clipping
                logger.warning("Audio clipping detected")

            return True

        except Exception as e:
            logger.error(f"Error validating audio quality: {e}")
            return False

    def _audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes"""
        try:
            # Create WAV file in memory
            buffer = io.BytesIO()

            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)

                # Convert float32 to int16
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting audio to bytes: {e}")
            return b""

    async def _detect_audio_capture_method(self) -> str:
        """Detect the best available audio capture method"""
        try:
            # Check for virtual audio devices
            if await self._check_virtual_audio_device():
                return "virtual_device"

            # Check for loopback capture capability
            if await self._check_loopback_capability():
                return "loopback"

            # Fall back to system audio
            return "system_default"

        except Exception as e:
            logger.error(f"Error detecting audio capture method: {e}")
            return "system_default"

    async def _check_virtual_audio_device(self) -> bool:
        """Check if virtual audio devices are available"""
        try:
            devices = sd.query_devices()
            for device in devices:
                if (
                    "virtual" in device["name"].lower()
                    or "loopback" in device["name"].lower()
                ):
                    self._audio_device_id = device["index"]
                    return True
            return False
        except Exception:
            return False

    async def _check_loopback_capability(self) -> bool:
        """Check if system supports audio loopback"""
        try:
            # Check for Windows WASAPI loopback
            if os.name == "nt":
                devices = sd.query_devices()
                for device in devices:
                    if (
                        device["max_input_channels"] > 0
                        and "stereo mix" in device["name"].lower()
                    ):
                        self._audio_device_id = device["index"]
                        return True

            # Check for PulseAudio monitor (Linux)
            elif os.name == "posix":
                try:
                    result = subprocess.run(
                        ["pactl", "list", "sources"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if "monitor" in result.stdout.lower():
                        return True
                except Exception:
                    pass

            return False
        except Exception:
            return False

    async def _setup_virtual_audio_device(self):
        """Setup virtual audio device capture"""
        logger.info("Setting up virtual audio device capture")
        # Virtual audio device is already configured in _check_virtual_audio_device

    async def _setup_loopback_capture(self):
        """Setup loopback audio capture"""
        logger.info("Setting up loopback audio capture")
        # Loopback device is already configured in _check_loopback_capability

    async def _setup_system_audio_capture(self):
        """Setup system default audio capture"""
        logger.info("Setting up system default audio capture")
        # Use default audio device
        self._audio_device_id = None

    async def _safe_callback(self, callback, *args):
        """Execute callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    def get_capture_stats(self) -> Dict[str, Any]:
        """Get current capture statistics"""
        return {
            "is_capturing": self.is_capturing,
            "total_chunks_sent": self.total_chunks_sent,
            "total_audio_duration": self.total_audio_duration,
            "meeting_id": self.meeting_id,
            "session_id": self.session_id,
            "last_audio_time": self.last_audio_time,
            "config": {
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "chunk_duration": self.config.chunk_duration,
            },
        }

    async def shutdown(self):
        """Shutdown browser audio capture"""
        try:
            logger.info("Shutting down browser audio capture")
            await self.stop_capture()
            logger.info("Browser audio capture shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Factory function for creating browser audio capture
def create_browser_audio_capture(
    config: Optional[BrowserAudioConfig] = None,
    orchestration_url: str = "http://localhost:3000",
) -> BrowserAudioCapture:
    """Create a browser audio capture instance"""
    if config is None:
        config = BrowserAudioConfig()

    return BrowserAudioCapture(config, orchestration_url)
