#!/usr/bin/env python3
"""
Audio Capture for Bot Container

Simplified audio capture from Google Meet browser sessions.
Based on modules/orchestration-service/src/bot/browser_audio_capture.py

This version is designed for Docker containers and focuses on:
1. Capturing audio from browser/system
2. Processing into chunks (16kHz, mono, float32)
3. Streaming to orchestration_client
4. Quality validation

Future enhancements (Phase 3.3c):
- Full sounddevice integration
- Multiple capture methods (virtual device, loopback, system)
- Advanced audio processing
- Quality monitoring
"""

import asyncio
import logging
import time
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio capture configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration: float = 2.0  # seconds
    audio_format: str = "float32"
    quality_threshold: float = 0.01  # Minimum RMS for valid audio

    # Capture method preferences (in order)
    prefer_virtual_device: bool = True
    prefer_loopback: bool = True
    fallback_to_system: bool = True


class AudioCapture:
    """
    Simplified audio capture for bot containers

    Usage:
        capture = AudioCapture(config, orchestration_client)
        await capture.initialize()
        await capture.start_capture()
        # ... audio streams automatically to orchestration_client
        await capture.stop_capture()
    """

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        orchestration_client=None  # OrchestrationClient instance
    ):
        self.config = config or AudioConfig()
        self.orchestration_client = orchestration_client

        # Audio capture state
        self.is_capturing = False
        self.audio_stream = None
        self.capture_task: Optional[asyncio.Task] = None

        # Audio buffer
        self.audio_buffer = []
        self.chunk_size = int(self.config.sample_rate * self.config.chunk_duration)

        # Statistics
        self.total_chunks_sent = 0
        self.total_audio_duration = 0.0
        self.last_audio_time = 0

        # Callback for audio events
        self.on_audio_error: Optional[Callable[[str], None]] = None

    async def initialize(self) -> bool:
        """
        Initialize audio capture system

        Phase 3.3c will implement:
        - Detect available audio capture methods
        - Setup virtual audio device if available
        - Configure loopback capture if available
        - Fallback to system default

        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing audio capture (stub)")

        # TODO Phase 3.3c: Implement actual initialization
        # 1. Check for sounddevice library
        # 2. Detect audio capture method:
        #    - Virtual audio device (best)
        #    - Loopback capture (good)
        #    - System default (fallback)
        # 3. Configure audio stream parameters

        # For now, simulate successful initialization
        logger.info("✅ Audio capture initialized (stub)")
        return True

    async def start_capture(self) -> bool:
        """
        Start capturing audio

        Phase 3.3c will implement:
        - Start sounddevice InputStream
        - Begin audio callback processing
        - Start chunk processing task
        - Begin streaming to orchestration_client

        Returns:
            bool: True if capture started successfully
        """
        if self.is_capturing:
            logger.warning("Audio capture already running")
            return True

        logger.info("Starting audio capture (stub)")

        # TODO Phase 3.3c: Implement actual capture start
        # 1. Configure audio stream:
        #    stream_config = {
        #        'callback': self._audio_callback,
        #        'channels': self.config.channels,
        #        'samplerate': self.config.sample_rate,
        #        'dtype': self.config.audio_format,
        #        'blocksize': self.chunk_size,
        #    }
        # 2. Start InputStream: sd.InputStream(**stream_config)
        # 3. Start processing task: asyncio.create_task(self._process_audio_chunks())

        self.is_capturing = True
        self.capture_task = asyncio.create_task(self._simulate_audio_chunks())

        logger.info("✅ Audio capture started (stub)")
        return True

    async def stop_capture(self):
        """
        Stop audio capture

        Phase 3.3c will implement:
        - Stop audio stream
        - Cancel processing task
        - Process remaining audio buffer
        - Cleanup resources
        """
        if not self.is_capturing:
            return

        logger.info("Stopping audio capture (stub)")

        self.is_capturing = False

        # Cancel processing task
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass

        # TODO Phase 3.3c: Implement actual cleanup
        # 1. Stop audio stream: self.audio_stream.stop()
        # 2. Close audio stream: self.audio_stream.close()
        # 3. Process remaining buffer: await self._process_audio_buffer()

        logger.info("✅ Audio capture stopped (stub)")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Audio stream callback (called by sounddevice)

        Phase 3.3c will implement:
        - Receive audio from sounddevice
        - Add to buffer
        - Handle audio stream errors

        Args:
            indata: Audio data array
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        # TODO Phase 3.3c: Implement callback
        # if status:
        #     logger.warning(f"Audio callback status: {status}")
        #
        # if self.is_capturing and len(indata) > 0:
        #     self.audio_buffer.extend(indata.flatten())
        #     self.last_audio_time = time.time()
        pass

    async def _process_audio_chunks(self):
        """
        Process audio chunks from buffer

        Phase 3.3c will implement:
        - Monitor audio buffer
        - Extract chunks when buffer is full
        - Validate audio quality
        - Send to orchestration_client
        """
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
        """
        Process the current audio buffer and send to orchestration

        Phase 3.3c will implement:
        - Extract chunk from buffer
        - Validate audio quality (RMS threshold)
        - Convert to bytes
        - Send via orchestration_client.send_audio_chunk()
        """
        if len(self.audio_buffer) < self.chunk_size:
            return

        # Extract chunk
        chunk_data = np.array(self.audio_buffer[:self.chunk_size], dtype=np.float32)
        self.audio_buffer = self.audio_buffer[self.chunk_size:]

        # Validate audio quality
        if not self._validate_audio_quality(chunk_data):
            logger.debug("Audio chunk quality too low, skipping")
            return

        # Convert to bytes
        audio_bytes = chunk_data.tobytes()

        # Send to orchestration client
        if self.orchestration_client:
            try:
                await self.orchestration_client.send_audio_chunk(audio_bytes)
                self.total_chunks_sent += 1
                self.total_audio_duration += self.config.chunk_duration
                logger.debug(f"Sent audio chunk {self.total_chunks_sent} "
                           f"({self.total_audio_duration:.1f}s total)")
            except Exception as e:
                logger.error(f"Failed to send audio chunk: {e}")
                if self.on_audio_error:
                    await self.on_audio_error(f"Failed to send audio: {e}")

    def _validate_audio_quality(self, audio_data: np.ndarray) -> bool:
        """
        Validate audio chunk quality

        Args:
            audio_data: Audio data array

        Returns:
            bool: True if audio quality is acceptable
        """
        # Check for silence (RMS too low)
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < self.config.quality_threshold:
            return False

        # Check for clipping (values near ±1.0)
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude > 0.99:
            logger.warning(f"Audio clipping detected: {max_amplitude:.2f}")

        return True

    async def _simulate_audio_chunks(self):
        """
        Simulate audio chunks for testing (stub implementation)

        This will be removed in Phase 3.3c when real audio capture is implemented.
        """
        try:
            logger.info("Simulating audio chunks (stub mode)")

            while self.is_capturing:
                # Generate simulated audio chunk
                chunk_data = np.random.randn(self.chunk_size).astype(np.float32) * 0.1
                audio_bytes = chunk_data.tobytes()

                # Send to orchestration client
                if self.orchestration_client:
                    try:
                        await self.orchestration_client.send_audio_chunk(audio_bytes)
                        self.total_chunks_sent += 1
                        self.total_audio_duration += self.config.chunk_duration
                        logger.debug(f"Sent simulated chunk {self.total_chunks_sent}")
                    except Exception as e:
                        logger.error(f"Failed to send simulated chunk: {e}")

                # Wait for chunk duration
                await asyncio.sleep(self.config.chunk_duration)

        except asyncio.CancelledError:
            logger.info("Audio simulation task cancelled")
        except Exception as e:
            logger.error(f"Audio simulation task failed: {e}")

    def get_stats(self) -> dict:
        """Get audio capture statistics"""
        return {
            "is_capturing": self.is_capturing,
            "total_chunks_sent": self.total_chunks_sent,
            "total_audio_duration": self.total_audio_duration,
            "last_audio_time": self.last_audio_time
        }


# Example usage
async def example_usage():
    """Example of using audio capture with orchestration client"""
    from orchestration_client import OrchestrationClient

    # Create orchestration client
    client = OrchestrationClient(
        orchestration_url="ws://localhost:3000/ws",
        user_token="test-token",
        meeting_id="test-meeting",
        connection_id="test-connection"
    )

    # Connect to orchestration
    await client.connect()

    # Create audio capture
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_duration=2.0
    )

    capture = AudioCapture(config, client)

    try:
        # Initialize audio capture
        await capture.initialize()

        # Start capturing
        await capture.start_capture()

        # Capture for 10 seconds
        await asyncio.sleep(10)

        # Stop capturing
        await capture.stop_capture()

        # Print stats
        stats = capture.get_stats()
        logger.info(f"Audio capture stats: {stats}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
