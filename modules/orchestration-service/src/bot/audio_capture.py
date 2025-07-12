#!/usr/bin/env python3
"""
Google Meet Audio Capture - Orchestration Service Integration

Real-time audio capture from Google Meet with streaming to whisper service.
Now integrated directly into the orchestration service for centralized control.

Features:
- Real-time audio capture from Google Meet sessions
- Multi-format audio processing and streaming
- Integration with whisper service for transcription
- Audio quality monitoring and optimization
- Session-based audio management
- Virtual audio device handling
"""

import os
import sys
import time
import logging
import asyncio
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
import io
import httpx
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio capture configuration."""

    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    blocksize: int = 1024
    device_name: Optional[str] = None
    audio_format: str = "wav"
    chunk_duration: float = 1.0  # seconds
    quality_threshold: float = 0.7


@dataclass
class MeetingInfo:
    """Google Meet meeting information."""

    meeting_id: str
    meeting_title: Optional[str] = None
    meeting_uri: Optional[str] = None
    organizer_email: Optional[str] = None
    participant_count: int = 0
    scheduled_start: Optional[datetime] = None


class AudioBuffer:
    """Thread-safe audio buffer for real-time processing."""

    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.RLock()
        self.sample_rate = 16000

    def add_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer."""
        with self.lock:
            self.buffer.append(audio_chunk.copy())

    def get_chunks(self, num_chunks: int) -> List[np.ndarray]:
        """Get specified number of recent chunks."""
        with self.lock:
            if len(self.buffer) < num_chunks:
                return list(self.buffer)
            return list(self.buffer)[-num_chunks:]

    def get_duration_audio(self, duration_seconds: float) -> np.ndarray:
        """Get audio data for specified duration."""
        samples_needed = int(duration_seconds * self.sample_rate)

        with self.lock:
            if not self.buffer:
                return np.array([], dtype=np.float32)

            # Concatenate recent chunks
            combined = np.concatenate(list(self.buffer))

            if len(combined) >= samples_needed:
                return combined[-samples_needed:]
            else:
                return combined

    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()


class AudioQualityAnalyzer:
    """Analyzes audio quality metrics."""

    def __init__(self):
        self.noise_threshold = 0.01
        self.silence_threshold = 0.005

    def analyze_chunk(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """Analyze audio chunk quality."""
        try:
            # RMS level
            rms = np.sqrt(np.mean(audio_chunk**2))

            # Peak level
            peak = np.max(np.abs(audio_chunk))

            # Zero crossing rate (indicates voice activity)
            zero_crossings = np.sum(np.diff(np.signbit(audio_chunk)))
            zcr = zero_crossings / len(audio_chunk)

            # Signal-to-noise ratio estimate
            signal_power = np.mean(audio_chunk**2)
            snr_estimate = 10 * np.log10(
                signal_power / max(self.noise_threshold, signal_power * 0.1)
            )

            # Voice activity detection
            is_voice = rms > self.silence_threshold and zcr > 0.01

            # Overall quality score
            quality_score = min(1.0, (rms * 2 + min(snr_estimate / 20, 1.0)) / 2)

            return {
                "rms_level": float(rms),
                "peak_level": float(peak),
                "zero_crossing_rate": float(zcr),
                "snr_estimate": float(snr_estimate),
                "voice_activity": bool(is_voice),
                "quality_score": float(quality_score),
                "clipping_detected": bool(peak > 0.98),
            }

        except Exception as e:
            logger.error(f"Error analyzing audio quality: {e}")
            return {
                "rms_level": 0.0,
                "peak_level": 0.0,
                "zero_crossing_rate": 0.0,
                "snr_estimate": 0.0,
                "voice_activity": False,
                "quality_score": 0.0,
                "clipping_detected": False,
            }


class GoogleMeetAudioCapture:
    """
    Google Meet audio capture system integrated with orchestration service.
    """

    def __init__(
        self,
        config: AudioConfig,
        whisper_service_url: str,
        bot_manager=None,
        database_manager=None,
    ):
        self.config = config
        self.whisper_service_url = whisper_service_url
        self.bot_manager = bot_manager
        self.database_manager = database_manager

        # Audio processing
        self.audio_buffer = AudioBuffer()
        self.quality_analyzer = AudioQualityAnalyzer()

        # Capture state
        self.is_capturing = False
        self.session_id = None
        self.meeting_info = None

        # Audio stream
        self.audio_stream = None
        self.capture_thread = None

        # Statistics
        self.total_chunks_captured = 0
        self.total_audio_duration = 0.0
        self.average_quality_score = 0.0
        self.start_time = None

        # Callbacks
        self.on_transcription_ready = None
        self.on_audio_chunk = None
        self.on_quality_alert = None
        self.on_error = None

        logger.info("Google Meet Audio Capture initialized")
        logger.info(f"  Sample rate: {config.sample_rate}Hz")
        logger.info(f"  Channels: {config.channels}")
        logger.info(f"  Chunk duration: {config.chunk_duration}s")

    def set_transcription_callback(self, callback: Callable[[Dict], None]):
        """Set callback for transcription results."""
        self.on_transcription_ready = callback

    def set_audio_chunk_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        """Set callback for raw audio chunks."""
        self.on_audio_chunk = callback

    def set_quality_alert_callback(self, callback: Callable[[Dict], None]):
        """Set callback for audio quality alerts."""
        self.on_quality_alert = callback

    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for error notifications."""
        self.on_error = callback

    async def start_capture(self, meeting_info: MeetingInfo) -> bool:
        """Start audio capture for a Google Meet session."""
        try:
            if self.is_capturing:
                logger.warning("Audio capture already running")
                return False

            self.meeting_info = meeting_info
            self.session_id = f"audio_{meeting_info.meeting_id}_{int(time.time())}"
            self.start_time = time.time()

            # Initialize audio capture
            success = await self._initialize_audio_capture()
            if not success:
                return False

            # Create whisper session
            success = await self._create_whisper_session()
            if not success:
                return False

            # Start capture thread
            self.is_capturing = True
            self.capture_thread = threading.Thread(
                target=self._capture_loop, daemon=True
            )
            self.capture_thread.start()

            logger.info(f"Started audio capture for meeting: {meeting_info.meeting_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            if self.on_error:
                self.on_error(f"Audio capture start failed: {e}")
            return False

    async def stop_capture(self) -> bool:
        """Stop audio capture."""
        try:
            if not self.is_capturing:
                logger.warning("Audio capture not running")
                return False

            self.is_capturing = False

            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            # Wait for capture thread to finish
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=5.0)

            # Close whisper session
            await self._close_whisper_session()

            # Calculate final statistics
            duration = time.time() - self.start_time if self.start_time else 0
            self.total_audio_duration = duration

            logger.info(f"Stopped audio capture for session: {self.session_id}")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Chunks captured: {self.total_chunks_captured}")
            logger.info(f"  Average quality: {self.average_quality_score:.2f}")

            return True

        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")
            if self.on_error:
                self.on_error(f"Audio capture stop failed: {e}")
            return False

    async def _initialize_audio_capture(self) -> bool:
        """Initialize audio capture devices and streams."""
        try:
            # Get available audio devices
            devices = sd.query_devices()
            logger.debug(f"Available audio devices: {len(devices)}")

            # Find appropriate input device
            device_id = None
            if self.config.device_name:
                for i, device in enumerate(devices):
                    if self.config.device_name.lower() in device["name"].lower():
                        if device["max_input_channels"] > 0:
                            device_id = i
                            break

            if device_id is None:
                # Use default input device
                device_id = sd.default.device[0]
                logger.info(
                    f"Using default audio input device: {devices[device_id]['name']}"
                )
            else:
                logger.info(
                    f"Using specified audio device: {devices[device_id]['name']}"
                )

            # Create audio stream
            self.audio_stream = sd.InputStream(
                device=device_id,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                callback=self._audio_callback,
            )

            # Start the stream
            self.audio_stream.start()

            return True

        except Exception as e:
            logger.error(f"Error initializing audio capture: {e}")
            return False

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback for real-time processing."""
        try:
            if status:
                logger.warning(f"Audio stream status: {status}")

            if not self.is_capturing:
                return

            # Convert to numpy array
            audio_chunk = indata[:, 0] if self.config.channels == 1 else indata

            # Add to buffer
            self.audio_buffer.add_chunk(audio_chunk)

            # Analyze quality
            quality_metrics = self.quality_analyzer.analyze_chunk(audio_chunk)

            # Update statistics
            self.total_chunks_captured += 1
            self.average_quality_score = (
                self.average_quality_score * (self.total_chunks_captured - 1)
                + quality_metrics["quality_score"]
            ) / self.total_chunks_captured

            # Quality alerts
            if quality_metrics["quality_score"] < self.config.quality_threshold:
                if self.on_quality_alert:
                    self.on_quality_alert(
                        {
                            "session_id": self.session_id,
                            "quality_score": quality_metrics["quality_score"],
                            "metrics": quality_metrics,
                            "alert_type": "low_quality",
                        }
                    )

            if quality_metrics["clipping_detected"]:
                if self.on_quality_alert:
                    self.on_quality_alert(
                        {
                            "session_id": self.session_id,
                            "alert_type": "clipping_detected",
                            "peak_level": quality_metrics["peak_level"],
                        }
                    )

            # Callback for raw audio
            if self.on_audio_chunk:
                self.on_audio_chunk(audio_chunk, quality_metrics)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def _capture_loop(self):
        """Main capture loop for processing audio chunks."""
        logger.info("Audio capture loop started")

        chunk_interval = self.config.chunk_duration
        last_process_time = time.time()

        try:
            while self.is_capturing:
                current_time = time.time()

                if current_time - last_process_time >= chunk_interval:
                    # Get audio chunk for processing
                    audio_data = self.audio_buffer.get_duration_audio(chunk_interval)

                    if len(audio_data) > 0:
                        # Process chunk
                        asyncio.create_task(
                            self._process_audio_chunk(audio_data, current_time)
                        )

                    last_process_time = current_time

                time.sleep(0.1)  # Small sleep to prevent busy waiting

        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
            if self.on_error:
                self.on_error(f"Capture loop error: {e}")

        logger.info("Audio capture loop ended")

    async def _process_audio_chunk(self, audio_data: np.ndarray, timestamp: float):
        """Process audio chunk and send to whisper service."""
        try:
            # Convert to bytes
            audio_bytes = self._audio_to_bytes(audio_data)

            # Store audio file if database is available
            if self.database_manager and self.bot_manager:
                try:
                    file_id = await self.bot_manager.store_audio_file(
                        self.session_id,
                        audio_bytes,
                        metadata={
                            "duration_seconds": len(audio_data)
                            / self.config.sample_rate,
                            "sample_rate": self.config.sample_rate,
                            "channels": self.config.channels,
                            "chunk_start_time": timestamp - self.config.chunk_duration,
                            "chunk_end_time": timestamp,
                            "audio_quality_score": self.average_quality_score,
                        },
                    )
                    if file_id:
                        logger.debug(f"Stored audio chunk: {file_id}")
                except Exception as e:
                    logger.warning(f"Failed to store audio chunk: {e}")

            # Send to whisper service
            await self._send_to_whisper_service(audio_bytes, timestamp)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    def _audio_to_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert audio data to bytes."""
        try:
            # Convert to int16 for efficient transmission
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Create WAV file in memory
            buffer = io.BytesIO()
            sf.write(buffer, audio_int16, self.config.sample_rate, format="WAV")
            buffer.seek(0)

            return buffer.read()

        except Exception as e:
            logger.error(f"Error converting audio to bytes: {e}")
            return b""

    async def _create_whisper_session(self) -> bool:
        """Create session with whisper service."""
        try:
            session_data = {
                "session_id": self.session_id,
                "meeting_info": asdict(self.meeting_info) if self.meeting_info else {},
                "audio_config": {
                    "sample_rate": self.config.sample_rate,
                    "channels": self.config.channels,
                    "format": self.config.audio_format,
                },
                "real_time_mode": True,
                "bot_mode": True,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.whisper_service_url}/api/sessions/create",
                    json=session_data,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info(f"Created whisper session: {self.session_id}")
                    return True
                else:
                    logger.error(
                        f"Failed to create whisper session: {response.status_code}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error creating whisper session: {e}")
            return False

    async def _send_to_whisper_service(self, audio_bytes: bytes, timestamp: float):
        """Send audio chunk to whisper service for transcription."""
        try:
            files = {"audio": ("chunk.wav", audio_bytes, "audio/wav")}

            data = {
                "session_id": self.session_id,
                "timestamp": timestamp,
                "chunk_duration": self.config.chunk_duration,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.whisper_service_url}/api/transcribe/stream",
                    files=files,
                    data=data,
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()

                    # Store transcript if available
                    if result.get("clean_text") and self.bot_manager:
                        try:
                            transcript_data = {
                                "text": result["clean_text"],
                                "source_type": "whisper_service",
                                "language": result.get("language", "en"),
                                "start_timestamp": result.get(
                                    "window_start_time",
                                    timestamp - self.config.chunk_duration,
                                ),
                                "end_timestamp": result.get(
                                    "window_end_time", timestamp
                                ),
                                "speaker_info": result.get("speaker_info"),
                                "metadata": {
                                    "confidence_score": result.get("confidence", 0.0),
                                    "processing_time": result.get(
                                        "processing_time", 0.0
                                    ),
                                    "chunk_id": result.get("segment_id"),
                                },
                            }

                            transcript_id = await self.bot_manager.store_transcript(
                                self.session_id, transcript_data
                            )

                            if transcript_id:
                                logger.debug(f"Stored transcript: {transcript_id}")

                        except Exception as e:
                            logger.warning(f"Failed to store transcript: {e}")

                    # Callback for transcription results
                    if self.on_transcription_ready and result.get("clean_text"):
                        self.on_transcription_ready(result)

                else:
                    logger.warning(f"Whisper service error: {response.status_code}")

        except Exception as e:
            logger.error(f"Error sending to whisper service: {e}")

    async def _close_whisper_session(self):
        """Close whisper service session."""
        try:
            if not self.session_id:
                return

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.whisper_service_url}/api/sessions/{self.session_id}/close",
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info(f"Closed whisper session: {self.session_id}")
                else:
                    logger.warning(
                        f"Failed to close whisper session: {response.status_code}"
                    )

        except Exception as e:
            logger.warning(f"Error closing whisper session: {e}")

    def get_capture_stats(self) -> Dict[str, Any]:
        """Get comprehensive capture statistics."""
        duration = time.time() - self.start_time if self.start_time else 0

        return {
            "session_id": self.session_id,
            "is_capturing": self.is_capturing,
            "total_chunks_captured": self.total_chunks_captured,
            "total_audio_duration": duration,
            "average_quality_score": self.average_quality_score,
            "config": asdict(self.config),
            "meeting_info": asdict(self.meeting_info) if self.meeting_info else None,
            "buffer_size": len(self.audio_buffer.buffer),
        }


# Factory function
def create_audio_capture(
    config: AudioConfig,
    whisper_service_url: str,
    bot_manager=None,
    database_manager=None,
) -> GoogleMeetAudioCapture:
    """Create a Google Meet audio capture instance."""
    return GoogleMeetAudioCapture(
        config, whisper_service_url, bot_manager, database_manager
    )


# Example usage
async def main():
    """Example usage of Google Meet audio capture."""
    config = AudioConfig(sample_rate=16000, channels=1, chunk_duration=2.0)

    capture = create_audio_capture(config, "http://localhost:5001")

    meeting = MeetingInfo(
        meeting_id="test-meeting-123", meeting_title="Test Audio Capture"
    )

    # Set callbacks
    def on_transcription(result):
        print(f"Transcription: {result.get('clean_text', 'No text')}")

    def on_quality_alert(alert):
        print(
            f"Quality alert: {alert['alert_type']} - Score: {alert.get('quality_score', 'N/A')}"
        )

    capture.set_transcription_callback(on_transcription)
    capture.set_quality_alert_callback(on_quality_alert)

    # Start capture
    success = await capture.start_capture(meeting)
    if success:
        print("Audio capture started")

        # Run for 30 seconds
        await asyncio.sleep(30)

        # Stop capture
        await capture.stop_capture()

        # Get stats
        stats = capture.get_capture_stats()
        print(f"Capture stats: {json.dumps(stats, indent=2, default=str)}")
    else:
        print("Failed to start audio capture")


if __name__ == "__main__":
    asyncio.run(main())
