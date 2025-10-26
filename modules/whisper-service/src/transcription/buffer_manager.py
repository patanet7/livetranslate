#!/usr/bin/env python3
"""
Audio Buffer Management

Simple audio buffer manager for legacy compatibility.
Extracted from whisper_service.py for better modularity.

NOTE: This simple AudioBufferManager class is deprecated - use RollingBufferManager
from buffer_manager.py instead for full functionality. Kept for backward compatibility only.
"""

import logging
import threading
import numpy as np
import webrtcvad
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SimpleAudioBufferManager:
    """
    DEPRECATED: Simple audio buffer manager (legacy)
    Use RollingBufferManager from buffer_manager.py for full functionality
    """

    def __init__(self, buffer_duration: float = 6.0, sample_rate: int = 16000, enable_vad: bool = True):
        """Initialize audio buffer manager"""
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        self.max_samples = int(buffer_duration * sample_rate)
        self.enable_vad = enable_vad

        # Rolling buffer for audio samples
        self.audio_buffer = deque(maxlen=self.max_samples)
        self.buffer_lock = threading.Lock()

        # VAD setup
        if enable_vad:
            try:
                self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
                self.vad_enabled = True
                logger.info("✓ Voice Activity Detection enabled")
            except:
                self.vad = None
                self.vad_enabled = False
                logger.warning("⚠ Voice Activity Detection not available")
        else:
            self.vad = None
            self.vad_enabled = False

        # Audio processing
        self.last_processed_time = 0

    def add_audio_chunk(self, audio_samples: np.ndarray) -> int:
        """Add new audio samples to the rolling buffer"""
        with self.buffer_lock:
            if isinstance(audio_samples, np.ndarray):
                # Convert to list and extend buffer
                samples_list = audio_samples.tolist()
                self.audio_buffer.extend(samples_list)
                return len(self.audio_buffer)
            return 0

    def get_buffer_audio(self) -> np.ndarray:
        """Get current buffer as numpy array"""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([])
            return np.array(list(self.audio_buffer))

    def find_speech_boundaries(self, audio_array: np.ndarray, chunk_duration: float = 0.02) -> Tuple[Optional[int], Optional[int]]:
        """Find speech boundaries using VAD"""
        if not self.vad_enabled or len(audio_array) == 0:
            return None, None

        try:
            # Convert to 16-bit PCM for VAD
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Process in 20ms chunks (VAD requirement)
            chunk_samples = int(self.sample_rate * chunk_duration)
            speech_chunks = []

            for i in range(0, len(audio_int16), chunk_samples):
                chunk = audio_int16[i:i + chunk_samples]
                if len(chunk) == chunk_samples:
                    # VAD expects specific sample rates
                    if self.sample_rate in [8000, 16000, 32000, 48000]:
                        is_speech = self.vad.is_speech(chunk.tobytes(), self.sample_rate)
                        speech_chunks.append((i, i + chunk_samples, is_speech))

            # Find speech boundaries
            speech_start = None
            speech_end = None

            for start, end, is_speech in speech_chunks:
                if is_speech and speech_start is None:
                    speech_start = start
                elif not is_speech and speech_start is not None:
                    speech_end = end
                    break

            return speech_start, speech_end

        except Exception as e:
            logger.debug(f"VAD processing failed: {e}")
            return None, None

    def clear_buffer(self):
        """Clear the audio buffer"""
        with self.buffer_lock:
            self.audio_buffer.clear()
