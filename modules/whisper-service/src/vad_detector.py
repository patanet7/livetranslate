#!/usr/bin/env python3
"""
Silero VAD Integration

Provides Voice Activity Detection using Silero VAD model to filter out
silence and noise before Whisper transcription.

Based on SimulStreaming reference: whisper_streaming/silero_vad_iterator.py
Implements FixedVADIterator pattern for chunk-by-chunk filtering.
"""

import logging
import os
import torch
import numpy as np
from typing import Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class VADIterator:
    """
    Silero VAD Iterator - detects speech start/end in streaming audio.

    Based on SimulStreaming's implementation.
    """

    def __init__(self,
                 model,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500,
                 speech_pad_ms: int = 100):
        """
        Initialize VAD Iterator.

        Args:
            model: Silero VAD model
            threshold: Speech probability threshold (0.5 default)
            sampling_rate: 16000 (required by Silero VAD)
            min_silence_duration_ms: Wait this long before ending speech
            speech_pad_ms: Padding around speech segments
        """
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in [8000, 16000]:
            raise ValueError('VADIterator only supports 8000 or 16000 Hz')

        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        self.reset_states()

    def reset_states(self):
        """Reset VAD state for new audio stream."""
        self.model.reset_states()
        self.triggered = False  # Currently in speech segment
        self.temp_end = 0       # Temporary end position
        self.current_sample = 0 # Current position in stream

    @torch.no_grad()
    def __call__(self, x: np.ndarray, return_seconds: bool = False) -> Optional[Dict]:
        """
        Process audio chunk and detect speech start/end.

        Args:
            x: Audio chunk (512 samples for Silero VAD)
            return_seconds: Return timestamps in seconds vs samples

        Returns:
            {'start': timestamp} - Speech started
            {'end': timestamp} - Speech ended
            None - No change in speech status
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        # Get speech probability from Silero VAD
        speech_prob = self.model(x, self.sampling_rate).item()

        # Cancel temp_end if speech continues
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        # Speech START detected
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {'start': int(speech_start) if not return_seconds else round(speech_start / self.sampling_rate, 1)}

        # Speech END detected (with hysteresis: threshold - 0.15)
        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample

            # Wait for min_silence_duration before confirming end
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False
                return {'end': int(speech_end) if not return_seconds else round(speech_end / self.sampling_rate, 1)}

        return None


class FixedVADIterator(VADIterator):
    """
    Fixed VAD Iterator - handles arbitrary chunk sizes.

    Silero VAD requires exactly 512-sample chunks, but incoming audio
    can be any size. This class buffers audio to 512-sample chunks.
    """

    def reset_states(self):
        """Reset state including internal buffer."""
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x: np.ndarray, return_seconds: bool = False) -> Optional[Dict]:
        """
        Process audio of any length by buffering to 512-sample chunks.

        Args:
            x: Audio chunk (any length)
            return_seconds: Return timestamps in seconds

        Returns:
            Dict with 'start' and/or 'end' keys, or None
        """
        incoming_len = len(x)
        self.buffer = np.append(self.buffer, x)
        buffer_size_after = len(self.buffer)

        logger.info(f"[FixedVAD] Received {incoming_len} samples, buffer now {buffer_size_after} samples")

        ret = None

        # Process all complete 512-sample chunks
        chunks_processed = 0
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            chunks_processed += 1

            if r is not None:
                logger.info(f"[FixedVAD] Chunk {chunks_processed} result: {r}")

            if ret is None:
                ret = r
            elif r is not None:
                # Merge multiple events
                if 'end' in r:
                    ret['end'] = r['end']  # Use latest end
                if 'start' in r and 'end' in ret:
                    # Remove end - merging segments
                    del ret['end']

        if chunks_processed > 0:
            logger.info(f"[FixedVAD] Processed {chunks_processed} chunks, {len(self.buffer)} samples remaining")

        return ret if ret != {} else None


class SileroVAD:
    """
    Silero VAD wrapper using FixedVADIterator pattern.

    Prevents Whisper hallucinations by filtering out silence BEFORE
    adding audio to the buffer (SimulStreaming pattern).
    """

    def __init__(self,
                 threshold: float = 0.5,
                 sampling_rate: int = 16000,
                 min_silence_duration_ms: int = 500):
        """
        Initialize Silero VAD with FixedVADIterator.

        Args:
            threshold: Speech probability threshold (0.0-1.0)
            sampling_rate: Audio sample rate (16000 Hz required)
            min_silence_duration_ms: Silence duration before ending speech
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        # Load Silero VAD model
        try:
            logger.info("[VAD] Loading Silero VAD model...")

            # Set custom cache directory to .models/silero-vad
            cache_dir = Path(__file__).parent.parent / '.models' / 'silero-vad'
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Set torch hub directory to our custom cache
            torch.hub.set_dir(str(cache_dir.parent))

            logger.info(f"[VAD] Using cache directory: {cache_dir.parent}")

            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            model.eval()

            # Create FixedVADIterator (handles arbitrary chunk sizes)
            self.vad_iterator = FixedVADIterator(
                model,
                threshold=threshold,
                sampling_rate=sampling_rate,
                min_silence_duration_ms=min_silence_duration_ms
            )

            logger.info(f"[VAD] ✅ Silero VAD loaded (threshold={threshold}, min_silence={min_silence_duration_ms}ms)")

        except Exception as e:
            logger.error(f"[VAD] ❌ Failed to load Silero VAD: {e}")
            self.vad_iterator = None

    def reset(self):
        """Reset VAD state for new session."""
        if self.vad_iterator:
            self.vad_iterator.reset_states()

    def check_speech(self, audio: np.ndarray) -> Optional[Dict]:
        """
        Check if audio chunk contains speech (pre-filter pattern).

        Args:
            audio: Audio chunk (any length, float32, mono, 16kHz)

        Returns:
            {'start': timestamp} - Speech started
            {'end': timestamp} - Speech ended
            None - No change in speech status
        """
        if self.vad_iterator is None:
            return None  # VAD not available

        try:
            # Log audio statistics for debugging
            rms = np.sqrt(np.mean(audio ** 2))
            max_amp = np.max(np.abs(audio))
            logger.info(f"[VAD] Audio chunk: RMS={rms:.6f}, Max={max_amp:.6f}, Length={len(audio)} samples")

            result = self.vad_iterator(audio, return_seconds=True)

            if result is not None:
                logger.info(f"[VAD] ✅ Detection: {result}")
            else:
                logger.info(f"[VAD] ⏸️ No state change (waiting for more audio)")

            return result
        except Exception as e:
            logger.warning(f"[VAD] Error checking speech: {e}")
            return None


# Global instance for easy access
_global_vad: Optional[SileroVAD] = None


def get_vad(threshold: float = 0.5) -> Optional[SileroVAD]:
    """
    Get or create global VAD instance.

    Args:
        threshold: Speech probability threshold

    Returns:
        SileroVAD instance or None if failed to load
    """
    global _global_vad

    if _global_vad is None:
        try:
            _global_vad = SileroVAD(threshold=threshold)
        except Exception as e:
            logger.error(f"[VAD] Failed to create VAD instance: {e}")
            return None

    return _global_vad
