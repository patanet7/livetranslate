#!/usr/bin/env python3
"""
Silero VAD Integration

Provides Voice Activity Detection using Silero VAD model to filter out
silence and noise before Whisper transcription.

Based on SimulStreaming reference: whisper_streaming/silero_vad_iterator.py
"""

import logging
import torch
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD wrapper for speech detection.

    Prevents Whisper hallucinations on silence/noise by detecting
    whether audio contains actual speech before transcription.
    """

    def __init__(self,
                 threshold: float = 0.5,          # Speech probability threshold
                 sampling_rate: int = 16000,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100):
        """
        Initialize Silero VAD.

        Args:
            threshold: Speech probability threshold (0.0-1.0)
            sampling_rate: Audio sample rate (must be 8000 or 16000)
            min_speech_duration_ms: Minimum speech duration to trigger speech
            min_silence_duration_ms: Minimum silence duration to trigger non-speech
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms

        # Load Silero VAD model
        try:
            logger.info("[VAD] Loading Silero VAD model...")
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )

            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils

            self.model.eval()
            logger.info(f"[VAD] ✅ Silero VAD loaded (threshold={threshold})")

        except Exception as e:
            logger.error(f"[VAD] ❌ Failed to load Silero VAD: {e}")
            self.model = None

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Check if audio contains speech.

        Args:
            audio: Audio array (float32, mono, 16kHz)

        Returns:
            True if speech detected, False otherwise
        """
        if self.model is None:
            # VAD not available, assume speech
            return True

        try:
            # Convert to torch tensor
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio).float()
            else:
                audio_tensor = audio

            # Silero VAD expects specific shape
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

            # Get speech probability
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sampling_rate).item()

            is_speech = speech_prob > self.threshold

            logger.debug(f"[VAD] Speech prob: {speech_prob:.3f}, is_speech: {is_speech}")

            return is_speech

        except Exception as e:
            logger.warning(f"[VAD] Error detecting speech: {e}, assuming speech")
            return True

    def has_speech_in_buffer(self, audio_segments: list) -> bool:
        """
        Check if buffer contains any speech.

        Args:
            audio_segments: List of audio tensors

        Returns:
            True if any segment contains speech
        """
        if not audio_segments:
            return False

        try:
            # Concatenate all segments
            full_audio = torch.cat(audio_segments, dim=0)

            # Check entire buffer
            return self.is_speech(full_audio.numpy())

        except Exception as e:
            logger.warning(f"[VAD] Error checking buffer: {e}, assuming speech")
            return True

    def get_speech_ratio(self, audio: np.ndarray, chunk_size_ms: int = 30) -> float:
        """
        Get ratio of speech frames in audio.

        Args:
            audio: Audio array (float32, mono, 16kHz)
            chunk_size_ms: Chunk size for frame-level analysis

        Returns:
            Ratio of speech frames (0.0-1.0)
        """
        if self.model is None:
            return 1.0

        try:
            # Calculate chunk size in samples
            chunk_samples = int(self.sampling_rate * chunk_size_ms / 1000)

            # Split into chunks
            speech_frames = 0
            total_frames = 0

            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i:i + chunk_samples]
                if len(chunk) < chunk_samples:
                    continue  # Skip incomplete chunks

                if self.is_speech(chunk):
                    speech_frames += 1
                total_frames += 1

            if total_frames == 0:
                return 0.0

            ratio = speech_frames / total_frames
            logger.debug(f"[VAD] Speech ratio: {ratio:.2%} ({speech_frames}/{total_frames} frames)")

            return ratio

        except Exception as e:
            logger.warning(f"[VAD] Error calculating speech ratio: {e}")
            return 1.0

    def filter_silence(self, audio: np.ndarray, min_speech_ratio: float = 0.3) -> Optional[np.ndarray]:
        """
        Filter out audio with low speech content.

        Args:
            audio: Audio array (float32, mono, 16kHz)
            min_speech_ratio: Minimum speech ratio to keep audio

        Returns:
            Audio if speech ratio > threshold, None otherwise
        """
        speech_ratio = self.get_speech_ratio(audio)

        if speech_ratio < min_speech_ratio:
            logger.info(f"[VAD] 🔇 Filtered silence (speech ratio: {speech_ratio:.2%} < {min_speech_ratio:.2%})")
            return None

        return audio

    def reset(self):
        """Reset VAD state (if needed)."""
        # Silero VAD is stateless for simple detection
        pass


# Global instance for easy access
_global_vad: Optional[SileroVAD] = None


def get_vad(threshold: float = 0.5) -> SileroVAD:
    """
    Get or create global VAD instance.

    Args:
        threshold: Speech probability threshold

    Returns:
        SileroVAD instance
    """
    global _global_vad

    if _global_vad is None:
        _global_vad = SileroVAD(threshold=threshold)

    return _global_vad
