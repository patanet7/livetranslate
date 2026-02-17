#!/usr/bin/env python3
"""
Audio Utility Functions

Provides audio preprocessing utilities for the Whisper service.
Extracted from whisper_service.py for better modularity and testability.
"""

import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
from livetranslate_common.logging import get_logger

logger = get_logger()


def load_audio_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load audio from bytes and convert to mono if stereo.

    Uses soundfile instead of librosa.load to avoid aifc dependency (Python 3.13 compatible).

    Args:
        audio_bytes: Audio data as bytes (WAV format)

    Returns:
        Tuple of (audio_data, sample_rate)
        - audio_data: Audio as numpy array (mono, float32)
        - sample_rate: Sample rate in Hz
    """
    # Create temporary file for soundfile to read
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file.flush()

        try:
            # Use soundfile instead of librosa.load to avoid aifc dependency (Python 3.13)
            audio_data, sample_rate = sf.read(tmp_file.name, dtype="float32")

            # soundfile returns stereo as (samples, channels), we need (samples,)
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]  # Take first channel
                logger.debug(f"[AUDIO] Converted stereo to mono (shape: {audio_data.shape})")

            return audio_data, sample_rate

        finally:
            # Clean up temporary file
            os.unlink(tmp_file.name)


def ensure_sample_rate(audio_data: np.ndarray, current_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to target sample rate if needed.

    Args:
        audio_data: Audio as numpy array
        current_sr: Current sample rate in Hz
        target_sr: Target sample rate in Hz

    Returns:
        Resampled audio data (or original if sample rates match)
    """
    if current_sr == target_sr:
        return audio_data

    logger.debug(f"[AUDIO] Resampling from {current_sr}Hz to {target_sr}Hz")
    resampled = librosa.resample(audio_data, orig_sr=current_sr, target_sr=target_sr)
    return resampled
