"""Audio downsampler for the meeting pipeline.

Converts native-quality audio (48kHz+ stereo) to 16kHz mono for the
transcription service. Uses scipy.signal.resample for quality.

The downsampled audio is ONLY for transcription — recordings always stay
at native quality.
"""
from __future__ import annotations

import numpy as np


def downsample_to_16k(
    audio: np.ndarray,
    source_rate: int,
    channels: int = 1,
    target_rate: int = 16000,
) -> np.ndarray:
    """Downsample audio to 16kHz mono float32.

    Args:
        audio: Input audio array. Shape ``(samples,)`` for mono or
               ``(samples, channels)`` for stereo / multi-channel.
        source_rate: Source sample rate in Hz.
        channels: Number of channels (1 = mono, 2 = stereo).  The value is
                  informational — channel count is always detected from the
                  actual array shape.
        target_rate: Target sample rate in Hz (default 16 000).

    Returns:
        1-D float32 array at ``target_rate``, mono.
    """
    # Mix stereo / multi-channel down to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # Fast-path: already at the target rate
    if source_rate == target_rate:
        return audio.astype(np.float32)

    # Resample using scipy for high-quality anti-aliased downsampling
    from scipy.signal import resample

    num_samples = int(round(len(audio) * target_rate / source_rate))
    resampled = resample(audio, num_samples)

    return resampled.astype(np.float32)
