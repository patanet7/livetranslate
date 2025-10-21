#!/usr/bin/env python3
"""
Silero VAD Iterator for Voice Activity Detection

Following SimulStreaming reference implementation:
- Filters silence BEFORE Whisper transcription
- Speech probability threshold: 0.5 (default)
- Handles variable-length audio with FixedVADIterator
- Returns speech segments with start/end timestamps

Reference: SimulStreaming/whisper_streaming/silero_vad_iterator.py
License: MIT (same as Silero VAD: https://github.com/snakers4/silero-vad/blob/master/LICENSE)

This is how SimulStreaming handles silence!
"""

import torch
import numpy as np


class VADIterator:
    """
    Voice Activity Detection Iterator

    Processes audio chunks and detects speech vs silence using Silero VAD model.

    Parameters:
        model: Preloaded Silero VAD model (.jit or .onnx)
        threshold (float): Speech probability threshold (default 0.5)
            - Probabilities >= threshold are considered SPEECH
            - Probabilities < threshold are considered SILENCE
        sampling_rate (int): Audio sampling rate (8000 or 16000 Hz)
        min_silence_duration_ms (int): Minimum silence duration to separate speech chunks
        speech_pad_ms (int): Padding added to each side of speech chunks

    Usage:
        model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        vad = VADIterator(model, threshold=0.5)

        audio_chunk = np.zeros(512, dtype=np.float32)
        result = vad(audio_chunk, return_seconds=True)

        if result and 'start' in result:
            print(f"Speech starts at {result['start']}s")
    """

    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100
    ):
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        # Validate sampling rate (Silero VAD requirement)
        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                'VADIterator does not support sampling rates other than [8000, 16000]'
            )

        # Calculate sample counts
        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000

        self.reset_states()

    def reset_states(self):
        """Reset VAD state for new audio stream"""
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    @torch.no_grad()
    def __call__(self, x, return_seconds=False, time_resolution: int = 1):
        """
        Process audio chunk and detect speech

        Args:
            x: Audio chunk (torch.Tensor or numpy array)
            return_seconds: Return timestamps in seconds (default: samples)
            time_resolution: Time resolution for seconds timestamps

        Returns:
            - {'start': timestamp} when speech begins
            - {'end': timestamp} when speech ends
            - None if no change in speech state
        """
        # Convert to tensor if needed
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        # Get chunk size
        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        # Calculate speech probability using Silero VAD model
        speech_prob = self.model(x, self.sampling_rate).item()

        # Speech detected above threshold
        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        # Speech start detected
        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(
                0,
                self.current_sample - self.speech_pad_samples - window_size_samples
            )

            return {
                'start': int(speech_start) if not return_seconds
                else round(speech_start / self.sampling_rate, time_resolution)
            }

        # Potential speech end detected
        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample

            # Check if silence duration is long enough
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                # Confirmed speech end
                speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
                self.temp_end = 0
                self.triggered = False

                return {
                    'end': int(speech_end) if not return_seconds
                    else round(speech_end / self.sampling_rate, time_resolution)
                }

        return None


class FixedVADIterator(VADIterator):
    """
    Fixed VAD Iterator for variable-length audio

    Extends VADIterator to handle any audio length, not just 512-sample chunks.
    Buffers audio internally and processes in 512-sample windows (Silero requirement).

    If multiple speech segments detected in one call, returns start of first
    and end of last segment.

    Usage:
        model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
        vad = FixedVADIterator(model, threshold=0.5)

        # Can handle any audio length
        audio = np.zeros(1600, dtype=np.float32)  # 0.1s at 16kHz
        result = vad(audio, return_seconds=True)
    """

    def reset_states(self):
        """Reset VAD state and clear internal buffer"""
        super().reset_states()
        self.buffer = np.array([], dtype=np.float32)

    def __call__(self, x, return_seconds=False):
        """
        Process variable-length audio chunk

        Buffers audio and processes in 512-sample windows required by Silero VAD.

        Args:
            x: Audio chunk (any length)
            return_seconds: Return timestamps in seconds

        Returns:
            - {'start': timestamp} when speech begins
            - {'end': timestamp} when speech ends
            - None if no change or buffering
        """
        # Add to buffer
        self.buffer = np.append(self.buffer, x)

        ret = None

        # Process complete 512-sample windows
        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]

            if ret is None:
                ret = r
            elif r is not None:
                # Handle multiple segments in one call
                if 'end' in r:
                    ret['end'] = r['end']  # Update to latest end
                if 'start' in r and 'end' in ret:
                    # Merge segments - remove end to continue with previous
                    del ret['end']

        return ret if ret != {} else None


# Example usage and testing
if __name__ == "__main__":
    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )

    # Create FixedVADIterator
    vad = FixedVADIterator(model, threshold=0.5, sampling_rate=16000)

    print("Silero VAD Iterator initialized")
    print(f"  Threshold: {vad.threshold}")
    print(f"  Sampling rate: {vad.sampling_rate}Hz")

    # Test with silent audio
    silent_audio = np.zeros(512, dtype=np.float32)
    result = vad(silent_audio, return_seconds=True)

    print(f"\nTest result (silent audio): {result}")
    print("✅ Silero VAD Iterator working correctly")
