#!/usr/bin/env python3
"""
Comprehensive Audio Test Data Fixtures

This module provides comprehensive test data generation and management for
audio processing tests, including various formats, quality levels, edge cases,
and realistic audio scenarios.
"""

import hashlib
import io
import json
import logging
import struct
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import UploadFile

logger = logging.getLogger(__name__)


@dataclass
class AudioTestCase:
    """Represents a single audio test case with metadata."""

    name: str
    description: str
    format_type: str
    sample_rate: int
    duration: float
    channels: int
    expected_text: str | None = None
    expected_language: str | None = "en"
    expected_speaker_count: int | None = 1
    quality_level: str = "medium"  # low, medium, high
    contains_speech: bool = True
    noise_level: float = 0.01
    corruption_type: str | None = None  # None, "partial", "header", "data"
    file_size_bytes: int | None = None
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class AudioSignalGenerator:
    """Advanced audio signal generation for testing various scenarios."""

    @staticmethod
    def generate_speech_like(
        duration: float,
        sample_rate: int = 16000,
        fundamental_freq: float = 120,
        formants: list[float] | None = None,
        amplitude: float = 0.7,
        noise_level: float = 0.01,
    ) -> np.ndarray:
        """Generate realistic speech-like audio signal."""
        if formants is None:
            formants = [800, 1200, 2400]  # Typical vowel formants

        t = np.arange(int(duration * sample_rate)) / sample_rate

        # Base fundamental frequency with natural variation
        freq_variation = 1 + 0.1 * np.sin(2 * np.pi * 2 * t)  # 2Hz variation
        instantaneous_freq = fundamental_freq * freq_variation

        # Generate fundamental with harmonics
        signal = np.zeros_like(t)

        # Fundamental frequency
        signal += 0.4 * amplitude * np.sin(2 * np.pi * instantaneous_freq * t)

        # Harmonics with decreasing amplitude
        for harmonic in range(2, 8):
            harmonic_amp = amplitude / (harmonic**1.5)
            signal += harmonic_amp * np.sin(2 * np.pi * instantaneous_freq * harmonic * t)

        # Add formant resonances
        for i, formant in enumerate(formants):
            formant_amp = amplitude * (0.3 - i * 0.1)  # Decreasing formant strength
            signal += formant_amp * np.sin(2 * np.pi * formant * t)

        # Add realistic envelope (attack, sustain, decay)
        envelope = np.ones_like(t)
        attack_samples = int(0.1 * sample_rate)
        decay_samples = int(0.1 * sample_rate)

        if len(t) > attack_samples:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if len(t) > decay_samples:
            envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

        signal *= envelope

        # Add realistic noise
        if noise_level > 0:
            noise = noise_level * np.random.randn(len(signal))
            signal += noise

        # Normalize to prevent clipping
        signal = signal / np.max(np.abs(signal)) * 0.9

        return signal.astype(np.float32)

    @staticmethod
    def generate_multi_speaker_audio(
        duration: float,
        speaker_configs: list[dict],
        sample_rate: int = 16000,
        overlap_factor: float = 0.1,
    ) -> np.ndarray:
        """Generate multi-speaker audio with overlapping speech."""
        total_samples = int(duration * sample_rate)
        signal = np.zeros(total_samples, dtype=np.float32)

        for speaker_config in speaker_configs:
            speaker_duration = speaker_config.get("duration", duration * 0.8)
            start_time = speaker_config.get("start_time", 0.0)
            fundamental_freq = speaker_config.get("fundamental_freq", 120)
            amplitude = speaker_config.get("amplitude", 0.6)

            start_sample = int(start_time * sample_rate)
            speaker_samples = int(speaker_duration * sample_rate)
            end_sample = min(start_sample + speaker_samples, total_samples)

            if start_sample < total_samples:
                speaker_signal = AudioSignalGenerator.generate_speech_like(
                    speaker_duration, sample_rate, fundamental_freq, amplitude=amplitude
                )

                # Trim to fit
                available_samples = end_sample - start_sample
                if len(speaker_signal) > available_samples:
                    speaker_signal = speaker_signal[:available_samples]

                # Add to main signal
                signal[start_sample : start_sample + len(speaker_signal)] += speaker_signal

        # Normalize to prevent clipping
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.9

        return signal

    @staticmethod
    def generate_music_like(
        duration: float,
        sample_rate: int = 16000,
        chord_progression: list[list[float]] | None = None,
        amplitude: float = 0.6,
    ) -> np.ndarray:
        """Generate music-like audio for non-speech testing."""
        if chord_progression is None:
            # Simple C major chord progression
            chord_progression = [
                [261.63, 329.63, 392.00],  # C major
                [293.66, 369.99, 440.00],  # D minor
                [329.63, 415.30, 493.88],  # E minor
                [349.23, 440.00, 523.25],  # F major
            ]

        t = np.arange(int(duration * sample_rate)) / sample_rate
        signal = np.zeros_like(t)

        chord_duration = duration / len(chord_progression)

        for i, chord in enumerate(chord_progression):
            start_time = i * chord_duration
            end_time = (i + 1) * chord_duration

            chord_mask = (t >= start_time) & (t < end_time)
            chord_t = t[chord_mask] - start_time

            # Generate chord
            chord_signal = np.zeros_like(chord_t)
            for freq in chord:
                chord_signal += amplitude * np.sin(2 * np.pi * freq * chord_t) / len(chord)

            signal[chord_mask] = chord_signal

        # Add some rhythmic variation
        rhythm_freq = 4  # 4 beats per second
        rhythm_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rhythm_freq * t)
        signal *= rhythm_envelope

        return signal.astype(np.float32)

    @staticmethod
    def add_noise(
        signal: np.ndarray,
        noise_type: str = "white",
        snr_db: float = 20,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Add various types of noise to audio signal."""
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        if noise_type == "white":
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        elif noise_type == "pink":
            # Generate pink noise (1/f spectrum)
            white_noise = np.random.randn(len(signal))
            fft_noise = np.fft.fft(white_noise)
            freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)
            # Apply 1/f filter (avoid division by zero)
            filter_response = 1 / np.sqrt(np.abs(freqs) + 1)
            pink_fft = fft_noise * filter_response
            noise = np.real(np.fft.ifft(pink_fft))
            noise = noise / np.std(noise) * np.sqrt(noise_power)
        elif noise_type == "brown":
            # Generate brown noise (1/f^2 spectrum)
            white_noise = np.random.randn(len(signal))
            fft_noise = np.fft.fft(white_noise)
            freqs = np.fft.fftfreq(len(signal), 1 / sample_rate)
            filter_response = 1 / (np.abs(freqs) + 1)
            brown_fft = fft_noise * filter_response
            noise = np.real(np.fft.ifft(brown_fft))
            noise = noise / np.std(noise) * np.sqrt(noise_power)
        elif noise_type == "environmental":
            # Simulate environmental noise (cars, air conditioning, etc.)
            low_freq_noise = np.random.randn(len(signal))
            # Low-pass filter for environmental noise
            from scipy import signal as scipy_signal

            b, a = scipy_signal.butter(4, 500 / (sample_rate / 2), "low")
            noise = scipy_signal.filtfilt(b, a, low_freq_noise)
            noise = noise / np.std(noise) * np.sqrt(noise_power)
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))

        return signal + noise.astype(np.float32)

    @staticmethod
    def apply_effects(
        signal: np.ndarray, effects: list[dict], sample_rate: int = 16000
    ) -> np.ndarray:
        """Apply various audio effects for testing."""
        processed_signal = signal.copy()

        for effect in effects:
            effect_type = effect.get("type")

            if effect_type == "reverb":
                # Simple reverb simulation
                delay_samples = int(effect.get("delay_ms", 50) * sample_rate / 1000)
                decay = effect.get("decay", 0.3)

                reverb_signal = np.zeros_like(processed_signal)
                if delay_samples < len(processed_signal):
                    reverb_signal[delay_samples:] = processed_signal[:-delay_samples] * decay
                processed_signal += reverb_signal

            elif effect_type == "echo":
                # Echo effect
                delay_samples = int(effect.get("delay_ms", 200) * sample_rate / 1000)
                feedback = effect.get("feedback", 0.4)

                echo_signal = np.zeros_like(processed_signal)
                if delay_samples < len(processed_signal):
                    echo_signal[delay_samples:] = processed_signal[:-delay_samples] * feedback
                processed_signal += echo_signal

            elif effect_type == "distortion":
                # Simple distortion
                gain = effect.get("gain", 2.0)
                threshold = effect.get("threshold", 0.7)

                amplified = processed_signal * gain
                processed_signal = np.clip(amplified, -threshold, threshold)

            elif effect_type == "lowpass":
                # Simple lowpass filter
                from scipy import signal as scipy_signal

                cutoff = effect.get("cutoff_hz", 4000)
                order = effect.get("order", 4)

                nyquist = sample_rate / 2
                normalized_cutoff = cutoff / nyquist
                b, a = scipy_signal.butter(order, normalized_cutoff, "low")
                processed_signal = scipy_signal.filtfilt(b, a, processed_signal)

            elif effect_type == "highpass":
                # Simple highpass filter
                from scipy import signal as scipy_signal

                cutoff = effect.get("cutoff_hz", 80)
                order = effect.get("order", 4)

                nyquist = sample_rate / 2
                normalized_cutoff = cutoff / nyquist
                b, a = scipy_signal.butter(order, normalized_cutoff, "high")
                processed_signal = scipy_signal.filtfilt(b, a, processed_signal)

        return processed_signal.astype(np.float32)


class AudioFormatEncoder:
    """Encode audio signals into various formats for testing."""

    @staticmethod
    def encode_wav(signal: np.ndarray, sample_rate: int = 16000, bit_depth: int = 16) -> bytes:
        """Encode signal as WAV format."""
        if bit_depth == 16:
            pcm_data = (signal * 32767).astype(np.int16)
            sample_width = 2
        elif bit_depth == 24:
            pcm_data = (signal * 8388607).astype(np.int32)
            sample_width = 3
        elif bit_depth == 32:
            pcm_data = signal.astype(np.float32)
            sample_width = 4
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)

            if bit_depth == 32:
                # For 32-bit float, write as bytes
                wav_file.writeframes(pcm_data.tobytes())
            else:
                wav_file.writeframes(pcm_data.tobytes())

        return buffer.getvalue()

    @staticmethod
    def encode_raw_pcm(
        signal: np.ndarray,
        sample_rate: int = 16000,
        bit_depth: int = 16,
        little_endian: bool = True,
    ) -> bytes:
        """Encode signal as raw PCM."""
        if bit_depth == 16:
            pcm_data = (signal * 32767).astype(np.int16)
        elif bit_depth == 24:
            pcm_data = (signal * 8388607).astype(np.int32)
        elif bit_depth == 32:
            pcm_data = signal.astype(np.float32)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")

        if little_endian:
            return pcm_data.tobytes()
        else:
            return pcm_data.byteswap().tobytes()

    @staticmethod
    def encode_simulated_mp3(signal: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Create a simulated MP3 file for testing (not actual MP3 encoding)."""
        # MP3 frame header simulation
        mp3_header = struct.pack("<4s", b"ID3\x03")  # ID3v2.3 header
        mp3_header += struct.pack("<H", 0)  # Flags
        mp3_header += struct.pack("<L", 0)  # Size (will be updated)

        # Add some MP3-like frame data (simplified)
        frame_header = b"\xff\xfb\x90\x00"  # Simplified MP3 frame header

        # Convert audio to bytes (simplified encoding)
        pcm_data = (signal * 32767).astype(np.int16).tobytes()

        # Combine header and data
        mp3_data = mp3_header + frame_header + pcm_data

        return mp3_data

    @staticmethod
    def create_corrupted_file(
        original_data: bytes, corruption_type: str, corruption_ratio: float = 0.1
    ) -> bytes:
        """Create corrupted audio files for error testing."""
        if corruption_type == "header":
            # Corrupt the header
            corrupted = bytearray(original_data)
            header_size = min(44, len(corrupted))  # WAV header is typically 44 bytes
            for i in range(0, header_size, 4):
                if np.random.random() < corruption_ratio:
                    corrupted[i] = 0xFF
            return bytes(corrupted)

        elif corruption_type == "data":
            # Corrupt random parts of the data
            corrupted = bytearray(original_data)
            corruption_points = int(len(corrupted) * corruption_ratio)

            for _ in range(corruption_points):
                pos = np.random.randint(0, len(corrupted))
                corrupted[pos] = np.random.randint(0, 256)

            return bytes(corrupted)

        elif corruption_type == "truncated":
            # Truncate the file
            truncate_point = int(len(original_data) * (1 - corruption_ratio))
            return original_data[:truncate_point]

        elif corruption_type == "padded":
            # Add random padding
            padding_size = int(len(original_data) * corruption_ratio)
            padding = bytes(np.random.randint(0, 256, padding_size))
            return original_data + padding

        else:
            return original_data


class AudioTestDataManager:
    """Comprehensive test data management and generation."""

    def __init__(self):
        self.test_cases = {}
        self.generated_files = {}
        self.cache_dir = None
        self._setup_cache()

    def _setup_cache(self):
        """Setup cache directory for generated test files."""
        self.cache_dir = Path(tempfile.gettempdir()) / "audio_test_cache"
        self.cache_dir.mkdir(exist_ok=True)

    def generate_comprehensive_test_suite(self) -> dict[str, AudioTestCase]:
        """Generate comprehensive test suite with various audio scenarios."""
        test_cases = {}

        # Basic speech tests
        test_cases.update(self._generate_speech_tests())

        # Multi-speaker tests
        test_cases.update(self._generate_multi_speaker_tests())

        # Quality variation tests
        test_cases.update(self._generate_quality_tests())

        # Format compatibility tests
        test_cases.update(self._generate_format_tests())

        # Error scenario tests
        test_cases.update(self._generate_error_tests())

        # Performance tests
        test_cases.update(self._generate_performance_tests())

        # Edge case tests
        test_cases.update(self._generate_edge_case_tests())

        self.test_cases = test_cases
        return test_cases

    def _generate_speech_tests(self) -> dict[str, AudioTestCase]:
        """Generate basic speech audio tests."""
        tests = {}

        # Standard speech test
        tests["standard_speech"] = AudioTestCase(
            name="standard_speech",
            description="Clear speech with standard quality",
            format_type="wav",
            sample_rate=16000,
            duration=3.0,
            channels=1,
            expected_text="Hello world this is a test",
            quality_level="high",
            noise_level=0.01,
            content_type="audio/wav",
        )

        # Different languages (simulated)
        for lang, _freq in [
            ("english", 120),
            ("spanish", 130),
            ("french", 125),
            ("german", 115),
        ]:
            tests[f"speech_{lang}"] = AudioTestCase(
                name=f"speech_{lang}",
                description=f"Speech in {lang} language",
                format_type="wav",
                sample_rate=16000,
                duration=3.0,
                channels=1,
                expected_language=lang[:2],
                quality_level="medium",
                noise_level=0.02,
                content_type="audio/wav",
            )

        # Different speaking styles
        for style, params in [
            ("fast", {"duration": 2.0, "fundamental_freq": 140}),
            ("slow", {"duration": 5.0, "fundamental_freq": 100}),
            ("whisper", {"duration": 3.0, "fundamental_freq": 90}),
            ("loud", {"duration": 3.0, "fundamental_freq": 150}),
        ]:
            tests[f"speech_{style}"] = AudioTestCase(
                name=f"speech_{style}",
                description=f"Speech with {style} characteristics",
                format_type="wav",
                sample_rate=16000,
                duration=params["duration"],
                channels=1,
                quality_level="medium",
                noise_level=0.02,
                content_type="audio/wav",
            )

        return tests

    def _generate_multi_speaker_tests(self) -> dict[str, AudioTestCase]:
        """Generate multi-speaker audio tests."""
        tests = {}

        for speaker_count in [2, 3, 4]:
            tests[f"multi_speaker_{speaker_count}"] = AudioTestCase(
                name=f"multi_speaker_{speaker_count}",
                description=f"Audio with {speaker_count} speakers",
                format_type="wav",
                sample_rate=16000,
                duration=5.0,
                channels=1,
                expected_speaker_count=speaker_count,
                quality_level="medium",
                noise_level=0.03,
                content_type="audio/wav",
            )

        # Overlapping speakers
        tests["overlapping_speakers"] = AudioTestCase(
            name="overlapping_speakers",
            description="Multiple speakers with overlapping speech",
            format_type="wav",
            sample_rate=16000,
            duration=4.0,
            channels=1,
            expected_speaker_count=2,
            quality_level="medium",
            noise_level=0.04,
            content_type="audio/wav",
        )

        return tests

    def _generate_quality_tests(self) -> dict[str, AudioTestCase]:
        """Generate tests with various quality levels."""
        tests = {}

        quality_configs = [
            ("high_quality", 25, 0.005),  # High SNR, low noise
            ("medium_quality", 15, 0.02),  # Medium SNR, medium noise
            ("low_quality", 8, 0.1),  # Low SNR, high noise
            ("very_noisy", 3, 0.3),  # Very low SNR, very high noise
        ]

        for name, _snr_db, noise_level in quality_configs:
            tests[name] = AudioTestCase(
                name=name,
                description=f"Audio with {name.replace('_', ' ')}",
                format_type="wav",
                sample_rate=16000,
                duration=3.0,
                channels=1,
                quality_level=name.split("_")[0],
                noise_level=noise_level,
                content_type="audio/wav",
            )

        return tests

    def _generate_format_tests(self) -> dict[str, AudioTestCase]:
        """Generate tests for different audio formats."""
        tests = {}

        formats = [
            ("wav", "audio/wav"),
            ("mp3", "audio/mpeg"),
            ("webm", "audio/webm"),
            ("ogg", "audio/ogg"),
            ("mp4", "audio/mp4"),
            ("flac", "audio/flac"),
        ]

        for format_type, content_type in formats:
            tests[f"format_{format_type}"] = AudioTestCase(
                name=f"format_{format_type}",
                description=f"Audio in {format_type.upper()} format",
                format_type=format_type,
                sample_rate=16000,
                duration=3.0,
                channels=1,
                quality_level="medium",
                noise_level=0.02,
                content_type=content_type,
            )

        # Different sample rates
        for sample_rate in [8000, 16000, 22050, 44100, 48000]:
            tests[f"sample_rate_{sample_rate}"] = AudioTestCase(
                name=f"sample_rate_{sample_rate}",
                description=f"Audio at {sample_rate}Hz sample rate",
                format_type="wav",
                sample_rate=sample_rate,
                duration=3.0,
                channels=1,
                quality_level="medium",
                noise_level=0.02,
                content_type="audio/wav",
            )

        return tests

    def _generate_error_tests(self) -> dict[str, AudioTestCase]:
        """Generate error scenario tests."""
        tests = {}

        # Corrupted files
        corruption_types = ["header", "data", "truncated", "padded"]
        for corruption in corruption_types:
            tests[f"corrupted_{corruption}"] = AudioTestCase(
                name=f"corrupted_{corruption}",
                description=f"Audio with {corruption} corruption",
                format_type="wav",
                sample_rate=16000,
                duration=3.0,
                channels=1,
                corruption_type=corruption,
                quality_level="medium",
                noise_level=0.02,
                content_type="audio/wav",
            )

        # Edge cases
        tests["empty_file"] = AudioTestCase(
            name="empty_file",
            description="Empty audio file",
            format_type="wav",
            sample_rate=16000,
            duration=0.0,
            channels=1,
            quality_level="medium",
            content_type="audio/wav",
        )

        tests["very_short"] = AudioTestCase(
            name="very_short",
            description="Very short audio (0.1 seconds)",
            format_type="wav",
            sample_rate=16000,
            duration=0.1,
            channels=1,
            quality_level="medium",
            content_type="audio/wav",
        )

        tests["very_long"] = AudioTestCase(
            name="very_long",
            description="Very long audio (10 minutes)",
            format_type="wav",
            sample_rate=16000,
            duration=600.0,
            channels=1,
            quality_level="medium",
            content_type="audio/wav",
        )

        return tests

    def _generate_performance_tests(self) -> dict[str, AudioTestCase]:
        """Generate performance testing scenarios."""
        tests = {}

        durations = [1.0, 5.0, 10.0, 30.0, 60.0]
        for duration in durations:
            tests[f"performance_{int(duration)}s"] = AudioTestCase(
                name=f"performance_{int(duration)}s",
                description=f"Performance test with {duration}s audio",
                format_type="wav",
                sample_rate=16000,
                duration=duration,
                channels=1,
                quality_level="medium",
                noise_level=0.02,
                content_type="audio/wav",
            )

        return tests

    def _generate_edge_case_tests(self) -> dict[str, AudioTestCase]:
        """Generate edge case tests."""
        tests = {}

        # Silence
        tests["silence"] = AudioTestCase(
            name="silence",
            description="Pure silence",
            format_type="wav",
            sample_rate=16000,
            duration=3.0,
            channels=1,
            contains_speech=False,
            quality_level="high",
            noise_level=0.0,
            content_type="audio/wav",
        )

        # Pure tones
        tests["pure_tone"] = AudioTestCase(
            name="pure_tone",
            description="Pure sine wave tone",
            format_type="wav",
            sample_rate=16000,
            duration=3.0,
            channels=1,
            contains_speech=False,
            quality_level="high",
            noise_level=0.0,
            content_type="audio/wav",
        )

        # Music
        tests["music"] = AudioTestCase(
            name="music",
            description="Music-like audio",
            format_type="wav",
            sample_rate=16000,
            duration=3.0,
            channels=1,
            contains_speech=False,
            quality_level="medium",
            noise_level=0.01,
            content_type="audio/wav",
        )

        # Clipped audio
        tests["clipped"] = AudioTestCase(
            name="clipped",
            description="Heavily clipped audio",
            format_type="wav",
            sample_rate=16000,
            duration=3.0,
            channels=1,
            quality_level="low",
            noise_level=0.05,
            content_type="audio/wav",
        )

        return tests

    def generate_audio_data(self, test_case: AudioTestCase) -> bytes:
        """Generate actual audio data for a test case."""
        # Check cache first
        cache_key = self._get_cache_key(test_case)
        cache_path = self.cache_dir / f"{cache_key}.{test_case.format_type}"

        if cache_path.exists():
            return cache_path.read_bytes()

        # Generate audio signal based on test case
        if test_case.duration == 0.0:
            # Empty file
            signal = np.array([], dtype=np.float32)
        elif not test_case.contains_speech:
            if test_case.name == "silence":
                signal = np.zeros(int(test_case.duration * test_case.sample_rate), dtype=np.float32)
            elif test_case.name == "pure_tone":
                t = (
                    np.arange(int(test_case.duration * test_case.sample_rate))
                    / test_case.sample_rate
                )
                signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            elif test_case.name == "music":
                signal = AudioSignalGenerator.generate_music_like(
                    test_case.duration, test_case.sample_rate
                )
            else:
                signal = np.zeros(int(test_case.duration * test_case.sample_rate), dtype=np.float32)
        else:
            # Speech-like audio
            if test_case.expected_speaker_count and test_case.expected_speaker_count > 1:
                # Multi-speaker audio
                speaker_configs = []
                for i in range(test_case.expected_speaker_count):
                    speaker_configs.append(
                        {
                            "fundamental_freq": 100 + i * 20,
                            "start_time": i
                            * test_case.duration
                            / test_case.expected_speaker_count
                            * 0.8,
                            "duration": test_case.duration * 0.6,
                            "amplitude": 0.6 / test_case.expected_speaker_count,
                        }
                    )

                signal = AudioSignalGenerator.generate_multi_speaker_audio(
                    test_case.duration, speaker_configs, test_case.sample_rate
                )
            else:
                # Single speaker
                fundamental_freq = 120
                if "fast" in test_case.name:
                    fundamental_freq = 140
                elif "slow" in test_case.name:
                    fundamental_freq = 100
                elif "whisper" in test_case.name:
                    fundamental_freq = 90
                elif "loud" in test_case.name:
                    fundamental_freq = 150

                signal = AudioSignalGenerator.generate_speech_like(
                    test_case.duration,
                    test_case.sample_rate,
                    fundamental_freq=fundamental_freq,
                    noise_level=test_case.noise_level,
                )

        # Apply quality adjustments
        if test_case.quality_level == "low" or "noisy" in test_case.name:
            snr_db = 8 if test_case.quality_level == "low" else 3
            signal = AudioSignalGenerator.add_noise(signal, "white", snr_db, test_case.sample_rate)

        # Apply effects for specific test cases
        if test_case.name == "clipped":
            signal = np.clip(signal * 3.0, -0.8, 0.8)

        # Encode to specified format
        if test_case.format_type == "wav":
            audio_data = AudioFormatEncoder.encode_wav(signal, test_case.sample_rate)
        elif test_case.format_type == "mp3":
            audio_data = AudioFormatEncoder.encode_simulated_mp3(signal, test_case.sample_rate)
        else:
            # For other formats, use WAV as base and add format-specific header
            audio_data = AudioFormatEncoder.encode_wav(signal, test_case.sample_rate)
            audio_data = self._add_format_header(audio_data, test_case.format_type)

        # Apply corruption if specified
        if test_case.corruption_type:
            audio_data = AudioFormatEncoder.create_corrupted_file(
                audio_data, test_case.corruption_type, 0.1
            )

        # Cache the result
        cache_path.write_bytes(audio_data)

        return audio_data

    def _add_format_header(self, wav_data: bytes, format_type: str) -> bytes:
        """Add format-specific headers to audio data."""
        if format_type == "webm":
            webm_header = b"\x1a\x45\xdf\xa3"
            return webm_header + wav_data[44:]  # Skip WAV header
        elif format_type == "ogg":
            ogg_header = b"OggS"
            return ogg_header + wav_data[44:]
        elif format_type == "mp4":
            mp4_header = b"\x00\x00\x00\x20ftypmp41"
            return mp4_header + wav_data[44:]
        elif format_type == "flac":
            flac_header = b"fLaC"
            return flac_header + wav_data[44:]
        else:
            return wav_data

    def _get_cache_key(self, test_case: AudioTestCase) -> str:
        """Generate cache key for test case."""
        # Create hash of test case parameters
        test_dict = test_case.to_dict()
        test_json = json.dumps(test_dict, sort_keys=True)
        return hashlib.md5(test_json.encode()).hexdigest()

    def create_upload_file(self, test_case: AudioTestCase) -> UploadFile:
        """Create FastAPI UploadFile from test case."""
        audio_data = self.generate_audio_data(test_case)
        file_obj = io.BytesIO(audio_data)

        filename = f"{test_case.name}.{test_case.format_type}"
        content_type = test_case.content_type or f"audio/{test_case.format_type}"

        return UploadFile(
            filename=filename,
            file=file_obj,
            content_type=content_type,
            size=len(audio_data),
        )

    def get_test_cases_by_category(self, category: str) -> dict[str, AudioTestCase]:
        """Get test cases filtered by category."""
        if not self.test_cases:
            self.generate_comprehensive_test_suite()

        filtered = {}
        for name, test_case in self.test_cases.items():
            if category in test_case.name or category in test_case.description.lower():
                filtered[name] = test_case

        return filtered

    def get_performance_test_cases(self) -> dict[str, AudioTestCase]:
        """Get test cases specifically for performance testing."""
        return self.get_test_cases_by_category("performance")

    def get_error_test_cases(self) -> dict[str, AudioTestCase]:
        """Get test cases specifically for error testing."""
        return self.get_test_cases_by_category("corrupted")

    def get_format_test_cases(self) -> dict[str, AudioTestCase]:
        """Get test cases specifically for format testing."""
        return self.get_test_cases_by_category("format")

    def cleanup_cache(self):
        """Clean up cached test files."""
        if self.cache_dir and self.cache_dir.exists():
            for file_path in self.cache_dir.glob("*"):
                file_path.unlink()


# Pytest fixtures
@pytest.fixture(scope="session")
def audio_test_data_manager():
    """Provide audio test data manager."""
    manager = AudioTestDataManager()
    yield manager
    manager.cleanup_cache()


@pytest.fixture
def comprehensive_test_cases(audio_test_data_manager):
    """Provide comprehensive test cases."""
    return audio_test_data_manager.generate_comprehensive_test_suite()


@pytest.fixture
def performance_test_cases(audio_test_data_manager):
    """Provide performance test cases."""
    return audio_test_data_manager.get_performance_test_cases()


@pytest.fixture
def error_test_cases(audio_test_data_manager):
    """Provide error scenario test cases."""
    return audio_test_data_manager.get_error_test_cases()


@pytest.fixture
def format_test_cases(audio_test_data_manager):
    """Provide format compatibility test cases."""
    return audio_test_data_manager.get_format_test_cases()


if __name__ == "__main__":
    # Example usage
    manager = AudioTestDataManager()
    test_cases = manager.generate_comprehensive_test_suite()

    print(f"Generated {len(test_cases)} test cases:")
    for name, test_case in test_cases.items():
        print(f"  {name}: {test_case.description}")

    # Generate a sample audio file
    sample_case = test_cases["standard_speech"]
    audio_data = manager.generate_audio_data(sample_case)
    print(f"\nGenerated {len(audio_data)} bytes of audio data for '{sample_case.name}'")
