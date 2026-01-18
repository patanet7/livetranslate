"""
Comprehensive Audio Validation Library for LiveTranslate

This module provides enterprise-grade audio format validation, corruption detection,
quality assessment, and format conversion capabilities for the LiveTranslate system.

Features:
- Multi-format support (WAV, MP3, WebM, OGG, MP4, FLAC, M4A)
- Audio corruption detection using signal analysis
- Sample rate, bit depth, and channel validation
- Quality assessment with scoring
- Format conversion with quality preservation
- Comprehensive metadata extraction
- Performance metrics and detailed reporting

Author: LiveTranslate Team
Version: 1.0.0
"""

import io
import logging
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import librosa
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError as e:
    AUDIO_LIBS_AVAILABLE = False
    MISSING_DEPS = str(e)

# Setup logging
logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """Base exception for audio validation errors"""

    pass


class AudioFormatError(AudioValidationError):
    """Raised when audio format is invalid or unsupported"""

    pass


class AudioCorruptionError(AudioValidationError):
    """Raised when audio corruption is detected"""

    pass


class AudioQualityError(AudioValidationError):
    """Raised when audio quality is below acceptable thresholds"""

    pass


class AudioFormat(Enum):
    """Supported audio formats"""

    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"
    OGG = "ogg"
    MP4 = "mp4"
    FLAC = "flac"
    M4A = "m4a"


class QualityLevel(Enum):
    """Audio quality levels"""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class AudioMetadata:
    """Audio metadata structure"""

    format: str
    sample_rate: int
    channels: int
    duration: float
    bit_depth: int | None = None
    bitrate: int | None = None
    codec: str | None = None
    file_size: int | None = None


@dataclass
class ValidationResult:
    """Audio validation result structure"""

    is_valid: bool
    format_valid: bool
    corruption_detected: bool
    quality_score: float
    quality_level: QualityLevel
    sample_rate_valid: bool
    metadata: AudioMetadata
    errors: list[str]
    warnings: list[str]
    processing_time: float
    recommendations: list[str]


@dataclass
class CorruptionAnalysis:
    """Audio corruption analysis results"""

    is_corrupted: bool
    corruption_type: str | None
    corruption_severity: float
    affected_regions: list[tuple[float, float]]
    confidence: float
    details: dict[str, Any]


class AudioValidator:
    """
    Comprehensive audio validation and processing library

    This class provides methods for validating audio format, detecting corruption,
    assessing quality, and converting between formats while preserving quality.
    """

    def __init__(
        self,
        default_sample_rate: int = 16000,
        quality_threshold: float = 0.7,
        enable_performance_metrics: bool = True,
    ):
        """
        Initialize AudioValidator

        Args:
            default_sample_rate: Default target sample rate for validation
            quality_threshold: Minimum quality score threshold (0.0-1.0)
            enable_performance_metrics: Whether to collect performance metrics
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError(f"Required audio libraries not available: {MISSING_DEPS}")

        self.default_sample_rate = default_sample_rate
        self.quality_threshold = quality_threshold
        self.enable_performance_metrics = enable_performance_metrics

        # Supported formats mapping
        self.format_extensions = {
            AudioFormat.WAV: [".wav", ".wave"],
            AudioFormat.MP3: [".mp3"],
            AudioFormat.WEBM: [".webm"],
            AudioFormat.OGG: [".ogg", ".oga"],
            AudioFormat.MP4: [".mp4", ".m4v"],
            AudioFormat.FLAC: [".flac"],
            AudioFormat.M4A: [".m4a", ".aac"],
        }

        # Quality assessment parameters
        self.quality_params = {
            "min_snr": 10.0,  # Minimum signal-to-noise ratio (dB)
            "max_thd": 0.1,  # Maximum total harmonic distortion
            "min_dynamic_range": 20.0,  # Minimum dynamic range (dB)
            "max_silence_ratio": 0.3,  # Maximum ratio of silence
        }

        logger.info(
            f"AudioValidator initialized with sample_rate={default_sample_rate}, "
            f"quality_threshold={quality_threshold}"
        )

    def validate_audio_format(
        self, audio_data: bytes | np.ndarray | str, expected_format: AudioFormat | None = None
    ) -> ValidationResult:
        """
        Validate audio format and basic properties

        Args:
            audio_data: Audio data as bytes, numpy array, or file path
            expected_format: Expected audio format for validation

        Returns:
            ValidationResult with comprehensive validation information
        """
        start_time = time.time()
        errors = []
        warnings = []
        recommendations = []

        try:
            # Load audio data
            if isinstance(audio_data, str):
                # File path provided
                audio_array, sr = librosa.load(audio_data, sr=None)
                file_format = self._detect_format_from_path(audio_data)
            elif isinstance(audio_data, bytes):
                # Bytes data provided
                audio_array, sr = self._load_audio_from_bytes(audio_data)
                file_format = self._detect_format_from_bytes(audio_data)
            elif isinstance(audio_data, np.ndarray):
                # Numpy array provided
                audio_array = audio_data
                sr = self.default_sample_rate
                file_format = AudioFormat.WAV  # Assume WAV for raw arrays
            else:
                raise AudioFormatError(f"Unsupported audio data type: {type(audio_data)}")

            # Extract metadata
            metadata = self._extract_metadata(audio_array, sr, file_format)

            # Validate format
            format_valid = True
            if expected_format and file_format != expected_format:
                format_valid = False
                errors.append(
                    f"Format mismatch: expected {expected_format.value}, got {file_format.value}"
                )

            # Validate sample rate
            sample_rate_valid = self._validate_sample_rate_internal(sr)
            if not sample_rate_valid:
                warnings.append(
                    f"Sample rate {sr} Hz differs from expected {self.default_sample_rate} Hz"
                )
                recommendations.append(f"Consider resampling to {self.default_sample_rate} Hz")

            # Check for corruption
            corruption_analysis = self.detect_audio_corruption(audio_array, sr)

            # Assess quality
            quality_score, quality_level = self._assess_audio_quality(audio_array, sr)

            # Generate recommendations
            if quality_score < self.quality_threshold:
                recommendations.append("Audio quality below threshold - consider noise reduction")

            if len(audio_array) < sr * 0.1:  # Less than 100ms
                warnings.append("Audio duration very short - may cause processing issues")

            processing_time = time.time() - start_time

            return ValidationResult(
                is_valid=format_valid
                and not corruption_analysis.is_corrupted
                and quality_score >= self.quality_threshold,
                format_valid=format_valid,
                corruption_detected=corruption_analysis.is_corrupted,
                quality_score=quality_score,
                quality_level=quality_level,
                sample_rate_valid=sample_rate_valid,
                metadata=metadata,
                errors=errors,
                warnings=warnings,
                processing_time=processing_time,
                recommendations=recommendations,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Audio validation failed: {e!s}")

            return ValidationResult(
                is_valid=False,
                format_valid=False,
                corruption_detected=True,
                quality_score=0.0,
                quality_level=QualityLevel.UNACCEPTABLE,
                sample_rate_valid=False,
                metadata=AudioMetadata("unknown", 0, 0, 0.0),
                errors=[str(e)],
                warnings=[],
                processing_time=processing_time,
                recommendations=["Unable to process audio - check format and integrity"],
            )

    def detect_audio_corruption(
        self, audio_data: np.ndarray | bytes | str, sample_rate: int | None = None
    ) -> CorruptionAnalysis:
        """
        Detect audio corruption using advanced signal analysis

        Args:
            audio_data: Audio data as numpy array, bytes, or file path
            sample_rate: Sample rate if audio_data is numpy array

        Returns:
            CorruptionAnalysis with detailed corruption information
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, (bytes, str)):
                if isinstance(audio_data, str):
                    audio_array, sr = librosa.load(audio_data, sr=None)
                else:
                    audio_array, sr = self._load_audio_from_bytes(audio_data)
            else:
                audio_array = audio_data
                sr = sample_rate or self.default_sample_rate

            corruption_indicators = []
            affected_regions = []
            severity_scores = []

            # 1. Check for digital artifacts (clipping)
            clipping_ratio = self._detect_clipping(audio_array)
            if clipping_ratio > 0.01:  # More than 1% clipping
                corruption_indicators.append("clipping")
                severity_scores.append(min(clipping_ratio * 10, 1.0))

            # 2. Check for unusual frequency content
            freq_anomalies = self._detect_frequency_anomalies(audio_array, sr)
            if freq_anomalies["score"] > 0.3:
                corruption_indicators.append("frequency_anomalies")
                severity_scores.append(freq_anomalies["score"])
                affected_regions.extend(freq_anomalies["regions"])

            # 3. Check for sudden amplitude changes (dropouts/pops)
            amplitude_anomalies = self._detect_amplitude_anomalies(audio_array, sr)
            if amplitude_anomalies["score"] > 0.2:
                corruption_indicators.append("amplitude_anomalies")
                severity_scores.append(amplitude_anomalies["score"])
                affected_regions.extend(amplitude_anomalies["regions"])

            # 4. Check for silence/zero regions
            silence_anomalies = self._detect_silence_anomalies(audio_array, sr)
            if silence_anomalies["score"] > 0.4:
                corruption_indicators.append("excessive_silence")
                severity_scores.append(silence_anomalies["score"])

            # 5. Check spectral centroid consistency
            spectral_anomalies = self._detect_spectral_anomalies(audio_array, sr)
            if spectral_anomalies["score"] > 0.3:
                corruption_indicators.append("spectral_inconsistency")
                severity_scores.append(spectral_anomalies["score"])

            # Calculate overall corruption metrics
            is_corrupted = len(corruption_indicators) > 0
            corruption_severity = np.mean(severity_scores) if severity_scores else 0.0
            confidence = min(corruption_severity * 2, 1.0) if is_corrupted else 0.95

            corruption_type = None
            if corruption_indicators:
                corruption_type = corruption_indicators[0]  # Primary corruption type

            details = {
                "clipping_ratio": clipping_ratio,
                "frequency_anomalies": freq_anomalies,
                "amplitude_anomalies": amplitude_anomalies,
                "silence_anomalies": silence_anomalies,
                "spectral_anomalies": spectral_anomalies,
                "corruption_indicators": corruption_indicators,
            }

            return CorruptionAnalysis(
                is_corrupted=is_corrupted,
                corruption_type=corruption_type,
                corruption_severity=corruption_severity,
                affected_regions=affected_regions,
                confidence=confidence,
                details=details,
            )

        except Exception as e:
            logger.error(f"Corruption detection failed: {e!s}")
            return CorruptionAnalysis(
                is_corrupted=True,
                corruption_type="analysis_failed",
                corruption_severity=1.0,
                affected_regions=[],
                confidence=0.5,
                details={"error": str(e)},
            )

    def validate_sample_rate(
        self, audio_data: np.ndarray | bytes | str, expected_rate: int = 16000
    ) -> tuple[bool, int, dict[str, Any]]:
        """
        Validate audio sample rate

        Args:
            audio_data: Audio data as numpy array, bytes, or file path
            expected_rate: Expected sample rate in Hz

        Returns:
            Tuple of (is_valid, actual_rate, analysis_details)
        """
        try:
            # Load audio and get sample rate
            if isinstance(audio_data, str):
                _, sr = librosa.load(audio_data, sr=None)
            elif isinstance(audio_data, bytes):
                _, sr = self._load_audio_from_bytes(audio_data)
            else:
                # For numpy arrays, we need to estimate or use provided rate
                sr = expected_rate  # Assume expected rate for arrays

            is_valid = sr == expected_rate
            tolerance = 0.1  # 10% tolerance
            is_close = abs(sr - expected_rate) / expected_rate <= tolerance

            analysis = {
                "actual_sample_rate": sr,
                "expected_sample_rate": expected_rate,
                "difference": sr - expected_rate,
                "relative_difference": (sr - expected_rate) / expected_rate,
                "within_tolerance": is_close,
                "needs_resampling": not is_valid,
            }

            return is_valid, sr, analysis

        except Exception as e:
            logger.error(f"Sample rate validation failed: {e!s}")
            return False, 0, {"error": str(e)}

    def validate_audio_quality(
        self, audio_data: np.ndarray | bytes | str, sample_rate: int | None = None
    ) -> tuple[float, QualityLevel, dict[str, Any]]:
        """
        Comprehensive audio quality assessment

        Args:
            audio_data: Audio data as numpy array, bytes, or file path
            sample_rate: Sample rate if audio_data is numpy array

        Returns:
            Tuple of (quality_score, quality_level, quality_metrics)
        """
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, (bytes, str)):
                if isinstance(audio_data, str):
                    audio_array, sr = librosa.load(audio_data, sr=None)
                else:
                    audio_array, sr = self._load_audio_from_bytes(audio_data)
            else:
                audio_array = audio_data
                sr = sample_rate or self.default_sample_rate

            quality_score, quality_level = self._assess_audio_quality(audio_array, sr)

            # Calculate detailed quality metrics
            metrics = self._calculate_quality_metrics(audio_array, sr)

            return quality_score, quality_level, metrics

        except Exception as e:
            logger.error(f"Quality validation failed: {e!s}")
            return 0.0, QualityLevel.UNACCEPTABLE, {"error": str(e)}

    def standardize_audio_format(
        self,
        audio_data: np.ndarray | bytes | str,
        target_format: AudioFormat,
        target_sample_rate: int | None = None,
        preserve_quality: bool = True,
    ) -> tuple[bytes, AudioMetadata]:
        """
        Convert audio to standardized format with quality preservation

        Args:
            audio_data: Input audio data
            target_format: Target audio format
            target_sample_rate: Target sample rate (None to keep original)
            preserve_quality: Whether to preserve quality during conversion

        Returns:
            Tuple of (converted_audio_bytes, metadata)
        """
        try:
            # Load audio data
            if isinstance(audio_data, str):
                audio_array, sr = librosa.load(audio_data, sr=None)
            elif isinstance(audio_data, bytes):
                audio_array, sr = self._load_audio_from_bytes(audio_data)
            else:
                audio_array = audio_data
                sr = target_sample_rate or self.default_sample_rate

            # Resample if needed
            if target_sample_rate and target_sample_rate != sr:
                if preserve_quality:
                    # Use high-quality resampling
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=sr,
                        target_sr=target_sample_rate,
                        res_type="kaiser_best",
                    )
                else:
                    # Use faster resampling
                    audio_array = librosa.resample(
                        audio_array, orig_sr=sr, target_sr=target_sample_rate
                    )
                sr = target_sample_rate

            # Convert to target format
            output_buffer = io.BytesIO()

            if target_format == AudioFormat.WAV:
                sf.write(output_buffer, audio_array, sr, format="WAV", subtype="PCM_16")
            elif target_format == AudioFormat.FLAC:
                sf.write(output_buffer, audio_array, sr, format="FLAC")
            elif target_format == AudioFormat.OGG:
                sf.write(output_buffer, audio_array, sr, format="OGG", subtype="VORBIS")
            else:
                # For other formats, use WAV as intermediate and warn
                sf.write(output_buffer, audio_array, sr, format="WAV", subtype="PCM_16")
                logger.warning(
                    f"Direct conversion to {target_format.value} not supported, using WAV"
                )

            converted_bytes = output_buffer.getvalue()

            # Create metadata for converted audio
            metadata = AudioMetadata(
                format=target_format.value,
                sample_rate=sr,
                channels=1 if len(audio_array.shape) == 1 else audio_array.shape[1],
                duration=len(audio_array) / sr,
                bit_depth=16,
                file_size=len(converted_bytes),
            )

            return converted_bytes, metadata

        except Exception as e:
            logger.error(f"Format conversion failed: {e!s}")
            raise AudioFormatError(f"Failed to convert to {target_format.value}: {e!s}") from e

    def get_audio_metadata(self, audio_data: bytes | str) -> AudioMetadata:
        """
        Extract comprehensive audio metadata

        Args:
            audio_data: Audio data as bytes or file path

        Returns:
            AudioMetadata with detailed information
        """
        try:
            if isinstance(audio_data, str):
                # File path provided
                audio_array, sr = librosa.load(audio_data, sr=None)
                file_format = self._detect_format_from_path(audio_data)

                # Get file size
                import os

                file_size = os.path.getsize(audio_data)
            else:
                # Bytes data provided
                audio_array, sr = self._load_audio_from_bytes(audio_data)
                file_format = self._detect_format_from_bytes(audio_data)
                file_size = len(audio_data)

            return self._extract_metadata(audio_array, sr, file_format, file_size)

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e!s}")
            return AudioMetadata("unknown", 0, 0, 0.0)

    # Private helper methods

    def _load_audio_from_bytes(self, audio_bytes: bytes) -> tuple[np.ndarray, int]:
        """Load audio from bytes data"""
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_array, sr = librosa.load(audio_buffer, sr=None)
            return audio_array, sr
        except Exception as e:
            # Try with soundfile as fallback
            try:
                audio_buffer = io.BytesIO(audio_bytes)
                audio_array, sr = sf.read(audio_buffer)
                if len(audio_array.shape) > 1:
                    audio_array = np.mean(audio_array, axis=1)  # Convert to mono
                return audio_array, sr
            except Exception:
                raise AudioFormatError(f"Unable to load audio from bytes: {e!s}") from e

    def _detect_format_from_path(self, file_path: str) -> AudioFormat:
        """Detect audio format from file path"""
        import os

        ext = os.path.splitext(file_path)[1].lower()

        for format_type, extensions in self.format_extensions.items():
            if ext in extensions:
                return format_type

        return AudioFormat.WAV  # Default fallback

    def _detect_format_from_bytes(self, audio_bytes: bytes) -> AudioFormat:
        """Detect audio format from bytes data (simplified detection)"""
        # Simple magic number detection
        if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
            return AudioFormat.WAV
        elif audio_bytes.startswith(b"fLaC"):
            return AudioFormat.FLAC
        elif audio_bytes.startswith(b"OggS"):
            return AudioFormat.OGG
        elif audio_bytes.startswith(b"ID3") or audio_bytes[0:2] == b"\xff\xfb":
            return AudioFormat.MP3
        else:
            return AudioFormat.WAV  # Default fallback

    def _extract_metadata(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        format_type: AudioFormat,
        file_size: int | None = None,
    ) -> AudioMetadata:
        """Extract metadata from audio array"""
        channels = 1 if len(audio_array.shape) == 1 else audio_array.shape[1]
        duration = len(audio_array) / sample_rate

        # Estimate bit depth and bitrate
        bit_depth = 16  # Default assumption
        bitrate = (int(file_size * 8 / duration) if duration > 0 else None) if file_size else None

        return AudioMetadata(
            format=format_type.value,
            sample_rate=sample_rate,
            channels=channels,
            duration=duration,
            bit_depth=bit_depth,
            bitrate=bitrate,
            file_size=file_size,
        )

    def _validate_sample_rate_internal(self, sample_rate: int) -> bool:
        """Internal sample rate validation"""
        return sample_rate == self.default_sample_rate

    def _assess_audio_quality(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> tuple[float, QualityLevel]:
        """Assess overall audio quality"""
        try:
            quality_metrics = self._calculate_quality_metrics(audio_array, sample_rate)

            # Weighted quality score calculation
            weights = {
                "snr_score": 0.3,
                "dynamic_range_score": 0.25,
                "thd_score": 0.2,
                "silence_score": 0.15,
                "spectral_score": 0.1,
            }

            quality_score = sum(weights[key] * quality_metrics[key] for key in weights)

            # Determine quality level
            if quality_score >= 0.9:
                quality_level = QualityLevel.EXCELLENT
            elif quality_score >= 0.75:
                quality_level = QualityLevel.GOOD
            elif quality_score >= 0.6:
                quality_level = QualityLevel.FAIR
            elif quality_score >= 0.4:
                quality_level = QualityLevel.POOR
            else:
                quality_level = QualityLevel.UNACCEPTABLE

            return quality_score, quality_level

        except Exception as e:
            logger.error(f"Quality assessment failed: {e!s}")
            return 0.0, QualityLevel.UNACCEPTABLE

    def _calculate_quality_metrics(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> dict[str, float]:
        """Calculate detailed quality metrics"""
        metrics = {}

        try:
            # Signal-to-noise ratio
            signal_power = np.mean(audio_array**2)
            noise_floor = np.percentile(np.abs(audio_array), 10)
            snr_db = 10 * np.log10(signal_power / (noise_floor**2 + 1e-10))
            metrics["snr_db"] = snr_db
            metrics["snr_score"] = min(max(snr_db / 30.0, 0.0), 1.0)  # Normalize to 0-1

            # Dynamic range
            dynamic_range = 20 * np.log10(
                np.max(np.abs(audio_array)) / (np.mean(np.abs(audio_array)) + 1e-10)
            )
            metrics["dynamic_range_db"] = dynamic_range
            metrics["dynamic_range_score"] = min(max(dynamic_range / 40.0, 0.0), 1.0)

            # Total harmonic distortion (simplified)
            fft = np.fft.fft(audio_array[: int(sample_rate)])  # Use first second
            freqs = np.fft.fftfreq(len(fft), 1 / sample_rate)
            magnitude = np.abs(fft)

            # Find fundamental frequency
            fundamental_idx = np.argmax(magnitude[1 : len(magnitude) // 2]) + 1
            freqs[fundamental_idx]

            # Calculate harmonics energy
            total_energy = np.sum(magnitude**2)
            harmonic_energy = 0
            for h in range(2, 6):  # 2nd to 5th harmonics
                harmonic_idx = int(fundamental_idx * h)
                if harmonic_idx < len(magnitude):
                    harmonic_energy += magnitude[harmonic_idx] ** 2

            thd = np.sqrt(harmonic_energy / (total_energy + 1e-10))
            metrics["thd"] = thd
            metrics["thd_score"] = max(1.0 - thd / 0.1, 0.0)  # Lower THD is better

            # Silence ratio
            silence_threshold = np.max(np.abs(audio_array)) * 0.01
            silence_ratio = np.mean(np.abs(audio_array) < silence_threshold)
            metrics["silence_ratio"] = silence_ratio
            metrics["silence_score"] = max(1.0 - silence_ratio / 0.3, 0.0)

            # Spectral centroid stability
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
            centroid_std = np.std(spectral_centroids)
            centroid_mean = np.mean(spectral_centroids)
            spectral_stability = 1.0 - min(centroid_std / centroid_mean, 1.0)
            metrics["spectral_stability"] = spectral_stability
            metrics["spectral_score"] = spectral_stability

        except Exception as e:
            logger.error(f"Quality metrics calculation failed: {e!s}")
            # Return default poor quality metrics
            metrics = {
                "snr_db": 0.0,
                "snr_score": 0.0,
                "dynamic_range_db": 0.0,
                "dynamic_range_score": 0.0,
                "thd": 1.0,
                "thd_score": 0.0,
                "silence_ratio": 1.0,
                "silence_score": 0.0,
                "spectral_stability": 0.0,
                "spectral_score": 0.0,
            }

        return metrics

    def _detect_clipping(self, audio_array: np.ndarray) -> float:
        """Detect digital clipping in audio"""
        max_val = np.max(np.abs(audio_array))
        clipping_threshold = 0.99 * max_val
        clipped_samples = np.sum(np.abs(audio_array) >= clipping_threshold)
        return clipped_samples / len(audio_array)

    def _detect_frequency_anomalies(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> dict[str, Any]:
        """Detect frequency domain anomalies"""
        try:
            # Compute spectrogram
            stft = librosa.stft(audio_array)
            magnitude = np.abs(stft)

            # Check for unusual frequency content
            freq_energy = np.mean(magnitude, axis=1)
            freq_deviation = np.std(freq_energy) / (np.mean(freq_energy) + 1e-10)

            # Look for sudden frequency changes
            freq_changes = np.diff(freq_energy)
            change_score = np.std(freq_changes) / (np.mean(np.abs(freq_changes)) + 1e-10)

            anomaly_score = min((freq_deviation + change_score) / 2.0, 1.0)

            return {
                "score": anomaly_score,
                "regions": [],  # Could be enhanced to detect specific regions
                "details": {"freq_deviation": freq_deviation, "change_score": change_score},
            }
        except Exception:
            return {"score": 0.0, "regions": [], "details": {}}

    def _detect_amplitude_anomalies(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> dict[str, Any]:
        """Detect amplitude anomalies (pops, clicks, dropouts)"""
        try:
            # Calculate envelope
            envelope = np.abs(audio_array)

            # Smooth envelope
            window_size = int(sample_rate * 0.01)  # 10ms window
            smoothed = np.convolve(envelope, np.ones(window_size) / window_size, mode="same")

            # Find sudden changes
            envelope_diff = np.abs(np.diff(smoothed))
            threshold = np.percentile(envelope_diff, 95)

            anomalies = envelope_diff > threshold * 3
            anomaly_score = np.sum(anomalies) / len(envelope_diff)

            # Find regions
            regions = []
            in_anomaly = False
            start_idx = 0

            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly and not in_anomaly:
                    start_idx = i
                    in_anomaly = True
                elif not is_anomaly and in_anomaly:
                    start_time = start_idx / sample_rate
                    end_time = i / sample_rate
                    regions.append((start_time, end_time))
                    in_anomaly = False

            return {
                "score": min(anomaly_score * 10, 1.0),
                "regions": regions,
                "details": {"threshold": threshold, "anomaly_count": np.sum(anomalies)},
            }
        except Exception:
            return {"score": 0.0, "regions": [], "details": {}}

    def _detect_silence_anomalies(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> dict[str, Any]:
        """Detect unusual silence patterns"""
        try:
            # Calculate RMS energy in frames
            frame_length = int(sample_rate * 0.025)  # 25ms frames
            hop_length = int(frame_length / 2)

            rms = librosa.feature.rms(
                y=audio_array, frame_length=frame_length, hop_length=hop_length
            )[0]

            # Define silence threshold
            silence_threshold = np.percentile(rms, 20)
            silent_frames = rms < silence_threshold

            silence_ratio = np.mean(silent_frames)

            # Check for long continuous silence
            max_silence_length = 0
            current_silence_length = 0

            for is_silent in silent_frames:
                if is_silent:
                    current_silence_length += 1
                else:
                    max_silence_length = max(max_silence_length, current_silence_length)
                    current_silence_length = 0

            max_silence_time = max_silence_length * hop_length / sample_rate

            # Anomaly score based on excessive silence
            anomaly_score = max(silence_ratio - 0.3, 0) / 0.7  # Above 30% silence is concerning
            if max_silence_time > 5.0:  # More than 5 seconds of continuous silence
                anomaly_score = max(anomaly_score, 0.8)

            return {
                "score": min(anomaly_score, 1.0),
                "regions": [],
                "details": {
                    "silence_ratio": silence_ratio,
                    "max_silence_time": max_silence_time,
                    "silence_threshold": silence_threshold,
                },
            }
        except Exception:
            return {"score": 0.0, "regions": [], "details": {}}

    def _detect_spectral_anomalies(
        self, audio_array: np.ndarray, sample_rate: int
    ) -> dict[str, Any]:
        """Detect spectral inconsistencies"""
        try:
            # Calculate spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate)[0]

            # Check for unusual variations
            centroid_var = np.var(spectral_centroids) / (np.mean(spectral_centroids) ** 2 + 1e-10)
            rolloff_var = np.var(spectral_rolloff) / (np.mean(spectral_rolloff) ** 2 + 1e-10)

            # Combined anomaly score
            anomaly_score = min((centroid_var + rolloff_var) / 2.0, 1.0)

            return {
                "score": anomaly_score,
                "regions": [],
                "details": {"centroid_variation": centroid_var, "rolloff_variation": rolloff_var},
            }
        except Exception:
            return {"score": 0.0, "regions": [], "details": {}}


# Convenience functions for easy usage


def validate_audio(
    audio_data: bytes | np.ndarray | str,
    expected_format: AudioFormat | None = None,
    expected_sample_rate: int = 16000,
) -> ValidationResult:
    """
    Convenience function for comprehensive audio validation

    Args:
        audio_data: Audio data to validate
        expected_format: Expected audio format
        expected_sample_rate: Expected sample rate

    Returns:
        ValidationResult with comprehensive analysis
    """
    validator = AudioValidator(default_sample_rate=expected_sample_rate)
    return validator.validate_audio_format(audio_data, expected_format)


def check_audio_corruption(audio_data: bytes | np.ndarray | str) -> CorruptionAnalysis:
    """
    Convenience function for audio corruption detection

    Args:
        audio_data: Audio data to check

    Returns:
        CorruptionAnalysis with detailed results
    """
    validator = AudioValidator()
    return validator.detect_audio_corruption(audio_data)


def convert_audio_format(
    audio_data: bytes | np.ndarray | str,
    target_format: AudioFormat,
    target_sample_rate: int | None = None,
) -> tuple[bytes, AudioMetadata]:
    """
    Convenience function for audio format conversion

    Args:
        audio_data: Input audio data
        target_format: Target format
        target_sample_rate: Target sample rate

    Returns:
        Tuple of (converted_audio_bytes, metadata)
    """
    validator = AudioValidator()
    return validator.standardize_audio_format(audio_data, target_format, target_sample_rate)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    validator = AudioValidator()

    # Test with a simple sine wave
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0

    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * frequency * t)

    # Validate the test audio
    result = validator.validate_audio_format(test_audio)
    print("Validation Result:")
    print(f"  Valid: {result.is_valid}")
    print(f"  Quality Score: {result.quality_score:.3f}")
    print(f"  Quality Level: {result.quality_level.value}")
    print(f"  Processing Time: {result.processing_time:.3f}s")

    # Check for corruption
    corruption = validator.detect_audio_corruption(test_audio, sample_rate)
    print("\nCorruption Analysis:")
    print(f"  Corrupted: {corruption.is_corrupted}")
    print(f"  Confidence: {corruption.confidence:.3f}")

    # Get metadata
    metadata = validator._extract_metadata(test_audio, sample_rate, AudioFormat.WAV)
    print("\nMetadata:")
    print(f"  Duration: {metadata.duration:.3f}s")
    print(f"  Sample Rate: {metadata.sample_rate} Hz")
    print(f"  Channels: {metadata.channels}")
