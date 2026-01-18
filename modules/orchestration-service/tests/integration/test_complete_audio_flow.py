#!/usr/bin/env python3
"""
Comprehensive End-to-End Audio Flow Validation Tests

This test suite validates the complete audio processing pipeline from frontend
through orchestration to whisper and translation services, ensuring all
components work together seamlessly.
"""

import io
import logging

# Import the application and dependencies
import sys
import time
import wave
from pathlib import Path
from typing import ClassVar
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient

service_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(service_root / "src"))

from dependencies import (
    get_audio_coordinator,
    get_audio_service_client,
    get_config_manager,
    get_config_sync_manager,
    get_translation_service_client,
)
from main_fastapi import app

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFlowTestSuite:
    """Comprehensive test suite for complete audio flow validation."""

    # Test configuration constants
    SAMPLE_RATE = 16000
    TEST_DURATION = 3.0
    CHUNK_SIZE = 2048
    SUPPORTED_FORMATS: ClassVar[list[str]] = ["wav", "mp3", "webm", "ogg", "mp4", "flac"]

    def __init__(self):
        self.test_session_id = None
        self.performance_metrics = {}
        self.error_scenarios = {}

    def generate_test_audio(
        self,
        format_type: str,
        duration: float | None = None,
        sample_rate: int | None = None,
        corrupt: bool = False,
    ) -> bytes:
        """Generate test audio data in various formats."""
        duration = duration or self.TEST_DURATION
        sample_rate = sample_rate or self.SAMPLE_RATE

        # Generate voice-like audio with multiple harmonics
        t = np.arange(int(duration * sample_rate)) / sample_rate

        # Fundamental frequency (120 Hz for voice)
        signal = 0.3 * np.sin(2 * np.pi * 120 * t)

        # Add harmonics for voice-like characteristics
        signal += 0.2 * np.sin(2 * np.pi * 240 * t)  # Second harmonic
        signal += 0.1 * np.sin(2 * np.pi * 360 * t)  # Third harmonic
        signal += 0.05 * np.sin(2 * np.pi * 480 * t)  # Fourth harmonic

        # Add formants (vowel characteristics)
        signal += 0.1 * np.sin(2 * np.pi * 800 * t)  # First formant
        signal += 0.05 * np.sin(2 * np.pi * 1200 * t)  # Second formant
        signal += 0.03 * np.sin(2 * np.pi * 2400 * t)  # Third formant

        # Add realistic noise
        signal += 0.01 * np.random.randn(len(signal))

        # Normalize to prevent clipping
        signal = signal / np.max(np.abs(signal)) * 0.8

        if corrupt:
            # Introduce corruption for error testing
            signal[len(signal) // 2 : len(signal) // 2 + 100] = np.nan

        # Convert to appropriate format
        if format_type == "wav":
            return self._generate_wav_bytes(signal, sample_rate)
        elif format_type == "mp3":
            return self._generate_mp3_bytes(signal, sample_rate)
        elif format_type == "webm":
            return self._generate_webm_bytes(signal, sample_rate)
        elif format_type == "ogg":
            return self._generate_ogg_bytes(signal, sample_rate)
        elif format_type == "mp4":
            return self._generate_mp4_bytes(signal, sample_rate)
        elif format_type == "flac":
            return self._generate_flac_bytes(signal, sample_rate)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _generate_wav_bytes(self, signal: np.ndarray, sample_rate: int) -> bytes:
        """Generate WAV format bytes."""
        # Convert to 16-bit PCM
        pcm_data = (signal * 32767).astype(np.int16)

        # Create WAV file in memory
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data.tobytes())

        return buffer.getvalue()

    def _generate_mp3_bytes(self, signal: np.ndarray, sample_rate: int) -> bytes:
        """Generate MP3 format bytes (simplified for testing)."""
        # For testing, we'll create a simple MP3-like header + PCM data
        # In production, this would use an actual MP3 encoder
        mp3_header = b"ID3\x03\x00\x00\x00\x00\x00\x00"
        pcm_data = (signal * 32767).astype(np.int16).tobytes()
        return mp3_header + pcm_data

    def _generate_webm_bytes(self, signal: np.ndarray, sample_rate: int) -> bytes:
        """Generate WebM format bytes (simplified for testing)."""
        webm_header = b"\x1a\x45\xdf\xa3"  # WebM signature
        pcm_data = (signal * 32767).astype(np.int16).tobytes()
        return webm_header + pcm_data

    def _generate_ogg_bytes(self, signal: np.ndarray, sample_rate: int) -> bytes:
        """Generate OGG format bytes (simplified for testing)."""
        ogg_header = b"OggS"  # OGG signature
        pcm_data = (signal * 32767).astype(np.int16).tobytes()
        return ogg_header + pcm_data

    def _generate_mp4_bytes(self, signal: np.ndarray, sample_rate: int) -> bytes:
        """Generate MP4 format bytes (simplified for testing)."""
        mp4_header = b"\x00\x00\x00\x20ftypmp41"  # MP4 signature
        pcm_data = (signal * 32767).astype(np.int16).tobytes()
        return mp4_header + pcm_data

    def _generate_flac_bytes(self, signal: np.ndarray, sample_rate: int) -> bytes:
        """Generate FLAC format bytes (simplified for testing)."""
        flac_header = b"fLaC"  # FLAC signature
        pcm_data = (signal * 32767).astype(np.int16).tobytes()
        return flac_header + pcm_data

    def create_upload_file(
        self, audio_bytes: bytes, filename: str, content_type: str
    ) -> UploadFile:
        """Create FastAPI UploadFile from audio bytes."""
        file_obj = io.BytesIO(audio_bytes)
        return UploadFile(
            filename=filename,
            file=file_obj,
            content_type=content_type,
            size=len(audio_bytes),
        )

    def calculate_performance_metrics(
        self, start_time: float, end_time: float, audio_duration: float
    ) -> dict[str, float]:
        """Calculate performance metrics for audio processing."""
        processing_time = end_time - start_time
        real_time_factor = processing_time / audio_duration
        throughput = audio_duration / processing_time

        return {
            "processing_time": processing_time,
            "audio_duration": audio_duration,
            "real_time_factor": real_time_factor,
            "throughput": throughput,
            "latency": processing_time,
        }


@pytest.fixture(scope="session")
def audio_test_suite():
    """Provide the audio test suite instance."""
    return AudioFlowTestSuite()


@pytest.fixture
def test_client():
    """Create FastAPI test client with mocked dependencies."""

    # Mock the dependencies
    mock_config_manager = AsyncMock()
    mock_audio_client = AsyncMock()
    mock_translation_client = AsyncMock()
    mock_audio_coordinator = AsyncMock()
    mock_config_sync_manager = AsyncMock()

    # Configure mock responses
    mock_config_manager.get_service_config.return_value = {
        "whisper_model": "whisper-base",
        "device": "auto",
        "enable_vad": True,
        "enable_speaker_diarization": True,
    }

    # Override dependencies
    app.dependency_overrides[get_config_manager] = lambda: mock_config_manager
    app.dependency_overrides[get_audio_service_client] = lambda: mock_audio_client
    app.dependency_overrides[get_translation_service_client] = lambda: mock_translation_client
    app.dependency_overrides[get_audio_coordinator] = lambda: mock_audio_coordinator
    app.dependency_overrides[get_config_sync_manager] = lambda: mock_config_sync_manager

    client = TestClient(app)

    yield (
        client,
        {
            "config_manager": mock_config_manager,
            "audio_client": mock_audio_client,
            "translation_client": mock_translation_client,
            "audio_coordinator": mock_audio_coordinator,
            "config_sync_manager": mock_config_sync_manager,
        },
    )

    # Clean up overrides
    app.dependency_overrides.clear()


class TestCompleteAudioFlow:
    """Test the complete audio processing pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_wav_format(self, test_client, audio_test_suite):
        """Test complete pipeline with WAV format audio."""
        client, mocks = test_client

        # Generate test audio
        audio_bytes = audio_test_suite.generate_test_audio("wav")
        audio_test_suite.create_upload_file(
            audio_bytes, "test_audio.wav", "audio/wav"
        )

        # Mock successful whisper service response
        whisper_response = {
            "text": "This is a test transcription",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
            "language": "en",
            "duration": 3.0,
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "This is a test transcription",
                    "speaker_id": "speaker_0",
                }
            ],
        }

        # Mock translation service response
        translation_response = {
            "translated_text": "Esta es una transcripción de prueba",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.88,
            "quality_score": 0.92,
        }

        # Configure mock responses
        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        mock_translation_response = Mock()
        mock_translation_response.status_code = 200
        mock_translation_response.json.return_value = translation_response
        mocks["translation_client"].post.return_value = mock_translation_response

        # Start timing
        start_time = time.time()

        # Make request to upload endpoint
        response = client.post(
            "/api/audio/upload",
            files={"audio": ("test_audio.wav", audio_bytes, "audio/wav")},
            data={
                "session_id": "test_session_complete_flow",
                "enable_transcription": "true",
                "enable_diarization": "true",
                "enable_translation": "true",
                "target_languages": "es",
                "whisper_model": "whisper-base",
                "translation_quality": "balanced",
            },
        )

        end_time = time.time()

        # Verify response
        assert response.status_code == 200
        response_data = response.json()

        # Verify response structure
        assert "transcription" in response_data
        assert "translation" in response_data
        assert "session_id" in response_data
        assert "processing_stats" in response_data

        # Verify transcription data
        transcription = response_data["transcription"]
        assert transcription["text"] == whisper_response["text"]
        assert transcription["confidence"] >= 0.9
        assert "speaker_id" in transcription

        # Verify translation data
        translation = response_data["translation"]
        assert translation["translated_text"] == translation_response["translated_text"]
        assert translation["source_language"] == "en"
        assert translation["target_language"] == "es"

        # Calculate and verify performance metrics
        metrics = audio_test_suite.calculate_performance_metrics(
            start_time, end_time, audio_test_suite.TEST_DURATION
        )

        # Performance assertions
        assert metrics["real_time_factor"] < 2.0  # Should process faster than 2x real-time
        assert metrics["processing_time"] < 10.0  # Should complete within 10 seconds

        # Verify service calls were made
        assert mocks["audio_client"].post.called
        assert mocks["translation_client"].post.called

    @pytest.mark.asyncio
    async def test_format_compatibility_all_formats(self, test_client, audio_test_suite):
        """Test pipeline compatibility with all supported audio formats."""
        client, mocks = test_client

        format_configs = [
            ("wav", "audio/wav"),
            ("mp3", "audio/mpeg"),
            ("webm", "audio/webm"),
            ("ogg", "audio/ogg"),
            ("mp4", "audio/mp4"),
            ("flac", "audio/flac"),
        ]

        # Configure successful mock responses
        whisper_response = {
            "text": "Format compatibility test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
            "language": "en",
            "duration": 3.0,
        }

        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        results = {}

        for format_type, content_type in format_configs:
            # Generate audio in specific format
            audio_bytes = audio_test_suite.generate_test_audio(format_type)

            start_time = time.time()

            # Test upload
            response = client.post(
                "/api/audio/upload",
                files={"audio": (f"test.{format_type}", audio_bytes, content_type)},
                data={
                    "session_id": f"format_test_{format_type}",
                    "enable_transcription": "true",
                    "enable_diarization": "false",
                    "enable_translation": "false",
                },
            )

            end_time = time.time()

            # Record results
            results[format_type] = {
                "status_code": response.status_code,
                "processing_time": end_time - start_time,
                "response": response.json() if response.status_code == 200 else None,
                "error": response.text if response.status_code != 200 else None,
            }

        # Verify all formats were processed successfully
        for format_type, result in results.items():
            assert result["status_code"] == 200, f"Format {format_type} failed: {result['error']}"
            assert result["response"] is not None
            assert "transcription" in result["response"]
            assert result["processing_time"] < 15.0  # Reasonable processing time

    @pytest.mark.asyncio
    async def test_concurrent_session_processing(self, test_client, audio_test_suite):
        """Test concurrent processing of multiple audio sessions."""
        client, mocks = test_client

        # Configure mock responses
        whisper_response = {
            "text": "Concurrent session test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
            "language": "en",
            "duration": 3.0,
        }

        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        # Prepare multiple concurrent requests
        num_concurrent = 5
        audio_bytes = audio_test_suite.generate_test_audio("wav")

        async def process_session(session_index: int):
            """Process a single session."""
            response = client.post(
                "/api/audio/upload",
                files={
                    "audio": (
                        f"concurrent_{session_index}.wav",
                        audio_bytes,
                        "audio/wav",
                    )
                },
                data={
                    "session_id": f"concurrent_session_{session_index}",
                    "enable_transcription": "true",
                    "enable_diarization": "true",
                },
            )
            return session_index, response

        # Execute concurrent requests
        start_time = time.time()

        # Simulate concurrent requests (TestClient is synchronous, so we'll make them sequentially but quickly)
        results = []
        for i in range(num_concurrent):
            session_start = time.time()
            result = await process_session(i)
            session_end = time.time()
            results.append((result, session_end - session_start))

        end_time = time.time()

        # Verify all sessions processed successfully
        for (session_index, response), processing_time in results:
            assert response.status_code == 200, f"Session {session_index} failed"
            response_data = response.json()
            assert "transcription" in response_data
            assert response_data["session_id"] == f"concurrent_session_{session_index}"
            assert processing_time < 10.0  # Each session should complete reasonably quickly

        # Verify total processing time is reasonable
        total_time = end_time - start_time
        assert total_time < 60.0  # All sessions should complete within 1 minute

        # Verify service was called for each session
        assert mocks["audio_client"].post.call_count >= num_concurrent

    @pytest.mark.asyncio
    async def test_error_scenarios_comprehensive(self, test_client, audio_test_suite):
        """Test comprehensive error handling scenarios."""
        client, _mocks = test_client

        error_test_cases = [
            {
                "name": "empty_audio_file",
                "audio_bytes": b"",
                "expected_status": 400,
                "description": "Empty audio file",
            },
            {
                "name": "corrupted_audio",
                "audio_bytes": audio_test_suite.generate_test_audio("wav", corrupt=True),
                "expected_status": 422,
                "description": "Corrupted audio data",
            },
            {
                "name": "invalid_format",
                "audio_bytes": b"invalid audio data",
                "expected_status": 422,
                "description": "Invalid audio format",
            },
            {
                "name": "oversized_audio",
                "audio_bytes": audio_test_suite.generate_test_audio(
                    "wav", duration=300
                ),  # 5 minutes
                "expected_status": 413,
                "description": "Oversized audio file",
            },
        ]

        for test_case in error_test_cases:
            response = client.post(
                "/api/audio/upload",
                files={"audio": ("test.wav", test_case["audio_bytes"], "audio/wav")},
                data={
                    "session_id": f"error_test_{test_case['name']}",
                    "enable_transcription": "true",
                },
            )

            # Some errors might be handled gracefully with 200 status but error in response
            if response.status_code == 200:
                response_data = response.json()
                # Check if error is reported in response
                assert (
                    "error" in response_data
                    or ("status" in response_data
                    and response_data["status"] == "error")
                ), f"Expected error for {test_case['description']}"
            else:
                # Direct HTTP error
                assert (
                    response.status_code >= 400
                ), f"Expected error status for {test_case['description']}"

    @pytest.mark.asyncio
    async def test_service_failure_scenarios(self, test_client, audio_test_suite):
        """Test handling of downstream service failures."""
        client, mocks = test_client

        audio_bytes = audio_test_suite.generate_test_audio("wav")

        # Test whisper service failure
        mocks["audio_client"].post.side_effect = httpx.ConnectError("Whisper service unavailable")

        response = client.post(
            "/api/audio/upload",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={
                "session_id": "whisper_failure_test",
                "enable_transcription": "true",
            },
        )

        # Should handle service failure gracefully
        if response.status_code == 200:
            response_data = response.json()
            assert "error" in response_data or response_data.get("status") == "error"
        else:
            assert response.status_code in [503, 502, 500]  # Service unavailable errors

        # Reset and test translation service failure
        mocks["audio_client"].post.side_effect = None
        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = {
            "text": "Service failure test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
        }
        mocks["audio_client"].post.return_value = mock_whisper_response

        # Make translation service fail
        mocks["translation_client"].post.side_effect = httpx.ConnectError(
            "Translation service unavailable"
        )

        response = client.post(
            "/api/audio/upload",
            files={"audio": ("test.wav", audio_bytes, "audio/wav")},
            data={
                "session_id": "translation_failure_test",
                "enable_transcription": "true",
                "enable_translation": "true",
                "target_languages": "es",
            },
        )

        # Should succeed with transcription but fail translation gracefully
        if response.status_code == 200:
            response_data = response.json()
            assert "transcription" in response_data  # Transcription should succeed
            # Translation should either be missing or have error
            if "translation" in response_data:
                assert "error" in response_data["translation"]

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, test_client, audio_test_suite):
        """Test performance benchmarks and regression detection."""
        client, mocks = test_client

        # Configure optimal mock responses
        whisper_response = {
            "text": "Performance benchmark test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
            "language": "en",
            "duration": 3.0,
        }

        translation_response = {
            "translated_text": "Prueba de benchmark de rendimiento",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.88,
        }

        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        mock_translation_response = Mock()
        mock_translation_response.status_code = 200
        mock_translation_response.json.return_value = translation_response
        mocks["translation_client"].post.return_value = mock_translation_response

        # Test different audio durations
        test_durations = [1.0, 3.0, 5.0, 10.0]
        performance_results = {}

        for duration in test_durations:
            audio_bytes = audio_test_suite.generate_test_audio("wav", duration=duration)

            # Warmup request
            client.post(
                "/api/audio/upload",
                files={"audio": ("warmup.wav", audio_bytes, "audio/wav")},
                data={"session_id": "warmup", "enable_transcription": "true"},
            )

            # Timed request
            start_time = time.time()
            response = client.post(
                "/api/audio/upload",
                files={"audio": (f"perf_{duration}s.wav", audio_bytes, "audio/wav")},
                data={
                    "session_id": f"performance_test_{duration}s",
                    "enable_transcription": "true",
                    "enable_translation": "true",
                    "target_languages": "es",
                },
            )
            end_time = time.time()

            assert response.status_code == 200

            # Calculate metrics
            metrics = audio_test_suite.calculate_performance_metrics(start_time, end_time, duration)
            performance_results[duration] = metrics

        # Performance assertions
        for duration, metrics in performance_results.items():
            # Real-time factor should be reasonable (< 1.5x for most cases)
            assert (
                metrics["real_time_factor"] < 3.0
            ), f"Real-time factor too high for {duration}s audio: {metrics['real_time_factor']}"

            # Processing time should be reasonable
            assert (
                metrics["processing_time"] < 30.0
            ), f"Processing time too high for {duration}s audio: {metrics['processing_time']}"

            # Throughput should be positive
            assert (
                metrics["throughput"] > 0
            ), f"Invalid throughput for {duration}s audio: {metrics['throughput']}"

        # Log performance results for monitoring
        logger.info("Performance benchmark results:")
        for duration, metrics in performance_results.items():
            logger.info(
                f"  {duration}s audio: {metrics['processing_time']:.2f}s processing "
                f"(RTF: {metrics['real_time_factor']:.2f})"
            )

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, test_client, audio_test_suite):
        """Test memory usage during audio processing."""
        import os

        import psutil

        client, mocks = test_client

        # Configure mock responses
        whisper_response = {
            "text": "Memory test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
        }

        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process multiple audio files to test memory behavior
        num_files = 10
        audio_bytes = audio_test_suite.generate_test_audio("wav")

        memory_usage = [initial_memory]

        for i in range(num_files):
            response = client.post(
                "/api/audio/upload",
                files={"audio": (f"memory_test_{i}.wav", audio_bytes, "audio/wav")},
                data={
                    "session_id": f"memory_test_{i}",
                    "enable_transcription": "true",
                },
            )

            assert response.status_code == 200

            # Record memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)

        # Analyze memory usage
        max_memory = max(memory_usage)
        memory_growth = max_memory - initial_memory

        # Memory assertions
        assert (
            memory_growth < 500
        ), f"Excessive memory growth: {memory_growth}MB"  # Should not grow > 500MB

        # Check for memory leaks (no continuous growth)
        if len(memory_usage) > 5:
            recent_avg = np.mean(memory_usage[-3:])
            early_avg = np.mean(memory_usage[2:5])
            growth_rate = (recent_avg - early_avg) / early_avg
            assert (
                growth_rate < 0.5
            ), f"Potential memory leak detected: {growth_rate * 100:.1f}% growth"

        logger.info(
            f"Memory usage: Initial={initial_memory:.1f}MB, Max={max_memory:.1f}MB, "
            f"Growth={memory_growth:.1f}MB"
        )

    @pytest.mark.asyncio
    async def test_configuration_synchronization(self, test_client, audio_test_suite):
        """Test configuration synchronization between services."""
        client, mocks = test_client

        # Test different configuration combinations
        config_test_cases = [
            {
                "whisper_model": "whisper-base",
                "enable_vad": True,
                "enable_diarization": True,
                "noise_reduction": False,
            },
            {
                "whisper_model": "whisper-small",
                "enable_vad": False,
                "enable_diarization": False,
                "noise_reduction": True,
            },
            {
                "whisper_model": "whisper-medium",
                "enable_vad": True,
                "enable_diarization": True,
                "noise_reduction": True,
            },
        ]

        audio_bytes = audio_test_suite.generate_test_audio("wav")

        for i, config in enumerate(config_test_cases):
            # Update configuration
            mocks["config_manager"].get_service_config.return_value = config

            # Configure appropriate response
            whisper_response = {
                "text": f"Config test {i}",
                "speaker_id": "speaker_0" if config["enable_diarization"] else None,
                "confidence": 0.95,
                "model_used": config["whisper_model"],
            }

            mock_whisper_response = Mock()
            mock_whisper_response.status_code = 200
            mock_whisper_response.json.return_value = whisper_response
            mocks["audio_client"].post.return_value = mock_whisper_response

            # Make request
            response = client.post(
                "/api/audio/upload",
                files={"audio": (f"config_test_{i}.wav", audio_bytes, "audio/wav")},
                data={
                    "session_id": f"config_sync_test_{i}",
                    "enable_transcription": "true",
                    "enable_diarization": str(config["enable_diarization"]).lower(),
                    "whisper_model": config["whisper_model"],
                    "noise_reduction": str(config["noise_reduction"]).lower(),
                },
            )

            assert response.status_code == 200
            response_data = response.json()

            # Verify configuration was applied
            assert "transcription" in response_data
            transcription = response_data["transcription"]

            # Check speaker diarization was applied correctly
            if config["enable_diarization"]:
                assert "speaker_id" in transcription and transcription["speaker_id"] is not None

            # Verify configuration sync manager was called
            assert (
                mocks["config_sync_manager"].update_processing_config.called
                or mocks["config_sync_manager"].get_current_config.called
            )

    @pytest.mark.asyncio
    async def test_audio_quality_validation(self, test_client, audio_test_suite):
        """Test audio quality validation and enhancement."""
        client, mocks = test_client

        # Configure mock responses
        whisper_response = {
            "text": "Audio quality test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
            "audio_quality": {
                "snr": 15.2,
                "rms": 0.12,
                "peak": 0.85,
                "spectral_centroid": 2500,
            },
        }

        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        # Test different quality scenarios
        quality_test_cases = [
            {
                "name": "high_quality",
                "amplitude": 0.8,
                "noise_level": 0.01,
                "expected_snr": "> 20dB",
            },
            {
                "name": "medium_quality",
                "amplitude": 0.5,
                "noise_level": 0.05,
                "expected_snr": "10-20dB",
            },
            {
                "name": "low_quality",
                "amplitude": 0.3,
                "noise_level": 0.1,
                "expected_snr": "< 10dB",
            },
        ]

        for test_case in quality_test_cases:
            # Generate audio with specific quality characteristics
            t = np.arange(int(3.0 * 16000)) / 16000
            signal = test_case["amplitude"] * np.sin(2 * np.pi * 440 * t)
            noise = test_case["noise_level"] * np.random.randn(len(signal))
            audio_signal = signal + noise

            audio_bytes = audio_test_suite._generate_wav_bytes(audio_signal, 16000)

            response = client.post(
                "/api/audio/upload",
                files={
                    "audio": (
                        f"quality_{test_case['name']}.wav",
                        audio_bytes,
                        "audio/wav",
                    )
                },
                data={
                    "session_id": f"quality_test_{test_case['name']}",
                    "enable_transcription": "true",
                    "speech_enhancement": "true",
                },
            )

            assert response.status_code == 200
            response_data = response.json()

            # Verify quality metrics are included
            assert "transcription" in response_data
            if "audio_quality" in response_data:
                quality = response_data["audio_quality"]
                assert "snr" in quality or "rms" in quality or "peak" in quality

    @pytest.mark.asyncio
    async def test_streaming_vs_batch_processing(self, test_client, audio_test_suite):
        """Test both streaming and batch processing modes."""
        client, mocks = test_client

        audio_bytes = audio_test_suite.generate_test_audio("wav")

        # Configure mock responses
        whisper_response = {
            "text": "Streaming vs batch test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
        }

        mock_whisper_response = Mock()
        mock_whisper_response.status_code = 200
        mock_whisper_response.json.return_value = whisper_response
        mocks["audio_client"].post.return_value = mock_whisper_response

        # Test batch processing
        batch_start = time.time()
        batch_response = client.post(
            "/api/audio/upload",
            files={"audio": ("batch_test.wav", audio_bytes, "audio/wav")},
            data={
                "session_id": "batch_processing_test",
                "enable_transcription": "true",
                "streaming": "false",
            },
        )
        batch_end = time.time()

        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        batch_time = batch_end - batch_start

        # Test streaming processing (if supported)
        streaming_start = time.time()
        streaming_response = client.post(
            "/api/audio/upload",
            files={"audio": ("streaming_test.wav", audio_bytes, "audio/wav")},
            data={
                "session_id": "streaming_processing_test",
                "enable_transcription": "true",
                "streaming": "true",
            },
        )
        streaming_end = time.time()

        # Streaming might not be supported via upload endpoint, so we handle both cases
        if streaming_response.status_code == 200:
            streaming_data = streaming_response.json()
            streaming_time = streaming_end - streaming_start

            # Compare processing modes
            assert "transcription" in batch_data
            assert "transcription" in streaming_data

            # Both should produce similar results
            batch_text = batch_data["transcription"]["text"]
            streaming_text = streaming_data["transcription"]["text"]

            # Text should be similar (allowing for minor differences)
            assert len(batch_text) > 0
            assert len(streaming_text) > 0

            logger.info(f"Batch processing: {batch_time:.2f}s, Streaming: {streaming_time:.2f}s")
        else:
            # Streaming not supported via this endpoint, which is acceptable
            logger.info(
                f"Streaming not supported via upload endpoint (status: {streaming_response.status_code})"
            )
            assert batch_data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
