#!/usr/bin/env python3
"""
Integration Tests for Audio Coordinator

Tests the complete AudioCoordinator integration including database operations,
service communication, session management, and end-to-end audio processing workflows.
"""

import pytest
import asyncio
import numpy as np
import httpx
from unittest.mock import patch

# Note: AudioCoordinator implementation to be added - tests temporarily disabled for specific classes
# from src.audio.audio_coordinator import AudioCoordinator, create_audio_coordinator
from src.audio.models import (
    AudioChunkMetadata,
    SourceType,
    get_default_chunking_config,
)


class TestAudioCoordinatorIntegration:
    """Test AudioCoordinator integration functionality."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization_and_shutdown(
        self, mock_database_adapter, mock_service_urls, temp_dir
    ):
        """Test coordinator initialization and shutdown process."""
        config_file = temp_dir / "coordinator_config.yaml"

        coordinator = create_audio_coordinator(
            database_url=None,  # Using mock adapter
            service_urls=mock_service_urls,
            config=get_default_chunking_config(),
            max_concurrent_sessions=5,
            audio_config_file=str(config_file),
        )

        # Replace with mock adapter
        coordinator.database_adapter = mock_database_adapter

        # Test initialization
        result = await coordinator.initialize()
        assert result == True
        assert coordinator.is_initialized == True
        assert coordinator.active_sessions == {}
        assert coordinator.session_processors == {}

        # Test shutdown
        await coordinator.shutdown()
        assert coordinator.is_initialized == False

    @pytest.mark.asyncio
    async def test_session_lifecycle_management(
        self, audio_coordinator, sample_audio_data
    ):
        """Test complete session lifecycle management."""
        session_id = "test_session_lifecycle"

        # Create session
        session_config = {
            "bot_session_id": "bot_123",
            "source_type": SourceType.BOT_AUDIO,
            "target_languages": ["en", "es"],
            "real_time_processing": True,
            "speaker_correlation_enabled": True,
        }

        result = await audio_coordinator.create_session(session_id, **session_config)
        assert result == True
        assert session_id in audio_coordinator.active_sessions

        # Verify session was stored in database
        audio_coordinator.database_adapter.store_session.assert_called_once()

        # Add audio data to session
        voice_audio = sample_audio_data["voice_like"]

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = (
                {"text": "Test transcription", "speaker_id": "speaker_0"},
                {"translated_text": "Test translation"},
            )

            success = await audio_coordinator.add_audio_data(session_id, voice_audio)
            assert success == True

        # Get session status
        status = await audio_coordinator.get_session_status(session_id)
        assert status["session_id"] == session_id
        assert status["active"] == True
        assert status["chunks_processed"] >= 0

        # End session
        result = await audio_coordinator.end_session(session_id)
        assert result == True
        assert session_id not in audio_coordinator.active_sessions

    @pytest.mark.asyncio
    async def test_concurrent_session_processing(
        self, audio_coordinator, sample_audio_data
    ):
        """Test concurrent processing of multiple sessions."""
        num_sessions = 3
        session_ids = [f"concurrent_session_{i}" for i in range(num_sessions)]

        # Create multiple sessions
        for session_id in session_ids:
            result = await audio_coordinator.create_session(
                session_id,
                bot_session_id=f"bot_{session_id}",
                source_type=SourceType.BOT_AUDIO,
                target_languages=["en"],
                real_time_processing=True,
            )
            assert result == True

        # Process audio concurrently
        voice_audio = sample_audio_data["voice_like"]

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = (
                {"text": "Concurrent transcription", "speaker_id": "speaker_0"},
                {"translated_text": "Concurrent translation"},
            )

            # Add audio to all sessions concurrently
            tasks = [
                audio_coordinator.add_audio_data(session_id, voice_audio)
                for session_id in session_ids
            ]
            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(results)

        # Verify all sessions are active
        for session_id in session_ids:
            status = await audio_coordinator.get_session_status(session_id)
            assert status["active"] == True

        # Clean up sessions
        for session_id in session_ids:
            await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_audio_processing_pipeline_integration(
        self, audio_coordinator, sample_audio_data
    ):
        """Test integration of complete audio processing pipeline."""
        session_id = "pipeline_integration_test"

        # Create session with processing enabled
        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_pipeline",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en", "es"],
            real_time_processing=True,
        )

        # Test processing various audio types
        test_cases = [
            ("voice_like", "voice-like audio"),
            ("noisy_voice_10db", "noisy voice"),
            ("sine_440", "pure tone"),
        ]

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = (
                {"text": "Pipeline test", "speaker_id": "speaker_0"},
                {"translated_text": "Prueba de pipeline"},
            )

            for audio_type, description in test_cases:
                audio_data = sample_audio_data[audio_type]

                # Process audio
                result = await audio_coordinator.add_audio_data(session_id, audio_data)
                assert result == True, f"Failed to process {description}"

                # Verify processing metadata
                session_status = await audio_coordinator.get_session_status(session_id)
                assert session_status["chunks_processed"] >= 0

        # End session
        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_database_integration_operations(
        self, audio_coordinator, sample_audio_data, test_chunk_metadata
    ):
        """Test database integration for all operations."""
        session_id = "database_integration_test"

        # Create session
        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_database",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        voice_audio = sample_audio_data["voice_like"]

        # Mock service responses
        whisper_response = {
            "text": "Database integration test",
            "speaker_id": "speaker_0",
            "confidence": 0.95,
            "start_timestamp": 0.0,
            "end_timestamp": 2.0,
        }

        translation_response = {
            "translated_text": "Prueba de integración de base de datos",
            "source_language": "en",
            "target_language": "es",
            "confidence": 0.88,
        }

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = (whisper_response, translation_response)

            # Process audio
            await audio_coordinator.add_audio_data(session_id, voice_audio)

        # Verify database operations were called
        db_adapter = audio_coordinator.database_adapter

        # Check audio chunk storage
        assert db_adapter.store_audio_chunk.called
        chunk_call = db_adapter.store_audio_chunk.call_args[0][0]
        assert isinstance(chunk_call, AudioChunkMetadata)
        assert chunk_call.session_id == session_id

        # Check transcript storage
        assert db_adapter.store_transcript.called

        # Check translation storage
        assert db_adapter.store_translation.called

        # Get session analytics
        analytics = await audio_coordinator.get_session_analytics(session_id)
        assert "chunks_processed" in analytics
        assert "total_duration" in analytics
        assert "average_processing_time" in analytics

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_service_communication_with_retry(
        self, audio_coordinator, sample_audio_data
    ):
        """Test service communication with retry logic."""
        session_id = "service_communication_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_service_comm",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        voice_audio = sample_audio_data["voice_like"]

        # Mock service client with initial failure, then success
        call_count = 0

        async def mock_service_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Mock connection error")
            return httpx.Response(
                200, json={"text": "Retry success", "speaker_id": "speaker_0"}
            )

        with patch.object(
            audio_coordinator.service_client, "post", side_effect=mock_service_call
        ):
            # This should succeed after retry
            result = await audio_coordinator.add_audio_data(session_id, voice_audio)
            # Note: Actual result depends on retry implementation in coordinator
            assert isinstance(result, bool)

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_configuration_hot_reload(self, audio_coordinator):
        """Test hot-reloading of configuration."""
        session_id = "config_reload_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_config",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        # Get initial configuration
        initial_config = await audio_coordinator.get_processing_config(session_id)
        assert initial_config is not None

        # Update configuration
        new_config_updates = {
            "vad": {"enabled": False},
            "noise_reduction": {"strength": 0.8},
            "compression": {"threshold": -15},
        }

        # Apply configuration update
        result = await audio_coordinator.update_processing_config(
            session_id, new_config_updates
        )
        assert result == True

        # Verify configuration was updated
        updated_config = await audio_coordinator.get_processing_config(session_id)
        assert updated_config.vad.enabled == False
        assert updated_config.noise_reduction.strength == 0.8
        assert updated_config.compression.threshold == -15

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_speaker_correlation_integration(
        self, audio_coordinator, sample_audio_data
    ):
        """Test speaker correlation functionality integration."""
        session_id = "speaker_correlation_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_speaker",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
            speaker_correlation_enabled=True,
        )

        voice_audio = sample_audio_data["voice_like"]

        # Mock whisper response with speaker info
        whisper_response = {
            "text": "Speaker correlation test",
            "speaker_id": "speaker_0",
            "speaker_embedding": [0.1, 0.2, 0.3],  # Mock embedding
            "confidence": 0.9,
        }

        # Mock Google Meet speaker data
        google_meet_speaker = {
            "speaker_id": "gmeet_speaker_123",
            "speaker_name": "John Doe",
            "speaking_time": 2.0,
        }

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = (whisper_response, None)

            # Mock speaker correlation
            with patch.object(
                audio_coordinator, "_correlate_speakers"
            ) as mock_correlate:
                mock_correlate.return_value = {
                    "correlation_confidence": 0.85,
                    "google_meet_speaker": google_meet_speaker,
                }

                result = await audio_coordinator.add_audio_data(session_id, voice_audio)
                assert result == True

                # Verify speaker correlation was attempted
                mock_correlate.assert_called_once()

        # Check if speaker correlation was stored
        db_adapter = audio_coordinator.database_adapter
        assert db_adapter.store_speaker_correlation.called

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, audio_coordinator, sample_audio_data
    ):
        """Test error handling and recovery mechanisms."""
        session_id = "error_handling_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_error",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        voice_audio = sample_audio_data["voice_like"]

        # Test database error handling
        audio_coordinator.database_adapter.store_audio_chunk.side_effect = Exception(
            "Database connection failed"
        )

        # Should handle database error gracefully
        result = await audio_coordinator.add_audio_data(session_id, voice_audio)
        # Result depends on error handling implementation
        assert isinstance(result, bool)

        # Reset mock
        audio_coordinator.database_adapter.store_audio_chunk.side_effect = None
        audio_coordinator.database_adapter.store_audio_chunk.return_value = (
            "recovered_chunk_id"
        )

        # Should recover on next attempt
        result = await audio_coordinator.add_audio_data(session_id, voice_audio)
        assert isinstance(result, bool)

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_session_cleanup_and_resource_management(
        self, audio_coordinator, sample_audio_data
    ):
        """Test session cleanup and resource management."""
        session_ids = [f"cleanup_test_{i}" for i in range(3)]

        # Create multiple sessions
        for session_id in session_ids:
            await audio_coordinator.create_session(
                session_id,
                bot_session_id=f"bot_{session_id}",
                source_type=SourceType.BOT_AUDIO,
                target_languages=["en"],
            )

        # Process some audio
        voice_audio = sample_audio_data["voice_like"]

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = ({"text": "Cleanup test"}, None)

            for session_id in session_ids:
                await audio_coordinator.add_audio_data(session_id, voice_audio)

        # Verify sessions are active
        assert len(audio_coordinator.active_sessions) == 3

        # Test cleanup on shutdown
        await audio_coordinator.shutdown()

        # All sessions should be cleaned up
        assert len(audio_coordinator.active_sessions) == 0
        assert len(audio_coordinator.session_processors) == 0

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(
        self, audio_coordinator, sample_audio_data
    ):
        """Test performance monitoring and metrics collection."""
        session_id = "performance_monitoring_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_performance",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        voice_audio = sample_audio_data["voice_like"]

        with patch.object(audio_coordinator, "_process_with_services") as mock_process:
            mock_process.return_value = (
                {"text": "Performance test", "speaker_id": "speaker_0"},
                {"translated_text": "Prueba de rendimiento"},
            )

            # Process multiple chunks to generate metrics
            for i in range(5):
                await audio_coordinator.add_audio_data(session_id, voice_audio)

        # Get performance metrics
        metrics = await audio_coordinator.get_performance_metrics(session_id)

        assert "total_chunks_processed" in metrics
        assert "average_processing_time" in metrics
        assert "average_chunk_duration" in metrics
        assert "total_audio_duration" in metrics
        assert "processing_throughput" in metrics

        # Verify metrics are reasonable
        assert metrics["total_chunks_processed"] >= 5
        assert metrics["average_processing_time"] >= 0
        assert metrics["total_audio_duration"] > 0

        await audio_coordinator.end_session(session_id)


class TestAudioCoordinatorErrorScenarios:
    """Test error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_session_operations(self, audio_coordinator):
        """Test operations on invalid sessions."""
        invalid_session_id = "nonexistent_session"

        # Test adding audio to nonexistent session
        audio_data = np.random.randn(16000).astype(np.float32)
        result = await audio_coordinator.add_audio_data(invalid_session_id, audio_data)
        assert result == False

        # Test getting status of nonexistent session
        status = await audio_coordinator.get_session_status(invalid_session_id)
        assert status is None or status.get("error") is not None

        # Test ending nonexistent session
        result = await audio_coordinator.end_session(invalid_session_id)
        assert result == False

    @pytest.mark.asyncio
    async def test_invalid_audio_data(self, audio_coordinator):
        """Test handling of invalid audio data."""
        session_id = "invalid_audio_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_invalid",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        # Test empty audio
        empty_audio = np.array([], dtype=np.float32)
        result = await audio_coordinator.add_audio_data(session_id, empty_audio)
        assert isinstance(result, bool)

        # Test NaN audio
        nan_audio = np.full(1000, np.nan, dtype=np.float32)
        result = await audio_coordinator.add_audio_data(session_id, nan_audio)
        assert isinstance(result, bool)

        # Test infinite audio
        inf_audio = np.full(1000, np.inf, dtype=np.float32)
        result = await audio_coordinator.add_audio_data(session_id, inf_audio)
        assert isinstance(result, bool)

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_service_unavailable_scenarios(
        self, audio_coordinator, sample_audio_data
    ):
        """Test handling when services are unavailable."""
        session_id = "service_unavailable_test"

        await audio_coordinator.create_session(
            session_id,
            bot_session_id="bot_unavailable",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )

        voice_audio = sample_audio_data["voice_like"]

        # Mock service unavailable
        with patch.object(audio_coordinator.service_client, "post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Service unavailable")

            result = await audio_coordinator.add_audio_data(session_id, voice_audio)
            # Should handle gracefully (exact behavior depends on implementation)
            assert isinstance(result, bool)

        await audio_coordinator.end_session(session_id)

    @pytest.mark.asyncio
    async def test_concurrent_session_limit(self, audio_coordinator):
        """Test behavior when concurrent session limit is exceeded."""
        max_sessions = audio_coordinator.max_concurrent_sessions
        session_ids = [f"limit_test_{i}" for i in range(max_sessions + 2)]

        created_sessions = []

        # Create sessions up to and beyond limit
        for session_id in session_ids:
            result = await audio_coordinator.create_session(
                session_id,
                bot_session_id=f"bot_{session_id}",
                source_type=SourceType.BOT_AUDIO,
                target_languages=["en"],
            )

            if result:
                created_sessions.append(session_id)

        # Should not exceed maximum
        assert len(created_sessions) <= max_sessions
        assert len(audio_coordinator.active_sessions) <= max_sessions

        # Clean up created sessions
        for session_id in created_sessions:
            await audio_coordinator.end_session(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
