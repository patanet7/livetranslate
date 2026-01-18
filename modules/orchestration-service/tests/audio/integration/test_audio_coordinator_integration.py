#!/usr/bin/env python3
"""
Integration Tests for Audio Coordinator

Tests the complete AudioCoordinator integration including session management,
audio processing, and end-to-end workflows using the actual API.
"""

import asyncio
from unittest.mock import patch, AsyncMock

import numpy as np
import pytest

from src.audio.audio_coordinator import create_audio_coordinator
from src.audio.models import (
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
            database_url=None,
            service_urls=mock_service_urls,
            config=get_default_chunking_config(),
            max_concurrent_sessions=5,
            audio_config_file=str(config_file),
        )

        # Replace with mock adapter
        coordinator.database_adapter = mock_database_adapter

        # Test initialization
        result = await coordinator.initialize()
        assert result
        assert coordinator.is_running

        # Test coordinator status
        status = coordinator.get_coordinator_status()
        assert status["is_running"] is True
        assert "session_statistics" in status

        # Test shutdown
        shutdown_stats = await coordinator.shutdown()
        assert not coordinator.is_running
        assert isinstance(shutdown_stats, dict)

    @pytest.mark.asyncio
    async def test_session_lifecycle_management(self, audio_coordinator, sample_audio_data):
        """Test complete session lifecycle management."""
        bot_session_id = "test_bot_lifecycle"

        # Create session
        session_id = await audio_coordinator.create_audio_session(
            bot_session_id=bot_session_id,
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en", "es"],
        )
        assert session_id is not None
        assert session_id.startswith("audio_")

        # Start session
        started = await audio_coordinator.start_audio_session(session_id)
        assert started

        # Get session status
        status = audio_coordinator.get_session_status(session_id)
        assert status is not None
        assert "session" in status
        assert "chunk_manager_status" in status

        # Stop session
        stop_stats = await audio_coordinator.stop_audio_session(session_id)
        assert isinstance(stop_stats, dict)

        # Session should no longer exist
        status_after = audio_coordinator.get_session_status(session_id)
        assert status_after is None

    @pytest.mark.asyncio
    async def test_concurrent_session_processing(self, audio_coordinator, sample_audio_data):
        """Test concurrent processing of multiple sessions."""
        num_sessions = 3
        session_ids = []

        # Create multiple sessions
        for i in range(num_sessions):
            session_id = await audio_coordinator.create_audio_session(
                bot_session_id=f"bot_concurrent_{i}",
                source_type=SourceType.BOT_AUDIO,
                target_languages=["en"],
            )
            assert session_id is not None
            session_ids.append(session_id)

            # Start session
            started = await audio_coordinator.start_audio_session(session_id)
            assert started

        # Verify all sessions are active
        all_statuses = audio_coordinator.get_all_sessions_status()
        assert len(all_statuses) >= num_sessions

        # Clean up sessions
        for session_id in session_ids:
            await audio_coordinator.stop_audio_session(session_id)

    @pytest.mark.asyncio
    async def test_audio_data_processing(self, audio_coordinator, sample_audio_data):
        """Test adding audio data to a session."""
        bot_session_id = "test_bot_audio_data"

        # Create and start session
        session_id = await audio_coordinator.create_audio_session(
            bot_session_id=bot_session_id,
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )
        assert session_id is not None
        await audio_coordinator.start_audio_session(session_id)

        # Add voice-like audio data
        voice_audio = sample_audio_data["voice_like"]
        result = await audio_coordinator.add_audio_data(session_id, voice_audio)
        assert result is True

        # Add more audio data
        for audio_type in ["sine_440", "noisy_voice_10db"]:
            audio_data = sample_audio_data[audio_type]
            result = await audio_coordinator.add_audio_data(session_id, audio_data)
            assert isinstance(result, bool)

        # Cleanup
        await audio_coordinator.stop_audio_session(session_id)

    @pytest.mark.asyncio
    async def test_coordinator_status_reporting(self, audio_coordinator, sample_audio_data):
        """Test coordinator status and statistics reporting."""
        # Get coordinator status
        status = audio_coordinator.get_coordinator_status()
        assert "is_running" in status
        assert "session_statistics" in status
        assert "config" in status

        # Create a session to have active stats
        session_id = await audio_coordinator.create_audio_session(
            bot_session_id="test_bot_status",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )
        await audio_coordinator.start_audio_session(session_id)

        # Status should reflect active session
        updated_status = audio_coordinator.get_coordinator_status()
        assert updated_status["session_statistics"]["active_sessions"] >= 1

        # Cleanup
        await audio_coordinator.stop_audio_session(session_id)


class TestAudioCoordinatorErrorScenarios:
    """Test error scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_session_operations(self, audio_coordinator):
        """Test operations on invalid sessions."""
        invalid_session_id = "nonexistent_session_12345"

        # Test adding audio to nonexistent session
        audio_data = np.random.randn(16000).astype(np.float32)
        result = await audio_coordinator.add_audio_data(invalid_session_id, audio_data)
        assert result is False

        # Test getting status of nonexistent session
        status = audio_coordinator.get_session_status(invalid_session_id)
        assert status is None

        # Test stopping nonexistent session
        result = await audio_coordinator.stop_audio_session(invalid_session_id)
        assert isinstance(result, dict)  # Returns empty dict on failure

    @pytest.mark.asyncio
    async def test_invalid_audio_data(self, audio_coordinator):
        """Test handling of invalid audio data."""
        # Create and start session
        session_id = await audio_coordinator.create_audio_session(
            bot_session_id="bot_invalid_audio",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )
        await audio_coordinator.start_audio_session(session_id)

        # Test empty audio
        empty_audio = np.array([], dtype=np.float32)
        result = await audio_coordinator.add_audio_data(session_id, empty_audio)
        assert isinstance(result, bool)

        # Test very small audio (may be rejected)
        tiny_audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = await audio_coordinator.add_audio_data(session_id, tiny_audio)
        assert isinstance(result, bool)

        # Cleanup
        await audio_coordinator.stop_audio_session(session_id)

    @pytest.mark.asyncio
    async def test_session_without_start(self, audio_coordinator, sample_audio_data):
        """Test adding audio to a session that wasn't started."""
        # Create session but don't start it
        session_id = await audio_coordinator.create_audio_session(
            bot_session_id="bot_no_start",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en"],
        )
        assert session_id is not None

        # Try to add audio - behavior depends on implementation
        voice_audio = sample_audio_data["voice_like"]
        result = await audio_coordinator.add_audio_data(session_id, voice_audio)
        # Should either succeed (auto-start) or fail gracefully
        assert isinstance(result, bool)

        # Cleanup
        await audio_coordinator.stop_audio_session(session_id)


class TestAudioCoordinatorConfiguration:
    """Test configuration handling."""

    @pytest.mark.asyncio
    async def test_audio_processing_config(self, audio_coordinator):
        """Test audio processing configuration retrieval."""
        # Get default config
        config = audio_coordinator.get_audio_processing_config()
        assert config is not None
        assert hasattr(config, "preset_name")

        # Get available presets
        presets = audio_coordinator.get_available_audio_presets()
        assert isinstance(presets, list)

        # Get config schema
        schema = audio_coordinator.get_audio_config_schema()
        assert isinstance(schema, dict)

    @pytest.mark.asyncio
    async def test_session_with_custom_config(self, audio_coordinator):
        """Test creating session with custom configuration."""
        custom_config = {
            "min_chunk_duration": 1.5,
            "max_chunk_duration": 4.0,
        }

        session_id = await audio_coordinator.create_audio_session(
            bot_session_id="bot_custom_config",
            source_type=SourceType.BOT_AUDIO,
            target_languages=["en", "es", "fr"],
            custom_config=custom_config,
        )
        assert session_id is not None

        # Cleanup
        await audio_coordinator.stop_audio_session(session_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
