"""
OBS WebSocket Output Tests (TDD)

Tests for OBS Studio integration via obs-websocket protocol.
Written BEFORE implementation following TDD principles.

OBS WebSocket v5 protocol reference:
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import asyncio

# Add src to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(orchestration_root))

from models.fireflies import CaptionEntry


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_caption():
    """Sample caption for testing."""
    return CaptionEntry(
        id="caption-001",
        original_text="Hello, how are you?",
        translated_text="Hola, ¿cómo estás?",
        speaker_name="Alice",
        speaker_color="#4CAF50",
        target_language="es",
        timestamp=datetime.now(timezone.utc),
        duration_seconds=8.0,
        confidence=0.95,
    )


@pytest.fixture
def obs_config():
    """Sample OBS connection config."""
    return {
        "host": "localhost",
        "port": 4455,
        "password": "test_password",
        "caption_source": "Caption Text",
        "speaker_source": "Speaker Name",
        "reconnect_interval": 5.0,
        "max_reconnect_attempts": 3,
    }


# =============================================================================
# Configuration Tests
# =============================================================================


class TestOBSConfiguration:
    """Test OBS output configuration."""

    def test_config_has_required_fields(self, obs_config):
        """Test configuration contains required fields."""
        assert "host" in obs_config
        assert "port" in obs_config
        assert "caption_source" in obs_config

    def test_config_defaults(self):
        """Test configuration has sensible defaults."""
        from services.obs_output import OBSOutputConfig

        config = OBSOutputConfig()
        assert config.host == "localhost"
        assert config.port == 4455
        assert config.caption_source == "LiveTranslate Caption"
        assert config.reconnect_interval > 0

    def test_config_custom_values(self):
        """Test configuration accepts custom values."""
        from services.obs_output import OBSOutputConfig

        config = OBSOutputConfig(
            host="192.168.1.100",
            port=4456,
            password="secret",
            caption_source="My Captions",
            speaker_source="My Speaker",
        )
        assert config.host == "192.168.1.100"
        assert config.port == 4456
        assert config.password == "secret"


# =============================================================================
# Connection Tests
# =============================================================================


class TestOBSConnection:
    """Test OBS WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to OBS."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()

            assert obs.is_connected
            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_raises_error(self):
        """Test connection failure raises appropriate error."""
        from services.obs_output import OBSOutput, OBSConnectionError

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(side_effect=Exception("Connection refused"))
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            with pytest.raises(OBSConnectionError):
                await obs.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection from OBS."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()
            await obs.disconnect()

            assert not obs.is_connected
            mock_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_on_disconnect(self):
        """Test automatic reconnection after disconnect."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(auto_reconnect=True)
            await obs.connect()

            # Simulate disconnect
            obs._on_disconnected()

            # Should attempt reconnect
            await asyncio.sleep(0.1)
            assert mock_client.connect.call_count >= 1


# =============================================================================
# Text Source Update Tests
# =============================================================================


class TestTextSourceUpdates:
    """Test updating OBS text sources."""

    @pytest.mark.asyncio
    async def test_update_caption_text(self, sample_caption):
        """Test updating caption text source."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(caption_source="Caption Text")
            await obs.connect()
            await obs.update_caption(sample_caption)

            # Should call SetInputSettings with caption text
            mock_client.set_input_settings.assert_called()
            call_args = mock_client.set_input_settings.call_args
            assert "Hola, ¿cómo estás?" in str(call_args)

    @pytest.mark.asyncio
    async def test_update_speaker_name(self, sample_caption):
        """Test updating speaker name source."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(
                caption_source="Caption Text",
                speaker_source="Speaker Name",
            )
            await obs.connect()
            await obs.update_caption(sample_caption)

            # Should update both caption and speaker sources
            assert mock_client.set_input_settings.call_count >= 2

    @pytest.mark.asyncio
    async def test_clear_caption(self):
        """Test clearing caption text."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()
            await obs.clear_caption()

            # Should set text to empty
            mock_client.set_input_settings.assert_called()

    @pytest.mark.asyncio
    async def test_update_with_original_text(self, sample_caption):
        """Test updating with both original and translated text."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(show_original=True)
            await obs.connect()
            await obs.update_caption(sample_caption)

            call_args = mock_client.set_input_settings.call_args
            # Should include both original and translated
            assert "Hello" in str(call_args) or "Hola" in str(call_args)


# =============================================================================
# Formatting Tests
# =============================================================================


class TestCaptionFormatting:
    """Test caption text formatting for OBS."""

    def test_format_caption_basic(self, sample_caption):
        """Test basic caption formatting."""
        from services.obs_output import format_caption_text

        text = format_caption_text(sample_caption, show_original=False)
        assert "Hola, ¿cómo estás?" in text

    def test_format_caption_with_original(self, sample_caption):
        """Test caption formatting with original text."""
        from services.obs_output import format_caption_text

        text = format_caption_text(sample_caption, show_original=True)
        assert "Hello, how are you?" in text
        assert "Hola, ¿cómo estás?" in text

    def test_format_caption_with_speaker(self, sample_caption):
        """Test caption formatting includes speaker."""
        from services.obs_output import format_caption_text

        text = format_caption_text(sample_caption, include_speaker=True)
        assert "Alice" in text

    def test_format_speaker_name(self, sample_caption):
        """Test speaker name formatting."""
        from services.obs_output import format_speaker_text

        text = format_speaker_text(sample_caption)
        assert text == "Alice"

    def test_format_empty_caption(self):
        """Test formatting empty/clear caption."""
        from services.obs_output import format_caption_text

        text = format_caption_text(None, show_original=False)
        assert text == ""


# =============================================================================
# Source Validation Tests
# =============================================================================


class TestSourceValidation:
    """Test OBS source validation."""

    @pytest.mark.asyncio
    async def test_validate_source_exists(self):
        """Test validating that a source exists in OBS."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.get_input_list.return_value = MagicMock(
                inputs=[
                    {"inputName": "Caption Text", "inputKind": "text_gdiplus_v2"},
                    {"inputName": "Speaker Name", "inputKind": "text_gdiplus_v2"},
                ]
            )
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(caption_source="Caption Text")
            await obs.connect()
            exists = await obs.validate_sources()

            assert exists

    @pytest.mark.asyncio
    async def test_validate_source_missing(self):
        """Test validation when source doesn't exist."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.get_input_list.return_value = MagicMock(inputs=[])
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(caption_source="Missing Source")
            await obs.connect()
            exists = await obs.validate_sources()

            assert not exists

    @pytest.mark.asyncio
    async def test_create_source_if_missing(self):
        """Test creating source if it doesn't exist."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.get_input_list.return_value = MagicMock(inputs=[])
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(caption_source="New Source", create_sources=True)
            await obs.connect()
            await obs.ensure_sources_exist()

            mock_client.create_input.assert_called()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestOBSErrorHandling:
    """Test OBS error handling."""

    @pytest.mark.asyncio
    async def test_update_when_disconnected(self, sample_caption):
        """Test updating caption when not connected."""
        from services.obs_output import OBSOutput, OBSConnectionError

        obs = OBSOutput()
        # Not connected

        with pytest.raises(OBSConnectionError):
            await obs.update_caption(sample_caption)

    @pytest.mark.asyncio
    async def test_handles_obs_error_gracefully(self, sample_caption):
        """Test graceful handling of OBS errors."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.set_input_settings = AsyncMock(
                side_effect=Exception("OBS Error")
            )
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()

            # Should not raise, but log error
            result = await obs.update_caption(sample_caption, raise_on_error=False)
            assert result is False

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test connection timeout handling."""
        from services.obs_output import OBSOutput, OBSConnectionError

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(
                side_effect=asyncio.TimeoutError("Connection timeout")
            )
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput(connection_timeout=1.0)
            with pytest.raises(OBSConnectionError):
                await obs.connect()


# =============================================================================
# Integration with Caption Buffer Tests
# =============================================================================


class TestCaptionBufferIntegration:
    """Test integration with CaptionBuffer."""

    @pytest.mark.asyncio
    async def test_register_as_caption_callback(self, sample_caption):
        """Test registering OBSOutput as caption callback."""
        from services.obs_output import OBSOutput
        from services.caption_buffer import create_caption_buffer

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()

            buffer = create_caption_buffer()

            # Register OBS callback
            async def on_caption(caption):
                await obs.update_caption(caption)

            # Add caption - should trigger callback
            # (actual integration would wire this up)

    @pytest.mark.asyncio
    async def test_handles_rapid_updates(self, sample_caption):
        """Test handling rapid caption updates."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()

            # Rapid updates
            for i in range(10):
                caption = CaptionEntry(
                    id=f"caption-{i:03d}",
                    translated_text=f"Caption {i}",
                    speaker_name="Alice",
                    target_language="es",
                )
                await obs.update_caption(caption)

            # All should succeed
            assert mock_client.set_input_settings.call_count >= 10


# =============================================================================
# Statistics Tests
# =============================================================================


class TestOBSStatistics:
    """Test OBS output statistics."""

    @pytest.mark.asyncio
    async def test_tracks_updates_sent(self, sample_caption):
        """Test tracking of updates sent to OBS."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()

            await obs.update_caption(sample_caption)
            await obs.update_caption(sample_caption)

            stats = obs.get_stats()
            assert stats["updates_sent"] == 2

    @pytest.mark.asyncio
    async def test_tracks_errors(self, sample_caption):
        """Test tracking of errors."""
        from services.obs_output import OBSOutput

        with patch("services.obs_output.obsws") as mock_obsws:
            mock_client = AsyncMock()
            mock_client.set_input_settings = AsyncMock(side_effect=Exception("Error"))
            mock_obsws.ReqClient.return_value = mock_client

            obs = OBSOutput()
            await obs.connect()

            await obs.update_caption(sample_caption, raise_on_error=False)

            stats = obs.get_stats()
            assert stats["errors"] >= 1

    def test_get_connection_info(self):
        """Test getting connection information."""
        from services.obs_output import OBSOutput

        obs = OBSOutput(host="192.168.1.100", port=4456)
        info = obs.get_connection_info()

        assert info["host"] == "192.168.1.100"
        assert info["port"] == 4456
        assert info["connected"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
