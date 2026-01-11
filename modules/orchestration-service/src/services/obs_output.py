"""
OBS WebSocket Output Service

Provides integration with OBS Studio via obs-websocket protocol v5.
Updates text sources with live captions for streaming overlays.

Protocol Reference:
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable, Awaitable

try:
    import obsws_python as obsws
except ImportError:
    obsws = None  # Will be mocked in tests

from models.fireflies import CaptionEntry

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class OBSConnectionError(Exception):
    """Raised when connection to OBS fails."""

    pass


class OBSSourceError(Exception):
    """Raised when source operations fail."""

    pass


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class OBSOutputConfig:
    """
    Configuration for OBS WebSocket output.

    Attributes:
        host: OBS WebSocket server host
        port: OBS WebSocket server port
        password: Authentication password (optional)
        caption_source: Name of text source for captions
        speaker_source: Name of text source for speaker name (optional)
        reconnect_interval: Seconds between reconnection attempts
        max_reconnect_attempts: Maximum reconnection attempts
        connection_timeout: Connection timeout in seconds
        auto_reconnect: Enable automatic reconnection
        create_sources: Create sources if they don't exist
        show_original: Show original text alongside translation
    """

    host: str = "localhost"
    port: int = 4455
    password: Optional[str] = None
    caption_source: str = "LiveTranslate Caption"
    speaker_source: Optional[str] = None
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 3
    connection_timeout: float = 10.0
    auto_reconnect: bool = False
    create_sources: bool = False
    show_original: bool = False


# =============================================================================
# Formatting Functions
# =============================================================================


def format_caption_text(
    caption: Optional[CaptionEntry],
    show_original: bool = False,
    include_speaker: bool = False,
) -> str:
    """
    Format caption text for OBS display.

    Args:
        caption: Caption entry to format, or None for empty
        show_original: Include original text above translation
        include_speaker: Include speaker name in output

    Returns:
        Formatted text string for OBS text source
    """
    if caption is None:
        return ""

    parts = []

    # Add speaker if requested
    if include_speaker and caption.speaker_name:
        parts.append(f"[{caption.speaker_name}]")

    # Add original text if requested
    if show_original and caption.original_text:
        parts.append(caption.original_text)

    # Add translated text
    if caption.translated_text:
        parts.append(caption.translated_text)

    return "\n".join(parts) if parts else ""


def format_speaker_text(caption: Optional[CaptionEntry]) -> str:
    """
    Format speaker name for OBS display.

    Args:
        caption: Caption entry with speaker info

    Returns:
        Speaker name string
    """
    if caption is None:
        return ""

    return caption.speaker_name or ""


# =============================================================================
# OBS Output Service
# =============================================================================


class OBSOutput:
    """
    OBS WebSocket output service.

    Connects to OBS Studio and updates text sources with live captions.
    Supports automatic reconnection and source validation.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 4455,
        password: Optional[str] = None,
        caption_source: str = "LiveTranslate Caption",
        speaker_source: Optional[str] = None,
        auto_reconnect: bool = False,
        create_sources: bool = False,
        show_original: bool = False,
        connection_timeout: float = 10.0,
        reconnect_interval: float = 5.0,
        max_reconnect_attempts: int = 3,
    ):
        """
        Initialize OBS output service.

        Args:
            host: OBS WebSocket server host
            port: OBS WebSocket server port
            password: Authentication password
            caption_source: Name of caption text source
            speaker_source: Name of speaker text source (optional)
            auto_reconnect: Enable automatic reconnection
            create_sources: Create sources if missing
            show_original: Show original text with translation
            connection_timeout: Connection timeout seconds
            reconnect_interval: Seconds between reconnects
            max_reconnect_attempts: Max reconnect attempts
        """
        self._host = host
        self._port = port
        self._password = password
        self._caption_source = caption_source
        self._speaker_source = speaker_source
        self._auto_reconnect = auto_reconnect
        self._create_sources = create_sources
        self._show_original = show_original
        self._connection_timeout = connection_timeout
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_attempts = max_reconnect_attempts

        self._client: Optional[Any] = None
        self._connected = False
        self._reconnect_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "updates_sent": 0,
            "errors": 0,
            "reconnections": 0,
            "connected_at": None,
            "last_update": None,
        }

        logger.info(
            f"OBSOutput initialized: host={host}, port={port}, "
            f"caption_source={caption_source}"
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to OBS."""
        return self._connected

    async def connect(self) -> None:
        """
        Connect to OBS WebSocket server.

        Raises:
            OBSConnectionError: If connection fails
        """
        if obsws is None:
            raise OBSConnectionError("obsws-python library not installed")

        try:
            self._client = obsws.ReqClient(
                host=self._host,
                port=self._port,
                password=self._password,
                timeout=self._connection_timeout,
            )
            await self._client.connect()
            self._connected = True
            self._stats["connected_at"] = datetime.now(timezone.utc)

            logger.info(f"Connected to OBS at {self._host}:{self._port}")

        except asyncio.TimeoutError as e:
            self._connected = False
            raise OBSConnectionError(f"Connection timeout: {e}")

        except Exception as e:
            self._connected = False
            raise OBSConnectionError(f"Failed to connect to OBS: {e}")

    async def disconnect(self) -> None:
        """Disconnect from OBS WebSocket server."""
        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._connected = False
        self._client = None

        logger.info("Disconnected from OBS")

    def _on_disconnected(self) -> None:
        """Handle disconnection event."""
        self._connected = False

        if self._auto_reconnect and not self._reconnect_task:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect to OBS."""
        attempts = 0

        while attempts < self._max_reconnect_attempts:
            try:
                await asyncio.sleep(self._reconnect_interval)
                await self.connect()
                self._stats["reconnections"] += 1
                logger.info("Reconnected to OBS")
                return

            except OBSConnectionError as e:
                attempts += 1
                logger.warning(
                    f"Reconnection attempt {attempts}/{self._max_reconnect_attempts} "
                    f"failed: {e}"
                )

        logger.error("Max reconnection attempts reached")

    async def update_caption(
        self,
        caption: CaptionEntry,
        raise_on_error: bool = True,
    ) -> bool:
        """
        Update OBS text source with caption.

        Args:
            caption: Caption entry to display
            raise_on_error: Raise exception on error (default True)

        Returns:
            True if update succeeded, False otherwise

        Raises:
            OBSConnectionError: If not connected (when raise_on_error=True)
        """
        if not self._connected or not self._client:
            if raise_on_error:
                raise OBSConnectionError("Not connected to OBS")
            return False

        try:
            # Format and update caption text
            caption_text = format_caption_text(
                caption,
                show_original=self._show_original,
            )

            await self._client.set_input_settings(
                name=self._caption_source,
                settings={"text": caption_text},
            )

            # Update speaker source if configured
            if self._speaker_source:
                speaker_text = format_speaker_text(caption)
                await self._client.set_input_settings(
                    name=self._speaker_source,
                    settings={"text": speaker_text},
                )

            self._stats["updates_sent"] += 1
            self._stats["last_update"] = datetime.now(timezone.utc)

            return True

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Failed to update OBS caption: {e}")

            if raise_on_error:
                raise

            return False

    async def clear_caption(self) -> bool:
        """
        Clear the caption text source.

        Returns:
            True if clear succeeded
        """
        if not self._connected or not self._client:
            return False

        try:
            await self._client.set_input_settings(
                name=self._caption_source,
                settings={"text": ""},
            )

            if self._speaker_source:
                await self._client.set_input_settings(
                    name=self._speaker_source,
                    settings={"text": ""},
                )

            return True

        except Exception as e:
            logger.error(f"Failed to clear OBS caption: {e}")
            return False

    async def validate_sources(self) -> bool:
        """
        Validate that required text sources exist in OBS.

        Returns:
            True if all sources exist
        """
        if not self._connected or not self._client:
            return False

        try:
            response = await self._client.get_input_list()
            input_names = {inp["inputName"] for inp in response.inputs}

            # Check caption source
            if self._caption_source not in input_names:
                logger.warning(f"Caption source '{self._caption_source}' not found")
                return False

            # Check speaker source if configured
            if self._speaker_source and self._speaker_source not in input_names:
                logger.warning(f"Speaker source '{self._speaker_source}' not found")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to validate sources: {e}")
            return False

    async def ensure_sources_exist(self) -> bool:
        """
        Create text sources if they don't exist.

        Returns:
            True if sources exist or were created
        """
        if not self._connected or not self._client:
            return False

        if not self._create_sources:
            return await self.validate_sources()

        try:
            response = await self._client.get_input_list()
            input_names = {inp["inputName"] for inp in response.inputs}

            # Create caption source if missing
            if self._caption_source not in input_names:
                await self._client.create_input(
                    scene_name="Scene",  # Default scene
                    input_name=self._caption_source,
                    input_kind="text_gdiplus_v2",
                    input_settings={"text": ""},
                )
                logger.info(f"Created caption source: {self._caption_source}")

            # Create speaker source if configured and missing
            if self._speaker_source and self._speaker_source not in input_names:
                await self._client.create_input(
                    scene_name="Scene",
                    input_name=self._speaker_source,
                    input_kind="text_gdiplus_v2",
                    input_settings={"text": ""},
                )
                logger.info(f"Created speaker source: {self._speaker_source}")

            return True

        except Exception as e:
            logger.error(f"Failed to create sources: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get output statistics.

        Returns:
            Dictionary with stats
        """
        return self._stats.copy()

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.

        Returns:
            Dictionary with connection details
        """
        return {
            "host": self._host,
            "port": self._port,
            "connected": self._connected,
            "caption_source": self._caption_source,
            "speaker_source": self._speaker_source,
            "auto_reconnect": self._auto_reconnect,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_obs_output(config: Optional[OBSOutputConfig] = None) -> OBSOutput:
    """
    Create OBS output instance from config.

    Args:
        config: Configuration, or None for defaults

    Returns:
        Configured OBSOutput instance
    """
    if config is None:
        config = OBSOutputConfig()

    return OBSOutput(
        host=config.host,
        port=config.port,
        password=config.password,
        caption_source=config.caption_source,
        speaker_source=config.speaker_source,
        auto_reconnect=config.auto_reconnect,
        create_sources=config.create_sources,
        show_original=config.show_original,
        connection_timeout=config.connection_timeout,
        reconnect_interval=config.reconnect_interval,
        max_reconnect_attempts=config.max_reconnect_attempts,
    )
