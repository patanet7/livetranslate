#!/usr/bin/env python3
"""
Unified Bot Manager - Orchestration Service

Provides a unified interface for bot management by coordinating between:
- Generic BotManager (orchestration service bot lifecycle)
- GoogleMeetBotManager (specialized Google Meet integration)

Uses dependency injection pattern to allow different bot implementations
while maintaining a consistent interface for the rest of the system.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

from .bot_manager import BotManager, MeetingRequest as GenericMeetingRequest
from src.bot.bot_manager import (
    GoogleMeetBotManager,
    MeetingRequest as GoogleMeetMeetingRequest,
)

logger = logging.getLogger(__name__)


class BotType(Enum):
    """Bot type enumeration for unified interface."""

    GENERIC = "generic"
    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    TEAMS = "teams"


@dataclass
class UnifiedMeetingRequest:
    """Unified meeting request that can be converted to specific implementations."""

    meeting_id: str
    meeting_title: Optional[str] = None
    meeting_uri: Optional[str] = None
    bot_type: BotType = BotType.GENERIC
    target_languages: List[str] = None
    enable_translation: bool = True
    enable_transcription: bool = True
    enable_virtual_webcam: bool = True
    audio_storage_enabled: bool = True
    cleanup_on_exit: bool = True
    organizer_email: Optional[str] = None
    scheduled_start: Optional[float] = None
    priority: str = "normal"
    requester_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = ["en"]
        if self.metadata is None:
            self.metadata = {}

    def to_generic_request(self) -> GenericMeetingRequest:
        """Convert to generic meeting request."""
        return GenericMeetingRequest(
            meeting_id=self.meeting_id,
            meeting_title=self.meeting_title,
            meeting_uri=self.meeting_uri,
            bot_type=self.bot_type,
            target_languages=self.target_languages,
            enable_translation=self.enable_translation,
            enable_transcription=self.enable_transcription,
            enable_virtual_webcam=self.enable_virtual_webcam,
            audio_storage_enabled=self.audio_storage_enabled,
            cleanup_on_exit=self.cleanup_on_exit,
            metadata=self.metadata,
        )

    def to_google_meet_request(self) -> GoogleMeetMeetingRequest:
        """Convert to Google Meet meeting request."""
        return GoogleMeetMeetingRequest(
            meeting_id=self.meeting_id,
            meeting_title=self.meeting_title,
            organizer_email=self.organizer_email,
            scheduled_start=self.scheduled_start,
            target_languages=self.target_languages,
            recording_enabled=self.audio_storage_enabled,
            auto_translation=self.enable_translation,
            priority=self.priority,
            requester_id=self.requester_id,
            metadata=self.metadata,
        )


class UnifiedBotManager:
    """
    Unified bot manager that coordinates between different bot implementations.

    Provides a single interface for bot management while delegating to
    appropriate specialized managers based on bot type.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize unified bot manager."""
        self.config = config or {}

        # Initialize specialized managers
        self.generic_manager = BotManager(self.config.get("generic_bot_config"))
        self.google_meet_manager = GoogleMeetBotManager(
            self.config.get("google_meet_config")
        )

        # Manager routing
        self.manager_map = {
            BotType.GENERIC: self.generic_manager,
            BotType.GOOGLE_MEET: self.google_meet_manager,
            BotType.ZOOM: self.generic_manager,  # Fallback to generic for now
            BotType.TEAMS: self.generic_manager,  # Fallback to generic for now
        }

        # Unified state tracking
        self.active_bots: Dict[
            str, Dict[str, Any]
        ] = {}  # bot_id -> {manager, type, info}
        self._initialized = False

        # Event callbacks
        self.unified_callbacks: List[Callable] = []

        # Setup cross-manager coordination
        self._setup_manager_coordination()

        logger.info("UnifiedBotManager initialized")

    async def initialize(self) -> bool:
        """Initialize all bot managers."""
        try:
            # Initialize generic manager
            await self.generic_manager.start()
            logger.info("Generic bot manager initialized")

            # Initialize Google Meet manager
            google_meet_success = await self.google_meet_manager.start()
            if google_meet_success:
                logger.info("Google Meet bot manager initialized")
            else:
                logger.warning("Google Meet bot manager initialization failed")

            self._initialized = True
            logger.info("Unified bot manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize unified bot manager: {e}")
            return False

    async def shutdown(self):
        """Shutdown all bot managers."""
        try:
            await self.generic_manager.stop()
            await self.google_meet_manager.stop()

            self._initialized = False
            logger.info("Unified bot manager shutdown completed")

        except Exception as e:
            logger.error(f"Error during unified bot manager shutdown: {e}")

    def _setup_manager_coordination(self):
        """Setup coordination between managers."""

        # Generic manager callbacks
        def on_generic_status_change(bot_id: str, old_status, new_status):
            self._handle_status_change(bot_id, "generic", old_status, new_status)

        def on_generic_lifecycle(bot_id: str, bot_info: Dict[str, Any]):
            self._handle_lifecycle_event(bot_id, "generic", bot_info)

        self.generic_manager.add_status_change_handler(on_generic_status_change)
        self.generic_manager.add_lifecycle_handler(on_generic_lifecycle)

        # Google Meet manager callbacks
        def on_google_meet_spawned(bot):
            self._handle_bot_spawned(bot.bot_id, "google_meet", bot)

        def on_google_meet_terminated(bot):
            self._handle_bot_terminated(bot.bot_id, "google_meet", bot)

        def on_google_meet_error(bot, error):
            self._handle_bot_error(bot.bot_id, "google_meet", bot, error)

        self.google_meet_manager.on_bot_spawned = on_google_meet_spawned
        self.google_meet_manager.on_bot_terminated = on_google_meet_terminated
        self.google_meet_manager.on_bot_error = on_google_meet_error

    def _handle_status_change(
        self, bot_id: str, manager_type: str, old_status, new_status
    ):
        """Handle status changes from managers."""
        try:
            if bot_id in self.active_bots:
                self.active_bots[bot_id]["status"] = new_status
                self.active_bots[bot_id]["last_update"] = (
                    asyncio.get_event_loop().time()
                )

                # Notify unified callbacks
                self._notify_unified_callbacks(
                    "status_change",
                    {
                        "bot_id": bot_id,
                        "manager_type": manager_type,
                        "old_status": old_status,
                        "new_status": new_status,
                    },
                )

        except Exception as e:
            logger.error(f"Error handling status change for bot {bot_id}: {e}")

    def _handle_lifecycle_event(
        self, bot_id: str, manager_type: str, bot_info: Dict[str, Any]
    ):
        """Handle lifecycle events from managers."""
        try:
            if bot_id in self.active_bots:
                self.active_bots[bot_id]["info"] = bot_info
                self.active_bots[bot_id]["last_update"] = (
                    asyncio.get_event_loop().time()
                )

                # Notify unified callbacks
                self._notify_unified_callbacks(
                    "lifecycle",
                    {
                        "bot_id": bot_id,
                        "manager_type": manager_type,
                        "bot_info": bot_info,
                    },
                )

        except Exception as e:
            logger.error(f"Error handling lifecycle event for bot {bot_id}: {e}")

    def _handle_bot_spawned(self, bot_id: str, manager_type: str, bot):
        """Handle bot spawned events."""
        try:
            self.active_bots[bot_id] = {
                "manager_type": manager_type,
                "manager": self.manager_map[BotType.GOOGLE_MEET],
                "bot_type": BotType.GOOGLE_MEET,
                "status": "active",
                "created_at": asyncio.get_event_loop().time(),
                "last_update": asyncio.get_event_loop().time(),
                "info": bot,
            }

            # Notify unified callbacks
            self._notify_unified_callbacks(
                "bot_spawned",
                {"bot_id": bot_id, "manager_type": manager_type, "bot": bot},
            )

        except Exception as e:
            logger.error(f"Error handling bot spawned for {bot_id}: {e}")

    def _handle_bot_terminated(self, bot_id: str, manager_type: str, bot):
        """Handle bot terminated events."""
        try:
            if bot_id in self.active_bots:
                self.active_bots[bot_id]["status"] = "terminated"
                self.active_bots[bot_id]["last_update"] = (
                    asyncio.get_event_loop().time()
                )

                # Notify unified callbacks
                self._notify_unified_callbacks(
                    "bot_terminated",
                    {"bot_id": bot_id, "manager_type": manager_type, "bot": bot},
                )

                # Remove from active bots after a delay
                asyncio.create_task(self._cleanup_terminated_bot(bot_id))

        except Exception as e:
            logger.error(f"Error handling bot terminated for {bot_id}: {e}")

    def _handle_bot_error(self, bot_id: str, manager_type: str, bot, error: str):
        """Handle bot error events."""
        try:
            if bot_id in self.active_bots:
                self.active_bots[bot_id]["status"] = "error"
                self.active_bots[bot_id]["error"] = error
                self.active_bots[bot_id]["last_update"] = (
                    asyncio.get_event_loop().time()
                )

                # Notify unified callbacks
                self._notify_unified_callbacks(
                    "bot_error",
                    {
                        "bot_id": bot_id,
                        "manager_type": manager_type,
                        "bot": bot,
                        "error": error,
                    },
                )

        except Exception as e:
            logger.error(f"Error handling bot error for {bot_id}: {e}")

    async def _cleanup_terminated_bot(self, bot_id: str):
        """Cleanup terminated bot after delay."""
        await asyncio.sleep(300)  # 5 minute delay
        if (
            bot_id in self.active_bots
            and self.active_bots[bot_id]["status"] == "terminated"
        ):
            del self.active_bots[bot_id]

    def _notify_unified_callbacks(self, event_type: str, event_data: Dict[str, Any]):
        """Notify unified callbacks of events."""
        for callback in self.unified_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(event_type, event_data))
                else:
                    callback(event_type, event_data)
            except Exception as e:
                logger.error(f"Error in unified callback: {e}")

    # Unified API Methods

    async def request_bot(
        self, meeting_request: UnifiedMeetingRequest
    ) -> Optional[str]:
        """Request a new bot using unified interface."""
        try:
            if not self._initialized:
                logger.error("Unified bot manager not initialized")
                return None

            # Determine which manager to use
            bot_type = meeting_request.bot_type
            manager = self.manager_map.get(bot_type)

            if not manager:
                logger.error(f"No manager available for bot type: {bot_type}")
                return None

            # Convert request to appropriate format and delegate
            if bot_type == BotType.GOOGLE_MEET:
                google_request = meeting_request.to_google_meet_request()
                bot_id = await self.google_meet_manager.request_bot(google_request)
            else:
                generic_request = meeting_request.to_generic_request()
                bot_id = await self.generic_manager.request_bot(generic_request)

            if bot_id and bot_id != "queued":
                # Track in unified state
                self.active_bots[bot_id] = {
                    "manager_type": bot_type.value,
                    "manager": manager,
                    "bot_type": bot_type,
                    "status": "spawning",
                    "created_at": asyncio.get_event_loop().time(),
                    "last_update": asyncio.get_event_loop().time(),
                    "meeting_request": meeting_request,
                }

            return bot_id

        except Exception as e:
            logger.error(f"Failed to request bot: {e}")
            return None

    async def terminate_bot(self, bot_id: str) -> bool:
        """Terminate a bot using unified interface."""
        try:
            if bot_id not in self.active_bots:
                logger.warning(f"Bot not found in unified manager: {bot_id}")
                return False

            bot_info = self.active_bots[bot_id]
            manager = bot_info["manager"]

            # Delegate to appropriate manager
            if isinstance(manager, GoogleMeetBotManager):
                return await manager.terminate_bot(bot_id)
            else:
                return await manager.terminate_bot(
                    bot_id, "Unified manager termination"
                )

        except Exception as e:
            logger.error(f"Failed to terminate bot {bot_id}: {e}")
            return False

    def get_bot_status(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get bot status using unified interface."""
        try:
            if bot_id not in self.active_bots:
                return None

            bot_info = self.active_bots[bot_id]
            manager = bot_info["manager"]

            # Get status from appropriate manager
            if isinstance(manager, GoogleMeetBotManager):
                manager_status = manager.get_bot_status(bot_id)
            else:
                manager_status = manager.get_bot_status(bot_id)

            # Combine with unified tracking info
            unified_status = {
                "bot_id": bot_id,
                "manager_type": bot_info["manager_type"],
                "bot_type": bot_info["bot_type"].value,
                "unified_status": bot_info["status"],
                "created_at": bot_info["created_at"],
                "last_update": bot_info["last_update"],
            }

            if manager_status:
                # Properly merge dictionaries
                for key, value in manager_status.items():
                    unified_status[key] = value

            return unified_status

        except Exception as e:
            logger.error(f"Failed to get bot status for {bot_id}: {e}")
            return None

    def get_all_bots_status(self) -> List[Dict[str, Any]]:
        """Get status of all bots using unified interface."""
        try:
            all_status = []

            for bot_id in self.active_bots.keys():
                status = self.get_bot_status(bot_id)
                if status:
                    all_status.append(status)

            return all_status

        except Exception as e:
            logger.error(f"Failed to get all bots status: {e}")
            return []

    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all managers."""
        try:
            # Get statistics from individual managers
            generic_stats = self.generic_manager.get_bot_stats()
            google_meet_stats = self.google_meet_manager.get_manager_statistics()

            # Unified statistics
            total_active = len(self.active_bots)
            by_type = {}
            by_status = {}

            for bot_info in self.active_bots.values():
                bot_type = bot_info["bot_type"].value
                status = bot_info["status"]

                by_type[bot_type] = by_type.get(bot_type, 0) + 1
                by_status[status] = by_status.get(status, 0) + 1

            return {
                "unified_manager": {
                    "initialized": self._initialized,
                    "total_active_bots": total_active,
                    "bots_by_type": by_type,
                    "bots_by_status": by_status,
                },
                "generic_manager": generic_stats,
                "google_meet_manager": google_meet_stats,
            }

        except Exception as e:
            logger.error(f"Failed to get manager statistics: {e}")
            return {"error": str(e)}

    def add_unified_callback(self, callback: Callable):
        """Add callback for unified events."""
        self.unified_callbacks.append(callback)

    def remove_unified_callback(self, callback: Callable):
        """Remove callback for unified events."""
        if callback in self.unified_callbacks:
            self.unified_callbacks.remove(callback)

    # Service client configuration
    def set_service_clients(self, **clients):
        """Set service clients for all managers."""
        self.generic_manager.set_service_clients(**clients)
        # Google Meet manager handles its own service integration

    # Health and monitoring
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all managers."""
        try:
            health_status = {
                "unified_manager": {
                    "status": "healthy" if self._initialized else "not_initialized",
                    "active_bots": len(self.active_bots),
                },
                "generic_manager": {
                    "status": "healthy" if self.generic_manager._running else "stopped"
                },
                "google_meet_manager": {
                    "status": "healthy"
                    if self.google_meet_manager.running
                    else "stopped"
                },
            }

            return health_status

        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Global unified bot manager instance
_unified_bot_manager: Optional[UnifiedBotManager] = None


async def get_unified_bot_manager() -> UnifiedBotManager:
    """Get the global unified bot manager instance."""
    global _unified_bot_manager

    if _unified_bot_manager is None:
        _unified_bot_manager = UnifiedBotManager()
        await _unified_bot_manager.initialize()

    return _unified_bot_manager


# Convenience functions
async def request_meeting_bot(meeting_request: UnifiedMeetingRequest) -> Optional[str]:
    """Request a bot for a meeting."""
    manager = await get_unified_bot_manager()
    return await manager.request_bot(meeting_request)


async def terminate_meeting_bot(bot_id: str) -> bool:
    """Terminate a meeting bot."""
    manager = await get_unified_bot_manager()
    return await manager.terminate_bot(bot_id)


async def get_meeting_bot_status(bot_id: str) -> Optional[Dict[str, Any]]:
    """Get status of a meeting bot."""
    manager = await get_unified_bot_manager()
    return manager.get_bot_status(bot_id)
