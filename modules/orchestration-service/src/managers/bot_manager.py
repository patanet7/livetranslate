"""
Bot Manager

Central bot lifecycle management for the orchestration service.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class BotStatus(Enum):
    """Bot status enumeration"""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECOVERING = "recovering"


class BotType(Enum):
    """Bot type enumeration"""

    GOOGLE_MEET = "google_meet"
    ZOOM = "zoom"
    TEAMS = "teams"
    GENERIC = "generic"


@dataclass
class MeetingRequest:
    """Meeting request data"""

    meeting_id: str
    meeting_title: str
    meeting_uri: str
    bot_type: BotType = BotType.GOOGLE_MEET
    target_languages: List[str] = field(default_factory=lambda: ["en"])
    enable_translation: bool = True
    enable_transcription: bool = True
    enable_virtual_webcam: bool = True
    audio_storage_enabled: bool = True
    cleanup_on_exit: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BotInstance:
    """Bot instance information"""

    bot_id: str
    meeting_request: MeetingRequest
    status: BotStatus
    created_at: float
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    last_activity: Optional[float] = None
    error_message: Optional[str] = None
    process_id: Optional[int] = None
    session_id: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "bot_id": self.bot_id,
            "meeting_id": self.meeting_request.meeting_id,
            "meeting_title": self.meeting_request.meeting_title,
            "meeting_uri": self.meeting_request.meeting_uri,
            "bot_type": self.meeting_request.bot_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_activity": self.last_activity,
            "error_message": self.error_message,
            "process_id": self.process_id,
            "session_id": self.session_id,
            "uptime": self._calculate_uptime(),
            "statistics": self.statistics,
        }

    def _calculate_uptime(self) -> Optional[float]:
        """Calculate bot uptime"""
        if not self.started_at:
            return None

        end_time = self.stopped_at if self.stopped_at else time.time()
        return end_time - self.started_at


@dataclass
class BotConfig:
    """Bot configuration"""

    max_concurrent_bots: int = 10
    bot_timeout: int = 3600
    audio_storage_path: str = "/tmp/audio"
    virtual_webcam_enabled: bool = True
    virtual_webcam_device: str = "/dev/video0"
    google_meet_credentials_path: str = ""
    cleanup_on_exit: bool = True
    recovery_attempts: int = 3
    recovery_delay: int = 60


class BotManager:
    """
    Central bot lifecycle management system
    """

    def __init__(self, config: BotConfig = None):
        self.config = config or BotConfig()

        # Bot instances
        self.bots: Dict[str, BotInstance] = {}
        self.bot_queue: List[MeetingRequest] = []

        # Event handlers
        self.status_change_handlers: List[Callable] = []
        self.lifecycle_handlers: List[Callable] = []

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.stats = {
            "total_bots_created": 0,
            "total_bots_completed": 0,
            "total_bots_failed": 0,
            "total_recovery_attempts": 0,
            "total_successful_recoveries": 0,
        }

        # Service clients (will be injected)
        self.audio_client = None
        self.translation_client = None
        self.database_client = None

    async def start(self):
        """Start bot manager"""
        if self._running:
            return

        self._running = True

        # Create storage directory
        storage_path = Path(self.config.audio_storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)

        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._queue_processor_task = asyncio.create_task(self._queue_processor_loop())

        logger.info("Bot manager started")

    async def stop(self):
        """Stop bot manager"""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._queue_processor_task:
            self._queue_processor_task.cancel()

        # Stop all bots
        await self._stop_all_bots()

        logger.info("Bot manager stopped")

    async def request_bot(self, meeting_request: MeetingRequest) -> str:
        """
        Request a new bot instance

        Returns:
            Bot ID
        """
        # Check if we're at capacity
        active_bots = [
            b
            for b in self.bots.values()
            if b.status in [BotStatus.RUNNING, BotStatus.STARTING]
        ]

        if len(active_bots) >= self.config.max_concurrent_bots:
            # Add to queue
            self.bot_queue.append(meeting_request)
            logger.info(f"Bot request queued: {meeting_request.meeting_id}")
            return "queued"

        # Create bot instance
        bot_id = str(uuid4())
        bot_instance = BotInstance(
            bot_id=bot_id,
            meeting_request=meeting_request,
            status=BotStatus.PENDING,
            created_at=time.time(),
        )

        self.bots[bot_id] = bot_instance
        self.stats["total_bots_created"] += 1

        # Start bot
        await self._start_bot(bot_id)

        logger.info(f"Bot requested: {bot_id} for meeting {meeting_request.meeting_id}")
        return bot_id

    async def _start_bot(self, bot_id: str):
        """Start a bot instance"""
        if bot_id not in self.bots:
            logger.error(f"Bot not found: {bot_id}")
            return

        bot_instance = self.bots[bot_id]

        try:
            # Update status
            await self._update_bot_status(bot_id, BotStatus.STARTING)

            # Initialize bot components
            await self._initialize_bot_components(bot_id)

            # Start bot processes
            await self._start_bot_processes(bot_id)

            # Update status
            bot_instance.started_at = time.time()
            bot_instance.last_activity = time.time()
            await self._update_bot_status(bot_id, BotStatus.RUNNING)

            logger.info(f"Bot started successfully: {bot_id}")

        except Exception as e:
            logger.error(f"Failed to start bot {bot_id}: {e}")
            bot_instance.error_message = str(e)
            await self._update_bot_status(bot_id, BotStatus.ERROR)

    async def _initialize_bot_components(self, bot_id: str):
        """Initialize bot components"""
        bot_instance = self.bots[bot_id]
        meeting_request = bot_instance.meeting_request

        # Create database session
        if self.database_client:
            session_id = await self.database_client.create_session(
                {
                    "bot_id": bot_id,
                    "meeting_id": meeting_request.meeting_id,
                    "meeting_title": meeting_request.meeting_title,
                    "meeting_uri": meeting_request.meeting_uri,
                    "bot_type": meeting_request.bot_type.value,
                    "target_languages": meeting_request.target_languages,
                    "metadata": meeting_request.metadata,
                }
            )
            bot_instance.session_id = session_id

        # Initialize audio service session
        if self.audio_client and meeting_request.enable_transcription:
            await self.audio_client.create_session(
                {
                    "bot_id": bot_id,
                    "session_id": bot_instance.session_id,
                    "enable_speaker_diarization": True,
                    "enable_vad": True,
                }
            )

        # Initialize translation service session
        if self.translation_client and meeting_request.enable_translation:
            await self.translation_client.create_session(
                {
                    "bot_id": bot_id,
                    "session_id": bot_instance.session_id,
                    "target_languages": meeting_request.target_languages,
                }
            )

    async def _start_bot_processes(self, bot_id: str):
        """Start bot processes"""
        bot_instance = self.bots[bot_id]
        meeting_request = bot_instance.meeting_request

        # This would start the actual bot processes
        # For now, simulate process startup
        await asyncio.sleep(1)

        # Update statistics
        bot_instance.statistics = {
            "audio_files_processed": 0,
            "transcripts_generated": 0,
            "translations_completed": 0,
            "errors_encountered": 0,
            "last_update": time.time(),
        }

        logger.info(f"Bot processes started for: {bot_id}")

    async def terminate_bot(self, bot_id: str, reason: str = "Manual termination"):
        """Terminate a bot instance"""
        if bot_id not in self.bots:
            logger.error(f"Bot not found: {bot_id}")
            return

        bot_instance = self.bots[bot_id]

        try:
            # Update status
            await self._update_bot_status(bot_id, BotStatus.STOPPING)

            # Stop bot processes
            await self._stop_bot_processes(bot_id)

            # Cleanup resources
            await self._cleanup_bot_resources(bot_id)

            # Update status
            bot_instance.stopped_at = time.time()
            await self._update_bot_status(bot_id, BotStatus.STOPPED)

            self.stats["total_bots_completed"] += 1

            logger.info(f"Bot terminated: {bot_id} - {reason}")

        except Exception as e:
            logger.error(f"Failed to terminate bot {bot_id}: {e}")
            bot_instance.error_message = str(e)
            await self._update_bot_status(bot_id, BotStatus.ERROR)

    async def _stop_bot_processes(self, bot_id: str):
        """Stop bot processes"""
        # This would stop the actual bot processes
        await asyncio.sleep(0.5)

        logger.info(f"Bot processes stopped for: {bot_id}")

    async def _cleanup_bot_resources(self, bot_id: str):
        """Cleanup bot resources"""
        bot_instance = self.bots[bot_id]
        meeting_request = bot_instance.meeting_request

        # Close service sessions
        if self.audio_client and bot_instance.session_id:
            await self.audio_client.close_session(bot_instance.session_id)

        if self.translation_client and bot_instance.session_id:
            await self.translation_client.close_session(bot_instance.session_id)

        # Cleanup files if requested
        if meeting_request.cleanup_on_exit:
            await self._cleanup_bot_files(bot_id)

        # Close database session
        if self.database_client and bot_instance.session_id:
            await self.database_client.close_session(bot_instance.session_id)

    async def _cleanup_bot_files(self, bot_id: str):
        """Cleanup bot files"""
        try:
            storage_path = Path(self.config.audio_storage_path) / bot_id
            if storage_path.exists():
                import shutil

                shutil.rmtree(storage_path)
                logger.info(f"Cleaned up files for bot: {bot_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup files for bot {bot_id}: {e}")

    async def _update_bot_status(self, bot_id: str, new_status: BotStatus):
        """Update bot status and notify handlers"""
        if bot_id not in self.bots:
            return

        bot_instance = self.bots[bot_id]
        old_status = bot_instance.status
        bot_instance.status = new_status

        # Notify status change handlers
        for handler in self.status_change_handlers:
            try:
                await handler(bot_id, old_status, new_status)
            except Exception as e:
                logger.error(f"Status change handler error: {e}")

        # Notify lifecycle handlers
        for handler in self.lifecycle_handlers:
            try:
                await handler(bot_id, bot_instance.to_dict())
            except Exception as e:
                logger.error(f"Lifecycle handler error: {e}")

    async def _monitor_loop(self):
        """Monitor bot health and activity"""
        while self._running:
            try:
                current_time = time.time()

                # Check bot timeouts
                for bot_id, bot_instance in list(self.bots.items()):
                    if bot_instance.status == BotStatus.RUNNING:
                        # Check timeout
                        if (
                            bot_instance.started_at
                            and current_time - bot_instance.started_at
                            > self.config.bot_timeout
                        ):
                            await self.terminate_bot(bot_id, "Timeout")

                        # Check activity
                        if (
                            bot_instance.last_activity
                            and current_time - bot_instance.last_activity > 300
                        ):  # 5 minutes
                            logger.warning(f"Bot {bot_id} inactive for 5+ minutes")
                            await self._attempt_bot_recovery(bot_id)

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

    async def _cleanup_loop(self):
        """Cleanup old bot instances"""
        while self._running:
            try:
                current_time = time.time()
                cleanup_threshold = 3600  # 1 hour

                # Remove old stopped/error bots
                bots_to_remove = []
                for bot_id, bot_instance in self.bots.items():
                    if bot_instance.status in [BotStatus.STOPPED, BotStatus.ERROR]:
                        if (
                            bot_instance.stopped_at
                            and current_time - bot_instance.stopped_at
                            > cleanup_threshold
                        ):
                            bots_to_remove.append(bot_id)

                for bot_id in bots_to_remove:
                    del self.bots[bot_id]

                if bots_to_remove:
                    logger.info(f"Cleaned up {len(bots_to_remove)} old bot instances")

                await asyncio.sleep(600)  # Clean up every 10 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)

    async def _queue_processor_loop(self):
        """Process bot queue when capacity is available"""
        while self._running:
            try:
                # Check if we have queue items and capacity
                if self.bot_queue:
                    active_bots = [
                        b
                        for b in self.bots.values()
                        if b.status in [BotStatus.RUNNING, BotStatus.STARTING]
                    ]

                    if len(active_bots) < self.config.max_concurrent_bots:
                        # Process next item in queue
                        meeting_request = self.bot_queue.pop(0)
                        await self.request_bot(meeting_request)
                        logger.info(
                            f"Processed queued bot request: {meeting_request.meeting_id}"
                        )

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
                await asyncio.sleep(10)

    async def _attempt_bot_recovery(self, bot_id: str):
        """Attempt to recover a bot"""
        if bot_id not in self.bots:
            return

        bot_instance = self.bots[bot_id]

        try:
            await self._update_bot_status(bot_id, BotStatus.RECOVERING)
            self.stats["total_recovery_attempts"] += 1

            # Attempt recovery actions
            success = await self._recover_bot(bot_id)

            if success:
                bot_instance.last_activity = time.time()
                await self._update_bot_status(bot_id, BotStatus.RUNNING)
                self.stats["total_successful_recoveries"] += 1
                logger.info(f"Successfully recovered bot: {bot_id}")
            else:
                bot_instance.error_message = "Recovery failed"
                await self._update_bot_status(bot_id, BotStatus.ERROR)
                logger.error(f"Failed to recover bot: {bot_id}")

        except Exception as e:
            bot_instance.error_message = f"Recovery error: {str(e)}"
            await self._update_bot_status(bot_id, BotStatus.ERROR)
            logger.error(f"Bot recovery error for {bot_id}: {e}")

    async def _recover_bot(self, bot_id: str) -> bool:
        """Recover a bot instance"""
        # This would implement actual recovery logic
        # For now, simulate recovery
        await asyncio.sleep(1)
        return True

    async def _stop_all_bots(self):
        """Stop all bot instances"""
        for bot_id in list(self.bots.keys()):
            if self.bots[bot_id].status in [BotStatus.RUNNING, BotStatus.STARTING]:
                await self.terminate_bot(bot_id, "Manager shutdown")

    def add_status_change_handler(self, handler: Callable):
        """Add status change handler"""
        self.status_change_handlers.append(handler)

    def add_lifecycle_handler(self, handler: Callable):
        """Add lifecycle handler"""
        self.lifecycle_handlers.append(handler)

    def get_bot_status(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get bot status"""
        if bot_id not in self.bots:
            return None

        return self.bots[bot_id].to_dict()

    def get_all_bots(self) -> List[Dict[str, Any]]:
        """Get all bot information"""
        return [bot.to_dict() for bot in self.bots.values()]

    def get_active_bots(self) -> List[Dict[str, Any]]:
        """Get active bot information"""
        return [
            bot.to_dict()
            for bot in self.bots.values()
            if bot.status in [BotStatus.RUNNING, BotStatus.STARTING]
        ]

    def get_bot_stats(self) -> Dict[str, Any]:
        """Get bot manager statistics"""
        active_bots = len(
            [
                b
                for b in self.bots.values()
                if b.status in [BotStatus.RUNNING, BotStatus.STARTING]
            ]
        )

        return {
            **self.stats,
            "active_bots": active_bots,
            "queued_requests": len(self.bot_queue),
            "total_bots": len(self.bots),
            "capacity_utilization": (active_bots / self.config.max_concurrent_bots)
            * 100,
            "success_rate": (
                self.stats["total_bots_completed"]
                / max(1, self.stats["total_bots_created"])
            )
            * 100,
            "recovery_rate": (
                self.stats["total_successful_recoveries"]
                / max(1, self.stats["total_recovery_attempts"])
            )
            * 100,
        }

    def set_service_clients(
        self, audio_client=None, translation_client=None, database_client=None
    ):
        """Set service clients"""
        self.audio_client = audio_client
        self.translation_client = translation_client
        self.database_client = database_client
