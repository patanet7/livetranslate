#!/usr/bin/env python3
"""
Google Meet Bot Manager

Central "brain" for spawning, managing, and coordinating Google Meet bots.
Provides intelligent lifecycle management, health monitoring, and meeting analytics.

Features:
- Bot instance lifecycle management (spawn, monitor, recover, cleanup)
- Google Meet API integration for official meeting coordination
- Service coordination with whisper/translation services
- Real-time meeting intelligence and analytics
- Automated bot deployment and scaling
"""

import os
import sys
import time
import logging
import asyncio
import threading
import uuid
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import httpx
from collections import defaultdict, deque

# Import Google Meet API client
from .google_meet_client import (
    GoogleMeetClient,
    BotManagerIntegration,
    create_google_meet_client,
)

# Import database manager
from ..database.bot_session_manager import (
    BotSessionDatabaseManager,
    create_bot_session_manager,
)

# Import enhanced lifecycle manager
from .bot_lifecycle_manager import BotLifecycleManager, create_lifecycle_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotStatus(Enum):
    """Bot instance status enumeration."""

    SPAWNING = "spawning"
    ACTIVE = "active"
    PAUSED = "paused"
    ENDING = "ending"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class MeetingRequest:
    """Request to join or create a meeting."""

    meeting_id: str
    meeting_title: Optional[str] = None
    organizer_email: Optional[str] = None
    scheduled_start: Optional[datetime] = None
    target_languages: List[str] = None
    recording_enabled: bool = False
    auto_translation: bool = True
    priority: str = "normal"  # 'high', 'normal', 'low'
    requester_id: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.target_languages is None:
            self.target_languages = ["en", "es", "fr", "de", "zh"]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BotInstance:
    """Represents an active Google Meet bot instance."""

    bot_id: str
    meeting_request: MeetingRequest
    status: BotStatus
    created_at: datetime
    google_meet_space_id: Optional[str] = None
    conference_record_id: Optional[str] = None
    session_id: Optional[str] = None
    participant_count: int = 0
    health_score: float = 1.0
    error_count: int = 0
    last_activity: Optional[datetime] = None
    performance_stats: Dict[str, Any] = None

    def __post_init__(self):
        if self.performance_stats is None:
            self.performance_stats = {
                "messages_processed": 0,
                "transcriptions_count": 0,
                "translations_count": 0,
                "average_latency": 0.0,
                "quality_score": 1.0,
                "uptime_seconds": 0,
            }


@dataclass
class MeetingAnalytics:
    """Real-time meeting analytics and insights."""

    meeting_id: str
    session_id: str
    participant_stats: Dict[str, Any]
    speaking_time_distribution: Dict[str, float]
    engagement_metrics: Dict[str, float]
    translation_quality: Dict[str, float]
    meeting_quality_score: float
    insights: List[str]
    recommendations: List[str]


class BotHealthMonitor:
    """Monitors bot health and performance."""

    def __init__(self):
        self.health_thresholds = {
            "error_rate_max": 0.05,  # 5% max error rate
            "response_time_max": 2.0,  # 2 seconds max response time
            "min_activity_interval": 300.0,  # 5 minutes max inactivity
            "memory_usage_max": 0.8,  # 80% max memory usage
        }
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

    def check_bot_health(self, bot: BotInstance) -> Tuple[float, List[str]]:
        """
        Check bot health and return health score with issues.

        Args:
            bot: Bot instance to check

        Returns:
            Tuple of (health_score, list_of_issues)
        """
        issues = []
        health_factors = []

        # Check error rate
        error_rate = bot.error_count / max(
            1, bot.performance_stats["messages_processed"]
        )
        if error_rate > self.health_thresholds["error_rate_max"]:
            issues.append(f"High error rate: {error_rate:.2%}")
            health_factors.append(0.3)
        else:
            health_factors.append(1.0)

        # Check response time
        avg_latency = bot.performance_stats.get("average_latency", 0.0)
        if avg_latency > self.health_thresholds["response_time_max"]:
            issues.append(f"High latency: {avg_latency:.2f}s")
            health_factors.append(0.5)
        else:
            health_factors.append(1.0)

        # Check activity
        if bot.last_activity:
            inactivity = (datetime.now() - bot.last_activity).total_seconds()
            if inactivity > self.health_thresholds["min_activity_interval"]:
                issues.append(f"Inactive for {inactivity:.0f}s")
                health_factors.append(0.6)
            else:
                health_factors.append(1.0)
        else:
            health_factors.append(0.8)  # No activity data

        # Check status
        if bot.status == BotStatus.ERROR:
            issues.append("Bot in error state")
            health_factors.append(0.1)
        elif bot.status == BotStatus.ACTIVE:
            health_factors.append(1.0)
        else:
            health_factors.append(0.7)

        # Calculate overall health score
        health_score = sum(health_factors) / len(health_factors)

        # Store in history
        self.health_history[bot.bot_id].append(
            {"timestamp": time.time(), "score": health_score, "issues": issues.copy()}
        )

        return health_score, issues

    def get_health_trend(self, bot_id: str) -> Dict[str, Any]:
        """Get health trend for a bot."""
        history = list(self.health_history[bot_id])
        if not history:
            return {"trend": "unknown", "average_score": 0.0}

        scores = [h["score"] for h in history]
        avg_score = sum(scores) / len(scores)

        if len(scores) >= 2:
            recent_avg = sum(scores[-5:]) / min(5, len(scores))
            older_avg = sum(scores[:-5]) / max(1, len(scores) - 5)

            if recent_avg > older_avg + 0.1:
                trend = "improving"
            elif recent_avg < older_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return {
            "trend": trend,
            "average_score": avg_score,
            "recent_score": scores[-1] if scores else 0.0,
            "history_length": len(history),
        }


class GoogleMeetBotManager:
    """
    Central manager for Google Meet bot lifecycle and coordination.

    The "brain" of the meeting intelligence system.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._load_default_config()

        # Bot management
        self.active_bots: Dict[str, BotInstance] = {}
        self.bot_queue: deque = deque()  # Pending bot spawn requests
        self.health_monitor = BotHealthMonitor()

        # Service integration
        self.whisper_service_url = self.config.get(
            "whisper_service_url", "http://localhost:5001"
        )
        self.translation_service_url = self.config.get(
            "translation_service_url", "http://localhost:5003"
        )

        # Google Meet API integration
        self.google_meet_client = None
        self.bot_manager_integration = None

        # Database integration
        self.database_manager = None

        # Enhanced lifecycle management
        self.lifecycle_manager = None

        # Performance tracking
        self.total_bots_spawned = 0
        self.successful_meetings = 0
        self.failed_meetings = 0
        self.average_meeting_duration = 0.0

        # Background tasks
        self.running = False
        self.management_thread = None
        self.health_check_thread = None

        # Event callbacks
        self.on_bot_spawned = None
        self.on_bot_terminated = None
        self.on_meeting_started = None
        self.on_meeting_ended = None
        self.on_bot_error = None

        # Thread safety
        self.lock = threading.RLock()

        logger.info(f"GoogleMeetBotManager initialized")
        logger.info(
            f"  Max concurrent bots: {self.config.get('max_concurrent_bots', 10)}"
        )
        logger.info(
            f"  Service URLs: whisper={self.whisper_service_url}, translation={self.translation_service_url}"
        )

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "max_concurrent_bots": 10,
            "bot_spawn_timeout": 60.0,
            "health_check_interval": 30.0,
            "auto_recovery_enabled": True,
            "meeting_timeout": 14400.0,  # 4 hours
            "whisper_service_url": "http://localhost:5001",
            "translation_service_url": "http://localhost:5003",
            "google_meet_credentials_path": None,
            "default_target_languages": ["en", "es", "fr", "de", "zh"],
            # Database configuration
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "livetranslate",
                "username": "postgres",
                "password": "livetranslate",
            },
            "audio_storage_path": "/data/livetranslate/audio",
        }

    async def start(self) -> bool:
        """Start the bot manager."""
        if self.running:
            logger.warning("Bot manager already running")
            return False

        try:
            self.running = True

            # Initialize Google Meet API client if credentials are provided
            credentials_path = self.config.get("google_meet_credentials_path")
            if credentials_path:
                try:
                    self.google_meet_client = create_google_meet_client(
                        credentials_path=credentials_path,
                        application_name="LiveTranslate Bot Manager",
                    )

                    success = await self.google_meet_client.initialize()
                    if success:
                        self.bot_manager_integration = BotManagerIntegration(
                            self.google_meet_client, self
                        )
                        logger.info("Google Meet API integration initialized")
                    else:
                        logger.warning(
                            "Google Meet API initialization failed - continuing without API access"
                        )
                        self.google_meet_client = None

                except Exception as e:
                    logger.warning(
                        f"Google Meet API setup failed: {e} - continuing without API access"
                    )
                    self.google_meet_client = None
            else:
                logger.info(
                    "No Google Meet credentials provided - API features disabled"
                )

            # Initialize database manager
            try:
                database_config = self.config.get("database", {})
                audio_storage_path = self.config.get(
                    "audio_storage_path", "/tmp/livetranslate/audio"
                )

                self.database_manager = create_bot_session_manager(
                    database_config, audio_storage_path
                )
                success = await self.database_manager.initialize()

                if success:
                    logger.info("Bot session database manager initialized")
                else:
                    logger.warning(
                        "Database manager initialization failed - continuing without database features"
                    )
                    self.database_manager = None

            except Exception as e:
                logger.warning(
                    f"Database setup failed: {e} - continuing without database features"
                )
                self.database_manager = None

            # Initialize enhanced lifecycle manager
            try:
                self.lifecycle_manager = create_lifecycle_manager(self)
                await self.lifecycle_manager.start_monitoring()
                logger.info(
                    "Enhanced lifecycle manager initialized and monitoring started"
                )
            except Exception as e:
                logger.warning(
                    f"Lifecycle manager setup failed: {e} - continuing with basic lifecycle management"
                )
                self.lifecycle_manager = None

            # Start background threads
            self.management_thread = threading.Thread(
                target=self._management_loop, daemon=True
            )
            self.management_thread.start()

            self.health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True
            )
            self.health_check_thread.start()

            logger.info("Google Meet Bot Manager started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start bot manager: {e}")
            self.running = False
            return False

    async def stop(self) -> bool:
        """Stop the bot manager and all bots."""
        logger.info("Stopping Google Meet Bot Manager...")

        self.running = False

        # Stop lifecycle manager
        if self.lifecycle_manager:
            try:
                await self.lifecycle_manager.stop_monitoring()
            except Exception as e:
                logger.error(f"Error stopping lifecycle manager: {e}")

        # Terminate all active bots
        termination_tasks = []
        with self.lock:
            for bot_id in list(self.active_bots.keys()):
                termination_tasks.append(self.terminate_bot(bot_id))

        # Wait for all bots to terminate
        if termination_tasks:
            await asyncio.gather(*termination_tasks, return_exceptions=True)

        # Wait for background threads
        if self.management_thread and self.management_thread.is_alive():
            self.management_thread.join(timeout=5.0)

        if self.health_check_thread and self.health_check_thread.is_alive():
            self.health_check_thread.join(timeout=5.0)

        logger.info("Google Meet Bot Manager stopped")
        return True

    async def request_bot(self, meeting_request: MeetingRequest) -> str:
        """
        Request a new bot for a meeting.

        Args:
            meeting_request: Meeting details and configuration

        Returns:
            Bot ID if successful, None if failed
        """
        try:
            # Validate request
            if not meeting_request.meeting_id:
                raise ValueError("Meeting ID is required")

            # Check if bot already exists for this meeting
            with self.lock:
                for bot in self.active_bots.values():
                    if bot.meeting_request.meeting_id == meeting_request.meeting_id:
                        logger.warning(
                            f"Bot already exists for meeting: {meeting_request.meeting_id}"
                        )
                        return bot.bot_id

                # Check capacity
                if len(self.active_bots) >= self.config["max_concurrent_bots"]:
                    logger.warning("Maximum concurrent bots reached, queuing request")
                    self.bot_queue.append(meeting_request)
                    return None

            # Generate bot ID
            bot_id = f"bot_{uuid.uuid4().hex[:8]}_{int(time.time())}"

            # Create bot instance
            bot = BotInstance(
                bot_id=bot_id,
                meeting_request=meeting_request,
                status=BotStatus.SPAWNING,
                created_at=datetime.now(),
            )

            # Add to active bots
            with self.lock:
                self.active_bots[bot_id] = bot
                self.total_bots_spawned += 1

            # Create database session record
            if self.database_manager:
                try:
                    session_data = {
                        "session_id": bot_id,  # Use bot_id as session_id for simplicity
                        "bot_id": bot_id,
                        "meeting_id": meeting_request.meeting_id,
                        "meeting_title": meeting_request.meeting_title,
                        "status": "spawning",
                        "start_time": bot.created_at,
                        "target_languages": meeting_request.target_languages,
                        "session_metadata": {
                            "meeting_request": asdict(meeting_request),
                            "bot_instance": {
                                "created_at": bot.created_at.isoformat(),
                                "priority": meeting_request.priority,
                                "requester_id": meeting_request.requester_id,
                            },
                        },
                        "performance_stats": bot.performance_stats,
                    }

                    db_session_id = await self.database_manager.create_bot_session(
                        session_data
                    )
                    if db_session_id:
                        logger.debug(f"Created database session for bot: {bot_id}")
                    else:
                        logger.warning(
                            f"Failed to create database session for bot: {bot_id}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Database session creation failed for bot {bot_id}: {e}"
                    )

            # Spawn bot asynchronously
            asyncio.create_task(self._spawn_bot(bot))

            logger.info(
                f"Bot spawn requested: {bot_id} for meeting {meeting_request.meeting_id}"
            )
            return bot_id

        except Exception as e:
            logger.error(f"Failed to request bot: {e}")
            return None

    async def terminate_bot(self, bot_id: str) -> bool:
        """
        Terminate a specific bot.

        Args:
            bot_id: ID of bot to terminate

        Returns:
            True if terminated successfully
        """
        try:
            with self.lock:
                if bot_id not in self.active_bots:
                    logger.warning(f"Bot not found: {bot_id}")
                    return False

                bot = self.active_bots[bot_id]

                # Update status
                bot.status = BotStatus.ENDING

            logger.info(f"Terminating bot: {bot_id}")

            # Perform cleanup
            success = await self._cleanup_bot(bot)

            # Remove from active bots
            with self.lock:
                if bot_id in self.active_bots:
                    bot = self.active_bots.pop(bot_id)
                    bot.status = BotStatus.TERMINATED

                    # Update database session
                    if self.database_manager:
                        try:
                            asyncio.create_task(
                                self.database_manager.update_bot_session(
                                    bot_id,
                                    {
                                        "status": "ended",
                                        "end_time": datetime.now(),
                                        "performance_stats": bot.performance_stats,
                                    },
                                )
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to update database session end for bot {bot_id}: {e}"
                            )

            # Callback
            if self.on_bot_terminated:
                self.on_bot_terminated(bot)

            # Process queued requests
            await self._process_bot_queue()

            logger.info(f"Bot terminated: {bot_id}")
            return success

        except Exception as e:
            logger.error(f"Failed to terminate bot {bot_id}: {e}")
            return False

    async def _spawn_bot(self, bot: BotInstance):
        """Spawn a new bot instance."""
        try:
            logger.info(f"Spawning bot: {bot.bot_id}")

            # Step 1: Google Meet integration
            success = await self._initialize_google_meet_integration(bot)
            if not success:
                raise Exception("Failed to initialize Google Meet integration")

            # Step 2: Service session creation
            success = await self._create_service_sessions(bot)
            if not success:
                raise Exception("Failed to create service sessions")

            # Step 3: Start bot processes
            success = await self._start_bot_processes(bot)
            if not success:
                raise Exception("Failed to start bot processes")

            # Update status
            with self.lock:
                bot.status = BotStatus.ACTIVE
                bot.last_activity = datetime.now()

            # Update database session
            if self.database_manager:
                try:
                    await self.database_manager.update_bot_session(
                        bot.bot_id,
                        {
                            "status": "active",
                            "google_meet_space_id": bot.google_meet_space_id,
                            "conference_record_id": bot.conference_record_id,
                            "session_id": bot.session_id,
                            "performance_stats": bot.performance_stats,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to update database session for bot {bot.bot_id}: {e}"
                    )

            # Callback
            if self.on_bot_spawned:
                self.on_bot_spawned(bot)

            if self.on_meeting_started:
                self.on_meeting_started(bot.meeting_request.meeting_id, bot.bot_id)

            logger.info(f"Bot spawned successfully: {bot.bot_id}")

        except Exception as e:
            logger.error(f"Failed to spawn bot {bot.bot_id}: {e}")

            with self.lock:
                bot.status = BotStatus.ERROR
                bot.error_count += 1

            if self.on_bot_error:
                self.on_bot_error(bot, str(e))

            # Schedule cleanup
            asyncio.create_task(self._cleanup_bot(bot))

    async def _initialize_google_meet_integration(self, bot: BotInstance) -> bool:
        """Initialize Google Meet API integration for the bot."""
        try:
            if not self.google_meet_client or not self.bot_manager_integration:
                logger.warning(
                    f"Google Meet API not available - using fallback mode for bot {bot.bot_id}"
                )
                # Fallback mode - generate mock IDs for testing
                bot.google_meet_space_id = f"spaces/mock_{uuid.uuid4().hex[:8]}"
                bot.conference_record_id = (
                    f"conferenceRecords/mock_{uuid.uuid4().hex[:8]}"
                )
                return True

            # Check if this is a request to join an existing meeting
            meeting_uri = (
                bot.meeting_request.metadata.get("meeting_uri")
                if bot.meeting_request.metadata
                else None
            )

            if meeting_uri:
                # Join existing meeting
                meeting_info = await self.bot_manager_integration.join_external_meeting(
                    meeting_uri
                )
                if meeting_info:
                    bot.google_meet_space_id = meeting_info.get("space", {}).get(
                        "name", f"external_{uuid.uuid4().hex[:8]}"
                    )
                    bot.conference_record_id = (
                        f"conferenceRecords/external_{uuid.uuid4().hex[:8]}"
                    )

                    # Store additional meeting information
                    if not bot.meeting_request.metadata:
                        bot.meeting_request.metadata = {}
                    bot.meeting_request.metadata.update(
                        {
                            "join_method": meeting_info.get("join_method"),
                            "api_access": meeting_info.get("api_access"),
                            "meeting_code": meeting_info.get("meeting_code"),
                        }
                    )

                    logger.info(
                        f"Bot {bot.bot_id} configured for external meeting: {meeting_uri}"
                    )
                else:
                    logger.error(
                        f"Failed to process external meeting URI: {meeting_uri}"
                    )
                    return False
            else:
                # Create new meeting space
                meeting_result = await self.bot_manager_integration.create_bot_meeting(
                    bot.meeting_request
                )
                if meeting_result:
                    space = meeting_result["space"]
                    meeting_info = meeting_result["meeting_info"]

                    bot.google_meet_space_id = space.name
                    bot.conference_record_id = (
                        f"conferenceRecords/pending_{uuid.uuid4().hex[:8]}"
                    )

                    # Update meeting request with created meeting info
                    bot.meeting_request.meeting_id = meeting_info["meeting_id"]
                    if not bot.meeting_request.metadata:
                        bot.meeting_request.metadata = {}
                    bot.meeting_request.metadata.update(
                        {
                            "meeting_uri": meeting_info["meeting_uri"],
                            "space_name": meeting_info["space_name"],
                            "created_by_api": True,
                        }
                    )

                    logger.info(
                        f"Created new meeting for bot {bot.bot_id}: {meeting_info['meeting_uri']}"
                    )
                else:
                    logger.error(f"Failed to create meeting space for bot {bot.bot_id}")
                    return False

            logger.debug(f"Google Meet integration initialized for bot {bot.bot_id}")
            return True

        except Exception as e:
            logger.error(f"Google Meet integration failed for bot {bot.bot_id}: {e}")
            return False

    async def _create_service_sessions(self, bot: BotInstance) -> bool:
        """Create sessions with whisper and translation services."""
        try:
            # Generate session ID
            bot.session_id = f"session_{bot.bot_id}_{int(time.time())}"

            # Create whisper service session
            whisper_success = await self._create_whisper_session(bot)
            if not whisper_success:
                return False

            # Create translation service session
            translation_success = await self._create_translation_session(bot)
            if not translation_success:
                return False

            logger.debug(f"Service sessions created for bot {bot.bot_id}")
            return True

        except Exception as e:
            logger.error(f"Service session creation failed for bot {bot.bot_id}: {e}")
            return False

    async def _create_whisper_session(self, bot: BotInstance) -> bool:
        """Create session with whisper service."""
        try:
            session_data = {
                "session_id": bot.session_id,
                "meeting_info": {
                    "meeting_id": bot.meeting_request.meeting_id,
                    "meeting_title": bot.meeting_request.meeting_title,
                    "organizer_email": bot.meeting_request.organizer_email,
                },
                "bot_mode": True,
                "audio_config": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "format": "float32",
                },
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.whisper_service_url}/api/sessions/create",
                    json=session_data,
                    timeout=10.0,
                )

                return response.status_code == 200

        except Exception as e:
            logger.error(f"Whisper session creation failed: {e}")
            return False

    async def _create_translation_session(self, bot: BotInstance) -> bool:
        """Create session with translation service."""
        try:
            session_data = {
                "session_id": bot.session_id,
                "target_languages": bot.meeting_request.target_languages,
                "source_language": "auto",
                "context_mode": "meeting",
                "bot_mode": True,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.translation_service_url}/api/sessions/create",
                    json=session_data,
                    timeout=10.0,
                )

                return response.status_code == 200

        except Exception as e:
            logger.error(f"Translation session creation failed: {e}")
            return False

    async def _start_bot_processes(self, bot: BotInstance) -> bool:
        """Start the actual bot processes."""
        try:
            # Initialize performance tracking
            bot.performance_stats.update(
                {
                    "start_time": time.time(),
                    "messages_processed": 0,
                    "transcriptions_count": 0,
                    "translations_count": 0,
                    "participants_seen": 0,
                }
            )

            # Start Google Meet API monitoring if available
            if (
                self.google_meet_client
                and self.bot_manager_integration
                and bot.google_meet_space_id
                and not bot.google_meet_space_id.startswith("mock_")
            ):
                try:
                    success = await self.bot_manager_integration.monitor_bot_meeting(
                        bot.google_meet_space_id, bot.bot_id
                    )
                    if success:
                        logger.info(
                            f"Started Google Meet monitoring for bot {bot.bot_id}"
                        )
                        bot.performance_stats["google_meet_monitoring"] = True
                    else:
                        logger.warning(
                            f"Failed to start Google Meet monitoring for bot {bot.bot_id}"
                        )
                        bot.performance_stats["google_meet_monitoring"] = False
                except Exception as e:
                    logger.warning(
                        f"Google Meet monitoring setup failed for bot {bot.bot_id}: {e}"
                    )
                    bot.performance_stats["google_meet_monitoring"] = False
            else:
                bot.performance_stats["google_meet_monitoring"] = False

            # TODO: Start actual bot components integration
            # This would include:
            # - Google Meet bot audio capture
            # - Caption processing integration
            # - Real-time transcription/translation pipeline

            logger.debug(f"Bot processes started for {bot.bot_id}")
            return True

        except Exception as e:
            logger.error(f"Bot process start failed for {bot.bot_id}: {e}")
            return False

    async def _cleanup_bot(self, bot: BotInstance) -> bool:
        """Clean up bot resources."""
        try:
            logger.info(f"Cleaning up bot: {bot.bot_id}")

            # Close service sessions
            await self._close_service_sessions(bot)

            # Update statistics
            with self.lock:
                if bot.status != BotStatus.ERROR:
                    self.successful_meetings += 1
                else:
                    self.failed_meetings += 1

            # Calculate meeting duration
            if bot.performance_stats.get("start_time"):
                duration = time.time() - bot.performance_stats["start_time"]
                bot.performance_stats["duration"] = duration

                # Update average
                total_meetings = self.successful_meetings + self.failed_meetings
                if total_meetings > 0:
                    self.average_meeting_duration = (
                        self.average_meeting_duration * (total_meetings - 1) + duration
                    ) / total_meetings

            # Callback
            if self.on_meeting_ended:
                self.on_meeting_ended(
                    bot.meeting_request.meeting_id, bot.bot_id, bot.performance_stats
                )

            return True

        except Exception as e:
            logger.error(f"Bot cleanup failed for {bot.bot_id}: {e}")
            return False

    async def _close_service_sessions(self, bot: BotInstance):
        """Close sessions with services."""
        if not bot.session_id:
            return

        # Close whisper session
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.whisper_service_url}/api/sessions/{bot.session_id}/close",
                    timeout=10.0,
                )
        except Exception as e:
            logger.warning(f"Failed to close whisper session: {e}")

        # Close translation session
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.translation_service_url}/api/sessions/{bot.session_id}/close",
                    timeout=10.0,
                )
        except Exception as e:
            logger.warning(f"Failed to close translation session: {e}")

    async def _process_bot_queue(self):
        """Process queued bot requests."""
        try:
            with self.lock:
                while (
                    self.bot_queue
                    and len(self.active_bots) < self.config["max_concurrent_bots"]
                ):
                    meeting_request = self.bot_queue.popleft()

            if "meeting_request" in locals():
                await self.request_bot(meeting_request)

        except Exception as e:
            logger.error(f"Error processing bot queue: {e}")

    def _management_loop(self):
        """Background management loop."""
        logger.info("Bot management loop started")

        while self.running:
            try:
                # Check for timeouts and cleanup
                self._check_bot_timeouts()

                # Auto-recovery if enabled
                if self.config.get("auto_recovery_enabled", True):
                    asyncio.run(self._attempt_bot_recovery())

                # Process queue
                asyncio.run(self._process_bot_queue())

                time.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Management loop error: {e}")
                time.sleep(5)

        logger.info("Bot management loop ended")

    def _health_check_loop(self):
        """Background health check loop."""
        logger.info("Bot health check loop started")

        while self.running:
            try:
                with self.lock:
                    bots_to_check = list(self.active_bots.values())

                for bot in bots_to_check:
                    health_score, issues = self.health_monitor.check_bot_health(bot)
                    bot.health_score = health_score

                    if health_score < 0.5:
                        logger.warning(
                            f"Bot {bot.bot_id} health poor: {health_score:.2f}, issues: {issues}"
                        )

                        if self.on_bot_error:
                            self.on_bot_error(bot, f"Poor health: {', '.join(issues)}")

                time.sleep(self.config.get("health_check_interval", 30.0))

            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(10)

        logger.info("Bot health check loop ended")

    def _check_bot_timeouts(self):
        """Check for bot timeouts and cleanup."""
        current_time = datetime.now()
        timeout_duration = timedelta(
            seconds=self.config.get("meeting_timeout", 14400.0)
        )

        bots_to_terminate = []

        with self.lock:
            for bot in self.active_bots.values():
                if current_time - bot.created_at > timeout_duration:
                    bots_to_terminate.append(bot.bot_id)

        for bot_id in bots_to_terminate:
            logger.info(f"Bot {bot_id} timed out, terminating")
            asyncio.run(self.terminate_bot(bot_id))

    async def _attempt_bot_recovery(self):
        """Attempt to recover failed bots."""
        with self.lock:
            error_bots = [
                bot
                for bot in self.active_bots.values()
                if bot.status == BotStatus.ERROR
            ]

        for bot in error_bots:
            if bot.error_count < 3:  # Max 3 recovery attempts
                logger.info(f"Attempting recovery for bot {bot.bot_id}")

                # Reset status and try restart
                bot.status = BotStatus.SPAWNING
                bot.error_count += 1

                asyncio.create_task(self._spawn_bot(bot))

    async def handle_meeting_event(self, bot_id: str, event_data: Dict[str, Any]):
        """Handle events from Google Meet API monitoring."""
        try:
            with self.lock:
                if bot_id not in self.active_bots:
                    logger.warning(f"Received event for unknown bot: {bot_id}")
                    return

                bot = self.active_bots[bot_id]

            event_type = event_data.get("event_type")

            if event_type == "participant_update":
                participants = event_data.get("participants", [])
                bot.participant_count = len(participants)
                bot.last_activity = datetime.now()

                logger.debug(
                    f"Bot {bot_id} participant update: {len(participants)} participants"
                )

                # Update performance stats
                bot.performance_stats["participants_seen"] = max(
                    bot.performance_stats.get("participants_seen", 0), len(participants)
                )

            elif event_type == "conference_ended":
                logger.info(f"Conference ended for bot {bot_id} - scheduling cleanup")
                asyncio.create_task(self.terminate_bot(bot_id))

            elif event_type == "error":
                error_msg = event_data.get("error", "Unknown Google Meet API error")
                logger.error(f"Google Meet API error for bot {bot_id}: {error_msg}")

                with self.lock:
                    bot.error_count += 1
                    bot.status = BotStatus.ERROR

                if self.on_bot_error:
                    self.on_bot_error(bot, error_msg)

        except Exception as e:
            logger.error(f"Error handling meeting event for bot {bot_id}: {e}")

    # Database access methods
    async def store_audio_file(
        self, session_id: str, audio_data: bytes, metadata: Dict = None
    ) -> Optional[str]:
        """Store audio file for a bot session."""
        if not self.database_manager:
            logger.warning("Database manager not available for audio storage")
            return None

        try:
            return await self.database_manager.audio_manager.store_audio_file(
                session_id, audio_data, metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error storing audio file: {e}")
            return None

    async def store_transcript(
        self, session_id: str, transcript_data: Dict
    ) -> Optional[str]:
        """Store transcript for a bot session."""
        if not self.database_manager:
            logger.warning("Database manager not available for transcript storage")
            return None

        try:
            return await self.database_manager.transcript_manager.store_transcript(
                session_id=session_id,
                source_type=transcript_data.get("source_type", "whisper_service"),
                transcript_text=transcript_data["text"],
                language_code=transcript_data.get("language", "en"),
                start_timestamp=transcript_data["start_timestamp"],
                end_timestamp=transcript_data["end_timestamp"],
                speaker_info=transcript_data.get("speaker_info"),
                audio_file_id=transcript_data.get("audio_file_id"),
                processing_metadata=transcript_data.get("metadata"),
            )
        except Exception as e:
            logger.error(f"Error storing transcript: {e}")
            return None

    async def store_translation(
        self, session_id: str, translation_data: Dict
    ) -> Optional[str]:
        """Store translation for a bot session."""
        if not self.database_manager:
            logger.warning("Database manager not available for translation storage")
            return None

        try:
            return await self.database_manager.translation_manager.store_translation(
                session_id=session_id,
                source_transcript_id=translation_data["source_transcript_id"],
                translated_text=translation_data["translated_text"],
                source_language=translation_data["source_language"],
                target_language=translation_data["target_language"],
                translation_service=translation_data.get(
                    "translation_service", "translation_service"
                ),
                speaker_info=translation_data.get("speaker_info"),
                timing_info=translation_data.get("timing_info"),
                processing_metadata=translation_data.get("metadata"),
            )
        except Exception as e:
            logger.error(f"Error storing translation: {e}")
            return None

    async def store_correlation(
        self, session_id: str, correlation_data: Dict
    ) -> Optional[str]:
        """Store time correlation for a bot session."""
        if not self.database_manager:
            logger.warning("Database manager not available for correlation storage")
            return None

        try:
            return await self.database_manager.correlation_manager.store_correlation(
                session_id=session_id,
                google_transcript_id=correlation_data.get("google_transcript_id"),
                inhouse_transcript_id=correlation_data.get("inhouse_transcript_id"),
                correlation_confidence=correlation_data.get(
                    "correlation_confidence", 0.0
                ),
                timing_offset=correlation_data.get("timing_offset", 0.0),
                correlation_type=correlation_data.get("correlation_type", "unknown"),
                correlation_method=correlation_data.get(
                    "correlation_method", "unknown"
                ),
                speaker_id=correlation_data.get("speaker_id"),
                start_timestamp=correlation_data.get("start_timestamp", 0.0),
                end_timestamp=correlation_data.get("end_timestamp", 0.0),
                correlation_metadata=correlation_data.get("metadata"),
            )
        except Exception as e:
            logger.error(f"Error storing correlation: {e}")
            return None

    async def get_session_comprehensive_data(self, session_id: str) -> Optional[Dict]:
        """Get comprehensive session data from database."""
        if not self.database_manager:
            logger.warning("Database manager not available")
            return None

        try:
            return await self.database_manager.get_comprehensive_session_data(
                session_id
            )
        except Exception as e:
            logger.error(f"Error getting comprehensive session data: {e}")
            return None

    async def cleanup_session_data(
        self, session_id: str, remove_files: bool = False
    ) -> bool:
        """Clean up session data from database."""
        if not self.database_manager:
            logger.warning("Database manager not available")
            return False

        try:
            return await self.database_manager.cleanup_session(session_id, remove_files)
        except Exception as e:
            logger.error(f"Error cleaning up session data: {e}")
            return False

    def get_bot_status(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific bot."""
        with self.lock:
            if bot_id not in self.active_bots:
                return None

            bot = self.active_bots[bot_id]
            health_trend = self.health_monitor.get_health_trend(bot_id)

            return {
                "bot_id": bot.bot_id,
                "status": bot.status.value,
                "meeting_id": bot.meeting_request.meeting_id,
                "meeting_title": bot.meeting_request.meeting_title,
                "created_at": bot.created_at.isoformat(),
                "session_id": bot.session_id,
                "participant_count": bot.participant_count,
                "health_score": bot.health_score,
                "health_trend": health_trend,
                "error_count": bot.error_count,
                "performance_stats": bot.performance_stats.copy(),
                "last_activity": bot.last_activity.isoformat()
                if bot.last_activity
                else None,
            }

    def get_all_bots_status(self) -> List[Dict[str, Any]]:
        """Get status of all active bots."""
        with self.lock:
            return [self.get_bot_status(bot_id) for bot_id in self.active_bots.keys()]

    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        with self.lock:
            active_count = len(self.active_bots)
            queued_count = len(self.bot_queue)

            # Calculate health distribution
            health_scores = [bot.health_score for bot in self.active_bots.values()]
            avg_health = (
                sum(health_scores) / len(health_scores) if health_scores else 0.0
            )

            # Status distribution
            status_counts = defaultdict(int)
            for bot in self.active_bots.values():
                status_counts[bot.status.value] += 1

            # Google Meet API status
            google_meet_status = {
                "api_available": self.google_meet_client is not None,
                "api_authenticated": False,
                "integration_ready": self.bot_manager_integration is not None,
            }

            if self.google_meet_client:
                google_meet_status.update(
                    self.google_meet_client.get_client_statistics()
                )

            # Database status
            database_status = {
                "manager_available": self.database_manager is not None,
                "connection_active": False,
            }

            if self.database_manager:
                try:
                    db_stats = await self.database_manager.get_database_statistics()
                    database_status.update(
                        {"connection_active": True, "statistics": db_stats}
                    )
                except Exception as e:
                    database_status["error"] = str(e)

            # Lifecycle manager status
            lifecycle_status = {
                "manager_available": self.lifecycle_manager is not None,
                "monitoring_active": False,
            }

            if self.lifecycle_manager:
                try:
                    lifecycle_stats = self.lifecycle_manager.get_lifecycle_statistics()
                    lifecycle_status.update(
                        {
                            "monitoring_active": self.lifecycle_manager.monitoring_active,
                            "statistics": lifecycle_stats,
                        }
                    )
                except Exception as e:
                    lifecycle_status["error"] = str(e)

            return {
                "manager_status": "running" if self.running else "stopped",
                "total_bots_spawned": self.total_bots_spawned,
                "active_bots": active_count,
                "queued_requests": queued_count,
                "successful_meetings": self.successful_meetings,
                "failed_meetings": self.failed_meetings,
                "success_rate": self.successful_meetings
                / max(1, self.total_bots_spawned),
                "average_meeting_duration": self.average_meeting_duration,
                "average_health_score": avg_health,
                "status_distribution": dict(status_counts),
                "google_meet_api": google_meet_status,
                "database": database_status,
                "lifecycle_management": lifecycle_status,
                "config": self.config.copy(),
            }


# Factory functions
def create_bot_manager(**config_kwargs) -> GoogleMeetBotManager:
    """Create a bot manager with optional configuration."""
    return GoogleMeetBotManager(config_kwargs)


# Example usage
async def main():
    """Example usage of the bot manager."""
    # Create bot manager
    manager = create_bot_manager(max_concurrent_bots=5, auto_recovery_enabled=True)

    # Set up callbacks
    def on_bot_spawned(bot):
        print(f"Bot spawned: {bot.bot_id} for meeting {bot.meeting_request.meeting_id}")

    def on_meeting_ended(meeting_id, bot_id, stats):
        print(f"Meeting ended: {meeting_id}, duration: {stats.get('duration', 0):.0f}s")

    def on_bot_error(bot, error):
        print(f"Bot error: {bot.bot_id} - {error}")

    manager.on_bot_spawned = on_bot_spawned
    manager.on_meeting_ended = on_meeting_ended
    manager.on_bot_error = on_bot_error

    # Start manager
    await manager.start()

    # Create meeting request
    meeting_request = MeetingRequest(
        meeting_id="test-meeting-123",
        meeting_title="Test Meeting",
        organizer_email="test@example.com",
        target_languages=["en", "es"],
    )

    # Request bot
    bot_id = await manager.request_bot(meeting_request)
    if bot_id:
        print(f"Bot requested: {bot_id}")

        # Wait a bit
        await asyncio.sleep(30)

        # Get status
        status = manager.get_bot_status(bot_id)
        print(f"Bot status: {json.dumps(status, indent=2, default=str)}")

        # Terminate bot
        await manager.terminate_bot(bot_id)

    # Get final statistics
    stats = manager.get_manager_statistics()
    print(f"Manager statistics: {json.dumps(stats, indent=2, default=str)}")

    # Stop manager
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
