#!/usr/bin/env python3
"""
Simplified Docker-Based Bot Manager

Phase 3.3: Simplified Bot Architecture
Replaces complex Python process management with Docker orchestration.

This manager:
- Spawns bot containers using Docker SDK
- Receives HTTP callbacks from bot containers
- Sends Redis commands to control bots
- Tracks bot status in database
- Provides simple health monitoring

Complexity: ~800 lines (vs 8,701 lines in old architecture)
Reduction: -60% complexity
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

try:
    import docker
    from docker.models.containers import Container
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    Container = None  # Type hint placeholder
    logger = logging.getLogger(__name__)
    logger.warning("Docker SDK not available - install with: pip install docker")

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Redis asyncio not available - install with: pip install redis")

# Configure logging
logger = logging.getLogger(__name__)


class BotStatus(Enum):
    """Bot container status"""
    SPAWNING = "spawning"          # Container starting
    STARTING = "starting"          # Container sent 'started' callback
    JOINING = "joining"            # Container joining meeting
    ACTIVE = "active"              # Container active in meeting
    STOPPING = "stopping"          # Stop command sent
    COMPLETED = "completed"        # Clean exit
    FAILED = "failed"              # Error exit
    UNKNOWN = "unknown"            # Status unknown


@dataclass
class BotConfig:
    """Bot container configuration"""
    # Required
    meeting_url: str
    connection_id: str
    user_token: str
    user_id: str

    # Optional
    orchestration_ws_url: str = "ws://orchestration:3000/ws"
    redis_url: Optional[str] = "redis://redis:6379"
    bot_manager_url: Optional[str] = "http://orchestration:3000"
    language: str = "en"
    task: str = "transcribe"
    enable_virtual_webcam: bool = False

    # Google Authentication (for restricted meetings)
    google_email: Optional[str] = None
    google_password: Optional[str] = None
    user_data_dir: Optional[str] = None
    headless: bool = True
    screenshots_enabled: bool = True
    screenshots_path: str = "/tmp/bot-screenshots"

    # Docker
    docker_image: str = "livetranslate-bot:latest"
    docker_network: str = "livetranslate_default"

    def to_env_vars(self) -> Dict[str, str]:
        """Convert config to Docker environment variables"""
        env_vars = {
            "MEETING_URL": self.meeting_url,
            "CONNECTION_ID": self.connection_id,
            "USER_TOKEN": self.user_token,
            "ORCHESTRATION_WS_URL": self.orchestration_ws_url,
            "REDIS_URL": self.redis_url or "",
            "BOT_MANAGER_URL": self.bot_manager_url or "",
            "LANGUAGE": self.language,
            "TASK": self.task,
            "ENABLE_VIRTUAL_WEBCAM": str(self.enable_virtual_webcam).lower(),
        }

        # Add Google authentication if provided
        if self.google_email:
            env_vars["GOOGLE_EMAIL"] = self.google_email
        if self.google_password:
            env_vars["GOOGLE_PASSWORD"] = self.google_password
        if self.user_data_dir:
            env_vars["USER_DATA_DIR"] = self.user_data_dir

        # Browser settings
        env_vars["HEADLESS"] = str(self.headless).lower()
        env_vars["SCREENSHOTS_ENABLED"] = str(self.screenshots_enabled).lower()
        env_vars["SCREENSHOTS_PATH"] = self.screenshots_path

        return env_vars


@dataclass
class BotInstance:
    """Represents a bot container instance"""
    # Identity
    connection_id: str
    user_id: str
    meeting_url: str

    # Container
    container_id: Optional[str] = None
    container_name: Optional[str] = None

    # Status
    status: BotStatus = BotStatus.SPAWNING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    active_at: Optional[float] = None
    stopped_at: Optional[float] = None

    # Health
    last_callback: Optional[float] = None
    error_message: Optional[str] = None
    exit_code: Optional[int] = None

    # Metadata
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-safe types"""
        data = asdict(self)
        # Ensure status is a string
        data["status"] = self.status.value if isinstance(self.status, BotStatus) else str(self.status)
        # Add computed properties
        data["uptime_seconds"] = self.uptime_seconds
        data["is_healthy"] = self.is_healthy
        # Remove config if it's None or empty to reduce response size
        if not data.get("config"):
            data.pop("config", None)
        return data

    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds"""
        if self.started_at:
            end_time = self.stopped_at or time.time()
            return end_time - self.started_at
        return 0.0

    @property
    def is_healthy(self) -> bool:
        """Check if bot is healthy"""
        if self.status == BotStatus.FAILED:
            return False

        # Check for stale callback (no callback in 5 minutes)
        if self.last_callback:
            inactivity = time.time() - self.last_callback
            if inactivity > 300:  # 5 minutes
                return False

        return True


class DockerBotManager:
    """
    Simplified Docker-based bot manager

    Manages bot lifecycle using Docker containers instead of Python processes.

    Usage:
        manager = DockerBotManager(
            orchestration_url="http://localhost:3000",
            redis_url="redis://localhost:6379"
        )

        # Start bot
        bot_id = await manager.start_bot(
            meeting_url="https://meet.google.com/abc-def-ghi",
            user_token="user-token-123",
            user_id="user-456"
        )

        # Bot sends callbacks:
        await manager.handle_bot_callback(bot_id, "started", {})
        await manager.handle_bot_callback(bot_id, "joining", {})
        await manager.handle_bot_callback(bot_id, "active", {})

        # Stop bot
        await manager.stop_bot(bot_id)

        # Bot sends callback:
        await manager.handle_bot_callback(bot_id, "completed", {})
    """

    def __init__(
        self,
        orchestration_url: str = "http://localhost:3000",
        redis_url: str = "redis://localhost:6379",
        docker_image: str = "livetranslate-bot:latest",
        docker_network: str = "livetranslate_default",
        enable_database: bool = False,  # Disabled by default for local testing
        # Google Authentication (for restricted meetings)
        google_email: Optional[str] = None,
        google_password: Optional[str] = None,
        user_data_dir: Optional[str] = None,
        headless: bool = True,
        screenshots_enabled: bool = True,
        screenshots_path: str = "/tmp/bot-screenshots"
    ):
        self.orchestration_url = orchestration_url
        self.redis_url = redis_url
        self.docker_image = docker_image
        self.docker_network = docker_network
        self.enable_database = enable_database

        # Google Authentication
        self.google_email = google_email
        self.google_password = google_password
        self.user_data_dir = user_data_dir
        self.headless = headless
        self.screenshots_enabled = screenshots_enabled
        self.screenshots_path = screenshots_path

        # Debug logging
        logger.info(f"🔧 DockerBotManager initialized with Google credentials: email={google_email}, password={'***' + google_password[-4:] if google_password else None}, user_data_dir={user_data_dir}")

        # Bot tracking
        self.bots: Dict[str, BotInstance] = {}

        # Docker client
        self.docker_client: Optional[docker.DockerClient] = None

        # Redis client
        self.redis_client: Optional[aioredis.Redis] = None

        # Database
        self.db_manager = None

        # Stats
        self.total_bots_started = 0
        self.total_bots_completed = 0
        self.total_bots_failed = 0

    async def initialize(self):
        """Initialize manager"""
        logger.info("Initializing Docker bot manager")

        # Initialize Docker client
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("✅ Docker client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Docker client: {e}")
                logger.warning("Bot manager will run in mock mode")
        else:
            logger.warning("Docker SDK not available - running in mock mode")

        # Initialize Redis client
        if REDIS_AVAILABLE and self.redis_url:
            try:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("✅ Redis client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
        else:
            logger.warning("Redis not available - commands will not work")

        # Initialize database (if enabled)
        if self.enable_database:
            try:
                from database.bot_session_manager import create_bot_session_manager
                from config import get_settings

                # Get database config from settings
                settings = get_settings()
                database_config = {
                    "host": settings.bot.database_host,
                    "port": settings.bot.database_port,
                    "database": settings.bot.database_name,
                    "username": settings.bot.database_user,
                    "password": settings.bot.database_password
                }
                audio_storage_path = settings.bot.audio_storage_path

                self.db_manager = await create_bot_session_manager(database_config, audio_storage_path)
                logger.info("✅ Database manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                logger.warning("Running without database persistence")
                self.db_manager = None

        logger.info("Docker bot manager initialized successfully")

    async def shutdown(self):
        """Shutdown manager"""
        logger.info("Shutting down Docker bot manager")

        # Stop all bots
        for connection_id in list(self.bots.keys()):
            try:
                await self.stop_bot(connection_id)
            except Exception as e:
                logger.error(f"Error stopping bot {connection_id}: {e}")

        # Close Redis
        if self.redis_client:
            await self.redis_client.close()

        logger.info("Docker bot manager shut down")

    async def start_bot(
        self,
        meeting_url: str,
        user_token: str,
        user_id: str,
        language: str = "en",
        task: str = "transcribe",
        enable_virtual_webcam: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a bot container

        Args:
            meeting_url: Google Meet URL
            user_token: User API token
            user_id: User ID
            language: Transcription language
            task: transcribe or translate
            enable_virtual_webcam: Enable virtual webcam output
            metadata: Optional metadata

        Returns:
            connection_id: Unique bot connection ID
        """
        # Generate connection ID
        connection_id = f"bot-{uuid.uuid4().hex[:12]}"

        # Create bot config
        # For Docker containers, replace localhost with host.docker.internal to reach host machine
        orchestration_url_for_container = self.orchestration_url.replace("localhost", "host.docker.internal")
        redis_url_for_container = self.redis_url.replace("localhost", "host.docker.internal") if self.redis_url else None

        config = BotConfig(
            meeting_url=meeting_url,
            connection_id=connection_id,
            user_token=user_token,
            user_id=user_id,
            orchestration_ws_url=orchestration_url_for_container.replace("http://", "ws://").replace("https://", "wss://") + "/ws",
            redis_url=redis_url_for_container,
            bot_manager_url=orchestration_url_for_container,
            language=language,
            task=task,
            enable_virtual_webcam=enable_virtual_webcam,
            # Google Authentication
            google_email=self.google_email,
            google_password=self.google_password,
            user_data_dir=self.user_data_dir,
            headless=self.headless,
            screenshots_enabled=self.screenshots_enabled,
            screenshots_path=self.screenshots_path,
            # Docker
            docker_image=self.docker_image,
            docker_network=self.docker_network
        )

        # Create bot instance
        bot = BotInstance(
            connection_id=connection_id,
            user_id=user_id,
            meeting_url=meeting_url,
            config=asdict(config),
            metadata=metadata or {}
        )

        # Track bot
        self.bots[connection_id] = bot
        self.total_bots_started += 1

        logger.info(f"Starting bot {connection_id} for meeting: {meeting_url}")

        # Start Docker container
        try:
            if self.docker_client:
                container = await self._start_container(config)
                bot.container_id = container.id
                bot.container_name = container.name
                logger.info(f"✅ Bot container started: {container.name} ({container.short_id})")
            else:
                # Mock mode
                bot.container_id = f"mock-{connection_id}"
                bot.container_name = f"bot-{connection_id}"
                logger.info(f"✅ Bot started (mock mode): {connection_id}")

        except Exception as e:
            logger.error(f"Failed to start bot container: {e}")
            bot.status = BotStatus.FAILED
            bot.error_message = str(e)
            self.total_bots_failed += 1
            raise

        # Save to database
        if self.db_manager:
            try:
                await self.db_manager.create_bot_session(
                    connection_id=connection_id,
                    user_id=user_id,
                    meeting_id=meeting_url,
                    meeting_url=meeting_url,
                    language=language,
                    task=task,
                    status=bot.status.value,
                    container_id=bot.container_id
                )
            except Exception as e:
                logger.error(f"Failed to save bot to database: {e}")

        return connection_id

    async def _start_container(self, config: BotConfig) -> Any:
        """Start Docker container"""
        if not self.docker_client:
            raise RuntimeError("Docker client not available")

        # Prepare container config
        container_config = {
            "image": config.docker_image,
            "name": f"bot-{config.connection_id}",
            "environment": config.to_env_vars(),
            "network": config.docker_network,
            "detach": True,
            "remove": False,  # Keep container for logs
            "labels": {
                "livetranslate.bot": "true",
                "livetranslate.connection_id": config.connection_id,
                "livetranslate.user_id": config.user_id
            }
        }

        # Start container
        container = self.docker_client.containers.run(**container_config)

        return container

    async def stop_bot(self, connection_id: str, timeout: int = 30):
        """
        Stop a bot by sending Redis leave command

        Args:
            connection_id: Bot connection ID
            timeout: Timeout in seconds
        """
        bot = self.bots.get(connection_id)
        if not bot:
            raise ValueError(f"Bot not found: {connection_id}")

        logger.info(f"Stopping bot {connection_id}")
        bot.status = BotStatus.STOPPING

        # Send Redis leave command
        if self.redis_client:
            try:
                command = {"action": "leave"}
                channel = f"bot_commands:{connection_id}"
                await self.redis_client.publish(channel, json.dumps(command))
                logger.info(f"Sent leave command to bot {connection_id}")
            except Exception as e:
                logger.error(f"Failed to send leave command: {e}")

        # Wait for container to stop (or force stop after timeout)
        if self.docker_client and bot.container_id:
            try:
                container = self.docker_client.containers.get(bot.container_id)
                container.wait(timeout=timeout)
                logger.info(f"Bot container stopped: {connection_id}")
            except Exception as e:
                logger.warning(f"Container did not stop gracefully, forcing: {e}")
                try:
                    container.stop(timeout=5)
                except Exception as e2:
                    logger.error(f"Failed to force stop container: {e2}")

        # Update database
        if self.db_manager:
            try:
                await self.db_manager.update_bot_status(
                    connection_id,
                    BotStatus.COMPLETED.value
                )
            except Exception as e:
                logger.error(f"Failed to update bot status in database: {e}")

    async def handle_bot_callback(
        self,
        connection_id: str,
        status: str,
        data: Dict[str, Any]
    ):
        """
        Handle HTTP callback from bot container

        Called by bot containers via:
        POST /bots/internal/callback/{status}

        Args:
            connection_id: Bot connection ID
            status: Callback status (started, joining, active, completed, failed)
            data: Callback data
        """
        bot = self.bots.get(connection_id)
        if not bot:
            logger.warning(f"Received callback for unknown bot: {connection_id}")
            return

        logger.info(f"Bot {connection_id} callback: {status}")

        # Update bot status
        bot.last_callback = time.time()

        if status == "started":
            bot.status = BotStatus.STARTING
            bot.started_at = time.time()

        elif status == "joining":
            bot.status = BotStatus.JOINING

        elif status == "active":
            bot.status = BotStatus.ACTIVE
            bot.active_at = time.time()

        elif status == "completed":
            bot.status = BotStatus.COMPLETED
            bot.stopped_at = time.time()
            self.total_bots_completed += 1

        elif status == "failed":
            bot.status = BotStatus.FAILED
            bot.stopped_at = time.time()
            bot.error_message = data.get("error", "Unknown error")
            bot.exit_code = data.get("exit_code")
            self.total_bots_failed += 1

        # Update database
        if self.db_manager:
            try:
                await self.db_manager.update_bot_status(
                    connection_id,
                    bot.status.value,
                    error_message=bot.error_message
                )
            except Exception as e:
                logger.error(f"Failed to update bot status in database: {e}")

        # Handle cleanup for terminated bots
        if status in ["completed", "failed"]:
            await self._cleanup_bot(connection_id)

    async def _cleanup_bot(self, connection_id: str):
        """Cleanup bot resources"""
        bot = self.bots.get(connection_id)
        if not bot:
            return

        logger.info(f"Cleaning up bot {connection_id}")

        # Remove container (DISABLED FOR DEBUGGING)
        # if self.docker_client and bot.container_id:
        #     try:
        #         container = self.docker_client.containers.get(bot.container_id)
        #         container.remove(force=True)
        #         logger.info(f"Removed container: {bot.container_id}")
        #     except Exception as e:
        #         logger.error(f"Failed to remove container: {e}")
        logger.info(f"Container cleanup disabled for debugging: {bot.container_id}")

        # Keep bot in memory for stats/history
        # (could implement removal after X hours)

    async def send_command(self, connection_id: str, command: Dict[str, Any]):
        """
        Send Redis command to bot

        Args:
            connection_id: Bot connection ID
            command: Command dict (e.g., {"action": "leave"})
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not available")

        channel = f"bot_commands:{connection_id}"
        await self.redis_client.publish(channel, json.dumps(command))
        logger.info(f"Sent command to bot {connection_id}: {command}")

    def get_bot(self, connection_id: str) -> Optional[BotInstance]:
        """Get bot by connection ID"""
        return self.bots.get(connection_id)

    def list_bots(
        self,
        status: Optional[BotStatus] = None,
        user_id: Optional[str] = None
    ) -> List[BotInstance]:
        """List bots with optional filters"""
        bots = list(self.bots.values())

        if status:
            bots = [b for b in bots if b.status == status]

        if user_id:
            bots = [b for b in bots if b.user_id == user_id]

        return bots

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        active_bots = len([b for b in self.bots.values() if b.status == BotStatus.ACTIVE])

        return {
            "total_bots": len(self.bots),
            "active_bots": active_bots,
            "total_started": self.total_bots_started,
            "total_completed": self.total_bots_completed,
            "total_failed": self.total_bots_failed,
            "success_rate": self.total_bots_completed / max(1, self.total_bots_started),
            "bots_by_status": {
                status.value: len([b for b in self.bots.values() if b.status == status])
                for status in BotStatus
            }
        }


# Singleton instance
_manager: Optional[DockerBotManager] = None


async def get_bot_manager() -> DockerBotManager:
    """Get or create bot manager singleton with configuration from settings"""
    global _manager
    if _manager is None:
        from config import get_settings
        settings = get_settings()

        _manager = DockerBotManager(
            orchestration_url=f"http://localhost:{settings.port}",
            redis_url=settings.redis.url,
            docker_image=settings.bot.docker_image,
            docker_network=settings.bot.docker_network,
            enable_database=settings.bot.enable_database,
            # Google Authentication
            google_email=settings.bot.google_email,
            google_password=settings.bot.google_password,
            user_data_dir=settings.bot.user_data_dir,
            headless=settings.bot.headless,
            screenshots_enabled=settings.bot.screenshots_enabled,
            screenshots_path=settings.bot.screenshots_path
        )
        await _manager.initialize()
    return _manager


# Example usage
async def example_usage():
    """Example of using Docker bot manager"""
    manager = DockerBotManager(
        orchestration_url="http://localhost:3000",
        redis_url="redis://localhost:6379"
    )

    await manager.initialize()

    try:
        # Start bot
        bot_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test-meeting",
            user_token="user-token-123",
            user_id="user-456",
            language="en",
            task="transcribe"
        )

        print(f"Started bot: {bot_id}")

        # Simulate callbacks from bot
        await asyncio.sleep(2)
        await manager.handle_bot_callback(bot_id, "started", {"container_id": "abc123"})

        await asyncio.sleep(2)
        await manager.handle_bot_callback(bot_id, "joining", {})

        await asyncio.sleep(2)
        await manager.handle_bot_callback(bot_id, "active", {})

        # Keep running
        await asyncio.sleep(30)

        # Stop bot
        await manager.stop_bot(bot_id)

        # Stats
        stats = manager.get_stats()
        print(f"Manager stats: {stats}")

    finally:
        await manager.shutdown()


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
