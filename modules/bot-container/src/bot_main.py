#!/usr/bin/env python3
"""
Bot Container Entry Point - "Headless Frontend"

This bot is a Docker container that:
1. Joins Google Meet via browser automation
2. Captures audio from the meeting
3. Streams to orchestration (SAME protocol as frontend!)
4. Receives transcription segments back
5. Optionally displays on virtual webcam

Architecture:
    Bot Container (this)
        ↓ audio chunks (WebSocket)
    Orchestration Service
        ↓ segments (WebSocket)
    Bot Container (receives processed segments)

Key Benefits:
- Isolated process (failures don't affect manager)
- Uses same orchestration infrastructure as frontend
- Consistent segment processing (deduplication, speaker grouping)
- Account tracking flows through orchestration
"""

import asyncio
import os
import logging
import signal
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import bot components (to be implemented)
# from browser_automation import GoogleMeetAutomation
# from audio_capture import AudioCapture
from orchestration_client import OrchestrationClient
# from redis_subscriber import RedisSubscriber

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Bot configuration"""
    meeting_url: str
    connection_id: str
    user_token: str
    orchestration_url: str
    redis_url: Optional[str] = None
    bot_manager_url: Optional[str] = None
    language: str = "en"
    task: str = "transcribe"
    enable_virtual_webcam: bool = False


class Bot:
    """
    Main bot class - orchestrates all bot components

    Components:
    - OrchestrationClient: WebSocket to orchestration
    - GoogleMeetAutomation: Browser automation (to be implemented)
    - AudioCapture: Audio extraction (to be implemented)
    - RedisSubscriber: Command listener (to be implemented)
    - VirtualWebcam: Display output (to be implemented)
    """

    def __init__(
        self,
        meeting_url: str,
        connection_id: str,
        user_token: str,
        orchestration_url: str,
        redis_url: Optional[str] = None,
        bot_manager_url: Optional[str] = None,
        language: str = "en",
        task: str = "transcribe",
        enable_virtual_webcam: bool = False
    ):
        """
        Initialize bot

        Args:
            meeting_url: Google Meet URL to join
            connection_id: Unique bot connection ID (from manager)
            user_token: User API token for orchestration auth
            orchestration_url: Orchestration WebSocket URL
            redis_url: Redis URL for commands (optional)
            bot_manager_url: Bot manager URL for callbacks (optional)
            language: Transcription language
            task: Transcription task ('transcribe' or 'translate')
            enable_virtual_webcam: Enable virtual webcam display
        """
        self.config = BotConfig(
            meeting_url=meeting_url,
            connection_id=connection_id,
            user_token=user_token,
            orchestration_url=orchestration_url,
            redis_url=redis_url,
            bot_manager_url=bot_manager_url,
            language=language,
            task=task,
            enable_virtual_webcam=enable_virtual_webcam
        )

        # Store config for easy access
        self.meeting_url = meeting_url
        self.connection_id = connection_id
        self.user_token = user_token
        self.orchestration_url = orchestration_url

        # Components (initialize as None, create in run())
        self.orchestration: Optional[OrchestrationClient] = None
        self.browser = None  # GoogleMeetAutomation instance
        # self.audio: Optional[AudioCapture] = None
        # self.redis: Optional[RedisSubscriber] = None
        # self.webcam: Optional[VirtualWebcam] = None

        # State
        self.running = False
        self.status = "initializing"  # initializing, connecting, joining, active, stopping, stopped

        # Graceful shutdown handler
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

    @classmethod
    def from_env(cls) -> "Bot":
        """
        Create bot from environment variables

        Expected environment:
        - MEETING_URL: Google Meet URL
        - CONNECTION_ID: Bot connection ID
        - USER_TOKEN: User API token
        - ORCHESTRATION_WS_URL: Orchestration WebSocket URL
        - REDIS_URL: Redis URL (optional)
        - BOT_MANAGER_URL: Bot manager URL (optional)
        - LANGUAGE: Transcription language (default: en)
        - TASK: Transcription task (default: transcribe)
        """
        return cls(
            meeting_url=os.environ["MEETING_URL"],
            connection_id=os.environ["CONNECTION_ID"],
            user_token=os.environ["USER_TOKEN"],
            orchestration_url=os.getenv("ORCHESTRATION_WS_URL", "ws://orchestration:3000/ws"),
            redis_url=os.getenv("REDIS_URL"),
            bot_manager_url=os.getenv("BOT_MANAGER_URL"),
            language=os.getenv("LANGUAGE", "en"),
            task=os.getenv("TASK", "transcribe"),
            enable_virtual_webcam=os.getenv("ENABLE_VIRTUAL_WEBCAM", "false").lower() == "true"
        )

    async def run(self):
        """
        Main bot execution loop

        Flow:
        1. Connect to orchestration (BEFORE joining meeting)
        2. Notify bot manager: started
        3. Join Google Meet
        4. Notify bot manager: joining → active
        5. Stream audio → orchestration
        6. Receive segments ← orchestration
        7. Optionally display on virtual webcam
        8. On exit: cleanup and notify manager
        """
        self.running = True
        logger.info(f"🤖 Bot {self.connection_id} starting...")
        logger.info(f"   Meeting: {self.config.meeting_url}")
        logger.info(f"   Orchestration: {self.config.orchestration_url}")

        try:
            # Phase 1: Connect to orchestration
            self.status = "connecting"
            await self._connect_to_orchestration()

            # Notify manager: started
            await self._notify_manager("started")

            # Phase 2: Join Google Meet
            self.status = "joining"
            await self._notify_manager("joining")
            await self._join_meeting()

            # Phase 3: Start audio streaming
            self.status = "active"
            await self._notify_manager("active")
            await self._start_audio_stream()

            # Phase 4: Main loop - keep bot alive
            await self._main_loop()

        except KeyboardInterrupt:
            logger.info("Bot interrupted by user")
            self.status = "stopping"

        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            self.status = "failed"
            await self._notify_manager("failed", error=str(e))

        finally:
            # Cleanup
            await self._cleanup()

    async def _connect_to_orchestration(self):
        """Connect to orchestration WebSocket service"""
        logger.info(f"Connecting to orchestration at {self.config.orchestration_url}")

        self.orchestration = OrchestrationClient(
            orchestration_url=self.config.orchestration_url,
            user_token=self.config.user_token,
            meeting_id=self.config.meeting_url,
            connection_id=self.config.connection_id
        )

        # Register segment handler
        self.orchestration.on_segment(self._handle_segment)
        self.orchestration.on_error(self._handle_error)

        # Connect
        connected = await self.orchestration.connect()
        if not connected:
            raise RuntimeError("Failed to connect to orchestration")

        logger.info("✅ Connected to orchestration service")

    async def _join_meeting(self):
        """Join Google Meet via browser automation"""
        logger.info(f"Joining Google Meet: {self.config.meeting_url}")

        from google_meet_automation import GoogleMeetAutomation, BrowserConfig, MeetingState

        # Configure browser automation with Google credentials from environment
        browser_config = BrowserConfig(
            headless=os.getenv("HEADLESS", "true").lower() == "true",
            audio_capture_enabled=True,
            video_enabled=False,
            microphone_enabled=False,
            screenshots_enabled=os.getenv("SCREENSHOTS_ENABLED", "true").lower() == "true",
            screenshots_path=os.getenv("SCREENSHOTS_PATH", "/tmp/bot-screenshots"),
            # Google Authentication
            google_email=os.getenv("GOOGLE_EMAIL"),
            google_password=os.getenv("GOOGLE_PASSWORD"),
            user_data_dir=os.getenv("USER_DATA_DIR")
        )

        # Create and initialize automation
        self.browser = GoogleMeetAutomation(browser_config)
        await self.browser.initialize()

        # Join meeting
        bot_name = f"LiveTranslate-{self.connection_id[:8]}"
        success = await self.browser.join_meeting(self.config.meeting_url, bot_name)

        if not success:
            raise RuntimeError("Failed to join Google Meet")

        # Wait for admission if in waiting room
        if self.browser.get_state() == MeetingState.WAITING_ROOM:
            logger.info("⏳ Bot is in waiting room, waiting for admission...")
            admitted = await self.browser.wait_for_active(timeout=300)
            if not admitted:
                logger.warning("⚠️  Bot was not admitted from waiting room")

        logger.info(f"✅ Bot joined Google Meet (state: {self.browser.get_state().value})")
        # self.browser = GoogleMeetAutomation()
        # await self.browser.join(self.config.meeting_url)

        logger.info("✅ Joined Google Meet")

    async def _start_audio_stream(self):
        """Start capturing and streaming audio"""
        logger.info("Starting audio stream...")

        # TODO: Implement audio capture
        # self.audio = AudioCapture()
        # async for audio_chunk in self.audio.capture_stream():
        #     await self.orchestration.send_audio_chunk(audio_chunk)

        logger.info("✅ Audio streaming started")

    async def _main_loop(self):
        """Main bot loop - keep alive and handle commands"""
        logger.info("Bot active - entering main loop")

        # TODO: Listen for Redis commands
        # await self.redis.listen()

        # For now, just keep alive
        while self.running:
            await asyncio.sleep(1)

    def _handle_segment(self, segment: Dict[str, Any]):
        """
        Handle transcription segment from orchestration

        Segment is already:
        - Deduplicated by absolute_start_time
        - Speaker grouped
        - Timestamped

        Args:
            segment: Segment dictionary with keys:
                - text: Transcribed text
                - speaker: Speaker ID (e.g., "SPEAKER_00")
                - absolute_start_time: ISO 8601 timestamp
                - absolute_end_time: ISO 8601 timestamp
                - is_final: Whether segment is final
                - confidence: Confidence score
        """
        logger.info(f"📄 Segment received:")
        logger.info(f"   Text: {segment.get('text')}")
        logger.info(f"   Speaker: {segment.get('speaker')}")
        logger.info(f"   Time: {segment.get('absolute_start_time')}")

        # TODO: Display on virtual webcam if enabled
        # if self.config.enable_virtual_webcam and self.webcam:
        #     await self.webcam.display_segment(segment)

    def _handle_error(self, error: str):
        """Handle error from orchestration"""
        logger.error(f"❌ Orchestration error: {error}")

    async def _notify_manager(self, status: str, error: Optional[str] = None):
        """
        Send HTTP callback to bot manager (like Vexa)

        Endpoints:
        - POST /api/bots/internal/callback/started
        - POST /api/bots/internal/callback/joining
        - POST /api/bots/internal/callback/active
        - POST /api/bots/internal/callback/completed
        - POST /api/bots/internal/callback/failed

        Args:
            status: Status to notify (started, joining, active, completed, failed)
            error: Optional error message (for failed status)
        """
        if not self.config.bot_manager_url:
            logger.debug(f"No bot manager URL configured - skipping {status} callback")
            return

        try:
            import httpx

            url = f"{self.config.bot_manager_url}/api/bots/internal/callback/{status}"
            payload = {
                "connection_id": self.config.connection_id,
                "container_id": os.getenv("HOSTNAME", "unknown")  # Docker container ID
            }

            if error:
                payload["error"] = error
                payload["exit_code"] = 1

            logger.info(f"Sending callback to bot manager: {status}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            logger.info(f"✅ Bot manager notified: {status}")

        except Exception as e:
            logger.error(f"Failed to notify bot manager ({status}): {e}")

    async def _cleanup(self):
        """Cleanup bot resources"""
        logger.info("Cleaning up bot resources...")

        # Disconnect from orchestration
        if self.orchestration:
            try:
                await self.orchestration.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from orchestration: {e}")

        # Leave Google Meet and cleanup browser
        if self.browser:
            try:
                await self.browser.leave_meeting()
                await self.browser.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up browser: {e}")

        # Original commented lines:
        # if self.browser:
        #     try:
        #         await self.browser.leave()
        #     except Exception as e:
        #         logger.error(f"Error leaving meeting: {e}")

        # Stop audio capture
        # if self.audio:
        #     try:
        #         await self.audio.stop()
        #     except Exception as e:
        #         logger.error(f"Error stopping audio: {e}")

        # Notify manager of completion
        if self.status != "failed":
            await self._notify_manager("completed")

        self.running = False
        self.status = "stopped"
        logger.info("✅ Bot cleanup complete")

    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals (SIGTERM, SIGINT)"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False


# Main entry point
async def main():
    """Main entry point for bot container"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger.info("=" * 60)
    logger.info("🤖 LiveTranslate Bot Container Starting")
    logger.info("=" * 60)

    try:
        # Create bot from environment
        bot = Bot.from_env()

        # Run bot
        await bot.run()

    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    logger.info("🤖 Bot container exiting")


if __name__ == "__main__":
    asyncio.run(main())
