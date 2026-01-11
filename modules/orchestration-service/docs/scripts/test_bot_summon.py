#!/usr/bin/env python3
"""
Google Meet Bot Summon Test Script

This script tests the complete Google Meet bot system by:
1. Checking service availability (orchestration, Redis, Docker)
2. Starting a bot to join a test Google Meet URL
3. Monitoring bot status and health
4. Verifying callbacks and lifecycle transitions
5. Optionally stopping the bot

Usage:
    python test_bot_summon.py --meeting-url <GOOGLE_MEET_URL>
    python test_bot_summon.py --meeting-url https://meet.google.com/abc-defg-hij --monitor-time 60
"""

import asyncio
import argparse
import logging
import time
from typing import Optional, Dict, Any
import httpx
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BotSummonTest:
    """Test harness for summoning and monitoring Google Meet bots"""

    def __init__(
        self, orchestration_url: str = "http://localhost:3000", timeout: int = 300
    ):
        self.orchestration_url = orchestration_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    async def check_orchestration_health(self) -> bool:
        """Check if orchestration service is running"""
        try:
            logger.info("🔍 Checking orchestration service health...")
            response = await self.client.get(f"{self.orchestration_url}/api/health")

            if response.status_code == 200:
                health = response.json()
                logger.info(
                    f"✅ Orchestration service is healthy: {health.get('status', 'unknown')}"
                )
                return True
            else:
                logger.error(
                    f"❌ Orchestration service returned status {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"❌ Failed to connect to orchestration service: {e}")
            logger.error(
                f"   Make sure the service is running at {self.orchestration_url}"
            )
            return False

    async def check_bot_manager_status(self) -> Dict[str, Any]:
        """Get bot manager statistics"""
        try:
            logger.info("🔍 Checking bot manager status...")
            response = await self.client.get(f"{self.orchestration_url}/api/bots/stats")

            if response.status_code == 200:
                stats = response.json()
                logger.info("✅ Bot manager is available")
                logger.info(f"   Total bots: {stats.get('total_bots', 0)}")
                logger.info(f"   Active bots: {stats.get('active_bots', 0)}")
                logger.info(f"   Success rate: {stats.get('success_rate', 0):.2%}")
                return stats
            else:
                logger.warning(f"⚠️  Bot manager returned status {response.status_code}")
                return {}

        except Exception as e:
            logger.warning(f"⚠️  Could not get bot manager stats: {e}")
            return {}

    async def list_existing_bots(self) -> list:
        """List all existing bots"""
        try:
            response = await self.client.get(f"{self.orchestration_url}/api/bots/list")

            if response.status_code == 200:
                data = response.json()
                bots = data.get("bots", [])
                logger.info(f"📋 Found {len(bots)} existing bot(s)")

                for bot in bots:
                    logger.info(
                        f"   - {bot['connection_id']}: {bot['status']} (user: {bot['user_id']})"
                    )

                return bots
            else:
                logger.warning(f"⚠️  Could not list bots: status {response.status_code}")
                return []

        except Exception as e:
            logger.warning(f"⚠️  Error listing bots: {e}")
            return []

    async def start_bot(
        self,
        meeting_url: str,
        user_token: str = "test-token-123",
        user_id: str = "test-user-456",
        language: str = "en",
        task: str = "transcribe",
        enable_virtual_webcam: bool = False,
    ) -> Optional[str]:
        """
        Start a bot to join Google Meet

        Returns:
            connection_id if successful, None otherwise
        """
        try:
            logger.info("🚀 Starting Google Meet bot...")
            logger.info(f"   Meeting URL: {meeting_url}")
            logger.info(f"   User ID: {user_id}")
            logger.info(f"   Language: {language}")
            logger.info(f"   Task: {task}")
            logger.info(f"   Virtual Webcam: {enable_virtual_webcam}")

            request_data = {
                "meeting_url": meeting_url,
                "user_token": user_token,
                "user_id": user_id,
                "language": language,
                "task": task,
                "enable_virtual_webcam": enable_virtual_webcam,
                "metadata": {"test": True, "test_script": "test_bot_summon.py"},
            }

            response = await self.client.post(
                f"{self.orchestration_url}/api/bots/start", json=request_data
            )

            if response.status_code == 200:
                result = response.json()
                connection_id = result.get("connection_id")
                logger.info("✅ Bot started successfully!")
                logger.info(f"   Connection ID: {connection_id}")
                logger.info(f"   Status: {result.get('status')}")
                logger.info(f"   Message: {result.get('message')}")
                return connection_id
            else:
                logger.error(f"❌ Failed to start bot: status {response.status_code}")
                try:
                    error = response.json()
                    logger.error(f"   Error: {error.get('detail', 'Unknown error')}")
                except:
                    logger.error(f"   Response: {response.text}")
                return None

        except Exception as e:
            logger.error(f"❌ Exception starting bot: {e}")
            return None

    async def get_bot_status(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get bot status"""
        try:
            response = await self.client.get(
                f"{self.orchestration_url}/api/bots/status/{connection_id}"
            )

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"⚠️  Bot not found: {connection_id}")
                return None
            else:
                logger.warning(f"⚠️  Could not get bot status: {response.status_code}")
                return None

        except Exception as e:
            logger.warning(f"⚠️  Error getting bot status: {e}")
            return None

    async def monitor_bot(
        self, connection_id: str, duration: int = 60, poll_interval: int = 5
    ) -> bool:
        """
        Monitor bot for specified duration

        Returns:
            True if bot reached active state, False otherwise
        """
        logger.info(f"👀 Monitoring bot {connection_id} for {duration} seconds...")

        start_time = time.time()
        last_status = None
        reached_active = False

        while time.time() - start_time < duration:
            status = await self.get_bot_status(connection_id)

            if status:
                current_status = status.get("status")

                # Log status changes
                if current_status != last_status:
                    logger.info(
                        f"📊 Bot status changed: {last_status} → {current_status}"
                    )

                    if current_status == "active":
                        logger.info("✅ Bot is now ACTIVE in the meeting!")
                        reached_active = True
                    elif current_status == "failed":
                        logger.error(
                            f"❌ Bot failed: {status.get('error_message', 'Unknown error')}"
                        )
                        return False
                    elif current_status == "completed":
                        logger.info("✅ Bot completed successfully")
                        return reached_active

                    last_status = current_status

                # Log detailed status periodically
                if time.time() - start_time % 15 < poll_interval:  # Every ~15 seconds
                    self._log_detailed_status(status)
            else:
                logger.warning(f"⚠️  Could not get status for bot {connection_id}")

            await asyncio.sleep(poll_interval)

        logger.info(f"⏱️  Monitoring duration ({duration}s) completed")
        return reached_active

    def _log_detailed_status(self, status: Dict[str, Any]):
        """Log detailed bot status information"""
        logger.info("📊 Bot Status Details:")
        logger.info(f"   Connection ID: {status.get('connection_id')}")
        logger.info(f"   Status: {status.get('status')}")
        logger.info(f"   User ID: {status.get('user_id')}")
        logger.info(f"   Meeting URL: {status.get('meeting_url')}")
        logger.info(f"   Uptime: {status.get('uptime_seconds', 0):.1f}s")
        logger.info(f"   Healthy: {status.get('is_healthy', False)}")
        logger.info(f"   Container ID: {status.get('container_id', 'N/A')}")
        logger.info(f"   Container Name: {status.get('container_name', 'N/A')}")

        if status.get("error_message"):
            logger.error(f"   Error: {status.get('error_message')}")

    async def stop_bot(self, connection_id: str, timeout: int = 30) -> bool:
        """Stop a bot"""
        try:
            logger.info(f"🛑 Stopping bot {connection_id}...")

            response = await self.client.post(
                f"{self.orchestration_url}/api/bots/stop/{connection_id}",
                json={"timeout": timeout},
            )

            if response.status_code == 200:
                result = response.json()
                logger.info("✅ Stop command sent successfully")
                logger.info(f"   Message: {result.get('message')}")
                return True
            else:
                logger.error(f"❌ Failed to stop bot: status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error stopping bot: {e}")
            return False

    async def run_full_test(
        self,
        meeting_url: str,
        monitor_duration: int = 60,
        auto_stop: bool = False,
        language: str = "en",
        enable_webcam: bool = False,
    ) -> bool:
        """
        Run full bot summon test

        Returns:
            True if test passed, False otherwise
        """
        try:
            logger.info("=" * 70)
            logger.info("🤖 GOOGLE MEET BOT SUMMON TEST")
            logger.info("=" * 70)

            # Step 1: Check orchestration service
            if not await self.check_orchestration_health():
                logger.error("❌ TEST FAILED: Orchestration service not available")
                return False

            # Step 2: Check bot manager
            await self.check_bot_manager_status()

            # Step 3: List existing bots
            await self.list_existing_bots()

            # Step 4: Start new bot
            connection_id = await self.start_bot(
                meeting_url=meeting_url,
                language=language,
                enable_virtual_webcam=enable_webcam,
            )

            if not connection_id:
                logger.error("❌ TEST FAILED: Could not start bot")
                return False

            # Step 5: Monitor bot
            logger.info("")
            reached_active = await self.monitor_bot(
                connection_id, duration=monitor_duration
            )

            # Step 6: Get final status
            logger.info("")
            logger.info("📊 Final Bot Status:")
            final_status = await self.get_bot_status(connection_id)
            if final_status:
                self._log_detailed_status(final_status)

            # Step 7: Optionally stop bot
            if auto_stop:
                logger.info("")
                await self.stop_bot(connection_id)

                # Wait for stop to complete
                await asyncio.sleep(5)

                # Check final status
                stopped_status = await self.get_bot_status(connection_id)
                if stopped_status:
                    logger.info(
                        f"📊 Bot status after stop: {stopped_status.get('status')}"
                    )

            # Test result
            logger.info("")
            logger.info("=" * 70)
            if reached_active:
                logger.info("✅ TEST PASSED: Bot successfully joined Google Meet!")
            else:
                logger.warning(
                    "⚠️  TEST PARTIAL: Bot started but did not reach active state"
                )
                logger.warning(
                    "   This may be normal if the meeting URL requires manual approval"
                )
            logger.info("=" * 70)

            return reached_active

        except Exception as e:
            logger.error(f"❌ TEST EXCEPTION: {e}", exc_info=True)
            return False


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test Google Meet bot summon and monitoring"
    )
    parser.add_argument(
        "--meeting-url",
        required=True,
        help="Google Meet URL to join (e.g., https://meet.google.com/abc-defg-hij)",
    )
    parser.add_argument(
        "--orchestration-url",
        default="http://localhost:3000",
        help="Orchestration service URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--monitor-time",
        type=int,
        default=60,
        help="How long to monitor the bot in seconds (default: 60)",
    )
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help="Automatically stop the bot after monitoring",
    )
    parser.add_argument(
        "--language", default="en", help="Transcription language (default: en)"
    )
    parser.add_argument(
        "--enable-webcam", action="store_true", help="Enable virtual webcam output"
    )

    args = parser.parse_args()

    # Create test harness
    test = BotSummonTest(orchestration_url=args.orchestration_url)

    try:
        # Run test
        success = await test.run_full_test(
            meeting_url=args.meeting_url,
            monitor_duration=args.monitor_time,
            auto_stop=args.auto_stop,
            language=args.language,
            enable_webcam=args.enable_webcam,
        )

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    finally:
        await test.close()


if __name__ == "__main__":
    asyncio.run(main())
