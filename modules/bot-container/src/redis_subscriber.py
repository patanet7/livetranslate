#!/usr/bin/env python3
"""
Redis Command Subscriber for Bot Container

Listens for commands from the bot manager via Redis pub/sub.
Allows manager to control bot lifecycle without direct communication.

Commands:
- leave: Stop bot and leave meeting
- reconfigure: Update bot configuration (language, task, etc.)
- status: Request status update

Future enhancements (Phase 3.3c):
- Full Redis integration
- Command validation
- Response publishing
- Error handling
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig:
    """Redis configuration"""

    url: str = "redis://localhost:6379"
    channel_prefix: str = "bot_commands"
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


class Command:
    """Command received from bot manager"""

    def __init__(self, action: str, data: dict[str, Any] | None = None):
        self.action = action
        self.data = data or {}

    @classmethod
    def from_json(cls, json_str: str) -> "Command":
        """Parse command from JSON string"""
        try:
            data = json.loads(json_str)
            return cls(action=data.get("action", "unknown"), data=data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse command JSON: {e}")
            return cls(action="invalid")

    def to_dict(self) -> dict[str, Any]:
        """Convert command to dictionary"""
        return {"action": self.action, **self.data}


class RedisSubscriber:
    """
    Redis pub/sub subscriber for bot commands

    Usage:
        subscriber = RedisSubscriber(config, connection_id)

        # Register command handler
        async def handle_command(command: Command):
            if command.action == "leave":
                await bot.shutdown()

        subscriber.on_command(handle_command)

        # Start listening
        await subscriber.start()
        # ... subscriber runs in background ...
        await subscriber.stop()
    """

    def __init__(self, config: RedisConfig | None = None, connection_id: str | None = None):
        self.config = config or RedisConfig()
        self.connection_id = connection_id

        # Redis connection state
        self.redis_client = None
        self.pubsub = None
        self.is_listening = False
        self.listener_task: asyncio.Task | None = None

        # Callbacks
        self.command_callback: Callable[[Command], None] | None = None
        self.error_callback: Callable[[str], None] | None = None

        # Statistics
        self.total_commands_received = 0
        self.reconnect_attempts = 0

    async def start(self) -> bool:
        """
        Start listening for commands

        Phase 3.3c will implement:
        - Connect to Redis server
        - Subscribe to bot command channel
        - Start background listener task

        Returns:
            bool: True if started successfully
        """
        if self.is_listening:
            logger.warning("Redis subscriber already listening")
            return True

        logger.info("Starting Redis subscriber (stub)")

        # TODO Phase 3.3c: Implement actual Redis connection
        # 1. Import redis.asyncio
        # 2. Connect to Redis:
        #    self.redis_client = redis.asyncio.from_url(self.config.url)
        # 3. Create pubsub:
        #    self.pubsub = self.redis_client.pubsub()
        # 4. Subscribe to channel:
        #    channel = f"{self.config.channel_prefix}:{self.connection_id}"
        #    await self.pubsub.subscribe(channel)
        # 5. Start listener task:
        #    self.listener_task = asyncio.create_task(self._listen_for_commands())

        self.is_listening = True
        self.listener_task = asyncio.create_task(self._simulate_commands())

        logger.info("✅ Redis subscriber started (stub)")
        return True

    async def stop(self):
        """
        Stop listening for commands

        Phase 3.3c will implement:
        - Unsubscribe from channel
        - Close Redis connection
        - Cancel listener task
        """
        if not self.is_listening:
            return

        logger.info("Stopping Redis subscriber (stub)")

        self.is_listening = False

        # Cancel listener task
        if self.listener_task:
            self.listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.listener_task

        # TODO Phase 3.3c: Implement actual cleanup
        # 1. Unsubscribe: await self.pubsub.unsubscribe()
        # 2. Close pubsub: await self.pubsub.close()
        # 3. Close Redis: await self.redis_client.close()

        logger.info("✅ Redis subscriber stopped (stub)")

    def on_command(self, callback: Callable[[Command], None]):
        """
        Register callback for commands

        Args:
            callback: Async function that receives Command objects
        """
        self.command_callback = callback

    def on_error(self, callback: Callable[[str], None]):
        """
        Register callback for errors

        Args:
            callback: Async function that receives error messages
        """
        self.error_callback = callback

    async def _listen_for_commands(self):
        """
        Background task to listen for commands

        Phase 3.3c will implement:
        - Listen for messages on subscribed channel
        - Parse command JSON
        - Invoke command callback
        - Handle errors and reconnection
        """
        try:
            logger.info("Redis listener started")

            while self.is_listening:
                try:
                    # TODO Phase 3.3c: Implement actual listening
                    # async for message in self.pubsub.listen():
                    #     if message['type'] == 'message':
                    #         command_json = message['data'].decode('utf-8')
                    #         command = Command.from_json(command_json)
                    #
                    #         logger.info(f"Received command: {command.action}")
                    #         self.total_commands_received += 1
                    #
                    #         if self.command_callback:
                    #             await self.command_callback(command)

                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Error in Redis listener: {e}")
                    if self.error_callback:
                        await self.error_callback(f"Listener error: {e}")

                    # Attempt reconnection
                    if self.reconnect_attempts < self.config.max_reconnect_attempts:
                        self.reconnect_attempts += 1
                        logger.info(f"Reconnecting... (attempt {self.reconnect_attempts})")
                        await asyncio.sleep(self.config.reconnect_delay)
                        # await self._reconnect()
                    else:
                        logger.error("Max reconnection attempts reached")
                        break

        except asyncio.CancelledError:
            logger.info("Redis listener cancelled")
        except Exception as e:
            logger.error(f"Redis listener failed: {e}")

    async def _simulate_commands(self):
        """
        Simulate commands for testing (stub implementation)

        This will be removed in Phase 3.3c when real Redis is implemented.
        """
        try:
            logger.info("Simulating Redis commands (stub mode)")

            # Simulate a few commands for testing
            await asyncio.sleep(30)  # Wait 30 seconds

            # Simulate status request
            if self.command_callback and self.is_listening:
                status_command = Command(action="status")
                logger.info("Simulated command: status")
                await self.command_callback(status_command)
                self.total_commands_received += 1

        except asyncio.CancelledError:
            logger.info("Redis simulation cancelled")
        except Exception as e:
            logger.error(f"Redis simulation failed: {e}")

    async def publish_response(self, response: dict[str, Any]):
        """
        Publish response to manager

        Phase 3.3c will implement:
        - Publish response to response channel
        - Include connection_id, status, data

        Args:
            response: Response dictionary to publish
        """
        # TODO Phase 3.3c: Implement response publishing
        # response_channel = f"bot_responses:{self.connection_id}"
        # response_json = json.dumps(response)
        # await self.redis_client.publish(response_channel, response_json)
        logger.debug(f"Would publish response: {response} (stub)")

    def get_stats(self) -> dict[str, Any]:
        """Get subscriber statistics"""
        return {
            "is_listening": self.is_listening,
            "total_commands_received": self.total_commands_received,
            "reconnect_attempts": self.reconnect_attempts,
            "connection_id": self.connection_id,
        }


# Example usage
async def example_usage():
    """Example of using Redis subscriber"""
    config = RedisConfig(url="redis://localhost:6379", channel_prefix="bot_commands")

    subscriber = RedisSubscriber(config, connection_id="test-bot-123")

    # Register command handler
    async def handle_command(command: Command):
        logger.info(f"Handling command: {command.action}")

        if command.action == "leave":
            logger.info("Received leave command - shutting down")
            # await bot.shutdown()

        elif command.action == "reconfigure":
            language = command.data.get("language")
            task = command.data.get("task")
            logger.info(f"Reconfiguring: language={language}, task={task}")
            # await bot.reconfigure(language=language, task=task)

        elif command.action == "status":
            logger.info("Received status request")
            # status = await bot.get_status()
            # await subscriber.publish_response(status)

    subscriber.on_command(handle_command)

    # Register error handler
    async def handle_error(error_msg: str):
        logger.error(f"Subscriber error: {error_msg}")

    subscriber.on_error(handle_error)

    try:
        # Start listening
        await subscriber.start()

        # Keep running
        await asyncio.sleep(60)

        # Stop listening
        await subscriber.stop()

        # Print stats
        stats = subscriber.get_stats()
        logger.info(f"Subscriber stats: {stats}")

    finally:
        await subscriber.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
