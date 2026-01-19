#!/usr/bin/env python3
"""
TDD Tests for Docker Bot Manager

Tests the simplified Docker-based bot management system.

Following TDD: Write tests FIRST, then implement!
"""

import asyncio

# Add src to path
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDockerBotManagerInitialization:
    """Test Docker bot manager initialization"""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager can be initialized"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(
            orchestration_url="http://localhost:3000",
            redis_url="redis://localhost:6379",
            enable_database=False,  # Disable for testing
        )

        assert manager.orchestration_url == "http://localhost:3000"
        assert manager.redis_url == "redis://localhost:6379"
        assert len(manager.bots) == 0

    @pytest.mark.asyncio
    async def test_manager_initialize_without_docker(self):
        """Test manager initializes in mock mode without Docker"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)

        # Should initialize without error even if Docker not available
        await manager.initialize()

        # Should have attempted initialization
        assert manager is not None


class TestDockerBotManagerStartBot:
    """Test starting bot containers"""

    @pytest.mark.asyncio
    async def test_start_bot_creates_bot_instance(self):
        """Test starting a bot creates a bot instance"""
        from bot.docker_bot_manager import BotStatus, DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None  # Mock mode

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token-123",
            user_id="user-456",
            language="en",
            task="transcribe",
        )

        # Verify bot created
        assert connection_id in manager.bots
        bot = manager.bots[connection_id]
        assert bot.user_id == "user-456"
        assert bot.meeting_url == "https://meet.google.com/test"
        assert bot.status == BotStatus.SPAWNING
        assert manager.total_bots_started == 1

    @pytest.mark.asyncio
    async def test_start_bot_generates_unique_connection_id(self):
        """Test each bot gets unique connection ID"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None  # Mock mode

        # Start multiple bots
        id1 = await manager.start_bot(
            meeting_url="https://meet.google.com/test1",
            user_token="token",
            user_id="user-1",
        )

        id2 = await manager.start_bot(
            meeting_url="https://meet.google.com/test2",
            user_token="token",
            user_id="user-2",
        )

        # IDs should be unique
        assert id1 != id2
        assert len(manager.bots) == 2


class TestDockerBotManagerCallbacks:
    """Test bot callback handling"""

    @pytest.mark.asyncio
    async def test_handle_started_callback(self):
        """Test handling 'started' callback from bot"""
        from bot.docker_bot_manager import BotStatus, DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Handle started callback
        await manager.handle_bot_callback(connection_id, "started", {"container_id": "abc123"})

        # Verify status updated
        bot = manager.bots[connection_id]
        assert bot.status == BotStatus.STARTING
        assert bot.started_at is not None
        assert bot.last_callback is not None

    @pytest.mark.asyncio
    async def test_handle_active_callback(self):
        """Test handling 'active' callback from bot"""
        from bot.docker_bot_manager import BotStatus, DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Simulate progression: started → joining → active
        await manager.handle_bot_callback(connection_id, "started", {})
        await manager.handle_bot_callback(connection_id, "joining", {})
        await manager.handle_bot_callback(connection_id, "active", {})

        # Verify status
        bot = manager.bots[connection_id]
        assert bot.status == BotStatus.ACTIVE
        assert bot.active_at is not None

    @pytest.mark.asyncio
    async def test_handle_completed_callback(self):
        """Test handling 'completed' callback from bot"""
        from bot.docker_bot_manager import BotStatus, DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Simulate completion
        await manager.handle_bot_callback(connection_id, "started", {})
        await manager.handle_bot_callback(connection_id, "active", {})
        await manager.handle_bot_callback(connection_id, "completed", {})

        # Verify status
        bot = manager.bots[connection_id]
        assert bot.status == BotStatus.COMPLETED
        assert bot.stopped_at is not None
        assert manager.total_bots_completed == 1

    @pytest.mark.asyncio
    async def test_handle_failed_callback(self):
        """Test handling 'failed' callback from bot"""
        from bot.docker_bot_manager import BotStatus, DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Simulate failure
        await manager.handle_bot_callback(
            connection_id, "failed", {"error": "Failed to join meeting", "exit_code": 1}
        )

        # Verify status
        bot = manager.bots[connection_id]
        assert bot.status == BotStatus.FAILED
        assert bot.error_message == "Failed to join meeting"
        assert bot.exit_code == 1
        assert manager.total_bots_failed == 1


class TestDockerBotManagerStopBot:
    """Test stopping bots"""

    @pytest.mark.asyncio
    async def test_stop_bot_sends_redis_command(self):
        """Test stopping bot sends Redis leave command"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Mock Redis client
        manager.redis_client = AsyncMock()

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Stop bot
        await manager.stop_bot(connection_id, timeout=5)

        # Verify Redis command sent
        manager.redis_client.publish.assert_called_once()
        call_args = manager.redis_client.publish.call_args
        channel = call_args[0][0]
        assert channel == f"bot_commands:{connection_id}"

        # Verify command contains leave action
        command_json = call_args[0][1]
        import json

        command = json.loads(command_json)
        assert command["action"] == "leave"

    @pytest.mark.asyncio
    async def test_stop_unknown_bot_raises_error(self):
        """Test stopping unknown bot raises ValueError"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)

        # Try to stop non-existent bot
        with pytest.raises(ValueError, match="Bot not found"):
            await manager.stop_bot("unknown-bot-id")


class TestDockerBotManagerCommands:
    """Test sending commands to bots"""

    @pytest.mark.asyncio
    async def test_send_command_publishes_to_redis(self):
        """Test send_command publishes to correct Redis channel"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)

        # Mock Redis client
        manager.redis_client = AsyncMock()

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Send reconfigure command
        await manager.send_command(
            connection_id,
            {"action": "reconfigure", "language": "es", "task": "translate"},
        )

        # Verify Redis publish called
        manager.redis_client.publish.assert_called_once()
        channel = manager.redis_client.publish.call_args[0][0]
        assert channel == f"bot_commands:{connection_id}"

    @pytest.mark.asyncio
    async def test_send_command_without_redis_raises_error(self):
        """Test send_command without Redis raises RuntimeError"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.redis_client = None  # No Redis

        # Try to send command
        with pytest.raises(RuntimeError, match="Redis client not available"):
            await manager.send_command("bot-123", {"action": "leave"})


class TestDockerBotManagerQueries:
    """Test querying bots"""

    @pytest.mark.asyncio
    async def test_get_bot_returns_bot_instance(self):
        """Test get_bot returns correct bot instance"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bot
        connection_id = await manager.start_bot(
            meeting_url="https://meet.google.com/test",
            user_token="token",
            user_id="user-123",
        )

        # Get bot
        bot = manager.get_bot(connection_id)
        assert bot is not None
        assert bot.connection_id == connection_id
        assert bot.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_get_bot_returns_none_for_unknown(self):
        """Test get_bot returns None for unknown bot"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)

        # Get unknown bot
        bot = manager.get_bot("unknown-id")
        assert bot is None

    @pytest.mark.asyncio
    async def test_list_bots_returns_all_bots(self):
        """Test list_bots returns all bots"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start multiple bots
        await manager.start_bot("https://meet.google.com/test1", "token", "user-1")
        await manager.start_bot("https://meet.google.com/test2", "token", "user-2")
        await manager.start_bot("https://meet.google.com/test3", "token", "user-3")

        # List all bots
        bots = manager.list_bots()
        assert len(bots) == 3

    @pytest.mark.asyncio
    async def test_list_bots_filters_by_status(self):
        """Test list_bots can filter by status"""
        from bot.docker_bot_manager import BotStatus, DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bots
        id1 = await manager.start_bot("https://meet.google.com/test1", "token", "user-1")
        id2 = await manager.start_bot("https://meet.google.com/test2", "token", "user-2")
        await manager.start_bot("https://meet.google.com/test3", "token", "user-3")

        # Make some active
        await manager.handle_bot_callback(id1, "active", {})
        await manager.handle_bot_callback(id2, "active", {})

        # List only active bots
        active_bots = manager.list_bots(status=BotStatus.ACTIVE)
        assert len(active_bots) == 2

        # List only spawning bots
        spawning_bots = manager.list_bots(status=BotStatus.SPAWNING)
        assert len(spawning_bots) == 1

    @pytest.mark.asyncio
    async def test_list_bots_filters_by_user_id(self):
        """Test list_bots can filter by user_id"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bots for different users
        await manager.start_bot("https://meet.google.com/test1", "token", "user-alice")
        await manager.start_bot("https://meet.google.com/test2", "token", "user-bob")
        await manager.start_bot("https://meet.google.com/test3", "token", "user-alice")

        # List bots for alice
        alice_bots = manager.list_bots(user_id="user-alice")
        assert len(alice_bots) == 2

        # List bots for bob
        bob_bots = manager.list_bots(user_id="user-bob")
        assert len(bob_bots) == 1


class TestDockerBotManagerStatistics:
    """Test manager statistics"""

    @pytest.mark.asyncio
    async def test_get_stats_returns_statistics(self):
        """Test get_stats returns correct statistics"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bots
        id1 = await manager.start_bot("https://meet.google.com/test1", "token", "user-1")
        id2 = await manager.start_bot("https://meet.google.com/test2", "token", "user-2")
        id3 = await manager.start_bot("https://meet.google.com/test3", "token", "user-3")

        # Simulate lifecycle
        await manager.handle_bot_callback(id1, "active", {})
        await manager.handle_bot_callback(id2, "active", {})
        await manager.handle_bot_callback(id1, "completed", {})
        await manager.handle_bot_callback(id3, "failed", {"error": "test"})

        # Get stats
        stats = manager.get_stats()

        assert stats["total_bots"] == 3
        assert stats["total_started"] == 3
        assert stats["total_completed"] == 1
        assert stats["total_failed"] == 1
        assert stats["active_bots"] == 1  # Only id2 still active
        assert stats["success_rate"] == 1 / 3  # 1 completed out of 3 started


class TestBotInstanceHealthChecks:
    """Test bot instance health monitoring"""

    @pytest.mark.asyncio
    async def test_bot_is_healthy_when_active(self):
        """Test bot is healthy when active with recent callback"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start and activate bot
        connection_id = await manager.start_bot("https://meet.google.com/test", "token", "user-123")
        await manager.handle_bot_callback(connection_id, "active", {})

        bot = manager.get_bot(connection_id)
        assert bot.is_healthy is True

    @pytest.mark.asyncio
    async def test_bot_is_unhealthy_when_failed(self):
        """Test bot is unhealthy when in failed state"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start and fail bot
        connection_id = await manager.start_bot("https://meet.google.com/test", "token", "user-123")
        await manager.handle_bot_callback(connection_id, "failed", {"error": "test"})

        bot = manager.get_bot(connection_id)
        assert bot.is_healthy is False

    @pytest.mark.asyncio
    async def test_bot_uptime_calculation(self):
        """Test bot uptime is calculated correctly"""
        from bot.docker_bot_manager import DockerBotManager

        manager = DockerBotManager(enable_database=False)
        manager.docker_client = None

        # Start bot
        connection_id = await manager.start_bot("https://meet.google.com/test", "token", "user-123")

        # Simulate started
        await manager.handle_bot_callback(connection_id, "started", {})

        # Wait a bit
        await asyncio.sleep(0.1)

        bot = manager.get_bot(connection_id)
        assert bot.uptime_seconds > 0
        assert bot.uptime_seconds < 1  # Should be less than 1 second


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
