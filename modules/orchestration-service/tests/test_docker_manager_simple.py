#!/usr/bin/env python3
"""
Simple Direct Tests for Docker Bot Manager

Tests without pytest to avoid import issues.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.docker_bot_manager import DockerBotManager, BotStatus


async def test_manager_creation():
    """Test manager can be created"""
    print("Test 1: Manager creation")
    manager = DockerBotManager(enable_database=False)
    assert len(manager.bots) == 0
    print("✅ PASS")


async def test_start_bot():
    """Test starting a bot"""
    print("\nTest 2: Start bot")
    manager = DockerBotManager(enable_database=False)
    manager.docker_client = None  # Mock mode

    connection_id = await manager.start_bot(
        meeting_url="https://meet.google.com/test",
        user_token="token-123",
        user_id="user-456",
    )

    assert connection_id in manager.bots
    assert manager.total_bots_started == 1
    bot = manager.bots[connection_id]
    assert bot.status == BotStatus.SPAWNING
    assert bot.user_id == "user-456"
    print(f"✅ PASS - Bot ID: {connection_id}")


async def test_callbacks():
    """Test bot callbacks"""
    print("\nTest 3: Bot callbacks")
    manager = DockerBotManager(enable_database=False)
    manager.docker_client = None

    # Start bot
    connection_id = await manager.start_bot(
        "https://meet.google.com/test", "token", "user-123"
    )

    # Simulate callback progression
    await manager.handle_bot_callback(connection_id, "started", {})
    assert manager.bots[connection_id].status == BotStatus.STARTING
    print("  ✓ Started callback")

    await manager.handle_bot_callback(connection_id, "joining", {})
    assert manager.bots[connection_id].status == BotStatus.JOINING
    print("  ✓ Joining callback")

    await manager.handle_bot_callback(connection_id, "active", {})
    assert manager.bots[connection_id].status == BotStatus.ACTIVE
    print("  ✓ Active callback")

    await manager.handle_bot_callback(connection_id, "completed", {})
    assert manager.bots[connection_id].status == BotStatus.COMPLETED
    assert manager.total_bots_completed == 1
    print("  ✓ Completed callback")

    print("✅ PASS - All callbacks handled correctly")


async def test_list_bots():
    """Test listing bots"""
    print("\nTest 4: List bots")
    manager = DockerBotManager(enable_database=False)
    manager.docker_client = None

    # Start multiple bots
    await manager.start_bot("https://meet.google.com/test1", "token", "user-1")
    await manager.start_bot("https://meet.google.com/test2", "token", "user-2")
    await manager.start_bot("https://meet.google.com/test3", "token", "user-3")

    # List all
    all_bots = manager.list_bots()
    assert len(all_bots) == 3
    print("  ✓ List all bots: 3 bots")

    # Filter by user
    user2_bots = manager.list_bots(user_id="user-2")
    assert len(user2_bots) == 1
    print("  ✓ Filter by user_id: 1 bot")

    print("✅ PASS")


async def test_stats():
    """Test statistics"""
    print("\nTest 5: Statistics")
    manager = DockerBotManager(enable_database=False)
    manager.docker_client = None

    # Start and complete some bots
    id1 = await manager.start_bot("https://meet.google.com/test1", "token", "user-1")
    id2 = await manager.start_bot("https://meet.google.com/test2", "token", "user-2")

    await manager.handle_bot_callback(id1, "completed", {})
    await manager.handle_bot_callback(id2, "failed", {"error": "test"})

    stats = manager.get_stats()
    assert stats["total_started"] == 2
    assert stats["total_completed"] == 1
    assert stats["total_failed"] == 1
    assert stats["success_rate"] == 0.5

    print(
        f"  ✓ Stats: {stats['total_started']} started, {stats['total_completed']} completed, {stats['total_failed']} failed"
    )
    print("✅ PASS")


async def test_health_check():
    """Test bot health checks"""
    print("\nTest 6: Bot health checks")
    manager = DockerBotManager(enable_database=False)
    manager.docker_client = None

    # Start and activate bot
    connection_id = await manager.start_bot(
        "https://meet.google.com/test", "token", "user-123"
    )
    await manager.handle_bot_callback(connection_id, "active", {})

    bot = manager.get_bot(connection_id)
    assert bot.is_healthy is True
    print("  ✓ Active bot is healthy")

    # Fail bot
    await manager.handle_bot_callback(connection_id, "failed", {"error": "test"})
    assert bot.is_healthy is False
    print("  ✓ Failed bot is unhealthy")

    print("✅ PASS")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Docker Bot Manager - Direct Tests")
    print("=" * 60)

    tests = [
        test_manager_creation,
        test_start_bot,
        test_callbacks,
        test_list_bots,
        test_stats,
        test_health_check,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL - {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
