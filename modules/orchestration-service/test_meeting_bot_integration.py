#!/usr/bin/env python3
"""
End-to-end test for Meeting Bot Service integration.

This test verifies that the Python orchestration service can successfully
call the Node.js meeting-bot-service to join Google Meet meetings.

Prerequisites:
1. Start the meeting-bot-service:
   cd modules/meeting-bot-service && npm run api

2. Run this test:
   python test_meeting_bot_integration.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from clients.meeting_bot_service_client import MeetingBotServiceClient


async def test_integration():
    """Test the full integration between Python and Node.js bot service."""

    print("=" * 70)
    print("🧪 Testing Meeting Bot Service Integration")
    print("=" * 70)
    print()

    # Initialize client
    client = MeetingBotServiceClient(base_url="http://localhost:5005")

    try:
        # Step 1: Health check
        print("1️⃣  Health Check")
        print("-" * 70)
        health = await client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Service: {health['service']}")
        print(f"   Active Bots: {health['activeBots']}")
        print(f"   ✅ Meeting bot service is healthy!")
        print()

        # Step 2: Join a meeting
        print("2️⃣  Join Meeting")
        print("-" * 70)

        meeting_url = "https://meet.google.com/oss-kqzr-ztg"
        bot_name = "LiveTranslate-Integration-Test"
        bot_id = "integration-test-bot-001"
        user_id = "test-user-001"

        print(f"   Meeting URL: {meeting_url}")
        print(f"   Bot Name: {bot_name}")
        print(f"   Bot ID: {bot_id}")
        print()

        response = await client.join_meeting(
            meeting_url=meeting_url,
            bot_name=bot_name,
            bot_id=bot_id,
            user_id=user_id,
            team_id="livetranslate-test",
            timezone="America/Los_Angeles"
        )

        print(f"   Success: {response.success}")
        print(f"   Bot ID: {response.botId}")
        print(f"   Correlation ID: {response.correlationId}")
        print(f"   Message: {response.message}")

        if response.success:
            print(f"   ✅ Bot join request successful!")
        else:
            print(f"   ❌ Bot join request failed: {response.error}")
            return False

        print()

        # Step 3: Check bot status
        print("3️⃣  Check Bot Status (after 5 seconds)")
        print("-" * 70)
        await asyncio.sleep(5)

        status = await client.get_bot_status(bot_id)
        print(f"   Success: {status.success}")
        print(f"   Bot ID: {status.botId}")
        print(f"   State: {status.state}")

        if status.success:
            print(f"   ✅ Bot status retrieved!")
        else:
            print(f"   ℹ️  Note: {status.error}")

        print()

        # Step 4: Keep bot running for observation
        print("4️⃣  Bot is running...")
        print("-" * 70)
        print("   The bot should now be visible in your Google Meet!")
        print("   Check your meeting to verify the bot joined successfully.")
        print()
        print("   Browser will stay open for 30 seconds for observation...")
        await asyncio.sleep(30)

        # Step 5: Leave meeting
        print()
        print("5️⃣  Leave Meeting")
        print("-" * 70)

        leave_result = await client.leave_meeting(bot_id)
        print(f"   Success: {leave_result.get('success')}")
        print(f"   Message: {leave_result.get('message')}")

        if leave_result.get('success'):
            print(f"   ✅ Bot left meeting successfully!")
        else:
            print(f"   ℹ️  Note: {leave_result.get('error')}")

        print()

        # Final summary
        print("=" * 70)
        print("✅ INTEGRATION TEST COMPLETE!")
        print("=" * 70)
        print()
        print("Summary:")
        print("  ✅ Python client can communicate with Node.js service")
        print("  ✅ Meeting bot service is operational")
        print("  ✅ Bot join flow executed successfully")
        print("  ✅ ScreenApp's battle-tested bot logic is integrated!")
        print()
        print("Next Steps:")
        print("  1. Wire this into the orchestration service bot manager")
        print("  2. Connect bot audio to whisper service")
        print("  3. Enable real-time transcription and translation")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ INTEGRATION TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure meeting-bot-service is running:")
        print("     cd modules/meeting-bot-service && npm run api")
        print("  2. Check that port 5005 is not blocked")
        print("  3. Verify Chrome is installed at the expected path")
        print()

        import traceback
        traceback.print_exc()

        return False


if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
