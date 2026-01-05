#!/usr/bin/env python3
"""
Test script to join a Google Meet meeting with visible browser
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from google_meet_automation import GoogleMeetAutomation, BrowserConfig


async def test_join_meeting():
    """Test joining a real Google Meet meeting"""

    email = os.getenv("GOOGLE_EMAIL", "thomas.patane@xbanker.ai")
    password = os.getenv("GOOGLE_PASSWORD", "8$bbG@E#CKK4s9Qx47Ty5ed2wss3Yn3f")
    meeting_url = "https://meet.google.com/oss-kqzr-ztg"
    profile_path = "/tmp/test-bot-profile-meeting"

    print("=" * 70)
    print("🧪 Testing Google Meet Join")
    print("=" * 70)
    print(f"Email: {email}")
    print(f"Password: {'*' * len(password)}")
    print(f"Meeting: {meeting_url}")
    print(f"Profile: {profile_path}")
    print("=" * 70)
    print()

    # Create config with browser VISIBLE
    config = BrowserConfig(
        headless=False,  # VISIBLE browser
        user_data_dir=profile_path,
        google_email=email,
        google_password=password,
        screenshots_enabled=True,
        screenshots_path="/tmp/test-screenshots-meeting"
    )

    automation = GoogleMeetAutomation(config)

    try:
        # Initialize browser
        print("🚀 Initializing browser...")
        await automation.initialize()
        print("✅ Browser initialized")
        print()

        # Join meeting
        print(f"🚪 Joining meeting: {meeting_url}")
        bot_name = "LiveTranslate-TestBot"
        success = await automation.join_meeting(meeting_url, bot_name)

        if success:
            print()
            print("=" * 70)
            print("✅ Successfully joined meeting!")
            print("=" * 70)
            print(f"Meeting state: {automation.get_state().value}")
            print()

            # Keep browser open to watch
            print("Browser will stay open for 2 minutes...")
            print("You should see the bot in your meeting!")
            await asyncio.sleep(120)
        else:
            print()
            print("=" * 70)
            print("❌ Failed to join meeting")
            print("=" * 70)
            return False

        return True

    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False

    finally:
        print()
        print("🧹 Cleaning up...")
        await automation.cleanup()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    success = asyncio.run(test_join_meeting())
    sys.exit(0 if success else 1)
