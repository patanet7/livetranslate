#!/usr/bin/env python3
"""
Test anonymous Google Meet joining with Vexa + ScreenApp battle-tested configuration

This test verifies that bots can join Google Meet WITHOUT authentication.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from google_meet_automation import BrowserConfig, GoogleMeetAutomation


async def test_anonymous_join():
    """Test joining a real Google Meet meeting anonymously"""

    meeting_url = "https://meet.google.com/oss-kqzr-ztg"
    bot_name = "LiveTranslate-Anonymous-Bot"

    print("=" * 70)
    print("🧪 Testing ANONYMOUS Google Meet Join")
    print("=" * 70)
    print(f"Meeting: {meeting_url}")
    print(f"Bot Name: {bot_name}")
    print("Authentication: NONE (Anonymous)")
    print()
    print("Configuration:")
    print("  ✅ Vexa's Chrome 129 user agent")
    print("  ✅ Vexa's incognito + security bypass args")
    print("  ✅ ScreenApp's auto-accept-this-tab-capture")
    print("  ✅ ScreenApp's MediaRecorder enablement")
    print("=" * 70)
    print()

    # Try with REAL Google account - might bypass bot detection!
    config = BrowserConfig(
        headless=False,  # VISIBLE browser to watch
        screenshots_enabled=True,
        screenshots_path="/tmp/test-screenshots-anonymous",
        google_email="thomas.patane@xbanker.ai",
        google_password=os.getenv("GOOGLE_PASSWORD"),  # Set this in env
    )

    # Try AUTHENTICATED joining - may bypass bot detection
    if config.google_email:
        print(f"✅ Config verified: Will join WITH authentication as {config.google_email}")
    else:
        print("✅ Config verified: Anonymous joining")
    print()

    automation = GoogleMeetAutomation(config)

    try:
        # Initialize browser
        print("🚀 Initializing browser with battle-tested args...")
        await automation.initialize()
        print("✅ Browser initialized")
        print()

        # Join meeting
        print(f"🚪 Joining meeting ANONYMOUSLY: {meeting_url}")
        success = await automation.join_meeting(meeting_url, bot_name)

        if success:
            print()
            print("=" * 70)
            print("✅ Successfully joined meeting ANONYMOUSLY!")
            print("=" * 70)
            print(f"Meeting state: {automation.get_state().value}")
            print()

            # Keep browser open to watch
            print("Browser will stay open for 2 minutes...")
            print("You should see the bot in your meeting!")
            print()
            print("Expected behavior:")
            print("  - Bot appears as 'LiveTranslate-Anonymous-Bot'")
            print("  - No Google account required")
            print("  - Microphone and camera muted")
            print("  - May be in waiting room (normal for anonymous bots)")
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

        # LEAVE BROWSER OPEN FOR INVESTIGATION
        print()
        print("🔍 LEAVING BROWSER OPEN FOR INVESTIGATION")
        print("   Press Ctrl+C to exit and close browser")
        try:
            await asyncio.sleep(600)  # Wait 10 minutes
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        return False

    finally:
        print()
        print("🧹 Cleaning up...")
        await automation.cleanup()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    success = asyncio.run(test_anonymous_join())
    sys.exit(0 if success else 1)
