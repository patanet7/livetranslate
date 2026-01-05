#!/usr/bin/env python3
"""
Simple test - just navigate and wait with browser visible
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from google_meet_automation import GoogleMeetAutomation, BrowserConfig


async def test_simple():
    """Simple test with manual observation"""

    meeting_url = "https://meet.google.com/oss-kqzr-ztg"
    profile_path = "/tmp/test-bot-profile-meeting"

    config = BrowserConfig(
        headless=False,
        user_data_dir=profile_path,
        google_email="thomas.patane@xbanker.ai",
        google_password="8$bbG@E#CKK4s9Qx47Ty5ed2wss3Yn3f",
        screenshots_enabled=True,
        screenshots_path="/tmp/test-screenshots-meeting"
    )

    automation = GoogleMeetAutomation(config)

    try:
        await automation.initialize()
        print("✅ Browser initialized")

        # Login first
        print("🔐 Logging in...")
        await automation.login_google_account()
        print("✅ Logged in")

        # Navigate to meeting
        print(f"🚪 Navigating to {meeting_url}")
        await automation.page.goto(meeting_url, wait_until='networkidle')
        print("✅ Page loaded")

        # Wait a bit for page to settle
        await asyncio.sleep(3)

        # Take screenshot
        await automation.page.screenshot(path="/tmp/test-screenshots-meeting/before-click.png")
        print("📸 Screenshot saved")

        # Try to find and click join button
        print("\n🔍 Looking for join button...")

        # List all buttons on the page
        buttons = await automation.page.query_selector_all('button')
        print(f"Found {len(buttons)} buttons on page")

        for i, button in enumerate(buttons):
            text = await button.inner_text()
            print(f"  Button {i}: '{text}'")

        # Try clicking by text
        print("\n🔘 Attempting to click 'Join now' button...")
        try:
            join_button = automation.page.get_by_text("Join now", exact=False)
            await join_button.click(timeout=5000)
            print("✅ Clicked join button!")
        except Exception as e:
            print(f"❌ Failed to click: {e}")

        # Keep browser open
        print("\n⏸️  Browser will stay open for 2 minutes...")
        await asyncio.sleep(120)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up...")
        await automation.cleanup()


if __name__ == "__main__":
    asyncio.run(test_simple())
