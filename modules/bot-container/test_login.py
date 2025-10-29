#!/usr/bin/env python3
"""
Quick test to verify Google login automation works
"""
import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from google_meet_automation import GoogleMeetAutomation, BrowserConfig


async def test_login():
    """Test Google login with actual credentials"""

    email = os.getenv("GOOGLE_EMAIL", "thomas.patane@xbanker.ai")
    password = os.getenv("GOOGLE_PASSWORD", "8$bbG@E#CKK4s9Qx47Ty5ed2wss3Yn3f")
    profile_path = "/tmp/test-bot-profile"

    print("=" * 70)
    print("🧪 Testing Google Account Login")
    print("=" * 70)
    print(f"Email: {email}")
    print(f"Password: {'*' * len(password)}")
    print(f"Profile: {profile_path}")
    print("=" * 70)
    print()

    # Create config
    config = BrowserConfig(
        headless=False,  # Show browser so we can see what happens
        user_data_dir=profile_path,
        google_email=email,
        google_password=password,
        screenshots_enabled=True,
        screenshots_path="/tmp/test-screenshots"
    )

    automation = GoogleMeetAutomation(config)

    try:
        # Initialize browser
        print("🚀 Initializing browser...")
        await automation.initialize()
        print("✅ Browser initialized")
        print()

        # Try to login
        print("🔐 Attempting to login to Google...")
        success = await automation.login_google_account()

        if success:
            print()
            print("=" * 70)
            print("✅ SUCCESS! Login completed")
            print("=" * 70)
            print()

            # Check if state was saved
            state_file = f"{profile_path}/state.json"
            if os.path.exists(state_file):
                size = os.path.getsize(state_file)
                print(f"✅ Browser state saved: {state_file} ({size} bytes)")
            else:
                print("⚠️  Warning: State file not found")
        else:
            print()
            print("=" * 70)
            print("❌ FAILED: Login did not complete")
            print("=" * 70)
            return False

        # Keep browser open for inspection
        print()
        print("Browser will stay open for 10 seconds for inspection...")
        await asyncio.sleep(10)

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
    success = asyncio.run(test_login())
    sys.exit(0 if success else 1)
