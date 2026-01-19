#!/usr/bin/env python3
"""
Manual Login Helper - One-Time Setup

This script opens a REAL browser window where you can:
1. Manually login to Google (handle 2FA, captcha, etc.)
2. Complete any security checks
3. Save the authenticated state

After this, all future bot runs will use the saved state - NO MORE LOGIN NEEDED!

Usage:
    python manual_login_helper.py --email your@gmail.com --profile-path /app/browser-profile
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from google_meet_automation import BrowserConfig, GoogleMeetAutomation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def manual_login_setup(email: str, profile_path: str):
    """
    Open browser in non-headless mode for manual Google login

    Args:
        email: Google email to login with
        profile_path: Path to save browser profile
    """
    print("=" * 70)
    print("🔐 Google Meet Bot - Manual Login Setup")
    print("=" * 70)
    print()
    print(f"Email: {email}")
    print(f"Profile Path: {profile_path}")
    print()
    print("INSTRUCTIONS:")
    print("1. A browser window will open")
    print("2. Login to Google with your credentials")
    print("3. Complete any 2FA/security checks")
    print("4. Once you see the Google homepage, press ENTER here")
    print("5. The authenticated state will be saved")
    print()
    print("After this setup, bots can join meetings without logging in!")
    print("=" * 70)
    input("Press ENTER to open browser...")

    # Create profile directory
    os.makedirs(profile_path, exist_ok=True)

    # Configure browser in NON-HEADLESS mode
    config = BrowserConfig(
        headless=False,  # Show browser window
        user_data_dir=profile_path,
        screenshots_enabled=True,
        screenshots_path="/tmp/bot-screenshots",
    )

    automation = GoogleMeetAutomation(config)

    try:
        # Initialize browser
        logger.info("🚀 Opening browser...")
        await automation.initialize()

        # Navigate to Google login
        logger.info("📧 Navigating to Google login page...")
        await automation.page.goto(
            "https://accounts.google.com/ServiceLogin?hl=en&continue=https://www.google.com/"
        )

        # Pre-fill email if provided
        try:
            await automation.page.wait_for_selector("#identifierId", timeout=5000)
            await automation.page.fill("#identifierId", email)
            logger.info(f"✅ Pre-filled email: {email}")
        except Exception:
            logger.info("⚠️  Could not pre-fill email, please enter manually")

        print()
        print("=" * 70)
        print("👁️  BROWSER WINDOW IS OPEN")
        print("=" * 70)
        print()
        print("Please complete the login in the browser window:")
        print("  1. Enter your password")
        print("  2. Complete 2FA if prompted")
        print("  3. Handle any security checks")
        print("  4. Wait until you see the Google homepage")
        print()
        print("Once you're logged in and see google.com,")
        input("press ENTER here to save the state...")

        # Save the authenticated state
        logger.info("💾 Saving authenticated browser state...")
        await automation._save_browser_state()

        state_file = f"{profile_path}/state.json"
        if os.path.exists(state_file):
            size = os.path.getsize(state_file)
            logger.info(f"✅ State saved successfully! ({size} bytes)")
            print()
            print("=" * 70)
            print("✅ SUCCESS! Authentication state has been saved.")
            print("=" * 70)
            print()
            print(f"Saved to: {state_file}")
            print()
            print("You can now use the bot without providing credentials!")
            print()
            print("Example API call:")
            print(f"""
curl -X POST http://localhost:3000/api/bots/start \\
  -H "Content-Type: application/json" \\
  -d '{{
    "meeting_url": "https://meet.google.com/xxx-xxxx-xxx",
    "user_token": "...",
    "user_id": "...",
    "user_data_dir": "{profile_path}"
  }}'
            """)
        else:
            logger.error("❌ Failed to save state file")
            return False

        # Keep browser open for a moment
        logger.info("Keeping browser open for 5 seconds...")
        await asyncio.sleep(5)

        return True

    except Exception as e:
        logger.error(f"❌ Setup failed: {e}", exc_info=True)
        return False

    finally:
        # Cleanup
        logger.info("🧹 Closing browser...")
        await automation.cleanup()


async def verify_saved_state(profile_path: str):
    """Verify that the saved state works"""
    print()
    print("=" * 70)
    print("🔍 Verifying saved authentication state...")
    print("=" * 70)

    state_file = f"{profile_path}/state.json"
    if not os.path.exists(state_file):
        logger.error(f"❌ State file not found: {state_file}")
        return False

    config = BrowserConfig(headless=False, user_data_dir=profile_path)

    automation = GoogleMeetAutomation(config)

    try:
        await automation.initialize()

        # Try to access Google - should already be logged in
        logger.info("🔑 Testing saved authentication...")
        await automation.page.goto("https://www.google.com/")
        await asyncio.sleep(3)

        # Check if we're logged in by looking for account button
        try:
            # Look for signs of being logged in
            await automation.page.wait_for_selector('[aria-label*="Google Account"]', timeout=5000)
            logger.info("✅ Successfully authenticated! Bot is logged in.")
            print()
            print("=" * 70)
            print("✅ VERIFICATION SUCCESSFUL")
            print("=" * 70)
            print()
            print("The bot can now join Google Meet without any credentials!")
            return True
        except Exception:
            logger.warning("⚠️  Could not verify login state")
            return False

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

    finally:
        await automation.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Manual login setup for Google Meet bot")
    parser.add_argument("--email", required=True, help="Google email address")
    parser.add_argument(
        "--profile-path",
        default="/tmp/bot-browser-profile",
        help="Path to save browser profile (default: /tmp/bot-browser-profile)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing saved state"
    )

    args = parser.parse_args()

    if args.verify_only:
        success = asyncio.run(verify_saved_state(args.profile_path))
    else:
        success = asyncio.run(manual_login_setup(args.email, args.profile_path))

        if success:
            # Optionally verify
            print()
            verify = input("Would you like to verify the saved state? (y/n): ")
            if verify.lower() == "y":
                asyncio.run(verify_saved_state(args.profile_path))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
