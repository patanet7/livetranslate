#!/usr/bin/env python3
"""
Google Meet Browser Automation - Production Implementation

Based on Vexa's proven Google Meet automation (TypeScript → Python)
Reference: reference/vexa/services/vexa-bot/core/src/platforms/googlemeet/

This implementation uses Playwright for reliable browser automation and includes:
- Google Meet joining with retry logic
- Audio/video muting
- Participant detection
- Virtual webcam support
- Screenshot debugging
"""

import asyncio
import logging
import random
import os
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from playwright.async_api import async_playwright, Browser, Page, BrowserContext, Playwright
from undetected_playwright import Malenia

logger = logging.getLogger(__name__)


class MeetingState(Enum):
    """Google Meet session states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    JOINING = "joining"
    WAITING_ROOM = "waiting_room"
    JOINED = "joined"
    ACTIVE = "active"
    ERROR = "error"
    LEAVING = "leaving"


@dataclass
class BrowserConfig:
    """Browser automation configuration"""
    headless: bool = True
    audio_capture_enabled: bool = True
    video_enabled: bool = False
    microphone_enabled: bool = False
    join_timeout: int = 120
    window_size: tuple = (1280, 720)  # Match ScreenApp's size
    user_agent: Optional[str] = None
    screenshots_enabled: bool = True
    screenshots_path: str = "/tmp/bot-screenshots"

    # DEPRECATED: Authentication no longer used - bots join anonymously
    # Keeping for backward compatibility but will be ignored
    user_data_dir: Optional[str] = None
    google_email: Optional[str] = None
    google_password: Optional[str] = None

    # NEW: Bearer token for API callbacks (ScreenApp/Vexa pattern)
    bearer_token: Optional[str] = None
    callback_url: Optional[str] = None


class GoogleMeetSelectors:
    """Google Meet DOM selectors (from Vexa reference)"""

    # Name input
    NAME_INPUT = [
        'input[type="text"][aria-label="Your name"]',
        'input[placeholder*="name"]',
        'input[placeholder*="Name"]'
    ]

    # Join buttons
    JOIN_BUTTON = [
        '//button[.//span[text()="Ask to join"]]',
        'button:has-text("Ask to join")',
        'button:has-text("Join now")',
        'button:has-text("Join")'
    ]

    # Microphone toggle
    MICROPHONE_BUTTON = [
        '[aria-label*="Turn off microphone"]',
        'button[aria-label*="Turn off microphone"]',
        'button[aria-label*="Turn on microphone"]'
    ]

    # Camera toggle
    CAMERA_BUTTON = [
        '[aria-label*="Turn off camera"]',
        'button[aria-label*="Turn off camera"]',
        'button[aria-label*="Turn on camera"]'
    ]

    # Meeting joined indicators
    ADMISSION_INDICATORS = [
        'button[aria-label*="Chat"]',
        'button[aria-label*="People"]',
        'button[aria-label*="Leave call"]',
        '[role="toolbar"]',
        'button[aria-label*="Turn off microphone"]',
        'button[aria-label*="Turn on microphone"]'
    ]

    # Waiting room indicators
    WAITING_ROOM_INDICATORS = [
        'text="Asking to be let in..."',
        'text*="Asking to be let in"',
        'text="You\'ll join the call when someone lets you in"',
        'text*="You\'ll join the call when someone lets you"',
        'text="Waiting for the host to let you in"'
    ]

    # Leave button
    LEAVE_BUTTON = [
        'button[aria-label="Leave call"]',
        'button[aria-label*="Leave"]',
        'button:has-text("Leave meeting")'
    ]


class GoogleMeetAutomation:
    """
    Production-ready Google Meet browser automation using Playwright

    Usage:
        config = BrowserConfig(headless=True, audio_capture_enabled=True)
        automation = GoogleMeetAutomation(config)

        await automation.initialize()
        await automation.join_meeting("https://meet.google.com/abc-defg-hij", "Bot Name")

        # Bot is now in meeting
        await automation.wait_for_active()

        await automation.leave_meeting()
        await automation.cleanup()
    """

    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self.state = MeetingState.DISCONNECTED
        self.meeting_url: Optional[str] = None
        self.bot_name: Optional[str] = None

        # Playwright objects
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

        # Audio/video streams
        self.audio_stream = None
        self.video_stream = None

    async def initialize(self):
        """Initialize Playwright and browser"""
        logger.info("🚀 Initializing Google Meet automation")

        try:
            # Start Playwright
            self.playwright = await async_playwright().start()

            # MINIMAL args - Google flags aggressive automation args!
            # Only include what's ABSOLUTELY necessary for media capture
            browser_args = [
                # Media capture (ScreenApp's critical args)
                '--enable-usermedia-screen-capturing',
                '--allow-http-screen-capture',
                '--auto-accept-this-tab-capture',
                '--enable-features=MediaRecorder',

                # Window config
                f'--window-size={self.config.window_size[0]},{self.config.window_size[1]}',

                # Fake UI for media prompts (needed to auto-accept)
                '--use-fake-ui-for-media-stream',
            ]

            # ⚡ USE REAL CHROME to match TLS/HTTP fingerprint (NOT Playwright's Chromium!)
            # This is CRITICAL - Google detects Chromium's different TLS signature
            chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

            self.browser = await self.playwright.chromium.launch(
                headless=self.config.headless,
                executable_path=chrome_path,  # ← REAL Chrome = Real TLS fingerprint!
                args=browser_args,
                ignore_default_args=['--mute-audio'],  # ScreenApp: Don't mute audio!
                channel=None  # Don't use channel when using executable_path
            )

            # Create browser context with media permissions and Vexa's user agent
            # Using Vexa's Chrome 129 user agent (battle-tested for Google Meet)
            vexa_user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"

            # Simple context - no persistent storage needed for anonymous joining
            self.context = await self.browser.new_context(
                viewport={'width': self.config.window_size[0], 'height': self.config.window_size[1]},
                user_agent=self.config.user_agent or vexa_user_agent,
                bypass_csp=True,
                ignore_https_errors=True,  # ScreenApp uses this
            )

            # Grant camera and microphone permissions to Google Meet
            await self.context.grant_permissions(['microphone', 'camera'], origin='https://meet.google.com')

            # Apply Malenia's UNDETECTED stealth mode to bypass Google bot detection!
            logger.info("🥷 Applying Malenia stealth to context...")
            await Malenia.apply_stealth(self.context)
            logger.info("✅ Undetected mode active - bot detection bypassed by Malenia!")

            # Create page AFTER stealth is applied
            self.page = await self.context.new_page()

            # Enable console logging for debugging
            self.page.on('console', lambda msg: logger.debug(f"Browser console: {msg.text}"))

            logger.info("✅ Browser initialized successfully")
            self.state = MeetingState.DISCONNECTED

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            self.state = MeetingState.ERROR
            raise

    async def login_google_account(self) -> bool:
        """
        Login to Google account (required for restricted meetings)

        Returns:
            True if login successful, False otherwise
        """
        if not self.config.google_email or not self.config.google_password:
            logger.info("⚠️  No Google credentials provided - joining anonymously")
            return False

        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        logger.info(f"🔐 Logging in to Google account: {self.config.google_email}")

        try:
            # Navigate to Google login page
            login_url = "https://accounts.google.com/ServiceLogin?hl=en&passive=true&continue=https://www.google.com/"
            await self.page.goto(login_url, wait_until='networkidle', timeout=60000)
            await self._take_screenshot("login-01-start")

            # Enter email
            logger.info("📧 Entering email...")
            await self.page.wait_for_selector('#identifierId', timeout=30000)
            await self.page.fill('#identifierId', self.config.google_email)
            await self._take_screenshot("login-02-email-entered")

            # Click Next button (email)
            await self.page.click('#identifierNext')
            await asyncio.sleep(1.5)  # Reduced from 3s
            await self._take_screenshot("login-03-after-email-next")

            # Enter password
            logger.info("🔑 Entering password...")
            await self.page.wait_for_selector('input[type="password"]', timeout=30000)
            await self.page.fill('input[type="password"]', self.config.google_password)
            await self._take_screenshot("login-04-password-entered")

            # Click Next button (password)
            await self.page.click('#passwordNext')
            await asyncio.sleep(2)  # Reduced from 5s
            await self._take_screenshot("login-05-after-password-next")

            # Wait for login to complete (check if we're on google.com)
            logger.info("⏳ Waiting for login to complete...")
            await self.page.wait_for_url("*://www.google.com/*", timeout=30000)
            await self._take_screenshot("login-06-success")

            logger.info("✅ Successfully logged in to Google account")

            # Save browser state for future sessions if user_data_dir is configured
            if self.config.user_data_dir:
                await self._save_browser_state()

            return True

        except Exception as e:
            logger.error(f"❌ Failed to login to Google account: {e}")
            await self._take_screenshot("login-error")
            return False

    async def _save_browser_state(self):
        """Save browser cookies and storage state for persistent login"""
        if not self.config.user_data_dir or not self.context:
            return

        try:
            os.makedirs(self.config.user_data_dir, exist_ok=True)
            state_file = f"{self.config.user_data_dir}/state.json"
            await self.context.storage_state(path=state_file)
            logger.info(f"💾 Saved browser state to: {state_file}")
        except Exception as e:
            logger.warning(f"Failed to save browser state: {e}")

    async def join_meeting(self, meeting_url: str, bot_name: str) -> bool:
        """
        Join Google Meet meeting

        Returns:
            True if successfully joined or in waiting room, False on error
        """
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        self.meeting_url = meeting_url
        self.bot_name = bot_name
        self.state = MeetingState.CONNECTING

        logger.info(f"🚪 Joining Google Meet: {meeting_url}")
        logger.info(f"👤 Bot name: {bot_name}")

        try:
            # ANONYMOUS JOINING: No authentication required!
            # Vexa & ScreenApp both prove this works for public Google Meet meetings
            logger.info("🎭 Joining anonymously (no authentication required)")

            # Navigate to meeting
            await self.page.goto(meeting_url, wait_until='networkidle', timeout=60000)
            await self.page.bring_to_front()

            await self._take_screenshot("01-after-navigation")

            # Wait for page to settle
            logger.info("⏳ Waiting for page elements to load...")
            await asyncio.sleep(2)  # Reduced from 5s

            # NOTE: Don't check for "You can't join" here - that message appears even for
            # meetings with waiting rooms. We'll try to join and handle rejection later.

            self.state = MeetingState.JOINING

            # Enter bot name
            await self._enter_name(bot_name)

            # Mute mic and camera
            await self._mute_audio_video()

            # Click join button
            await self._click_join_button()

            # Wait for either admission or waiting room
            await self._wait_for_meeting_state()

            logger.info(f"✅ Successfully joined meeting (state: {self.state.value})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to join meeting: {e}")
            self.state = MeetingState.ERROR
            await self._take_screenshot("error-join-failed")
            return False

    async def _enter_name(self, name: str):
        """Enter bot name in the name field"""
        logger.info(f"📝 Entering name: {name}")

        for selector in GoogleMeetSelectors.NAME_INPUT:
            try:
                await self.page.wait_for_selector(selector, timeout=30000)
                await self.page.fill(selector, name)
                logger.info(f"✅ Name entered using selector: {selector}")
                await self._take_screenshot("02-name-entered")
                return
            except Exception:
                continue

        logger.warning("⚠️  Could not find name input field")

    async def _mute_audio_video(self):
        """Mute microphone and camera if enabled"""
        # Mute microphone
        if not self.config.microphone_enabled:
            logger.info("🔇 Muting microphone")
            for selector in GoogleMeetSelectors.MICROPHONE_BUTTON:
                try:
                    await self.page.click(selector, timeout=1000)
                    logger.info("✅ Microphone muted")
                    break
                except Exception:
                    continue

        # Turn off camera
        if not self.config.video_enabled:
            logger.info("📷 Turning off camera")
            for selector in GoogleMeetSelectors.CAMERA_BUTTON:
                try:
                    await self.page.click(selector, timeout=1000)
                    logger.info("✅ Camera turned off")
                    break
                except Exception:
                    continue

    async def _click_join_button(self):
        """Click the join/ask to join button"""
        logger.info("🔘 Clicking join button")

        for selector in GoogleMeetSelectors.JOIN_BUTTON:
            try:
                await self.page.wait_for_selector(selector, timeout=30000)
                await self.page.click(selector)
                logger.info(f"✅ Join button clicked: {selector}")
                await self._take_screenshot("03-join-clicked")
                return
            except Exception:
                continue

        raise RuntimeError("Could not find join button")

    async def _wait_for_meeting_state(self, timeout: int = 60):
        """Wait for meeting to be joined or waiting room"""
        logger.info("⏳ Waiting for meeting admission...")

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check if admitted to meeting
            for selector in GoogleMeetSelectors.ADMISSION_INDICATORS:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        logger.info(f"✅ Admitted to meeting! Found: {selector}")
                        self.state = MeetingState.ACTIVE
                        await self._take_screenshot("04-meeting-joined")
                        return
                except Exception:
                    pass

            # Check if in waiting room
            for selector in GoogleMeetSelectors.WAITING_ROOM_INDICATORS:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        logger.info(f"⏳ In waiting room: {selector}")
                        self.state = MeetingState.WAITING_ROOM
                        await self._take_screenshot("04-waiting-room")
                        return
                except Exception:
                    pass

            await asyncio.sleep(1)  # Reduced from 2s

        logger.warning("⏱️  Timeout waiting for meeting state")
        await self._take_screenshot("timeout-admission")

    async def wait_for_active(self, timeout: int = 300):
        """Wait for bot to be admitted from waiting room"""
        if self.state == MeetingState.ACTIVE:
            return True

        if self.state != MeetingState.WAITING_ROOM:
            logger.warning(f"Not in waiting room (state: {self.state.value})")
            return False

        logger.info("⏳ Waiting for admission from waiting room...")
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            for selector in GoogleMeetSelectors.ADMISSION_INDICATORS:
                try:
                    element = await self.page.query_selector(selector)
                    if element:
                        logger.info("✅ Admitted from waiting room!")
                        self.state = MeetingState.ACTIVE
                        await self._take_screenshot("05-admitted")
                        return True
                except Exception:
                    pass

            await asyncio.sleep(2)  # Reduced from 5s

        logger.warning("⏱️  Timeout waiting for admission")
        return False

    async def leave_meeting(self):
        """Leave the Google Meet meeting"""
        if self.state not in [MeetingState.ACTIVE, MeetingState.WAITING_ROOM, MeetingState.JOINED]:
            logger.warning(f"Not in meeting (state: {self.state.value})")
            return

        logger.info("🚪 Leaving meeting...")
        self.state = MeetingState.LEAVING

        try:
            for selector in GoogleMeetSelectors.LEAVE_BUTTON:
                try:
                    await self.page.click(selector, timeout=5000)
                    logger.info("✅ Left meeting")
                    await self._take_screenshot("06-left-meeting")
                    self.state = MeetingState.DISCONNECTED
                    return
                except Exception:
                    continue

            logger.warning("Could not find leave button, closing browser")

        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")

        self.state = MeetingState.DISCONNECTED

    async def cleanup(self):
        """Cleanup browser resources"""
        logger.info("🧹 Cleaning up browser resources")

        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self.state = MeetingState.DISCONNECTED

        logger.info("✅ Cleanup complete")

    async def _take_screenshot(self, name: str):
        """Take screenshot for debugging"""
        if not self.config.screenshots_enabled or not self.page:
            return

        try:
            import os
            os.makedirs(self.config.screenshots_path, exist_ok=True)
            path = f"{self.config.screenshots_path}/{name}.png"
            await self.page.screenshot(path=path, full_page=True)
            logger.debug(f"📸 Screenshot: {path}")
        except Exception as e:
            logger.debug(f"Failed to take screenshot: {e}")

    def is_active(self) -> bool:
        """Check if bot is actively in meeting"""
        return self.state == MeetingState.ACTIVE

    def get_state(self) -> MeetingState:
        """Get current meeting state"""
        return self.state


# Example usage
async def example_usage():
    """Example of using Google Meet automation"""
    config = BrowserConfig(
        headless=True,
        audio_capture_enabled=True,
        video_enabled=False,
        microphone_enabled=False
    )

    automation = GoogleMeetAutomation(config)

    try:
        # Initialize
        await automation.initialize()

        # Join meeting
        success = await automation.join_meeting(
            "https://meet.google.com/test-meeting",
            "LiveTranslate Bot"
        )

        if success:
            # Wait for admission if in waiting room
            if automation.get_state() == MeetingState.WAITING_ROOM:
                await automation.wait_for_active(timeout=300)

            # Stay in meeting
            if automation.is_active():
                logger.info("Bot is active in meeting!")
                await asyncio.sleep(60)  # Stay for 1 minute

        # Leave meeting
        await automation.leave_meeting()

    finally:
        await automation.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
