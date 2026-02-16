"""
Google Meet Browser Automation

This module provides browser automation capabilities for joining Google Meet sessions,
capturing audio, and extracting captions in real-time.
"""

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from livetranslate_common.logging import get_logger

# Browser automation imports
try:
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException, TimeoutException
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logger = get_logger()


class MeetingState(Enum):
    """Google Meet session states"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    JOINING = "joining"
    JOINED = "joined"
    RECORDING = "recording"
    ERROR = "error"
    LEAVING = "leaving"


@dataclass
class GoogleMeetConfig:
    """Configuration for Google Meet automation"""

    headless: bool = True
    audio_capture_enabled: bool = True
    video_enabled: bool = False
    microphone_enabled: bool = False
    join_timeout: int = 30
    chrome_profile_path: str | None = None
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    window_size: tuple = (1920, 1080)
    chrome_binary_path: str | None = None


class GoogleMeetAutomation:
    """
    Google Meet browser automation for joining meetings and capturing content
    """

    def __init__(self, config: GoogleMeetConfig):
        self.config = config
        self.driver: webdriver.Chrome | None = None
        self.meeting_state = MeetingState.DISCONNECTED
        self.meeting_id: str | None = None
        self.meeting_url: str | None = None
        self.participants: dict[str, dict[str, Any]] = {}

        # Callbacks
        self.on_state_change: Callable[[MeetingState], None] | None = None
        self.on_participant_change: Callable[[dict[str, Any]], None] | None = None
        self.on_caption_received: Callable[[str, str, float], None] | None = None
        self.on_audio_data: Callable[[bytes], None] | None = None

        # Internal state
        self._last_caption_timestamp = 0
        self._caption_elements: list = []
        self._is_monitoring = False
        self._monitor_task: asyncio.Task | None = None

        # Background task tracking (prevents garbage collection and enables cleanup)
        self._background_tasks: set[asyncio.Task] = set()

        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is required for Google Meet automation")

    async def initialize(self):
        """Initialize browser automation"""
        try:
            logger.info("Initializing Google Meet automation")

            # Set up Chrome options
            chrome_options = ChromeOptions()

            if self.config.headless:
                chrome_options.add_argument("--headless")

            # Essential Chrome arguments for meeting functionality
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-features=VizDisplayCompositor")
            chrome_options.add_argument(
                f"--window-size={self.config.window_size[0]},{self.config.window_size[1]}"
            )
            chrome_options.add_argument(f"--user-agent={self.config.user_agent}")

            # Audio/video permissions
            chrome_options.add_argument("--use-fake-ui-for-media-stream")
            chrome_options.add_argument("--use-fake-device-for-media-stream")

            if self.config.audio_capture_enabled:
                chrome_options.add_argument("--allow-running-insecure-content")
                chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
                chrome_options.add_argument("--disable-web-security")
                chrome_options.add_argument("--disable-features=VizDisplayCompositor")

            # Chrome profile
            if self.config.chrome_profile_path:
                chrome_options.add_argument(f"--user-data-dir={self.config.chrome_profile_path}")

            # Chrome binary path
            if self.config.chrome_binary_path:
                chrome_options.binary_location = self.config.chrome_binary_path

            # Initialize Chrome driver
            try:
                self.driver = webdriver.Chrome(options=chrome_options)
                logger.info("Chrome driver initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Chrome driver: {e}")
                raise

            # Set timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)

            logger.info("Google Meet automation initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Google Meet automation: {e}")
            raise

    async def join_meeting(self, meeting_url: str) -> bool:
        """Join a Google Meet meeting"""
        try:
            logger.info(f"Joining Google Meet: {meeting_url}")
            self.meeting_url = meeting_url
            self.meeting_id = self._extract_meeting_id(meeting_url)

            self._update_state(MeetingState.CONNECTING)

            # Navigate to meeting
            self.driver.get(meeting_url)

            # Wait for page to load
            await asyncio.sleep(2)

            # Check if already in meeting or need to join
            if await self._is_in_meeting():
                logger.info("Already in meeting")
                self._update_state(MeetingState.JOINED)
                await self._start_monitoring()
                return True

            # Handle join process
            if await self._handle_join_process():
                self._update_state(MeetingState.JOINED)
                await self._start_monitoring()
                return True
            else:
                self._update_state(MeetingState.ERROR)
                return False

        except Exception as e:
            logger.error(f"Failed to join meeting: {e}")
            self._update_state(MeetingState.ERROR)
            return False

    async def _handle_join_process(self) -> bool:
        """Handle the meeting join process"""
        try:
            wait = WebDriverWait(self.driver, self.config.join_timeout)

            # Disable camera if not needed
            if not self.config.video_enabled:
                await self._disable_camera()

            # Disable microphone if not needed
            if not self.config.microphone_enabled:
                await self._disable_microphone()

            # Look for join button
            join_selectors = [
                "button[data-testid='join-button']",
                "button[aria-label*='Join']",
                "button[jsname='Qx7uuf']",  # Common Google Meet join button
                "div[role='button'][aria-label*='Join']",
                "span:contains('Join now')",
                "button:contains('Join now')",
            ]

            for selector in join_selectors:
                try:
                    if selector.startswith("span:contains") or selector.startswith(
                        "button:contains"
                    ):
                        # Handle text-based selectors
                        elements = self.driver.find_elements(
                            By.XPATH, "//*[contains(text(), 'Join now')]"
                        )
                        if elements:
                            elements[0].click()
                            logger.info("Clicked join button via text")
                            break
                    else:
                        element = wait.until(
                            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                        )
                        element.click()
                        logger.info(f"Clicked join button: {selector}")
                        break
                except TimeoutException:
                    continue

            # Wait for meeting to load
            await asyncio.sleep(5)

            # Verify we're in the meeting
            return await self._is_in_meeting()

        except Exception as e:
            logger.error(f"Error in join process: {e}")
            return False

    async def _disable_camera(self):
        """Disable camera before joining"""
        try:
            camera_selectors = [
                "button[data-testid='camera-button']",
                "button[aria-label*='camera']",
                "button[aria-label*='Turn off camera']",
                "div[role='button'][aria-label*='camera']",
            ]

            for selector in camera_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element.is_enabled():
                        element.click()
                        logger.info("Camera disabled")
                        break
                except NoSuchElementException:
                    continue
        except Exception as e:
            logger.warning(f"Could not disable camera: {e}")

    async def _disable_microphone(self):
        """Disable microphone before joining"""
        try:
            mic_selectors = [
                "button[data-testid='microphone-button']",
                "button[aria-label*='microphone']",
                "button[aria-label*='Turn off microphone']",
                "div[role='button'][aria-label*='microphone']",
            ]

            for selector in mic_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element.is_enabled():
                        element.click()
                        logger.info("Microphone disabled")
                        break
                except NoSuchElementException:
                    continue
        except Exception as e:
            logger.warning(f"Could not disable microphone: {e}")

    async def _is_in_meeting(self) -> bool:
        """Check if we're currently in a meeting"""
        try:
            # Check for meeting indicators
            meeting_indicators = [
                "div[data-testid='meeting-content']",
                "div[jsname='eOWUhd']",  # Meeting container
                "div[aria-label*='meeting']",
                "div[class*='meeting']",
                "div[class*='call-content']",
            ]

            for indicator in meeting_indicators:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, indicator)
                    if element.is_displayed():
                        return True
                except NoSuchElementException:
                    continue

            # Check URL for meeting indicators
            current_url = self.driver.current_url
            return bool("meet.google.com" in current_url and len(current_url.split("/")[-1]) > 5)

        except Exception as e:
            logger.error(f"Error checking meeting status: {e}")
            return False

    async def _start_monitoring(self):
        """Start monitoring meeting for captions and participant changes"""
        if self._is_monitoring:
            return

        self._is_monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_meeting())
        logger.info("Started meeting monitoring")

    async def _monitor_meeting(self):
        """Monitor meeting for captions and participant changes"""
        try:
            while self._is_monitoring and self.meeting_state == MeetingState.JOINED:
                try:
                    # Monitor captions
                    await self._monitor_captions()

                    # Monitor participants
                    await self._monitor_participants()

                    # Check meeting status
                    if not await self._is_in_meeting():
                        logger.warning("No longer in meeting")
                        self._update_state(MeetingState.DISCONNECTED)
                        break

                    # Wait before next check
                    await asyncio.sleep(0.5)  # Check every 500ms

                except Exception as e:
                    logger.error(f"Error in meeting monitoring: {e}")
                    await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Meeting monitoring failed: {e}")
        finally:
            self._is_monitoring = False

    async def _monitor_captions(self):
        """Monitor and extract captions from the meeting"""
        try:
            # Common Google Meet caption selectors
            caption_selectors = [
                "div[data-testid='captions-container']",
                "div[jsname='dsyhDe']",  # Common caption container
                "div[class*='caption']",
                "div[class*='subtitle']",
                "div[aria-live='polite']",  # Live caption regions
                "div[role='log']",
            ]

            for selector in caption_selectors:
                try:
                    caption_container = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if caption_container.is_displayed():
                        await self._extract_captions_from_container(caption_container)
                        break
                except NoSuchElementException:
                    continue

        except Exception as e:
            logger.error(f"Error monitoring captions: {e}")

    async def _extract_captions_from_container(self, container):
        """Extract captions from a container element"""
        try:
            # Get all caption elements
            caption_elements = container.find_elements(By.CSS_SELECTOR, "div, span, p")

            for element in caption_elements:
                try:
                    text = element.text.strip()
                    if text and len(text) > 0:
                        # Try to extract speaker name
                        speaker = self._extract_speaker_from_element(element)
                        timestamp = time.time()

                        # Check if this is a new caption
                        if timestamp > self._last_caption_timestamp:
                            self._last_caption_timestamp = timestamp

                            # Callback for caption received
                            if self.on_caption_received:
                                await self._safe_callback(
                                    self.on_caption_received,
                                    speaker or "Unknown",
                                    text,
                                    timestamp,
                                )

                except Exception as e:
                    logger.warning(f"Error extracting caption from element: {e}")

        except Exception as e:
            logger.error(f"Error extracting captions: {e}")

    def _extract_speaker_from_element(self, element) -> str | None:
        """Extract speaker name from caption element"""
        try:
            # Try to find speaker name in parent or sibling elements
            parent = element.find_element(By.XPATH, "..")
            siblings = parent.find_elements(By.CSS_SELECTOR, "*")

            for sibling in siblings:
                text = sibling.text.strip()
                # Look for patterns like "John Smith:" or "John Smith said:"
                if ":" in text and len(text) < 50:
                    potential_speaker = text.split(":")[0].strip()
                    if len(potential_speaker) > 0 and len(potential_speaker) < 30:
                        return potential_speaker

            return None

        except Exception:
            return None

    async def _monitor_participants(self):
        """Monitor participants in the meeting"""
        try:
            # Common participant selectors
            participant_selectors = [
                "div[data-testid='participant-item']",
                "div[jsname='oJeWuf']",  # Participant container
                "div[class*='participant']",
                "div[class*='attendee']",
            ]

            current_participants = {}

            for selector in participant_selectors:
                try:
                    participant_elements = self.driver.find_elements(By.CSS_SELECTOR, selector)

                    for element in participant_elements:
                        try:
                            name = element.text.strip()
                            if name:
                                current_participants[name] = {
                                    "name": name,
                                    "timestamp": time.time(),
                                    "element": element,
                                }
                        except Exception as e:
                            logger.warning(f"Error extracting participant: {e}")

                except NoSuchElementException:
                    continue

            # Check for changes
            if current_participants != self.participants:
                self.participants = current_participants

                if self.on_participant_change:
                    await self._safe_callback(self.on_participant_change, self.participants)

        except Exception as e:
            logger.error(f"Error monitoring participants: {e}")

    async def leave_meeting(self):
        """Leave the current meeting"""
        try:
            if self.meeting_state != MeetingState.JOINED:
                return

            logger.info("Leaving meeting")
            self._update_state(MeetingState.LEAVING)

            # Stop monitoring
            self._is_monitoring = False
            if self._monitor_task:
                self._monitor_task.cancel()

            # Try to find and click leave button
            leave_selectors = [
                "button[data-testid='leave-button']",
                "button[aria-label*='Leave']",
                "button[jsname='CQylAd']",  # Common leave button
                "div[role='button'][aria-label*='Leave']",
            ]

            for selector in leave_selectors:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element.is_enabled():
                        element.click()
                        logger.info("Clicked leave button")
                        break
                except NoSuchElementException:
                    continue

            # Wait for leave to complete
            await asyncio.sleep(3)

            self._update_state(MeetingState.DISCONNECTED)

        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
            self._update_state(MeetingState.ERROR)

    async def shutdown(self):
        """Shutdown browser automation"""
        try:
            logger.info("Shutting down Google Meet automation")

            # Stop monitoring
            self._is_monitoring = False
            if self._monitor_task:
                self._monitor_task.cancel()

            # Leave meeting if joined
            if self.meeting_state == MeetingState.JOINED:
                await self.leave_meeting()

            # Close browser
            if self.driver:
                self.driver.quit()
                self.driver = None

            self._update_state(MeetingState.DISCONNECTED)
            logger.info("Google Meet automation shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def _extract_meeting_id(self, meeting_url: str) -> str:
        """Extract meeting ID from URL"""
        try:
            # Extract meeting ID from Google Meet URL
            match = re.search(r"meet\.google\.com/([a-zA-Z0-9\-_]+)", meeting_url)
            if match:
                return match.group(1)
            return "unknown"
        except Exception:
            return "unknown"

    def _update_state(self, new_state: MeetingState):
        """Update meeting state and trigger callback"""
        if self.meeting_state != new_state:
            old_state = self.meeting_state
            self.meeting_state = new_state
            logger.info(f"Meeting state changed: {old_state} -> {new_state}")

            if self.on_state_change:
                task = asyncio.create_task(self._safe_callback(self.on_state_change, new_state))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

    async def _safe_callback(self, callback, *args):
        """Execute callback safely"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to a meeting"""
        return self.meeting_state == MeetingState.JOINED

    @property
    def meeting_info(self) -> dict[str, Any]:
        """Get current meeting information"""
        return {
            "meeting_id": self.meeting_id,
            "meeting_url": self.meeting_url,
            "state": self.meeting_state.value,
            "participants": self.participants,
            "is_connected": self.is_connected,
        }


# Factory function for creating Google Meet automation
def create_google_meet_automation(
    config: GoogleMeetConfig | None = None,
) -> GoogleMeetAutomation:
    """Create a Google Meet automation instance"""
    if config is None:
        config = GoogleMeetConfig()

    return GoogleMeetAutomation(config)
