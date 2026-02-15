"""
Browser E2E: Settings Tab

Tests API key management, service status display, and activity log.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestSettingsTab:
    """Tests for the Settings tab of the Fireflies dashboard."""

    def test_dashboard_loads(self, browser, dashboard_url, test_output_dir, timestamp):
        """Dashboard loads and shows the correct title."""
        browser.open(dashboard_url)
        title = browser.get_title()
        assert "Fireflies Dashboard" in title

        browser.screenshot(str(test_output_dir / f"{timestamp}_settings_dashboard_loaded.png"))

    def test_navigate_to_settings(self, browser, dashboard_url):
        """Can navigate to the Settings tab."""
        browser.open(dashboard_url)
        # Click the Settings tab button
        browser.click("text=Settings")
        # Verify settings content is visible
        assert browser.wait_for_text("Fireflies API Configuration", timeout=5)

    def test_save_api_key(
        self, browser, dashboard_url, mock_fireflies_server, test_output_dir, timestamp
    ):
        """Fill API key, save, and verify masked display appears."""
        browser.open(dashboard_url)
        browser.click("text=Settings")
        browser.wait("500")

        api_key = mock_fireflies_server["api_key"]

        # Fill the API key input
        browser.fill("#apiKeyInput", api_key)

        # Click Save
        browser.click("text=Save API Key")
        browser.wait("500")

        # Verify the saved key display is visible with masked value
        assert browser.wait_for_text("Saved", timeout=5)

        browser.screenshot(str(test_output_dir / f"{timestamp}_settings_api_key_saved.png"))

    def test_clear_api_key(self, browser, dashboard_url, mock_fireflies_server):
        """Clearing API key removes the saved display."""
        browser.open(dashboard_url)
        browser.click("text=Settings")
        browser.wait("500")

        # First save a key
        browser.fill("#apiKeyInput", mock_fireflies_server["api_key"])
        browser.click("text=Save API Key")
        browser.wait("500")

        # Now clear it
        browser.click("text=Clear")
        browser.wait("500")

        # The saved key display should be hidden (input should be empty)
        value = browser.get_value("#apiKeyInput")
        assert value == ""

    def test_activity_log_shows_init(self, browser, dashboard_url):
        """Activity log shows initialization message on load."""
        browser.open(dashboard_url)
        browser.click("text=Settings")
        browser.wait("1000")

        # The activity log should contain the init message
        log_text = browser.get_text("#logPanel")
        assert "Dashboard initialized" in log_text

    def test_service_status_section(self, browser, dashboard_url, orchestration_server):
        """Service status section renders."""
        browser.open(dashboard_url)
        browser.click("text=Settings")
        browser.wait("1000")

        # Service status section should exist
        assert browser.wait_for_text("Service Status", timeout=5)

        # Take screenshot for visual inspection
        snapshot = browser.snapshot()
        assert "Service Status" in snapshot
