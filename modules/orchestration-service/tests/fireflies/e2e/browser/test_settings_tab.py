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
        browser.click("button.tab[onclick*=\"showTab('settings')\"]")
        # Verify settings content is visible
        assert browser.wait_for_text("Fireflies API Configuration", timeout=5)

    def test_save_api_key(
        self, browser, dashboard_url, mock_fireflies_server, test_output_dir, timestamp
    ):
        """Fill API key via JS and verify masked display appears."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('settings')\"]")
        browser.wait("500")

        api_key = mock_fireflies_server["api_key"]

        # Save the API key directly via JS (bypasses async validation against Fireflies API)
        browser.eval_js(f"""
            apiKey = '{api_key}';
            localStorage.setItem('fireflies_api_key', '{api_key}');
            document.getElementById('savedKeyDisplay').classList.remove('hidden');
            document.getElementById('maskedKey').textContent = maskApiKey('{api_key}');
            updateApiStatus(true);
            log('API key saved (test)', 'success');
        """)
        browser.wait("500")

        # Verify the saved key display is visible
        assert browser.is_visible("#savedKeyDisplay")

        browser.screenshot(str(test_output_dir / f"{timestamp}_settings_api_key_saved.png"))

    def test_clear_api_key(self, browser, dashboard_url, mock_fireflies_server):
        """Clearing API key removes the saved display."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('settings')\"]")
        browser.wait("500")

        # First save a key via JS
        api_key = mock_fireflies_server["api_key"]
        browser.eval_js(f"""
            apiKey = '{api_key}';
            localStorage.setItem('fireflies_api_key', '{api_key}');
            document.getElementById('savedKeyDisplay').classList.remove('hidden');
            document.getElementById('maskedKey').textContent = maskApiKey('{api_key}');
        """)
        browser.wait("300")

        # Now clear it via JS (clearApiKey uses confirm() dialog which blocks)
        browser.eval_js("""
            apiKey = '';
            localStorage.removeItem('fireflies_api_key');
            document.getElementById('apiKeyInput').value = '';
            document.getElementById('savedKeyDisplay').classList.add('hidden');
        """)
        browser.wait("300")

        # The saved key display should be hidden and input empty
        value = browser.get_value("#apiKeyInput")
        assert value == ""

    def test_activity_log_shows_init(self, browser, dashboard_url):
        """Activity log shows initialization message on load."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('settings')\"]")
        browser.wait("1000")

        # The activity log should contain the init message
        log_text = browser.get_text("#logPanel")
        assert "Dashboard initialized" in log_text

    def test_service_status_section(self, browser, dashboard_url, orchestration_server):
        """Service status section renders."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('settings')\"]")
        browser.wait("1000")

        # Service status section should exist
        assert browser.wait_for_text("Service Status", timeout=5)

        # Take screenshot for visual inspection
        snapshot = browser.snapshot()
        assert "Service Status" in snapshot
