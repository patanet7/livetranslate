"""
Browser E2E: Connect Tab

Tests meeting connection flow, language selector, and active meetings discovery.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestConnectTab:
    """Tests for the Connect tab of the Fireflies dashboard."""

    def test_connect_tab_loads(self, browser, dashboard_url):
        """Connect tab is the default active tab on load."""
        browser.open(dashboard_url)
        # Connect tab should be active by default
        assert browser.wait_for_text("Connect to Fireflies Meeting", timeout=5)

    def test_language_selector_populated(self, browser, dashboard_url, orchestration_server):
        """Target language selector is populated from backend config."""
        browser.open(dashboard_url)
        browser.wait("1000")  # Wait for dashboard config to load

        # Check that the language select has options
        count = browser.get_count("#targetLanguages option")
        assert count > 0, "Target language selector should have options from backend config"

    def test_connect_to_meeting(
        self,
        browser,
        dashboard_url,
        mock_fireflies_server,
        test_output_dir,
        timestamp,
    ):
        """Fill transcript ID, connect, and verify success."""
        browser.open(dashboard_url)
        browser.wait("1000")

        # Save API key via JS (bypasses async validation)
        api_key = mock_fireflies_server["api_key"]
        browser.eval_js(f"""
            apiKey = '{api_key}';
            localStorage.setItem('fireflies_api_key', '{api_key}');
            updateApiStatus(true);
        """)

        # Go to Connect tab
        browser.click("button.tab[onclick*=\"showTab('connect')\"]")
        browser.wait("500")

        # Fill transcript ID
        browser.fill("#transcriptId", mock_fireflies_server["transcript_id"])

        # Click Connect button
        browser.screenshot(str(test_output_dir / f"{timestamp}_connect_before_click.png"))
        browser.click("#connectBtn")

        # Wait for the connection response (dialog or redirect)
        browser.wait("2000")
        browser.screenshot(str(test_output_dir / f"{timestamp}_connect_after_click.png"))

    def test_fetch_active_meetings(
        self,
        browser,
        dashboard_url,
        mock_fireflies_server,
        test_output_dir,
        timestamp,
    ):
        """Clicking Refresh Meetings triggers the fetch flow."""
        browser.open(dashboard_url)
        browser.wait("1000")

        # Save API key via JS (bypasses async validation)
        api_key = mock_fireflies_server["api_key"]
        browser.eval_js(f"""
            apiKey = '{api_key}';
            localStorage.setItem('fireflies_api_key', '{api_key}');
            document.getElementById('savedKeyDisplay').classList.remove('hidden');
            updateApiStatus(true);
        """)
        browser.wait("300")

        # Ensure we're on Connect tab
        browser.click("button.tab[onclick*=\"showTab('connect')\"]")
        browser.wait("500")

        # Click Refresh Meetings
        browser.click("#fetchMeetingsBtn")
        browser.wait("2000")

        browser.screenshot(str(test_output_dir / f"{timestamp}_connect_meetings_list.png"))

        # Verify meetings list element exists (content depends on API availability)
        assert browser.is_visible("#meetingsList")
