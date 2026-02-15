"""
Browser E2E: Live Feed Tab

Tests side-by-side original transcript + translation feed display.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestLiveFeedTab:
    """Tests for the Live Feed tab."""

    def test_livefeed_tab_loads(self, browser, dashboard_url):
        """Live Feed tab shows the dual-panel layout."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.wait_for_text("Live Transcript & Translation Feed", timeout=5)

    def test_livefeed_has_dual_panels(self, browser, dashboard_url):
        """Live Feed shows original transcript and translation panels."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.wait_for_text("Original Transcript", timeout=5)
        assert browser.wait_for_text("Translation", timeout=5)

    def test_livefeed_session_selector(self, browser, dashboard_url):
        """Live Feed has a session selector dropdown."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.is_visible("#feedSessionSelect")

    def test_livefeed_status_badge(self, browser, dashboard_url):
        """Live Feed shows disconnected status by default."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        status_text = browser.get_text("#feedStatus")
        assert "Disconnected" in status_text

    def test_livefeed_empty_state(self, browser, dashboard_url):
        """Panels show empty state message before connecting."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        original_text = browser.get_text("#originalFeed")
        assert "Connect to a session" in original_text

    def test_livefeed_has_save_export_buttons(self, browser, dashboard_url):
        """Save Feed and Export JSON buttons are present."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        snap = browser.snapshot()
        assert "Save Feed" in snap
        assert "Export JSON" in snap
