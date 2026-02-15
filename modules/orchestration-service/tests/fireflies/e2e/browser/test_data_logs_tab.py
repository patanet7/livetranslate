"""
Browser E2E: Data & Logs Tab

Tests session data viewer with transcript and translation panels.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestDataLogsTab:
    """Tests for the Data & Logs tab."""

    def test_data_tab_loads(self, browser, dashboard_url):
        """Data & Logs tab shows session data viewer."""
        browser.open(dashboard_url)
        browser.click("text=Data & Logs")
        browser.wait("500")

        assert browser.wait_for_text("Session Data Viewer", timeout=5)

    def test_session_dropdown(self, browser, dashboard_url):
        """Session selector dropdown exists."""
        browser.open(dashboard_url)
        browser.click("text=Data & Logs")
        browser.wait("500")

        assert browser.is_visible("#dataSessionSelect")

    def test_dual_panels(self, browser, dashboard_url):
        """Transcripts and Translations panels are both present."""
        browser.open(dashboard_url)
        browser.click("text=Data & Logs")
        browser.wait("500")

        assert browser.wait_for_text("Transcripts", timeout=5)
        assert browser.wait_for_text("Translations", timeout=5)

    def test_empty_state_messages(self, browser, dashboard_url):
        """Empty state messages shown when no session selected."""
        browser.open(dashboard_url)
        browser.click("text=Data & Logs")
        browser.wait("500")

        transcripts_text = browser.get_text("#transcriptsPanel")
        assert "Select a session" in transcripts_text

        translations_text = browser.get_text("#translationsPanel")
        assert "Select a session" in translations_text
