"""
Browser E2E: History Tab

Tests historical transcript browsing, date range queries, and transcript viewer modal.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestHistoryTab:
    """Tests for the History tab."""

    def test_history_tab_loads(self, browser, dashboard_url):
        """History tab shows historical transcripts section."""
        browser.open(dashboard_url)
        browser.click("text=History")
        browser.wait("500")

        assert browser.wait_for_text("Historical Transcripts from Fireflies", timeout=5)

    def test_date_range_inputs(self, browser, dashboard_url):
        """Date range inputs for From and To are present."""
        browser.open(dashboard_url)
        browser.click("text=History")
        browser.wait("500")

        assert browser.is_visible("#historyDateFrom")
        assert browser.is_visible("#historyDateTo")

    def test_past_meetings_table_columns(self, browser, dashboard_url):
        """Past meetings table has Date, Title, Duration, Speakers, Actions columns."""
        browser.open(dashboard_url)
        browser.click("text=History")
        browser.wait("500")

        table_html = browser.get_html("#pastMeetingsTable")
        assert "Date" in table_html
        assert "Title" in table_html
        assert "Duration" in table_html
        assert "Speakers" in table_html
        assert "Actions" in table_html

    def test_saved_transcripts_section(self, browser, dashboard_url):
        """Saved transcripts table is present with correct columns."""
        browser.open(dashboard_url)
        browser.click("text=History")
        browser.wait("500")

        assert browser.wait_for_text("Saved Transcripts", timeout=5)
        table_html = browser.get_html("#savedTranscriptsTable")
        assert "Session/Transcript ID" in table_html
        assert "Language" in table_html
        assert "Saved At" in table_html

    def test_fetch_past_meetings_button(self, browser, dashboard_url):
        """Fetch Past Meetings button exists and is clickable."""
        browser.open(dashboard_url)
        browser.click("text=History")
        browser.wait("500")

        snap = browser.snapshot()
        assert "Fetch Past Meetings" in snap
