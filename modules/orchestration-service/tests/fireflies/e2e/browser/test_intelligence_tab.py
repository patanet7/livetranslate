"""
Browser E2E: Intelligence Tab

Tests meeting notes, AI analysis, post-meeting insights, and Q&A agent.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


def _open_intelligence_tab(browser, dashboard_url):
    """Navigate to the intelligence tab and wait for it to render."""
    browser.open(dashboard_url)
    browser.wait("1000")
    browser.click("button.tab[onclick*=\"showTab('intelligence')\"]")
    browser.wait("1000")


class TestIntelligenceTab:
    """Tests for the Intelligence tab."""

    def test_intelligence_tab_loads(self, browser, dashboard_url):
        """Intelligence tab shows meeting notes section."""
        _open_intelligence_tab(browser, dashboard_url)

        html = browser.get_html("#tab-intelligence")
        assert "Meeting Notes" in html

    def test_session_selector(self, browser, dashboard_url):
        """Intelligence tab has a session selector."""
        _open_intelligence_tab(browser, dashboard_url)

        count = browser.get_count("#intelSessionSelect")
        assert count >= 1, "Session selector should exist"

    def test_manual_note_input(self, browser, dashboard_url):
        """Manual note input and Add button are present."""
        _open_intelligence_tab(browser, dashboard_url)

        count = browser.get_count("#manualNoteInput")
        assert count >= 1, "Manual note input should exist"
        html = browser.get_html("#tab-intelligence")
        assert "Add Note" in html

    def test_analyze_prompt_input(self, browser, dashboard_url):
        """AI analysis prompt input and Analyze button are present."""
        _open_intelligence_tab(browser, dashboard_url)

        count = browser.get_count("#analyzePromptInput")
        assert count >= 1, "Analyze prompt input should exist"
        html = browser.get_html("#tab-intelligence")
        assert "Analyze" in html

    def test_post_meeting_insights_section(self, browser, dashboard_url):
        """Post-meeting insights section with template selector."""
        _open_intelligence_tab(browser, dashboard_url)

        html = browser.get_html("#tab-intelligence")
        assert "Post-Meeting Insights" in html
        count = browser.get_count("#insightTemplateSelect")
        assert count >= 1, "Insight template selector should exist"

    def test_generate_insight_buttons(self, browser, dashboard_url):
        """Generate Insight and Generate All buttons are present."""
        _open_intelligence_tab(browser, dashboard_url)

        html = browser.get_html("#tab-intelligence")
        assert "Generate Insight" in html
        assert "Generate All" in html

    def test_meeting_qa_agent_section(self, browser, dashboard_url):
        """Meeting Q&A Agent section with chat input."""
        _open_intelligence_tab(browser, dashboard_url)

        html = browser.get_html("#tab-intelligence")
        assert "Meeting Q&amp;A Agent" in html or "Meeting Q&A Agent" in html
        count = browser.get_count("#agentChatInput")
        assert count >= 1, "Agent chat input should exist"

    def test_qa_agent_suggested_queries(self, browser, dashboard_url):
        """Suggested queries area exists."""
        _open_intelligence_tab(browser, dashboard_url)

        count = browser.get_count("#suggestedQueries")
        assert count >= 1, "Suggested queries container should exist"
