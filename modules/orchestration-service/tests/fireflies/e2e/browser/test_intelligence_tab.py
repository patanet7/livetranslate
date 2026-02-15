"""
Browser E2E: Intelligence Tab

Tests meeting notes, AI analysis, post-meeting insights, and Q&A agent.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestIntelligenceTab:
    """Tests for the Intelligence tab."""

    def test_intelligence_tab_loads(self, browser, dashboard_url):
        """Intelligence tab shows meeting notes section."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.wait_for_text("Meeting Notes", timeout=5)

    def test_session_selector(self, browser, dashboard_url):
        """Intelligence tab has a session selector."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.is_visible("#intelSessionSelect")

    def test_manual_note_input(self, browser, dashboard_url):
        """Manual note input and Add button are present."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.is_visible("#manualNoteInput")
        snap = browser.snapshot()
        assert "Add Note" in snap

    def test_analyze_prompt_input(self, browser, dashboard_url):
        """AI analysis prompt input and Analyze button are present."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.is_visible("#analyzePromptInput")
        snap = browser.snapshot()
        assert "Analyze" in snap

    def test_post_meeting_insights_section(self, browser, dashboard_url):
        """Post-meeting insights section with template selector."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.wait_for_text("Post-Meeting Insights", timeout=5)
        assert browser.is_visible("#insightTemplateSelect")

    def test_generate_insight_buttons(self, browser, dashboard_url):
        """Generate Insight and Generate All buttons are present."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        snap = browser.snapshot()
        assert "Generate Insight" in snap
        assert "Generate All" in snap

    def test_meeting_qa_agent_section(self, browser, dashboard_url):
        """Meeting Q&A Agent section with chat input."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.wait_for_text("Meeting Q&A Agent", timeout=5)
        assert browser.is_visible("#agentChatInput")

    def test_qa_agent_suggested_queries(self, browser, dashboard_url):
        """Suggested queries area exists."""
        browser.open(dashboard_url)
        browser.click("text=Intelligence")
        browser.wait("500")

        assert browser.is_visible("#suggestedQueries")
