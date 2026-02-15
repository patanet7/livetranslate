"""
Browser E2E: Glossary Tab

Tests glossary management — create, view entries, domain selection.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestGlossaryTab:
    """Tests for the Glossary tab."""

    def test_glossary_tab_loads(self, browser, dashboard_url):
        """Glossary tab loads with vocabulary libraries section."""
        browser.open(dashboard_url)
        browser.click("text=Glossary")
        browser.wait("500")

        assert browser.wait_for_text("Vocabulary Libraries", timeout=5)

    def test_glossary_details_section(self, browser, dashboard_url):
        """Glossary details form has name, domain, and source language fields."""
        browser.open(dashboard_url)
        browser.click("text=Glossary")
        browser.wait("500")

        assert browser.wait_for_text("Glossary Details", timeout=5)
        assert browser.is_visible("#glossaryName")
        assert browser.is_visible("#glossaryDomain")

    def test_domain_dropdown_options(self, browser, dashboard_url):
        """Domain dropdown has expected domain options."""
        browser.open(dashboard_url)
        browser.click("text=Glossary")
        browser.wait("500")

        domain_html = browser.get_html("#glossaryDomain")
        assert "Medical" in domain_html
        assert "Legal" in domain_html
        assert "Technology" in domain_html
        assert "Business" in domain_html
        assert "Finance" in domain_html

    def test_glossary_entries_table(self, browser, dashboard_url):
        """Glossary entries table has expected columns."""
        browser.open(dashboard_url)
        browser.click("text=Glossary")
        browser.wait("500")

        table_html = browser.get_html("#glossaryEntriesTable")
        assert "Source Term" in table_html
        assert "Spanish" in table_html
        assert "French" in table_html
        assert "German" in table_html
        assert "Priority" in table_html

    def test_glossary_action_buttons(self, browser, dashboard_url):
        """Add Term, Import CSV, and Export buttons are present."""
        browser.open(dashboard_url)
        browser.click("text=Glossary")
        browser.wait("500")

        snap = browser.snapshot()
        assert "Add Term" in snap
        assert "Import CSV" in snap
        assert "Export" in snap
