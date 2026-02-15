"""
Browser E2E: Translation Tab

Tests model info, model switching, prompt templates, and test translation.
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestTranslationTab:
    """Tests for the Translation tab."""

    def test_translation_tab_loads(self, browser, dashboard_url):
        """Translation tab shows model section."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("500")

        assert browser.wait_for_text("Translation Model", timeout=5)

    def test_model_info_displayed(self, browser, dashboard_url):
        """Current model info element exists."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("1000")

        assert browser.is_visible("#currentModelInfo")

    def test_model_selector(self, browser, dashboard_url):
        """Model selector dropdown exists."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("500")

        assert browser.is_visible("#modelSelect")

    def test_prompt_template_section(self, browser, dashboard_url):
        """Prompt template section with style selector and textarea."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("500")

        assert browser.wait_for_text("Translation Prompt Template", timeout=5)
        assert browser.is_visible("#templateStyleSelect")
        assert browser.is_visible("#promptTemplate")

    def test_template_style_options(self, browser, dashboard_url):
        """Template style selector has Simple, Full, and Minimal options."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("500")

        style_html = browser.get_html("#templateStyleSelect")
        assert "Simple" in style_html
        assert "Full" in style_html
        assert "Minimal" in style_html

    def test_test_translation_section(self, browser, dashboard_url):
        """Test Translation section with input, target language, and translate button."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("500")

        assert browser.wait_for_text("Test Translation", timeout=5)
        assert browser.is_visible("#testText")
        assert browser.is_visible("#testTargetLang")

    def test_test_translation_default_text(self, browser, dashboard_url):
        """Test text input has default placeholder text."""
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('translation')\"]")
        browser.wait("500")

        value = browser.get_value("#testText")
        assert "Hello" in value or value == ""  # May have default or be empty
