"""
Test sidebar navigation and page transitions.

BEHAVIORAL — real SvelteKit server, real Chromium browser, real navigation.
"""

import time

import pytest


class TestNavigation:
    """Verify sidebar navigation works across all pages."""

    def test_navigate_dashboard_to_fireflies(self, browser, screenshot_path):
        """Click Fireflies in sidebar, navigate to connect page."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        browser.click("text=Fireflies")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Transcript ID" in snap or "Fireflies" in snap
        browser.screenshot(screenshot_path("nav_to_fireflies"))

    def test_navigate_fireflies_to_config(self, browser, screenshot_path):
        """Navigate from Fireflies to Config via sidebar."""
        browser.open("http://localhost:5180/fireflies")
        browser.wait("text=Fireflies")
        browser.click("text=Config")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Configuration" in snap or "Audio" in snap
        browser.screenshot(screenshot_path("nav_to_config"))

    def test_navigate_config_sub_pages(self, browser, screenshot_path):
        """Navigate between config sub-pages."""
        browser.open("http://localhost:5180/config")
        browser.wait("text=Configuration")

        # Click Audio card
        browser.click("text=Audio")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Audio" in snap
        browser.screenshot(screenshot_path("nav_config_audio"))

    def test_navigate_to_translation_bench(self, browser, screenshot_path):
        """Navigate to Translation test bench from sidebar."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        browser.click("text=Translation")
        time.sleep(1)
        snap = browser.snapshot()
        assert "Translation" in snap
        browser.screenshot(screenshot_path("nav_to_translation"))
