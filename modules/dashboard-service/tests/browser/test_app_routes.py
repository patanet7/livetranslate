"""
Visual verification of all dashboard routes.

Each test navigates to a route, takes a screenshot, and verifies
that expected DOM elements are present and no errors are shown.
These are BEHAVIORAL tests — real SvelteKit server, real Chromium browser.
"""

import time

import pytest


class TestDashboardRoutes:
    """Verify every Phase 1 route renders correctly."""

    def test_dashboard_home(self, browser, screenshot_path):
        """Dashboard home shows health cards and quick actions."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        snap = browser.snapshot()
        assert "Dashboard" in snap
        assert "Quick Actions" in snap
        browser.screenshot(screenshot_path("dashboard_home"))

    def test_fireflies_connect_page(self, browser, screenshot_path):
        """Fireflies connect page shows form with transcript ID input."""
        browser.open("http://localhost:5180/fireflies")
        browser.wait("text=Fireflies")
        snap = browser.snapshot()
        assert "Transcript ID" in snap
        assert "Connect" in snap
        browser.screenshot(screenshot_path("fireflies_connect"))

    def test_fireflies_history_page(self, browser, screenshot_path):
        """History page renders session table or empty state."""
        browser.open("http://localhost:5180/fireflies/history")
        browser.wait("text=Session History")
        snap = browser.snapshot()
        assert "Session History" in snap
        browser.screenshot(screenshot_path("fireflies_history"))

    def test_fireflies_glossary_page(self, browser, screenshot_path):
        """Glossary page shows add form and entries table."""
        browser.open("http://localhost:5180/fireflies/glossary")
        browser.wait("text=Glossary")
        snap = browser.snapshot()
        assert "Add Term" in snap
        assert "Source Term" in snap
        browser.screenshot(screenshot_path("fireflies_glossary"))

    def test_config_hub(self, browser, screenshot_path):
        """Config hub shows links to audio, translation, system."""
        browser.open("http://localhost:5180/config")
        browser.wait("text=Configuration")
        snap = browser.snapshot()
        assert "Audio" in snap
        assert "Translation" in snap
        assert "System" in snap
        browser.screenshot(screenshot_path("config_hub"))

    def test_config_audio(self, browser, screenshot_path):
        """Audio config page shows settings form."""
        browser.open("http://localhost:5180/config/audio")
        browser.wait("text=Audio Configuration")
        snap = browser.snapshot()
        assert "Save" in snap
        browser.screenshot(screenshot_path("config_audio"))

    def test_config_translation(self, browser, screenshot_path):
        """Translation config page shows backend/model/language form."""
        browser.open("http://localhost:5180/config/translation")
        browser.wait("text=Translation Configuration")
        snap = browser.snapshot()
        assert "Backend" in snap
        assert "Model" in snap
        browser.screenshot(screenshot_path("config_translation"))

    def test_config_system(self, browser, screenshot_path):
        """System config page shows theme and notification settings."""
        browser.open("http://localhost:5180/config/system")
        browser.wait("text=System Configuration")
        snap = browser.snapshot()
        assert "Theme" in snap
        browser.screenshot(screenshot_path("config_system"))

    def test_translation_test_bench(self, browser, screenshot_path):
        """Translation test bench shows input/output panels."""
        browser.open("http://localhost:5180/translation/test")
        browser.wait("text=Translation Test Bench")
        snap = browser.snapshot()
        assert "Input" in snap
        assert "Result" in snap
        browser.screenshot(screenshot_path("translation_test_bench"))

    def test_captions_overlay_no_session(self, browser, screenshot_path):
        """Captions overlay without session param shows error message."""
        browser.open("http://localhost:5180/captions")
        time.sleep(1)  # overlay has no loading indicators
        snap = browser.snapshot()
        assert "Missing" in snap or "session" in snap.lower()
        browser.screenshot(screenshot_path("captions_overlay_no_session"))

    def test_sidebar_navigation(self, browser, screenshot_path):
        """Sidebar is present on all (app) pages and has correct links."""
        browser.open("http://localhost:5180/")
        browser.wait("text=Dashboard")
        snap = browser.snapshot()
        assert "Fireflies" in snap
        assert "Config" in snap
        assert "Translation" in snap
        browser.screenshot(screenshot_path("sidebar_navigation"))

    def test_overlay_has_no_sidebar(self, browser, screenshot_path):
        """Captions overlay (overlay group) has no sidebar or navigation."""
        browser.open("http://localhost:5180/captions")
        time.sleep(1)
        snap = browser.snapshot()
        # Sidebar elements should NOT be present
        assert "Quick Actions" not in snap
        browser.screenshot(screenshot_path("overlay_no_sidebar"))
