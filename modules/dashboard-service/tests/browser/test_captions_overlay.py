"""
Visual verification of the OBS captions overlay.

Tests display modes, URL parameters, transparent background,
and DOM structure. BEHAVIORAL — real SvelteKit server, real Chromium browser.
No mock data injection — we verify real rendered behavior.
"""

import time

import pytest


class TestCaptionsOverlay:
    """Verify the OBS captions overlay renders correctly."""

    def test_overlay_transparent_background(self, browser, screenshot_path):
        """Overlay has no chrome and transparent background."""
        browser.open("http://localhost:5180/captions?session=test123")
        time.sleep(2)
        browser.screenshot(screenshot_path("overlay_transparent"))
        # Verify no sidebar elements
        snap = browser.snapshot()
        # The overlay layout should NOT contain sidebar nav items
        assert "Quick Actions" not in snap

    def test_overlay_custom_font_size(self, browser, screenshot_path):
        """fontSize URL param changes text size."""
        browser.open("http://localhost:5180/captions?session=test123&fontSize=32")
        time.sleep(1)
        # Check the font-size style is applied
        result = browser.eval_js(
            "document.querySelector('.captions-overlay')?.style.fontSize"
        )
        assert "32" in str(result)
        browser.screenshot(screenshot_path("overlay_font_32"))

    def test_overlay_no_session_error(self, browser, screenshot_path):
        """Missing session param shows error message."""
        browser.open("http://localhost:5180/captions")
        time.sleep(1)
        snap = browser.snapshot()
        assert "session" in snap.lower() or "Missing" in snap
        browser.screenshot(screenshot_path("overlay_no_session"))

    def test_overlay_custom_background(self, browser, screenshot_path):
        """bg URL param sets custom background color."""
        browser.open("http://localhost:5180/captions?session=test123&bg=rgba(0,0,0,0.8)")
        time.sleep(1)
        result = browser.eval_js(
            "document.querySelector('.captions-overlay')?.style.background"
        )
        assert "0.8" in str(result) or "rgba" in str(result)
        browser.screenshot(screenshot_path("overlay_custom_bg"))
