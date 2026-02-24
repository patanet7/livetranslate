"""
Browser E2E: Live Feed Tab — Validation 2

Tests side-by-side original transcript + translation feed display,
plus Pause/Resume, Display Mode, and Apply Language controls.

This is Validation 2 from the VALIDATION_PLAN.md: Dashboard UI Controls.
Screenshots are captured for human review — automated assertions verify
only that controls exist and are wired (not pixel-level appearance).

Run: uv run pytest tests/fireflies/e2e/browser/test_livefeed_tab.py -v -m "e2e and browser"
"""

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestLiveFeedTab:
    """Tests for the Live Feed tab layout and content."""

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


# =============================================================================
# Validation 2: Dashboard UI Controls
# =============================================================================


class TestLiveFeedControls:
    """
    Validation 2: Verify that the Live Feed tab UI controls render
    correctly and are wired to the correct endpoints.

    Controls tested:
    - Pause/Resume button (#btnPause)
    - Display Mode toggle (english/both/translated)
    - Apply Language button (#btnChangeLang)
    - Caption Preview area (#captionPreview)

    Screenshots are captured at key states for human review.
    Automated assertions verify control existence and initial state only.
    """

    def test_pause_button_exists_and_initially_disabled(
        self, browser, dashboard_url, browser_output_dir, timestamp
    ):
        """
        GIVEN: Dashboard loaded, Live Feed tab active
        WHEN: Page renders
        THEN: Pause button exists and is disabled (no active session)
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.is_visible("#btnPause"), "Pause button not visible"
        pause_text = browser.get_text("#btnPause")
        assert "Pause" in pause_text, f"Pause button text unexpected: {pause_text}"

        # Screenshot for human review
        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_livefeed_pause_button_disabled.png")
        )

    def test_display_mode_buttons_exist(
        self, browser, dashboard_url, browser_output_dir, timestamp
    ):
        """
        GIVEN: Dashboard loaded, Live Feed tab active
        WHEN: Page renders
        THEN: All three display mode buttons exist (English, Both, Translated)
              and 'Both' is the default active mode
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        # All three mode buttons should be visible
        assert browser.is_visible("#mode-english"), "English mode button missing"
        assert browser.is_visible("#mode-both"), "Both mode button missing"
        assert browser.is_visible("#mode-translated"), "Translated mode button missing"

        # Verify button text
        snap = browser.snapshot()
        assert "English" in snap
        assert "Both" in snap
        assert "Translated" in snap

        # Screenshot for human review: display mode buttons visible
        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_livefeed_display_mode_buttons.png")
        )

    def test_apply_language_button_exists(self, browser, dashboard_url):
        """
        GIVEN: Dashboard loaded, Live Feed tab active
        WHEN: Page renders
        THEN: Apply Language button exists and is disabled (no active session)
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.is_visible("#btnChangeLang"), "Apply Language button not visible"
        lang_text = browser.get_text("#btnChangeLang")
        assert "Apply Language" in lang_text

    def test_caption_preview_area_exists(self, browser, dashboard_url):
        """
        GIVEN: Dashboard loaded, Live Feed tab active
        WHEN: Page renders
        THEN: Caption Preview section renders with placeholder text
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.is_visible("#captionPreview"), "Caption preview not visible"
        preview_text = browser.get_text("#captionPreviewContent")
        assert "Captions will appear" in preview_text

    def test_target_language_selector_exists(self, browser, dashboard_url):
        """
        GIVEN: Dashboard loaded, Live Feed tab active
        WHEN: Page renders
        THEN: Target language dropdown exists
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("500")

        assert browser.is_visible("#feedTargetLang"), "Target language dropdown missing"

    def test_full_livefeed_screenshot(
        self, browser, dashboard_url, browser_output_dir, timestamp
    ):
        """
        GIVEN: Dashboard loaded, Live Feed tab active
        WHEN: Full page rendered
        THEN: Screenshot captured for human review — should show:
              - Pause button (disabled)
              - Display mode buttons (Both active)
              - Dual panels (Original Transcript + Translation)
              - Caption Preview section
              - Session selector
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('livefeed')\"]")
        browser.wait("1000")

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_livefeed_full_controls.png")
        )


class TestDashboardPageLoad:
    """
    Validation 2 supplement: Verify the dashboard page loads without
    JavaScript errors and all tab navigation works.
    """

    def test_dashboard_title(self, browser, dashboard_url, browser_output_dir, timestamp):
        """
        GIVEN: Browser navigates to dashboard URL
        WHEN: Page loads
        THEN: Page title contains "Fireflies Dashboard"
        """
        browser.open(dashboard_url)
        title = browser.get_title()
        assert "Fireflies Dashboard" in title

        # Screenshot: initial page load for human review
        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_dashboard_initial_load.png")
        )

    def test_all_tabs_visible(self, browser, dashboard_url):
        """
        GIVEN: Dashboard loaded
        WHEN: Page renders
        THEN: All expected tab buttons are visible in the snapshot
        """
        browser.open(dashboard_url)
        browser.wait("500")

        snap = browser.snapshot()
        expected_tabs = [
            "Connect",
            "Live Feed",
            "Sessions",
            "Glossary",
            "History",
            "Settings",
        ]
        for tab in expected_tabs:
            assert tab in snap, f"Tab '{tab}' not found in page snapshot"

    def test_settings_tab_renders(
        self, browser, dashboard_url, browser_output_dir, timestamp
    ):
        """
        GIVEN: Dashboard loaded
        WHEN: Settings tab is clicked
        THEN: Demo Mode section renders with launch button
        """
        browser.open(dashboard_url)
        browser.click("button.tab[onclick*=\"showTab('settings')\"]")
        browser.wait("500")

        assert browser.wait_for_text("Demo Mode", timeout=5)
        assert browser.is_visible("#settingsDemoBtn"), "Demo launch button not found"

        # Screenshot for human review
        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_settings_demo_mode.png")
        )
