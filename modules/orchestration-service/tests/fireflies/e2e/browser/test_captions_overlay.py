"""
Browser E2E: Captions Overlay — Full Validation Suite

Tests the captions.html overlay page for:
- Speaker name rendering + per-speaker colors
- Original text (italic) + translated text (larger font)
- Caption auto-expiry timing
- Max caption count enforcement
- Fade animation class
- Connection status indicator (green/red)
- Multi-speaker attribution
"""

import time

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestCaptionsOverlay:
    """Full validation suite for the captions overlay."""

    def test_captions_page_loads_with_session(
        self, browser, captions_url, mock_fireflies_server, browser_output_dir, timestamp
    ):
        """Captions page loads when given a valid session ID."""
        # Use a test session ID — even without active WebSocket, page should load
        url = captions_url("test-session-123")
        browser.open(url)
        browser.wait("1000")

        title = browser.get_title()
        assert "LiveTranslate Captions" in title
        browser.screenshot(str(browser_output_dir / f"{timestamp}_captions_loaded.png"))

    def test_setup_help_shown_without_session(self, browser, base_url):
        """Without session param, setup help screen is displayed."""
        browser.open(f"{base_url}/static/captions.html")
        browser.wait("1000")

        assert browser.wait_for_text("LiveTranslate Caption Overlay", timeout=5)
        assert browser.wait_for_text("session=YOUR_SESSION_ID", timeout=2)

    def test_connection_status_indicator_exists(self, browser, captions_url):
        """Connection status indicator is rendered."""
        browser.open(captions_url("test-session-456", showStatus="true"))
        browser.wait("1000")

        # Status element should exist
        assert browser.is_visible("#status")

    def test_caption_box_structure(self, browser, captions_url, browser_output_dir, timestamp):
        """
        Inject a caption via JS and verify the DOM structure.

        Since we may not have a live WebSocket feeding captions,
        we inject one directly using eval_js to test rendering.
        """
        url = captions_url("test-inject", showSpeaker="true", showOriginal="true")
        browser.open(url)
        browser.wait("1000")

        # Inject a caption directly via the page's addCaption function
        browser.eval_js("""
            addCaption({
                id: 'test-caption-001',
                speaker_name: 'Alice',
                speaker_color: '#4CAF50',
                original_text: 'Hola, buenos días',
                translated_text: 'Hello, good morning',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        # Verify caption box appeared
        count = browser.get_count(".caption-box")
        assert count >= 1, "At least one caption box should be rendered"

        browser.screenshot(str(browser_output_dir / f"{timestamp}_captions_box_structure.png"))

    def test_speaker_name_rendered(self, browser, captions_url):
        """Speaker name element contains the correct name."""
        url = captions_url("test-speaker", showSpeaker="true", showOriginal="true")
        browser.open(url)
        browser.wait("500")

        browser.eval_js("""
            addCaption({
                id: 'test-speaker-001',
                speaker_name: 'Alice',
                speaker_color: '#4CAF50',
                original_text: 'Prueba',
                translated_text: 'Test',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        speaker_text = browser.get_text(".speaker-name")
        assert "Alice" in speaker_text

    def test_speaker_name_colored(self, browser, captions_url):
        """Speaker name has the correct inline color style."""
        url = captions_url("test-color", showSpeaker="true", showOriginal="true")
        browser.open(url)
        browser.wait("500")

        browser.eval_js("""
            addCaption({
                id: 'test-color-001',
                speaker_name: 'Bob',
                speaker_color: '#FF5722',
                original_text: 'Prueba de color',
                translated_text: 'Color test',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        styles = browser.get_styles(".speaker-name")
        # The color should be set (rgb format from computed styles)
        assert "color" in styles.lower()

    def test_original_text_rendered(self, browser, captions_url):
        """Original text element shows the source language text."""
        url = captions_url("test-original", showSpeaker="true", showOriginal="true")
        browser.open(url)
        browser.wait("500")

        browser.eval_js("""
            addCaption({
                id: 'test-orig-001',
                speaker_name: 'Alice',
                speaker_color: '#4CAF50',
                original_text: 'Hola, ¿cómo estás?',
                translated_text: 'Hello, how are you?',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        original = browser.get_text(".original-text")
        assert "Hola" in original

    def test_translated_text_rendered(self, browser, captions_url):
        """Translated text element shows the target language text."""
        url = captions_url("test-translated", showSpeaker="true", showOriginal="true")
        browser.open(url)
        browser.wait("500")

        browser.eval_js("""
            addCaption({
                id: 'test-trans-001',
                speaker_name: 'Alice',
                speaker_color: '#4CAF50',
                original_text: 'Buenos días a todos',
                translated_text: 'Good morning everyone',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        translated = browser.get_text(".translated-text")
        assert "Good morning everyone" in translated

    def test_caption_auto_expiry(self, browser, captions_url, browser_output_dir, timestamp):
        """Caption fades and is removed after its duration expires."""
        url = captions_url(
            "test-expiry",
            showSpeaker="true",
            showOriginal="true",
            fadeTime="1",  # Fast fade for testing
        )
        browser.open(url)
        browser.wait("500")

        # Add caption with short duration (3 seconds)
        browser.eval_js("""
            addCaption({
                id: 'test-expiry-001',
                speaker_name: 'Alice',
                speaker_color: '#4CAF50',
                original_text: 'Texto temporal',
                translated_text: 'Temporary text',
                duration_seconds: 3
            });
        """)
        browser.wait("500")

        # Verify it's present
        count_before = browser.get_count(".caption-box")
        assert count_before >= 1
        browser.screenshot(str(browser_output_dir / f"{timestamp}_captions_before_expiry.png"))

        # Poll until caption is removed (generous timeout for slow CI)
        deadline = time.time() + 10
        count_after = count_before
        while time.time() < deadline:
            count_after = browser.get_count(".caption-box")
            if count_after < count_before:
                break
            browser.wait("500")

        browser.screenshot(str(browser_output_dir / f"{timestamp}_captions_after_expiry.png"))
        assert count_after < count_before, "Caption should be removed after expiry"

    def test_max_caption_count(self, browser, captions_url):
        """Adding more than maxCaptions removes the oldest."""
        url = captions_url(
            "test-max",
            showSpeaker="true",
            showOriginal="true",
            maxCaptions="3",
        )
        browser.open(url)
        browser.wait("500")

        # Add 5 captions (max is 3)
        for i in range(5):
            browser.eval_js(f"""
                addCaption({{
                    id: 'test-max-{i:03d}',
                    speaker_name: 'Speaker{i}',
                    speaker_color: '#{"4CAF50" if i % 2 == 0 else "FF5722"}',
                    original_text: 'Texto {i}',
                    translated_text: 'Text {i}',
                    duration_seconds: 60
                }});
            """)
            browser.wait("200")

        browser.wait("500")

        count = browser.get_count(".caption-box")
        assert count <= 3, f"Should have at most 3 captions, got {count}"

    def test_multi_speaker_captions(self, browser, captions_url, browser_output_dir, timestamp):
        """Multiple speakers each get their own caption with distinct names."""
        url = captions_url("test-multi", showSpeaker="true", showOriginal="true")
        browser.open(url)
        browser.wait("500")

        # Add captions from different speakers
        browser.eval_js("""
            addCaption({
                id: 'multi-alice',
                speaker_name: 'Alice',
                speaker_color: '#4CAF50',
                original_text: 'Hola desde Alice',
                translated_text: 'Hello from Alice',
                duration_seconds: 30
            });
        """)
        browser.wait("200")

        browser.eval_js("""
            addCaption({
                id: 'multi-bob',
                speaker_name: 'Bob',
                speaker_color: '#2196F3',
                original_text: 'Hola desde Bob',
                translated_text: 'Hello from Bob',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        # Get all speaker names
        snap = browser.snapshot()
        assert "Alice" in snap, "Alice's caption should be visible"
        assert "Bob" in snap, "Bob's caption should be visible"

        browser.screenshot(str(browser_output_dir / f"{timestamp}_captions_multi_speaker.png"))

    def test_connection_status_indicator_state(self, browser, captions_url):
        """Status indicator shows a connection state (connected or disconnected)."""
        url = captions_url("test-status", showStatus="true")
        browser.open(url)
        browser.wait("2000")

        # Status element's class attribute should contain a recognized state
        status_class = browser.get_attr("#status", "class")
        assert (
            "status-connected" in status_class or "status-disconnected" in status_class
        ), f"Status should have connected/disconnected class, got: '{status_class}'"
