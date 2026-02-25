"""
Browser E2E: Live Caption Rendering — Visual Display Verification

Tests the visual rendering of captions in the browser using the live pipeline:
- Display modes (both, translated, english)
- Speaker color coding
- Caption expiry / removal
- Max captions limit enforcement

These tests use the REAL pipeline with a mock Fireflies server. Captions arrive
via the WebSocket pipeline as interim_caption events (word-by-word streaming).

Since the test environment may not have an LLM, caption_added (final translated)
events may not appear. Tests work with what IS available:
- interim_caption events arrive reliably from the pipeline
- Final captions can be injected via addCaption() eval_js to test display mode
  filtering of .final-caption elements

Display mode behavior in captions.html:
- mode=both: .interim-caption visible, .final-caption visible
- mode=translated: .interim-caption hidden (handleInterimCaption returns early),
                    .final-caption visible (.translated-text shown)
- mode=english: .interim-caption visible,
                .final-caption hidden (style.display='none' set in addCaption)
"""

import json
import time

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


def _js_str(raw: str) -> str:
    """Strip embedded JS string quotes from agent-browser eval_js output.

    agent-browser returns JS string values wrapped in double quotes,
    e.g. eval_js("'hello'") returns '"hello"' in Python. This helper
    normalizes to a clean Python string.
    """
    return raw.strip().strip('"').strip("'")


class TestCaptionsLiveRendering:
    """Visual caption rendering tests using the live pipeline."""

    def test_display_mode_both(
        self,
        live_session,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """mode=both: interim captions AND final captions (original + translated) are visible.

        Opens captions.html with mode=both (default), waits for interim captions
        from the live pipeline, then injects a final caption to verify both types
        render and are visible.
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, mode="both")

        browser.open(url)

        # Wait for interim captions from the live pipeline
        deadline = time.time() + 30
        has_interim = False
        while time.time() < deadline:
            try:
                count_str = browser.eval_js(
                    "document.querySelectorAll('.interim-caption').length"
                ).strip()
                if int(count_str) >= 1:
                    has_interim = True
                    break
            except Exception:
                pass
            time.sleep(1)

        assert has_interim, (
            "Expected at least 1 .interim-caption element via the live pipeline "
            "in mode=both"
        )

        # Verify the interim captions are visible (not hidden)
        interim_display = _js_str(browser.eval_js(
            "getComputedStyle(document.querySelector('.interim-caption')).display"
        ))
        assert interim_display != "none", (
            f"In mode=both, .interim-caption should be visible, got display={interim_display}"
        )

        # Inject a final caption to test that both types render together
        browser.eval_js("""
            addCaption({
                id: 'both-mode-final-001',
                speaker_name: 'TestSpeaker',
                speaker_color: '#4CAF50',
                original_text: 'Texto original de prueba',
                translated_text: 'Test original text',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        # Verify the final caption exists and has both original and translated text
        final_count_str = browser.eval_js(
            "document.querySelectorAll('.final-caption').length"
        ).strip()
        assert int(final_count_str) >= 1, "Final caption should be visible in mode=both"

        # Verify .original-text is visible on the final caption
        original_display = _js_str(browser.eval_js(
            "document.querySelector('.final-caption .original-text') "
            "? getComputedStyle(document.querySelector('.final-caption .original-text')).display "
            ": 'missing'"
        ))
        assert original_display != "none" and original_display != "missing", (
            f"In mode=both, .original-text should be visible, got: {original_display}"
        )

        # Verify .translated-text is visible on the final caption
        translated_display = _js_str(browser.eval_js(
            "document.querySelector('.final-caption .translated-text') "
            "? getComputedStyle(document.querySelector('.final-caption .translated-text')).display "
            ": 'missing'"
        ))
        assert translated_display != "none" and translated_display != "missing", (
            f"In mode=both, .translated-text should be visible, got: {translated_display}"
        )

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_mode_both.png")
        )

    def test_display_mode_translated_only(
        self,
        live_session,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """mode=translated: interim captions are hidden, only final translated text shown.

        In translated mode, handleInterimCaption() returns early without creating
        DOM elements. Any existing .interim-caption elements are set to display:none
        by applyDisplayMode(). Final captions (.final-caption) remain visible with
        .translated-text shown.
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, mode="translated")

        browser.open(url)

        # Give the pipeline time to produce events — in translated mode,
        # handleInterimCaption returns early, so no .interim-caption elements
        # should appear at all.
        browser.wait("5000")

        # Verify no interim captions exist in the DOM (they are never created
        # because handleInterimCaption returns early in translated mode)
        interim_count_str = browser.eval_js(
            "document.querySelectorAll('.interim-caption').length"
        ).strip()
        interim_count = int(interim_count_str)
        assert interim_count == 0, (
            f"In mode=translated, .interim-caption should not be created "
            f"(handleInterimCaption returns early), but found {interim_count}"
        )

        # Inject a final caption to verify translated text is shown
        browser.eval_js("""
            addCaption({
                id: 'translated-mode-final-001',
                speaker_name: 'TestSpeaker',
                speaker_color: '#4CAF50',
                original_text: 'Texto solo traducido',
                translated_text: 'Translated only text',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        # Verify .final-caption is visible (not hidden)
        final_display = _js_str(browser.eval_js(
            "document.querySelector('.final-caption') "
            "? getComputedStyle(document.querySelector('.final-caption')).display "
            ": 'missing'"
        ))
        assert final_display != "none" and final_display != "missing", (
            f"In mode=translated, .final-caption should be visible, got: {final_display}"
        )

        # Verify .translated-text is visible
        translated_text = _js_str(browser.eval_js(
            "document.querySelector('.final-caption .translated-text')?.textContent || ''"
        ))
        assert "Translated only text" in translated_text, (
            f"In mode=translated, .translated-text should show, got: '{translated_text}'"
        )

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_mode_translated.png")
        )

    def test_display_mode_english_only(
        self,
        live_session,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """mode=english: interim captions visible, final captions hidden.

        In english mode:
        - .interim-caption elements are visible (handleInterimCaption proceeds normally)
        - .final-caption elements are hidden (addCaption sets display:none in english mode)
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, mode="english")

        browser.open(url)

        # Wait for interim captions from the live pipeline
        deadline = time.time() + 30
        has_interim = False
        while time.time() < deadline:
            try:
                count_str = browser.eval_js(
                    "document.querySelectorAll('.interim-caption').length"
                ).strip()
                if int(count_str) >= 1:
                    has_interim = True
                    break
            except Exception:
                pass
            time.sleep(1)

        assert has_interim, (
            "Expected at least 1 .interim-caption element via the live pipeline "
            "in mode=english"
        )

        # Verify interim captions are visible
        interim_display = _js_str(browser.eval_js(
            "getComputedStyle(document.querySelector('.interim-caption')).display"
        ))
        assert interim_display != "none", (
            f"In mode=english, .interim-caption should be visible, "
            f"got display={interim_display}"
        )

        # Inject a final caption and verify it is hidden in english mode
        browser.eval_js("""
            addCaption({
                id: 'english-mode-final-001',
                speaker_name: 'TestSpeaker',
                speaker_color: '#4CAF50',
                original_text: 'Solo texto en ingles',
                translated_text: 'English only text',
                duration_seconds: 30
            });
        """)
        browser.wait("500")

        # In english mode, addCaption sets element.style.display = 'none' on
        # .final-caption elements
        final_display = _js_str(browser.eval_js(
            "document.querySelector('.final-caption') "
            "? document.querySelector('.final-caption').style.display "
            ": 'missing'"
        ))
        assert final_display == "none", (
            f"In mode=english, .final-caption should be hidden (display:none), "
            f"got: '{final_display}'"
        )

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_mode_english.png")
        )

    def test_multi_speaker_color_coding(
        self,
        live_session,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """Multiple speakers get distinct .speaker-name elements with different names.

        The mock server uses 2 speakers (Alice and Bob) so we should see at least
        2 distinct speaker names in the DOM once enough captions have arrived.
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, showSpeaker="true")

        browser.open(url)

        # Wait for speaker names to appear. The mock scenario has Alice and Bob,
        # so we need enough pipeline time for both speakers' chunks to arrive.
        deadline = time.time() + 40
        speaker_names = set()
        while time.time() < deadline:
            try:
                names_json = browser.eval_js(
                    "JSON.stringify(Array.from(document.querySelectorAll('.speaker-name'))"
                    ".map(function(el) { return el.textContent.trim(); }))"
                ).strip()
                names = json.loads(names_json)
                speaker_names = set(n for n in names if n)
                if len(speaker_names) >= 2:
                    break
            except Exception:
                pass
            time.sleep(2)

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_multi_speaker.png")
        )

        # We need at least 2 distinct speaker names to verify color coding
        assert len(speaker_names) >= 2, (
            f"Expected at least 2 distinct speaker names (Alice and Bob), "
            f"got {len(speaker_names)}: {speaker_names}"
        )

    def test_caption_expiry_removes_old(
        self,
        live_session,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """Captions are removed from the DOM after their duration expires.

        Opens the page with fadeTime=1 (short fade), injects a caption with a
        short duration_seconds=2, verifies it appears, then waits for it to
        be removed from the DOM.
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, fadeTime="1")

        browser.open(url)
        browser.wait("1000")

        # Inject a caption with a short duration so it expires quickly.
        # duration_seconds=2, fadeTime=1 => fadeDelay = (2000 - 1000) = 1000ms,
        # then fading takes 1000ms, so total ~2s until removed.
        browser.eval_js("""
            addCaption({
                id: 'expiry-test-001',
                speaker_name: 'ExpiryTest',
                speaker_color: '#FF5722',
                original_text: 'Texto temporal',
                translated_text: 'Temporary caption',
                duration_seconds: 2
            });
        """)
        browser.wait("500")

        # Verify the caption is present
        has_caption = _js_str(browser.eval_js(
            "document.querySelector('[data-id=\"expiry-test-001\"]') !== null ? 'yes' : 'no'"
        ))
        assert has_caption == "yes", (
            f"Injected caption should be present initially, got: '{has_caption}'"
        )

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_before_expiry.png")
        )

        # Wait for the caption to expire and be removed.
        # Total expiry time: fadeDelay (1s) + fadeTime (1s) = 2s, plus margin.
        deadline = time.time() + 10
        removed = False
        while time.time() < deadline:
            check = _js_str(browser.eval_js(
                "document.querySelector('[data-id=\"expiry-test-001\"]') !== null ? 'yes' : 'no'"
            ))
            if check == "no":
                removed = True
                break
            time.sleep(0.5)

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_after_expiry.png")
        )

        assert removed, (
            "Caption 'expiry-test-001' should be removed from the DOM after expiry"
        )

    def test_max_captions_limit(
        self,
        live_session,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """At most maxCaptions=3 .caption-box elements are visible at any time.

        Opens with maxCaptions=3, lets the live pipeline stream captions, then
        injects additional captions to exceed the limit. Verifies the DOM never
        has more than 3 .caption-box elements.
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, maxCaptions="3")

        browser.open(url)

        # Wait for the pipeline to produce some captions
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                count_str = browser.eval_js(
                    "document.querySelectorAll('.caption-box').length"
                ).strip()
                if int(count_str) >= 1:
                    break
            except Exception:
                pass
            time.sleep(1)

        # Now inject 5 more captions to push beyond the limit
        for i in range(5):
            browser.eval_js(f"""
                addCaption({{
                    id: 'max-test-{i:03d}',
                    speaker_name: 'Speaker{i}',
                    speaker_color: '#{"4CAF50" if i % 2 == 0 else "FF5722"}',
                    original_text: 'Texto {i}',
                    translated_text: 'Text {i}',
                    duration_seconds: 60
                }});
            """)
            browser.wait("200")

        browser.wait("500")

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_rendering_max_captions.png")
        )

        # Verify at most 3 .caption-box elements
        count_str = browser.eval_js(
            "document.querySelectorAll('.caption-box').length"
        ).strip()
        count = int(count_str)
        assert count <= 3, (
            f"With maxCaptions=3, should have at most 3 .caption-box elements, "
            f"got {count}"
        )
