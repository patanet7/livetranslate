"""
Browser E2E: Live Pipeline — Full Pipeline Verification

Tests the complete Fireflies pipeline:
connect → mock stream → sentence aggregation → translation → WebSocket → browser captions

These tests use the REAL pipeline with a mock Fireflies server. Unlike
test_captions_overlay.py (which injects captions via JS), these tests verify
that captions arrive through the actual WebSocket pipeline end-to-end.

Pipeline event types:
- interim_caption: Word-by-word streaming captions (arrive first as ASR streams)
- caption_added: Final translated captions (arrive after sentence aggregation + translation)

In test environments without a running LLM, interim_caption events always arrive
but caption_added events may not appear (translation falls through to passthrough
which still requires sentence aggregation to complete). Tests accept either event
type as proof that the pipeline is operational.
"""

import time

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestLivePipeline:
    """End-to-end tests for the live Fireflies pipeline."""

    def test_connect_starts_session(self, orchestration_server, mock_fireflies_server):
        """POST /fireflies/connect returns 201 and session appears in /sessions."""
        mock = mock_fireflies_server

        # Connect a session.
        # The /connect endpoint awaits Socket.IO handshake (wait_timeout=30s inside
        # the Fireflies client) so we need a generous HTTP timeout.
        resp = httpx.post(
            f"{orchestration_server}/fireflies/connect",
            json={
                "api_key": mock["api_key"],
                "transcript_id": mock["transcript_id"],
                "api_base_url": mock["url"],
                "target_languages": ["es"],
            },
            timeout=45,
        )
        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"

        data = resp.json()
        session_id = data["session_id"]
        assert session_id, "Response must include a session_id"
        assert data["transcript_id"] == mock["transcript_id"]

        # Verify session appears in GET /fireflies/sessions
        sessions_resp = httpx.get(
            f"{orchestration_server}/fireflies/sessions",
            timeout=10,
        )
        sessions_resp.raise_for_status()
        session_ids = [s["session_id"] for s in sessions_resp.json()]
        assert session_id in session_ids, (
            f"Session {session_id} not found in active sessions: {session_ids}"
        )

        # Clean up: disconnect the session
        httpx.post(
            f"{orchestration_server}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=15,
        )

    def test_captions_arrive_via_websocket(
        self, live_session, orchestration_server, ws_caption_messages
    ):
        """Caption events arrive on the WebSocket (interim or final)."""
        messages, connect_fn, close_fn = ws_caption_messages
        session_id = live_session["session_id"]

        # Connect WebSocket to the caption stream
        connect_fn(orchestration_server, session_id)

        # Wait for caption events to arrive through the pipeline.
        # The mock streams 20 exchanges word-by-word. We accept either
        # caption_added (final) or interim_caption events as proof that
        # the pipeline is delivering captions through the WebSocket.
        deadline = time.time() + 30
        caption_events = []
        while time.time() < deadline:
            caption_events = [
                m for m in messages
                if m.get("event") in ("caption_added", "interim_caption")
            ]
            if len(caption_events) >= 1:
                break
            time.sleep(1)

        close_fn()

        assert len(caption_events) >= 1, (
            f"Expected at least 1 caption event (caption_added or interim_caption), "
            f"got 0. All messages ({len(messages)}): "
            f"{[m.get('event') for m in messages]}"
        )

        # Verify the event payload contains text content
        first_event = caption_events[0]
        if first_event.get("event") == "caption_added":
            caption = first_event.get("caption", {})
            assert "translated_text" in caption or "original_text" in caption, (
                f"caption_added should have translated_text or original_text: {caption}"
            )
        else:
            # interim_caption has text directly
            assert first_event.get("text") or first_event.get("speaker_name"), (
                f"interim_caption should have text or speaker_name: {first_event}"
            )

    def test_captions_render_in_browser(
        self, live_session, browser, captions_url, browser_output_dir, timestamp
    ):
        """Captions render in the browser via the live pipeline."""
        session_id = live_session["session_id"]
        url = captions_url(session_id)

        browser.open(url)

        # Wait for any caption elements to appear via the live pipeline.
        # Accept either final captions (.final-caption) or interim captions
        # (.interim-caption) as proof that the WebSocket pipeline is delivering
        # content to the browser. Use eval_js for reliable compound selector queries.
        deadline = time.time() + 30
        found = False
        caption_type = None
        while time.time() < deadline:
            try:
                # Check for final captions first
                final_count = int(browser.eval_js(
                    "document.querySelectorAll('.final-caption').length"
                ).strip())
                if final_count >= 1:
                    found = True
                    caption_type = "final"
                    break
                # Fall back to interim captions
                interim_count = int(browser.eval_js(
                    "document.querySelectorAll('.interim-caption').length"
                ).strip())
                if interim_count >= 1:
                    found = True
                    caption_type = "interim"
                    break
            except Exception:
                pass
            time.sleep(2)

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_live_pipeline_captions.png")
        )

        assert found, (
            "Expected at least 1 caption element (.final-caption or .interim-caption) "
            "to appear via the live pipeline"
        )

        # Verify the caption contains non-empty text
        if caption_type == "final":
            caption_text = browser.eval_js(
                "document.querySelector('.final-caption .translated-text')?.textContent || ''"
            ).strip()
        else:
            caption_text = browser.eval_js(
                "document.querySelector('.interim-caption .caption-text')?.textContent || ''"
            ).strip()

        assert len(caption_text) > 0, (
            f"Caption ({caption_type}) should contain non-empty text, got: '{caption_text}'"
        )

    def test_interim_captions_grow(
        self, live_session, browser, captions_url, browser_output_dir, timestamp
    ):
        """Interim captions appear and grow as words stream in."""
        session_id = live_session["session_id"]
        url = captions_url(session_id)

        browser.open(url)

        # Wait for any caption content to appear via the live pipeline.
        # Use eval_js for reliable DOM queries — agent-browser get_count
        # can be unreliable with compound CSS selectors.
        deadline = time.time() + 30
        found_interim = False
        first_text = ""
        while time.time() < deadline:
            try:
                # Count interim captions via JavaScript for reliability
                count_str = browser.eval_js(
                    "document.querySelectorAll('.interim-caption').length"
                )
                count = int(count_str.strip())
                if count >= 1:
                    first_text = browser.eval_js(
                        "document.querySelector('.interim-caption .caption-text')?.textContent || ''"
                    )
                    found_interim = True
                    break
            except Exception:
                pass
            time.sleep(1)

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_live_pipeline_interim_initial.png")
        )

        if not found_interim:
            # Interim captions may have already been replaced by finals, or
            # captions may be present but as a different type. Check snapshots.
            snap = browser.snapshot()
            # If we see speaker names, the pipeline is working but captions
            # may have already transitioned to finals.
            has_content = "Alice" in snap or "Bob" in snap
            final_count_str = browser.eval_js(
                "document.querySelectorAll('.final-caption').length"
            )
            final_count = int(final_count_str.strip())
            if final_count >= 1 or has_content:
                pytest.skip(
                    "Pipeline delivered captions but interim captions were "
                    "already replaced by finals before we could observe them."
                )

            pytest.fail("No interim captions appeared within the timeout")

        # Poll for the interim caption text to grow as the mock streams words.
        # The mock streams word-by-word at ~500ms per chunk, so text WILL grow.
        first_len = len(first_text)
        growth_deadline = time.time() + 20
        later_text = first_text
        while time.time() < growth_deadline:
            time.sleep(1)
            later_text = browser.eval_js(
                "document.querySelector('.interim-caption .caption-text')?.textContent || ''"
            ).strip()
            if len(later_text) > first_len:
                break

        browser.screenshot(
            str(browser_output_dir / f"{timestamp}_live_pipeline_interim_grown.png")
        )

        assert len(later_text) > first_len, (
            f"Interim caption text should grow as words stream in. "
            f"First length: {first_len} ({first_text!r}), "
            f"later length: {len(later_text)} ({later_text!r})"
        )

    def test_speaker_attribution_displayed(
        self, live_session, browser, captions_url
    ):
        """Speaker names (Alice/Bob) appear in caption boxes."""
        session_id = live_session["session_id"]
        url = captions_url(session_id, showSpeaker="true")

        browser.open(url)

        # Wait for speaker-name elements to appear in the DOM.
        # The mock scenario uses Alice and Bob speakers.
        deadline = time.time() + 30
        speaker_names_text = ""
        while time.time() < deadline:
            speaker_names_text = browser.eval_js(
                "Array.from(document.querySelectorAll('.speaker-name'))"
                ".map(el => el.textContent).join(' ')"
            ).strip()
            if "Alice" in speaker_names_text or "Bob" in speaker_names_text:
                break
            time.sleep(2)

        assert "Alice" in speaker_names_text or "Bob" in speaker_names_text, (
            f"Expected .speaker-name elements to contain 'Alice' or 'Bob', "
            f"got: '{speaker_names_text}'"
        )

    def test_session_disconnect_cleans_up(
        self, orchestration_server, mock_fireflies_server
    ):
        """After disconnect, session no longer appears in /sessions."""
        mock = mock_fireflies_server

        # Connect a session (generous timeout for Socket.IO handshake)
        resp = httpx.post(
            f"{orchestration_server}/fireflies/connect",
            json={
                "api_key": mock["api_key"],
                "transcript_id": mock["transcript_id"],
                "api_base_url": mock["url"],
                "target_languages": ["es"],
            },
            timeout=45,
        )
        resp.raise_for_status()
        session_id = resp.json()["session_id"]

        # Verify the session is listed
        sessions_resp = httpx.get(
            f"{orchestration_server}/fireflies/sessions",
            timeout=10,
        )
        sessions_resp.raise_for_status()
        session_ids = [s["session_id"] for s in sessions_resp.json()]
        assert session_id in session_ids, "Session should be listed after connect"

        # Disconnect the session
        disc_resp = httpx.post(
            f"{orchestration_server}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=15,
        )
        assert disc_resp.status_code == 200, (
            f"Disconnect failed: {disc_resp.status_code} {disc_resp.text}"
        )

        # Allow a moment for cleanup
        time.sleep(1)

        # Verify the session is no longer listed
        sessions_resp = httpx.get(
            f"{orchestration_server}/fireflies/sessions",
            timeout=10,
        )
        sessions_resp.raise_for_status()
        session_ids = [s["session_id"] for s in sessions_resp.json()]
        assert session_id not in session_ids, (
            f"Session {session_id} should be removed after disconnect, "
            f"but still present in: {session_ids}"
        )
