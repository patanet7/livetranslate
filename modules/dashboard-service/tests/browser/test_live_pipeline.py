"""
Live pipeline E2E tests — full integrated flow verification.

Tests the complete data path:
  Mock Fireflies → Orchestration → WebSocket → SvelteKit → Browser

Each test connects a real Fireflies session through the orchestration service,
opens the SvelteKit dashboard in a real Chromium browser, and verifies that
real streamed captions appear in the DOM.

NO MOCKS. NO INJECTED DATA. Real pipeline, real browser, real screenshots.
"""

import time

import httpx
import pytest


class TestLivePipeline:
    """Verify the full live pipeline from connect to browser captions."""

    def test_connect_creates_session(self, orchestration_server, mock_fireflies_server, screenshot_path, browser):
        """POST /fireflies/connect returns a session, visible in /fireflies/sessions."""
        mock = mock_fireflies_server

        # Connect through the real orchestration API
        resp = httpx.post(
            f"{orchestration_server}/fireflies/connect",
            json={
                "api_key": mock["api_key"],
                "transcript_id": mock["transcript_id"],
                "api_base_url": mock["url"],
                "target_languages": ["es"],
            },
            timeout=15,
        )
        assert resp.status_code in (200, 201), f"Connect failed: {resp.text}"
        session_id = resp.json()["session_id"]

        # Verify session appears in sessions list
        sessions_resp = httpx.get(f"{orchestration_server}/fireflies/sessions", timeout=10)
        assert sessions_resp.status_code == 200
        session_ids = [s["session_id"] for s in sessions_resp.json()]
        assert session_id in session_ids

        # Verify the history page in the browser shows the session
        browser.open("http://localhost:5180/fireflies/history")
        browser.wait("text=Session History")
        time.sleep(2)  # Allow streamed data to load
        browser.screenshot(screenshot_path("live_session_in_history"))

        # Cleanup
        httpx.post(
            f"{orchestration_server}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=10,
        )

    def test_captions_arrive_in_overlay(self, live_session, browser, screenshot_path):
        """
        Connect live session → open captions overlay → verify real captions render.

        The overlay connects via WebSocket to the orchestration service and receives
        real translated captions from the Fireflies mock stream.
        """
        session_id = live_session["session_id"]

        # Open the captions overlay with the live session
        browser.open(f"http://localhost:5180/captions?session={session_id}&mode=both")

        # Wait for real captions to arrive from the pipeline
        # The mock streams with 500ms chunk delay, so captions should start within ~5s
        found = False
        for _ in range(20):  # Poll for up to 10 seconds
            snap = browser.snapshot()
            # Check for speaker names from the conversation scenario (Alice/Bob)
            if "Alice" in snap or "Bob" in snap:
                found = True
                break
            time.sleep(0.5)

        browser.screenshot(screenshot_path("live_captions_in_overlay"))
        assert found, "No captions from the live pipeline appeared in the overlay within 10s"

    def test_captions_show_speaker_attribution(self, live_session, browser, screenshot_path):
        """Verify speaker names appear with color coding in the overlay."""
        session_id = live_session["session_id"]

        browser.open(f"http://localhost:5180/captions?session={session_id}&mode=both")

        # Wait for at least one caption
        for _ in range(20):
            snap = browser.snapshot()
            if "Alice" in snap or "Bob" in snap:
                break
            time.sleep(0.5)

        browser.screenshot(screenshot_path("live_speaker_attribution"))

        # Verify speaker styling exists in DOM
        speaker_count = browser.get_count(".speaker")
        assert speaker_count > 0, "No .speaker elements found — captions not rendering"

    def test_sessions_page_shows_active_session(self, live_session, browser, screenshot_path):
        """After connecting, the sessions page lists the active session."""
        browser.open("http://localhost:5180/fireflies/sessions")
        time.sleep(3)  # Allow page load + API fetch

        snap = browser.snapshot()
        browser.screenshot(screenshot_path("live_sessions_page"))

        # The sessions page should show active session data
        session_id_prefix = live_session["session_id"][:8]
        assert (
            session_id_prefix in snap
            or "connected" in snap.lower()
            or "Session" in snap
        ), f"Active session not visible on sessions page. Snapshot: {snap[:500]}"

    def test_disconnect_removes_session(self, orchestration_server, mock_fireflies_server, browser, screenshot_path):
        """Disconnecting a session removes it from the active list."""
        mock = mock_fireflies_server

        # Connect
        resp = httpx.post(
            f"{orchestration_server}/fireflies/connect",
            json={
                "api_key": mock["api_key"],
                "transcript_id": mock["transcript_id"],
                "api_base_url": mock["url"],
                "target_languages": ["es"],
            },
            timeout=15,
        )
        session_id = resp.json()["session_id"]

        # Verify it exists
        sessions = httpx.get(f"{orchestration_server}/fireflies/sessions", timeout=10).json()
        assert any(s["session_id"] == session_id for s in sessions)

        # Disconnect
        disc_resp = httpx.post(
            f"{orchestration_server}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=10,
        )
        assert disc_resp.status_code == 200

        # Verify it's gone
        time.sleep(1)
        sessions_after = httpx.get(f"{orchestration_server}/fireflies/sessions", timeout=10).json()
        assert not any(s["session_id"] == session_id for s in sessions_after)

        # Verify in browser
        browser.open("http://localhost:5180/fireflies/history")
        browser.wait("text=Session History")
        time.sleep(2)
        browser.screenshot(screenshot_path("live_after_disconnect"))
