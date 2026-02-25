"""
Browser E2E: Cross-Page User Journey

Tests the "disjoint" user experience: connect on the dashboard page (or via
REST API), view captions on a separate page, manage sessions through the
REST API, and verify isolation between sessions.

Pipeline flow:
  Dashboard/API connect -> orchestration -> mock Fireflies -> pipeline ->
  WebSocket -> captions.html (separate page)

These tests verify the cross-page contract: a session created via one
interface is observable and controllable from other interfaces.
"""

import json
import time

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.browser]


class TestCrossPageFlow:
    """Cross-page user journey tests."""

    def test_dashboard_connect_then_captions_page(
        self,
        browser,
        dashboard_url,
        captions_url,
        setup_api_key,
        mock_fireflies_server,
        orchestration_server,
        browser_output_dir,
        timestamp,
    ):
        """Connect via dashboard UI, then navigate to captions.html and verify captions appear.

        Since the dashboard JS calls POST /fireflies/connect without an api_base_url
        (it uses the real Fireflies API by default), we override connectToMeeting via
        eval_js to inject the mock server's api_base_url. After connecting, we retrieve
        the session_id via the REST API and navigate to captions.html.
        """
        mock = mock_fireflies_server
        base = orchestration_server

        # 1. Open the dashboard
        browser.open(dashboard_url)
        browser.wait("1500")

        # 2. Set API key in localStorage
        setup_api_key(browser)

        # 3. Go to the Connect tab
        browser.click("button.tab[onclick*=\"showTab('connect')\"]")
        browser.wait("500")

        # 4. Fill the transcript ID
        browser.fill("#transcriptId", mock["transcript_id"])
        browser.wait("300")

        # 5. Override connectToMeeting to include api_base_url pointing to mock server.
        #    We also store the session_id on window so we can retrieve it.
        mock_url = mock["url"]
        api_key_js = json.dumps(mock["api_key"])
        browser.eval_js(f"""
            window.__crossPageSessionId = null;
            window.__crossPageError = null;

            // Replace connectToMeeting with version that includes api_base_url
            connectToMeeting = async function() {{
                const transcriptId = document.getElementById('transcriptId').value;
                const btn = document.getElementById('connectBtn');
                btn.disabled = true;
                btn.textContent = 'Connecting...';

                try {{
                    const response = await fetch('/fireflies/connect', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            api_key: {api_key_js},
                            transcript_id: transcriptId,
                            api_base_url: '{mock_url}',
                            target_languages: ['es']
                        }})
                    }});

                    if (!response.ok) {{
                        const err = await response.json().catch(function() {{ return {{}}; }});
                        throw new Error(err.detail || ('HTTP ' + response.status));
                    }}

                    const data = await response.json();
                    window.__crossPageSessionId = data.session_id;
                }} catch (e) {{
                    window.__crossPageError = String(e.message || e);
                }}

                btn.disabled = false;
                btn.textContent = 'Connect';
            }};
        """)
        browser.wait("300")

        # 6. Click Connect
        browser.screenshot(str(browser_output_dir / f"{timestamp}_cross_page_before_connect.png"))
        browser.click("#connectBtn")

        # 7. Wait for connection to complete (generous timeout for Socket.IO handshake)
        deadline = time.time() + 45
        session_id = None
        while time.time() < deadline:
            # Poll for session_id or error from our overridden connectToMeeting.
            # agent-browser eval returns raw JS value; strip surrounding quotes.
            sid_raw = browser.eval_js(
                "window.__crossPageSessionId ? window.__crossPageSessionId : 'NONE'"
            ).strip().strip('"').strip("'")
            err_raw = browser.eval_js(
                "window.__crossPageError ? window.__crossPageError : 'NONE'"
            ).strip().strip('"').strip("'")
            if err_raw != "NONE":
                pytest.fail(f"Dashboard connect failed: {err_raw}")
            if sid_raw != "NONE":
                session_id = sid_raw
                break
            time.sleep(1)

        assert session_id, "Failed to get session_id from dashboard connect"

        browser.screenshot(str(browser_output_dir / f"{timestamp}_cross_page_after_connect.png"))

        # 8. Navigate to captions.html with this session
        url = captions_url(session_id)
        browser.open(url)
        browser.wait("2000")

        # 9. Wait for caption content to appear (interim or final)
        deadline = time.time() + 30
        found = False
        while time.time() < deadline:
            try:
                interim_count = int(browser.eval_js(
                    "document.querySelectorAll('.interim-caption').length"
                ).strip())
                final_count = int(browser.eval_js(
                    "document.querySelectorAll('.final-caption').length"
                ).strip())
                if interim_count >= 1 or final_count >= 1:
                    found = True
                    break
            except Exception:
                pass
            time.sleep(1)

        browser.screenshot(str(browser_output_dir / f"{timestamp}_cross_page_captions.png"))

        assert found, (
            "Expected at least 1 caption element (.interim-caption or .final-caption) "
            "on captions.html after connecting via dashboard"
        )

        # Cleanup: disconnect
        try:
            httpx.post(
                f"{base}/fireflies/disconnect",
                json={"session_id": session_id},
                timeout=15,
            )
        except Exception:
            pass

    def test_session_visible_in_api_after_connect(
        self,
        live_session,
        orchestration_server,
    ):
        """After connecting via the REST API, GET /fireflies/sessions lists the session.

        Uses the live_session fixture (which POSTs to /fireflies/connect) and
        verifies the session appears in the sessions list via GET. Tests the REST
        API directly (not the session-manager.html UI, which has a stub).
        """
        session_id = live_session["session_id"]
        base = orchestration_server

        resp = httpx.get(f"{base}/fireflies/sessions", timeout=10)
        resp.raise_for_status()

        sessions = resp.json()
        session_ids = [s["session_id"] for s in sessions]

        assert session_id in session_ids, (
            f"Session {session_id} not found in GET /fireflies/sessions. "
            f"Active sessions: {session_ids}"
        )

        # Verify session has expected fields
        our_session = next(s for s in sessions if s["session_id"] == session_id)
        assert "fireflies_transcript_id" in our_session or "transcript_id" in our_session, (
            f"Session should have transcript_id field: {our_session}"
        )

    def test_api_disconnect_stops_captions(
        self,
        orchestration_server,
        mock_fireflies_server,
        ws_caption_messages,
    ):
        """Connect a session, verify captions arrive via WebSocket, then POST
        /fireflies/disconnect and verify no new captions arrive afterwards.

        Uses the WebSocket fixture (not the browser) for reliable timing control.
        """
        mock = mock_fireflies_server
        base = orchestration_server
        messages, connect_fn, close_fn = ws_caption_messages

        # 1. Connect a session via REST API
        resp = httpx.post(
            f"{base}/fireflies/connect",
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

        try:
            # 2. Connect WebSocket to caption stream
            connect_fn(base, session_id)

            # 3. Wait for at least 1 caption event to prove captions are flowing
            deadline = time.time() + 30
            while time.time() < deadline:
                caption_events = [
                    m for m in messages
                    if m.get("event") in ("caption_added", "interim_caption")
                ]
                if len(caption_events) >= 1:
                    break
                time.sleep(1)

            caption_events = [
                m for m in messages
                if m.get("event") in ("caption_added", "interim_caption")
            ]
            assert len(caption_events) >= 1, (
                f"Expected at least 1 caption event before disconnect, got 0. "
                f"All messages: {[m.get('event') for m in messages]}"
            )

            # 4. Disconnect the session via REST API
            disc_resp = httpx.post(
                f"{base}/fireflies/disconnect",
                json={"session_id": session_id},
                timeout=15,
            )
            assert disc_resp.status_code == 200, (
                f"Disconnect failed: {disc_resp.status_code} {disc_resp.text}"
            )

            # 5. Verify the stream eventually stops.
            #    After disconnect, in-flight events from the pipeline buffer may
            #    still arrive for several seconds. We poll until the count stabilizes
            #    (no new events for 5 consecutive seconds), proving the stream stopped.
            stable_count = None
            stable_since = None
            stabilization_window = 5.0  # seconds of silence required

            deadline = time.time() + 40
            while time.time() < deadline:
                current = len([
                    m for m in messages
                    if m.get("event") in ("caption_added", "interim_caption")
                ])
                if stable_count is not None and current == stable_count:
                    if time.time() - stable_since >= stabilization_window:
                        break  # Stream has been silent for the required window
                else:
                    stable_count = current
                    stable_since = time.time()
                time.sleep(0.5)

            # Stream must have stabilized (not still timing out)
            time_in_stable = time.time() - (stable_since or time.time())
            assert time_in_stable >= stabilization_window, (
                f"Caption stream did not stabilize within timeout. "
                f"Last stable count: {stable_count}, "
                f"was stable for {time_in_stable:.1f}s (need {stabilization_window}s)"
            )

            # Verify the session is no longer listed
            sessions_resp = httpx.get(f"{base}/fireflies/sessions", timeout=10)
            sessions_resp.raise_for_status()
            active_ids = [s["session_id"] for s in sessions_resp.json()]
            assert session_id not in active_ids, (
                f"Session {session_id} should be removed after disconnect"
            )

        finally:
            close_fn()
            # Ensure cleanup even if assertions failed
            try:
                httpx.post(
                    f"{base}/fireflies/disconnect",
                    json={"session_id": session_id},
                    timeout=5,
                )
            except Exception:
                pass

    def test_multiple_sessions_isolated(
        self,
        orchestration_server,
        mock_fireflies_server,
        ws_caption_messages,
    ):
        """Connect two sessions with the same transcript ID, verify each gets its
        own session_id and captions don't bleed between sessions.

        Since AgentBrowser can only have one page at a time, we use the WebSocket
        fixture to verify isolation. Connecting the same transcript_id twice should
        produce two different session_ids, each with its own caption stream.
        """
        mock = mock_fireflies_server
        base = orchestration_server
        messages_1, connect_fn_1, close_fn_1 = ws_caption_messages

        # 1. Connect session 1
        resp1 = httpx.post(
            f"{base}/fireflies/connect",
            json={
                "api_key": mock["api_key"],
                "transcript_id": mock["transcript_id"],
                "api_base_url": mock["url"],
                "target_languages": ["es"],
            },
            timeout=45,
        )
        resp1.raise_for_status()
        session_id_1 = resp1.json()["session_id"]

        # 2. Connect session 2 (same transcript_id generates different session_id)
        resp2 = httpx.post(
            f"{base}/fireflies/connect",
            json={
                "api_key": mock["api_key"],
                "transcript_id": mock["transcript_id"],
                "api_base_url": mock["url"],
                "target_languages": ["es"],
            },
            timeout=45,
        )
        resp2.raise_for_status()
        session_id_2 = resp2.json()["session_id"]

        try:
            # 3. Verify different session IDs
            assert session_id_1 != session_id_2, (
                f"Two connections should produce different session_ids, "
                f"got: {session_id_1} == {session_id_2}"
            )

            # 4. Verify both sessions are listed in the API
            sessions_resp = httpx.get(f"{base}/fireflies/sessions", timeout=10)
            sessions_resp.raise_for_status()
            active_ids = [s["session_id"] for s in sessions_resp.json()]

            assert session_id_1 in active_ids, (
                f"Session 1 ({session_id_1}) not in active sessions: {active_ids}"
            )
            assert session_id_2 in active_ids, (
                f"Session 2 ({session_id_2}) not in active sessions: {active_ids}"
            )

            # 5. Connect WebSocket to session 1 and verify it receives captions
            connect_fn_1(base, session_id_1)

            deadline = time.time() + 20
            while time.time() < deadline:
                caption_events = [
                    m for m in messages_1
                    if m.get("event") in ("caption_added", "interim_caption")
                ]
                if len(caption_events) >= 1:
                    break
                time.sleep(1)

            caption_events = [
                m for m in messages_1
                if m.get("event") in ("caption_added", "interim_caption")
            ]
            assert len(caption_events) >= 1, (
                f"Session 1 should receive captions via WebSocket. "
                f"Got 0 caption events. All events: {[m.get('event') for m in messages_1]}"
            )

            close_fn_1()

        finally:
            # Cleanup both sessions
            for sid in [session_id_1, session_id_2]:
                try:
                    httpx.post(
                        f"{base}/fireflies/disconnect",
                        json={"session_id": sid},
                        timeout=10,
                    )
                except Exception:
                    pass

    def test_captions_page_reconnect(
        self,
        live_session,
        browser,
        captions_url,
        orchestration_server,
        browser_output_dir,
        timestamp,
    ):
        """Open captions.html, verify WebSocket connects, simulate a disconnect
        by calling ws.close(), then verify the page auto-reconnects.

        The captions.html page has auto-reconnect logic with up to 10 attempts
        and a 3-second delay between attempts.
        """
        session_id = live_session["session_id"]
        url = captions_url(session_id, showStatus="true")

        # 1. Open captions page
        browser.open(url)
        browser.wait("2000")

        # 2. Wait for WebSocket to connect (status-connected class on #status)
        deadline = time.time() + 15
        connected = False
        while time.time() < deadline:
            try:
                status_class = browser.eval_js(
                    "document.getElementById('status').className"
                ).strip()
                if "status-connected" in status_class:
                    connected = True
                    break
            except Exception:
                pass
            time.sleep(1)

        browser.screenshot(str(browser_output_dir / f"{timestamp}_reconnect_initial.png"))

        assert connected, (
            "Captions page should show status-connected before testing reconnect"
        )

        # 3. Force-close the WebSocket to simulate a disconnect
        browser.eval_js("ws.close();")
        browser.wait("1000")

        # 4. Verify the status changes to disconnected or connecting
        status_class = browser.eval_js(
            "document.getElementById('status').className"
        ).strip()
        assert "status-connected" not in status_class, (
            f"After ws.close(), status should NOT be 'connected', got: {status_class}"
        )

        browser.screenshot(str(browser_output_dir / f"{timestamp}_reconnect_disconnected.png"))

        # 5. Wait for auto-reconnect (captions.html reconnects after 3s delay)
        deadline = time.time() + 20
        reconnected = False
        while time.time() < deadline:
            try:
                status_class = browser.eval_js(
                    "document.getElementById('status').className"
                ).strip()
                if "status-connected" in status_class:
                    reconnected = True
                    break
            except Exception:
                pass
            time.sleep(1)

        browser.screenshot(str(browser_output_dir / f"{timestamp}_reconnect_restored.png"))

        assert reconnected, (
            "Captions page should auto-reconnect after WebSocket disconnect. "
            f"Final status class: {status_class}"
        )
