"""
Browser E2E: Ordering & Dedup -- Captured Realtime Data Tests

Tests ordering, deduplication, and ASR correction handling using
real captured Fireflies stream data (not synthetic scenarios).

Uses captured_realtime_scenario() which models ACTUAL Fireflies API
behavior including:
- Word-by-word streaming with same chunk_id
- ASR text corrections mid-stream ("key cat" -> "kitty cat")
- Multiple chunk_ids active simultaneously (interleaving)
- ~18 updates per chunk before finalization

Pipeline event flow:
  mock server -> Socket.IO -> FirefliesRealtimeClient -> LiveCaptionManager
  -> grow-only filter -> WebSocket broadcast -> captions.html

The grow-only filter in LiveCaptionManager:
- type "grow": text is new or longer (allowed)
- type "correction": text was rewritten but is longer (allowed)
- type "final": last update for this chunk_id (always allowed)
- Shrinks are suppressed (not broadcast)
"""

import asyncio
import json
import logging
import threading
import time

import httpx
import pytest

from fireflies.mocks.fireflies_mock_server import (
    FirefliesMockServer,
    MockTranscriptScenario,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.e2e, pytest.mark.browser]

# Use a DIFFERENT port from the session-scoped mock server (8090)
CAPTURED_MOCK_PORT = 8091

TEST_API_KEY = "test-api-key"  # pragma: allowlist secret

# Total chunks in captured scenario: 18 + 18 + 5 + 2 = 43
# At 50ms each = ~2.2s streaming, but pipeline processing adds latency
STREAM_WAIT_SECONDS = 30


# =============================================================================
# Helper: connect a captured session (non-fixture, for manual lifecycle)
# =============================================================================


def _connect_captured_session(orchestration_server: str, mock: dict) -> dict:
    """Connect a Fireflies session using the captured mock data.

    Returns dict with session_id and transcript_id.
    The caller is responsible for disconnecting on cleanup.
    """
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
    data = resp.json()
    return {
        "session_id": data["session_id"],
        "transcript_id": mock["transcript_id"],
    }


def _disconnect_session(orchestration_server: str, session_id: str) -> None:
    """Disconnect a Fireflies session. Ignores errors."""
    try:
        httpx.post(
            f"{orchestration_server}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=10,
        )
    except Exception:
        logger.warning("disconnect failed for session %s", session_id, exc_info=True)


# =============================================================================
# Fixtures: Captured realtime mock server (module-scoped)
# =============================================================================


@pytest.fixture(scope="module")
def captured_mock_server():
    """Mock server with captured realtime data (real Fireflies behavior).

    Runs on port 8091 (different from session-scoped mock on 8090).
    Uses captured_realtime_scenario() with 50ms word delay for fast testing.
    """
    server = FirefliesMockServer(
        host="localhost",
        port=CAPTURED_MOCK_PORT,
        valid_api_keys={TEST_API_KEY},
    )
    scenario = MockTranscriptScenario.captured_realtime_scenario(word_delay_ms=50)
    transcript_id = server.add_scenario(scenario)

    # Start server event loop in background thread
    loop = asyncio.new_event_loop()
    loop.run_until_complete(server.start())

    # run_forever() in daemon thread keeps the event loop processing I/O
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()

    logger.info(
        f"Captured mock server started on port {CAPTURED_MOCK_PORT}, "
        f"transcript_id={transcript_id}"
    )

    yield {
        "server": server,
        "url": f"http://localhost:{CAPTURED_MOCK_PORT}",
        "transcript_id": transcript_id,
        "api_key": TEST_API_KEY,
    }

    # Shutdown
    asyncio.run_coroutine_threadsafe(server.stop(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join(timeout=5)
    if not loop.is_closed():
        loop.close()
    logger.info("Captured mock server stopped")


@pytest.fixture
def captured_live_session(orchestration_server, captured_mock_server):
    """Connect a live session using captured realtime data.

    POSTs to /fireflies/connect with the captured mock server URL.
    Yields session info. Disconnects on teardown.
    """
    info = _connect_captured_session(orchestration_server, captured_mock_server)

    yield info

    _disconnect_session(orchestration_server, info["session_id"])


# =============================================================================
# Test Class
# =============================================================================


class TestOrderingDedup:
    """Ordering, deduplication, and ASR correction tests with captured realtime data."""

    def test_no_duplicate_captions(
        self, captured_live_session, orchestration_server, ws_caption_messages
    ):
        """Stream captured data, verify no duplicate (chunk_id, text) pairs in interim events.

        The LiveCaptionManager dedup filter suppresses pure duplicate text
        for the same chunk_id. After the grow-only filter, every broadcast
        interim_caption event should have unique (chunk_id, text) pairs.
        """
        messages, connect_fn, close_fn = ws_caption_messages
        session_id = captured_live_session["session_id"]

        connect_fn(orchestration_server, session_id)

        # Wait for interim caption events to arrive
        deadline = time.time() + STREAM_WAIT_SECONDS
        while time.time() < deadline:
            interim_events = [
                m for m in messages if m.get("event") == "interim_caption"
            ]
            # We expect events from multiple chunk_ids (65174, 65175, 65176, 65177)
            chunk_ids_seen = {m.get("chunk_id") for m in interim_events}
            if len(chunk_ids_seen) >= 2 and len(interim_events) >= 5:
                break
            time.sleep(1)

        close_fn()

        interim_events = [m for m in messages if m.get("event") == "interim_caption"]
        assert len(interim_events) >= 3, (
            f"Expected at least 3 interim_caption events, got {len(interim_events)}. "
            f"All events: {[m.get('event') for m in messages]}"
        )

        # Check for duplicate (chunk_id, text) pairs
        seen_pairs: set[tuple[str, str]] = set()
        duplicates: list[dict] = []

        for event in interim_events:
            chunk_id = event.get("chunk_id", "")
            text = event.get("text", "")
            pair = (chunk_id, text)

            if pair in seen_pairs:
                duplicates.append(event)
            else:
                seen_pairs.add(pair)

        assert len(duplicates) == 0, (
            f"Found {len(duplicates)} duplicate (chunk_id, text) pairs: "
            f"{[(d.get('chunk_id'), d.get('text')[:50]) for d in duplicates]}"
        )

    def test_captions_in_chronological_order(
        self, captured_live_session, orchestration_server, ws_caption_messages
    ):
        """Verify chunk_ids appear in chronological order (65174 before 65175 before 65176).

        The captured scenario streams chunks in order: 65174, 65175, 65176, 65177.
        The FIRST appearance of each chunk_id in interim_caption events should
        follow this ordering.
        """
        messages, connect_fn, close_fn = ws_caption_messages
        session_id = captured_live_session["session_id"]

        connect_fn(orchestration_server, session_id)

        # Wait for events from multiple chunk_ids
        deadline = time.time() + STREAM_WAIT_SECONDS
        while time.time() < deadline:
            interim_events = [
                m for m in messages if m.get("event") == "interim_caption"
            ]
            chunk_ids_seen = {m.get("chunk_id") for m in interim_events}
            if len(chunk_ids_seen) >= 3:
                break
            time.sleep(1)

        close_fn()

        interim_events = [m for m in messages if m.get("event") == "interim_caption"]
        assert len(interim_events) >= 3, (
            f"Expected at least 3 interim events, got {len(interim_events)}"
        )

        # Extract first-appearance order of chunk_ids
        first_appearance_order: list[str] = []
        seen: set[str] = set()
        for event in interim_events:
            cid = event.get("chunk_id", "")
            if cid and cid not in seen:
                first_appearance_order.append(cid)
                seen.add(cid)

        # Verify chronological order: each chunk_id should be >= previous
        for i in range(1, len(first_appearance_order)):
            prev = int(first_appearance_order[i - 1])
            curr = int(first_appearance_order[i])
            assert curr >= prev, (
                f"Chunk IDs not in chronological order: "
                f"{first_appearance_order[i - 1]} appeared before "
                f"{first_appearance_order[i]}, but {curr} < {prev}. "
                f"Full order: {first_appearance_order}"
            )

    def test_interim_never_shrinks_in_browser(
        self,
        captured_live_session,
        orchestration_server,
        ws_caption_messages,
    ):
        """Verify the grow-only filter never delivers shrinking text to the browser.

        The server-side grow-only filter in LiveCaptionManager suppresses shrinks.
        We verify this via the WebSocket caption stream: for any given chunk_id,
        the text length of interim_caption events should only increase or stay
        the same (never decrease).

        This directly validates what the browser would render, since captions.html
        uses the WebSocket stream as its sole data source.
        """
        messages, connect_fn, close_fn = ws_caption_messages
        session_id = captured_live_session["session_id"]

        connect_fn(orchestration_server, session_id)

        # Wait for interim caption events to arrive
        deadline = time.time() + STREAM_WAIT_SECONDS
        while time.time() < deadline:
            interim_events = [
                m for m in messages if m.get("event") == "interim_caption"
            ]
            chunk_ids_seen = {m.get("chunk_id") for m in interim_events}
            if len(chunk_ids_seen) >= 2 and len(interim_events) >= 5:
                break
            time.sleep(1)

        close_fn()

        interim_events = [m for m in messages if m.get("event") == "interim_caption"]
        assert len(interim_events) >= 3, (
            f"Expected at least 3 interim_caption events, got {len(interim_events)}. "
            f"All events: {[m.get('event') for m in messages]}"
        )

        # Track max text length per chunk_id and look for shrinks
        max_lengths: dict[str, int] = {}
        shrinks: list[dict] = []

        for event in interim_events:
            chunk_id = event.get("chunk_id", "")
            text = event.get("text", "")
            text_len = len(text)

            prev_max = max_lengths.get(chunk_id, 0)
            if text_len < prev_max:
                shrinks.append({
                    "chunk_id": chunk_id,
                    "prev_max": prev_max,
                    "new_len": text_len,
                    "text": text[:50],
                    "type": event.get("type", ""),
                })
            if text_len > prev_max:
                max_lengths[chunk_id] = text_len

        assert len(shrinks) == 0, (
            f"Grow-only filter failed: {len(shrinks)} shrink(s) detected in "
            f"interim_caption events: {shrinks[:5]}. "
            f"Max lengths: {max_lengths}"
        )

    def test_asr_corrections_handled(
        self, captured_live_session, orchestration_server, ws_caption_messages
    ):
        """Verify ASR correction from "key cat" to "kitty cat" in chunk 65174.

        The captured_realtime_scenario has chunk_id=65174 with text progression
        that includes "key cat" early on and "kitty cat" as the final correction.
        The grow-only filter allows corrections (text changed but longer), so
        the final interim text for chunk 65174 should contain "kitty cat".
        """
        messages, connect_fn, close_fn = ws_caption_messages
        session_id = captured_live_session["session_id"]

        connect_fn(orchestration_server, session_id)

        # Wait for chunk 65174 events to arrive
        deadline = time.time() + STREAM_WAIT_SECONDS
        while time.time() < deadline:
            chunk_65174_events = [
                m
                for m in messages
                if m.get("event") == "interim_caption" and m.get("chunk_id") == "65174"
            ]
            # The scenario has 18 updates for 65174, but many may be suppressed
            # by the grow-only filter. Wait until we see a few.
            if len(chunk_65174_events) >= 3:
                break
            time.sleep(1)

        close_fn()

        chunk_65174_events = [
            m
            for m in messages
            if m.get("event") == "interim_caption" and m.get("chunk_id") == "65174"
        ]
        assert len(chunk_65174_events) >= 1, (
            f"Expected events for chunk_id=65174, got none. "
            f"All chunk_ids: {set(m.get('chunk_id') for m in messages if m.get('event') == 'interim_caption')}"
        )

        # The last broadcast event for chunk 65174 should contain the corrected text.
        # Due to the grow-only filter, the correction "key cat" -> "kitty cat" should
        # be classified as "correction" (longer text) and broadcast.
        last_65174_text = chunk_65174_events[-1].get("text", "")

        assert "kitty cat" in last_65174_text, (
            f"Final interim text for chunk 65174 should contain 'kitty cat' "
            f"(ASR correction), but got: '{last_65174_text}'. "
            f"All 65174 texts: {[e.get('text', '')[-40:] for e in chunk_65174_events]}"
        )

    def test_interleaved_chunks_produce_separate_captions(
        self,
        orchestration_server,
        captured_mock_server,
        browser,
        captions_url,
        browser_output_dir,
        timestamp,
    ):
        """Chunk 65174 and 65175 overlap in time; verify they produce distinct elements.

        In the captured data, chunk 65175 starts before 65174 is finalized.
        Both should appear as separate .interim-caption elements with different IDs.

        Monkey-patches handleInterimCaption to track every chunk_id seen,
        even if DOM elements are short-lived due to enforceMaxCaptions.
        """
        session_info = _connect_captured_session(
            orchestration_server, captured_mock_server
        )
        session_id = session_info["session_id"]

        try:
            url = captions_url(session_id)
            browser.open(url)

            # Monkey-patch handleInterimCaption to track all chunk_ids seen.
            browser.eval_js("""
                window.__interleaveTracker = { allChunkIds: [], updates: 0 };

                var _origHandleInterim2 = handleInterimCaption;
                handleInterimCaption = function(data) {
                    window.__interleaveTracker.updates++;
                    var cid = data.chunk_id;
                    if (window.__interleaveTracker.allChunkIds.indexOf(cid) === -1) {
                        window.__interleaveTracker.allChunkIds.push(cid);
                    }
                    _origHandleInterim2(data);
                };
            """)

            # Wait for multiple chunk_ids to be observed
            deadline = time.time() + STREAM_WAIT_SECONDS
            while time.time() < deadline:
                try:
                    count_str = browser.eval_js(
                        "String(window.__interleaveTracker.allChunkIds.length)"
                    ).strip()
                    count = int(count_str) if count_str.isdigit() else 0
                    if count >= 2:
                        break
                except Exception as e:
                    logger.debug(f"Polling tracker: {e}")
                time.sleep(0.5)

            browser.screenshot(
                str(browser_output_dir / f"{timestamp}_ordering_dedup_interleaved.png")
            )

            # Read all tracked chunk_ids
            ids_str = browser.eval_js(
                "JSON.stringify(window.__interleaveTracker.allChunkIds)"
            ).strip()
            all_chunk_ids = json.loads(ids_str)
            updates_str = browser.eval_js(
                "String(window.__interleaveTracker.updates)"
            ).strip()

            assert len(all_chunk_ids) >= 2, (
                f"Expected at least 2 different chunk_ids from interleaved streaming, "
                f"got {len(all_chunk_ids)}: {all_chunk_ids}. "
                f"Total updates: {updates_str}"
            )

            # Verify the chunk_ids are distinct numeric values
            unique_ids = set(all_chunk_ids)
            assert len(unique_ids) >= 2, (
                f"Expected at least 2 unique chunk_ids, got: {unique_ids}"
            )

        finally:
            _disconnect_session(orchestration_server, session_id)

    def test_small_fragments_grow(
        self, captured_live_session, orchestration_server, ws_caption_messages
    ):
        """Very short initial fragments grow into longer text.

        Chunk 65175 starts with just "but" (1 word). The grow-only filter
        allows short text as long as it is growing. Verify that the FINAL
        interim text for each chunk_id is longer than its initial text.
        """
        messages, connect_fn, close_fn = ws_caption_messages
        session_id = captured_live_session["session_id"]

        connect_fn(orchestration_server, session_id)

        # Wait for substantial data to arrive
        deadline = time.time() + STREAM_WAIT_SECONDS
        while time.time() < deadline:
            interim_events = [
                m for m in messages if m.get("event") == "interim_caption"
            ]
            chunk_ids_seen = {m.get("chunk_id") for m in interim_events}
            if len(chunk_ids_seen) >= 2 and len(interim_events) >= 5:
                break
            time.sleep(1)

        close_fn()

        interim_events = [m for m in messages if m.get("event") == "interim_caption"]
        assert len(interim_events) >= 3, (
            f"Expected at least 3 interim events, got {len(interim_events)}"
        )

        # Group events by chunk_id, track first and last text
        chunk_texts: dict[str, list[str]] = {}
        for event in interim_events:
            cid = event.get("chunk_id", "")
            text = event.get("text", "")
            if cid:
                chunk_texts.setdefault(cid, []).append(text)

        # For each chunk_id that has multiple updates, verify final > initial
        chunks_that_grew = 0
        for cid, texts in chunk_texts.items():
            if len(texts) >= 2:
                first_text = texts[0]
                last_text = texts[-1]
                assert len(last_text) > len(first_text), (
                    f"chunk_id={cid}: final text should be longer than initial. "
                    f"Initial ({len(first_text)}): '{first_text}', "
                    f"Final ({len(last_text)}): '{last_text}'"
                )
                chunks_that_grew += 1

        assert chunks_that_grew >= 1, (
            f"Expected at least 1 chunk_id with text growth, "
            f"got {chunks_that_grew}. "
            f"Chunk update counts: {[(k, len(v)) for k, v in chunk_texts.items()]}"
        )
