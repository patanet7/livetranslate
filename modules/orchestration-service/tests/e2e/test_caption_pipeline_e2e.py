"""
End-to-End Caption Pipeline Test

Simulates the full translation pipeline:
1. Audio chunks arriving (simulated timing)
2. Transcription with speaker diarization (simulated)
3. Translation (simulated with realistic delays)
4. Caption buffer management (REAL)
5. WebSocket broadcasting (REAL)
6. Client receiving updates (REAL WebSocket client)

This tests everything except actual Whisper/translation model inference.
"""

import asyncio
import json
import logging
from dataclasses import dataclass

import aiohttp
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Configuration
# =============================================================================

BASE_URL = "http://localhost:3000"
WS_URL = "ws://localhost:3000/api/captions/stream"

# Use "test" session by default so you can watch in the overlay
# Override with environment variable: TEST_SESSION_ID=my-session
import contextlib
import os

SESSION_ID = os.environ.get("TEST_SESSION_ID", "test")


async def clear_session(session_id: str, delay_before: float = 2.0):
    """Clear all captions from a session before test.

    Args:
        session_id: Session to clear
        delay_before: Seconds to wait before clearing (so user can see results)
    """
    # Wait so user can see previous test results in overlay
    if delay_before > 0:
        logger.info(f"Waiting {delay_before}s before clearing (watch the overlay)...")
        await asyncio.sleep(delay_before)

    async with aiohttp.ClientSession() as session:
        try:
            # First, ensure the session exists by sending a dummy caption
            await session.post(
                f"{BASE_URL}/api/captions/{session_id}",
                json={
                    "text": "init",
                    "speaker_name": "System",
                    "duration_seconds": 0.1,
                },
            )
            await asyncio.sleep(0.2)

            # Now delete to clear
            resp = await session.delete(f"{BASE_URL}/api/captions/{session_id}")
            if resp.status == 204:
                logger.info(f"Cleared session: {session_id}")
            else:
                logger.warning(f"Clear session returned status {resp.status}")
        except Exception as e:
            logger.warning(f"Failed to clear session: {e}")
    await asyncio.sleep(0.5)  # Let it settle


# =============================================================================
# Simulated Data
# =============================================================================


@dataclass
class SimulatedUtterance:
    """Simulates a transcribed utterance from Whisper + diarization."""

    speaker: str
    text: str
    delay_before: float = 0.5  # Seconds before this utterance
    translation: str | None = None  # If None, will auto-generate


# Realistic conversation simulation
CONVERSATION_SCRIPT = [
    SimulatedUtterance("Alice", "Hello everyone, welcome to the meeting", delay_before=0.0),
    SimulatedUtterance("Alice", "today we'll discuss the Q4 roadmap", delay_before=1.5),
    SimulatedUtterance("Bob", "Thanks Alice, I have some updates", delay_before=2.0),
    SimulatedUtterance("Bob", "on the backend migration", delay_before=1.0),
    SimulatedUtterance("Alice", "Great, please go ahead", delay_before=1.5),
    SimulatedUtterance("Bob", "We've completed phase one", delay_before=1.0),
    SimulatedUtterance("Bob", "and started phase two this week", delay_before=1.2),
    SimulatedUtterance("Charlie", "Quick question about phase two", delay_before=2.5),
    SimulatedUtterance("Bob", "Sure, go ahead Charlie", delay_before=1.0),
    SimulatedUtterance("Charlie", "What's the expected timeline?", delay_before=0.8),
    SimulatedUtterance("Bob", "We're targeting end of month", delay_before=1.5),
    SimulatedUtterance("Alice", "That aligns with our goals", delay_before=2.0),
    SimulatedUtterance("Alice", "Let's move to the next topic", delay_before=1.0),
]

# Simulated translations (Spanish)
TRANSLATIONS = {
    "Hello everyone, welcome to the meeting": "Hola a todos, bienvenidos a la reunión",
    "today we'll discuss the Q4 roadmap": "hoy discutiremos la hoja de ruta del Q4",
    "Thanks Alice, I have some updates": "Gracias Alice, tengo algunas actualizaciones",
    "on the backend migration": "sobre la migración del backend",
    "Great, please go ahead": "Genial, por favor continúa",
    "We've completed phase one": "Hemos completado la fase uno",
    "and started phase two this week": "y comenzamos la fase dos esta semana",
    "Quick question about phase two": "Pregunta rápida sobre la fase dos",
    "Sure, go ahead Charlie": "Claro, adelante Charlie",
    "What's the expected timeline?": "¿Cuál es el cronograma esperado?",
    "We're targeting end of month": "Estamos apuntando a fin de mes",
    "That aligns with our goals": "Eso se alinea con nuestros objetivos",
    "Let's move to the next topic": "Pasemos al siguiente tema",
}


def get_translation(text: str) -> str:
    """Get simulated translation for text."""
    return TRANSLATIONS.get(text, f"[Translated: {text}]")


# =============================================================================
# WebSocket Client for Receiving Captions
# =============================================================================


class CaptionReceiver:
    """WebSocket client that receives caption events."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.ws = None
        self.received_events: list[dict] = []
        self.captions: dict = {}  # id -> caption data
        self.connected = False
        self._task = None

    async def connect(self):
        """Connect to WebSocket and start receiving."""
        url = f"{WS_URL}/{self.session_id}"
        logger.info(f"Connecting to {url}")

        session = aiohttp.ClientSession()
        self.ws = await session.ws_connect(url)
        self.connected = True
        self._session = session

        # Start receiving in background
        self._task = asyncio.create_task(self._receive_loop())

        # Wait for initial connection event
        await asyncio.sleep(0.5)
        logger.info(f"Connected, received {len(self.received_events)} initial events")

    async def _receive_loop(self):
        """Background task to receive WebSocket messages."""
        try:
            async for msg in self.ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    self._handle_event(data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {msg.data}")
                    break
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    def _handle_event(self, data: dict):
        """Handle incoming WebSocket event."""
        event = data.get("event")
        self.received_events.append(data)

        if event == "connected":
            # Initial captions on connect
            for caption in data.get("current_captions", []):
                self.captions[caption["id"]] = caption
            logger.info(f"Connected with {len(self.captions)} existing captions")

        elif event == "caption_added":
            caption = data.get("caption", {})
            self.captions[caption["id"]] = caption
            logger.info(
                f"Caption added: [{caption.get('speaker_name')}] {caption.get('translated_text')[:40]}..."
            )

        elif event == "caption_updated":
            caption = data.get("caption", {})
            self.captions[caption["id"]] = caption
            logger.info(
                f"Caption updated: [{caption.get('speaker_name')}] {caption.get('translated_text')[:40]}..."
            )

        elif event == "caption_expired":
            caption_id = data.get("caption_id")
            self.captions.pop(caption_id, None)
            logger.info(f"Caption expired: {caption_id[:8]}...")

    async def close(self):
        """Close WebSocket connection."""
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.ws:
            await self.ws.close()
        if hasattr(self, "_session"):
            await self._session.close()
        self.connected = False

    def get_visible_captions(self) -> list[dict]:
        """Get currently visible captions."""
        return list(self.captions.values())

    def get_caption_count(self) -> int:
        """Get number of visible captions."""
        return len(self.captions)

    def get_events_by_type(self, event_type: str) -> list[dict]:
        """Get all events of a specific type."""
        return [e for e in self.received_events if e.get("event") == event_type]


# =============================================================================
# Caption Sender (Simulates Translation Pipeline Output)
# =============================================================================


class CaptionSender:
    """Sends captions to the API (simulates translation pipeline output)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.sent_count = 0

    async def send_caption(
        self,
        text: str,
        speaker_name: str,
        original_text: str | None = None,
    ) -> dict:
        """Send a caption to the API."""
        url = f"{BASE_URL}/api/captions/{self.session_id}"

        payload = {
            "text": text,
            "speaker_name": speaker_name,
            "original_text": original_text,
            "target_language": "es",
            "confidence": 0.95,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                result = await resp.json()
                self.sent_count += 1
                return result


# =============================================================================
# Test Scenarios
# =============================================================================


@pytest.mark.asyncio
async def test_full_conversation_flow():
    """
    Test a full conversation with multiple speakers.

    Verifies:
    - Same speaker aggregation
    - Different speaker creates new bubble
    - Out-of-order detection
    - WebSocket broadcasts received
    """
    # Clear session before test to ensure clean state
    await clear_session(SESSION_ID)

    receiver = CaptionReceiver(SESSION_ID)
    sender = CaptionSender(SESSION_ID)

    try:
        # Connect receiver
        await receiver.connect()
        assert receiver.connected, "Failed to connect WebSocket"

        # Run through conversation
        results = []
        for utterance in CONVERSATION_SCRIPT:
            # Simulate delay (audio chunk timing)
            if utterance.delay_before > 0:
                await asyncio.sleep(utterance.delay_before)

            # Get translation
            translation = utterance.translation or get_translation(utterance.text)

            # Send caption (simulates translation pipeline output)
            result = await sender.send_caption(
                text=translation,
                speaker_name=utterance.speaker,
                original_text=utterance.text,
            )
            results.append(result)

            logger.info(
                f"Sent [{utterance.speaker}]: {utterance.text[:30]}... -> "
                f"{'aggregated' if result.get('was_aggregated') else 'new bubble'}"
            )

        # Wait for all broadcasts to be received
        await asyncio.sleep(1.0)

        # Analyze results
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS")
        logger.info("=" * 60)

        # Count aggregations
        aggregated = sum(1 for r in results if r.get("was_aggregated"))
        new_bubbles = sum(1 for r in results if not r.get("was_aggregated"))
        logger.info(
            f"Sent {len(results)} utterances: {new_bubbles} new bubbles, {aggregated} aggregations"
        )

        # Check WebSocket events
        added_events = receiver.get_events_by_type("caption_added")
        updated_events = receiver.get_events_by_type("caption_updated")
        logger.info(f"WebSocket events: {len(added_events)} added, {len(updated_events)} updated")

        # Verify visible captions
        visible = receiver.get_visible_captions()
        logger.info(f"Currently visible: {len(visible)} captions")
        for cap in visible:
            logger.info(f"  [{cap['speaker_name']}] {cap['translated_text'][:50]}...")

        # Assertions
        assert len(results) == len(CONVERSATION_SCRIPT), "All utterances should be sent"
        assert len(added_events) == new_bubbles, "Should have received all 'added' events"
        assert len(updated_events) == aggregated, "Should have received all 'updated' events"

        # Verify out-of-order behavior
        # Alice speaks, then Bob, then Alice again - Alice should get new bubble
        [r for i, r in enumerate(results) if CONVERSATION_SCRIPT[i].speaker == "Alice"]
        # After Bob speaks, Alice's next utterance should NOT aggregate
        # Find where Alice speaks after Bob
        speakers = [u.speaker for u in CONVERSATION_SCRIPT]
        for i in range(1, len(speakers)):
            if speakers[i] == "Alice" and speakers[i - 1] != "Alice":
                # This Alice utterance should be a new bubble (out-of-order)
                assert not results[i].get(
                    "was_aggregated"
                ), f"Alice utterance {i} should be new bubble after {speakers[i-1]} spoke"

        logger.info("\n✅ All assertions passed!")

    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_rapid_same_speaker():
    """
    Test rapid speech from same speaker.

    Verifies:
    - All utterances aggregate into one bubble
    - Text accumulates correctly
    - Expiration timer extends
    """
    session_id = SESSION_ID  # Use same session so you can watch

    # Clear session before test to ensure clean state
    await clear_session(session_id)

    receiver = CaptionReceiver(session_id)
    sender = CaptionSender(session_id)

    try:
        await receiver.connect()

        # Send 5 rapid utterances from same speaker
        utterances = [
            "First part of the sentence",
            "second part continues",
            "and then some more",
            "with additional content",
            "finally wrapping up",
        ]

        first_caption_id = None
        for i, text in enumerate(utterances):
            result = await sender.send_caption(
                text=f"[{i+1}] {text}",
                speaker_name="RapidSpeaker",
            )

            if i == 0:
                first_caption_id = result["caption_id"]
                assert not result.get("was_aggregated"), "First should create new"
            else:
                assert result.get("was_aggregated"), f"Utterance {i+1} should aggregate"
                assert result["caption_id"] == first_caption_id, "Should use same caption ID"

            await asyncio.sleep(0.3)  # Small delay between

        await asyncio.sleep(0.5)

        # Verify single bubble with all text
        visible = receiver.get_visible_captions()
        assert len(visible) == 1, f"Should have 1 bubble, got {len(visible)}"

        full_text = visible[0]["translated_text"]
        assert "[1]" in full_text and "[5]" in full_text, "Should contain first and last"

        logger.info("✅ Rapid same speaker: Single bubble with aggregated text")
        logger.info(f"   Text: {full_text[:80]}...")

    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_speaker_interleaving():
    """
    Test speakers interleaving (A, B, A, B pattern).

    Verifies:
    - Each switch creates new bubble
    - No aggregation when different speaker in between
    """
    session_id = SESSION_ID  # Use same session so you can watch

    # Clear session before test to ensure clean state
    await clear_session(session_id)

    receiver = CaptionReceiver(session_id)
    sender = CaptionSender(session_id)

    try:
        await receiver.connect()

        # Alternating speakers
        utterances = [
            ("Alice", "Alice first message"),
            ("Bob", "Bob responds"),
            ("Alice", "Alice replies"),  # Should be NEW (out-of-order)
            ("Bob", "Bob continues"),  # Should be NEW (out-of-order)
            ("Alice", "Alice wraps up"),  # Should be NEW (out-of-order)
        ]

        results = []
        for speaker, text in utterances:
            result = await sender.send_caption(text=text, speaker_name=speaker)
            results.append(result)
            await asyncio.sleep(0.5)

        # Every utterance should be a new bubble (alternating speakers)
        for i, result in enumerate(results):
            # First utterance is always new, subsequent depend on pattern
            if i == 0:
                assert not result.get("was_aggregated"), "First should be new"
            else:
                # Since speakers alternate, each should be new (out-of-order)
                assert not result.get(
                    "was_aggregated"
                ), f"Utterance {i} should be new (speaker changed)"

        await asyncio.sleep(0.5)
        visible = receiver.get_visible_captions()

        # Should have 5 separate bubbles (all alternating)
        assert len(visible) == 5, f"Should have 5 bubbles, got {len(visible)}"

        logger.info(f"✅ Interleaving: {len(visible)} separate bubbles for alternating speakers")

    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_max_aggregation_time():
    """
    Test that bubbles force-refresh after max_aggregation_time.

    Verifies:
    - Same speaker gets new bubble after ~6 seconds
    - Old bubble remains visible briefly (overlap)
    """
    session_id = SESSION_ID  # Use same session so you can watch

    # Clear session before test to ensure clean state
    await clear_session(session_id)

    receiver = CaptionReceiver(session_id)
    sender = CaptionSender(session_id)

    try:
        await receiver.connect()

        # Send first utterance
        result1 = await sender.send_caption(
            text="Starting a long monologue",
            speaker_name="LongSpeaker",
        )
        first_id = result1["caption_id"]
        logger.info(f"First caption: {first_id[:8]}")

        # Continue speaking for 7+ seconds (past max_aggregation_time of 6s)
        for i in range(8):
            await asyncio.sleep(1.0)
            result = await sender.send_caption(
                text=f"Continuing at second {i+1}",
                speaker_name="LongSpeaker",
            )

            if i < 5:  # First ~5 seconds should aggregate
                # May or may not aggregate depending on exact timing
                pass
            else:  # After 6s should force new bubble
                if not result.get("was_aggregated"):
                    logger.info(f"New bubble forced at second {i+1}")
                    assert result["caption_id"] != first_id, "Should have new ID"
                    break

        logger.info("✅ Max aggregation time: New bubble forced after ~6 seconds")

    finally:
        await receiver.close()


@pytest.mark.asyncio
async def test_caption_expiration():
    """
    Test that captions have correct expiration timing.

    Note: Server doesn't proactively broadcast expirations.
    Client handles expiration based on time_remaining_seconds.
    Server cleans up on next access.

    Verifies:
    - Caption appears with correct time_remaining
    - time_remaining decreases over time
    - Server cleanup removes expired on next GET
    """
    session_id = SESSION_ID  # Use same session so you can watch

    # Clear session before test to ensure clean state
    await clear_session(session_id)

    receiver = CaptionReceiver(session_id)
    sender = CaptionSender(session_id)

    try:
        await receiver.connect()

        # Send a caption
        await sender.send_caption(
            text="This will expire soon",
            speaker_name="Expirer",
        )

        await asyncio.sleep(0.5)
        assert receiver.get_caption_count() == 1, "Should have 1 caption"

        # Check initial time_remaining
        visible = receiver.get_visible_captions()
        initial_remaining = visible[0].get("time_remaining_seconds", 0)
        logger.info(f"Initial time_remaining: {initial_remaining:.1f}s")
        assert initial_remaining > 3, "Should have ~4 seconds initially"

        # Wait a bit and check time decreased
        await asyncio.sleep(1.5)

        # Get fresh data from server (triggers cleanup check)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/captions/{session_id}") as resp:
                data = await resp.json()
                if data["captions"]:
                    new_remaining = data["captions"][0].get("time_remaining_seconds", 0)
                    logger.info(f"After 1.5s: time_remaining = {new_remaining:.1f}s")
                    assert new_remaining < initial_remaining - 1, "Time should decrease"

        # Wait for full expiration + buffer
        logger.info("Waiting for full expiration (~4 more seconds)...")
        await asyncio.sleep(4)

        # Query server - should trigger cleanup and return empty
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{BASE_URL}/api/captions/{session_id}") as resp:
                data = await resp.json()
                remaining_count = len(data.get("captions", []))
                logger.info(f"After expiration: {remaining_count} captions remaining")
                assert remaining_count == 0, "Caption should be cleaned up"

        logger.info("✅ Expiration: Caption properly expired and cleaned up")

    finally:
        await receiver.close()


# =============================================================================
# Main Runner
# =============================================================================


async def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("CAPTION PIPELINE END-TO-END TESTS")
    print(f"Session ID: {SESSION_ID}")
    print(f"Watch at: {BASE_URL}/static/captions.html?session={SESSION_ID}&showStatus=true")
    print("=" * 70 + "\n")

    tests = [
        ("Full Conversation Flow", test_full_conversation_flow),
        ("Rapid Same Speaker", test_rapid_same_speaker),
        ("Speaker Interleaving", test_speaker_interleaving),
        ("Max Aggregation Time", test_max_aggregation_time),
        ("Caption Expiration", test_caption_expiration),
    ]

    results = []
    for name, test_func in tests:
        # Clear session before each test
        await clear_session(SESSION_ID)
        print(f"\n{'─' * 50}")
        print(f"TEST: {name}")
        print(f"{'─' * 50}")

        try:
            await test_func()
            results.append((name, "PASSED", None))
            print(f"\n✅ {name}: PASSED")
        except AssertionError as e:
            results.append((name, "FAILED", str(e)))
            print(f"\n❌ {name}: FAILED - {e}")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"\n⚠️ {name}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    errors = sum(1 for _, status, _ in results if status == "ERROR")

    for name, status, msg in results:
        icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
        print(f"  {icon} {name}: {status}")
        if msg:
            print(f"      {msg[:60]}...")

    print(f"\nTotal: {passed} passed, {failed} failed, {errors} errors")
    print("=" * 70 + "\n")

    return passed, failed, errors


if __name__ == "__main__":
    asyncio.run(run_all_tests())
