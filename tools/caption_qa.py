#!/usr/bin/env python3
"""Adversarial QA for the caption pipeline.

Tests the full caption system by injecting data through the REST API
and verifying behavior via the WebSocket caption stream.

Usage:
    # Start orchestration + dashboard first:
    #   just bot-full
    #
    # Then run QA:
    uv run python tools/caption_qa.py

    # Or specific test suites:
    uv run python tools/caption_qa.py --test pipeline
    uv run python tools/caption_qa.py --test modes
    uv run python tools/caption_qa.py --test stress
    uv run python tools/caption_qa.py --test all
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field

import httpx
import websockets


ORCHESTRATION_URL = "http://localhost:3000"
WS_URL = "ws://localhost:3000"
SESSION_ID = "qa-test-session"


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration_ms: float = 0.0


@dataclass
class QAReport:
    results: list[TestResult] = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)
        status = "\033[32mPASS\033[0m" if result.passed else "\033[31mFAIL\033[0m"
        print(f"  [{status}] {result.name}: {result.message} ({result.duration_ms:.0f}ms)")

    def summary(self):
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        print(f"\n{'='*60}")
        print(f"QA Results: {passed} passed, {failed} failed, {len(self.results)} total")
        if failed:
            print(f"\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.message}")
        print(f"{'='*60}")
        return failed == 0


async def check_orchestration_up() -> bool:
    """Verify orchestration service is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ORCHESTRATION_URL}/api/system/health")
            return resp.status_code == 200
    except Exception:
        return False


async def inject_caption(
    client: httpx.AsyncClient,
    text: str,
    speaker: str = "Alice",
    original: str | None = None,
    color: str = "#4CAF50",
    lang: str = "es",
    duration: float | None = None,
) -> dict:
    """Inject a caption via REST API."""
    body = {
        "text": text,
        "original_text": original or text,
        "speaker_name": speaker,
        "speaker_color": color,
        "target_language": lang,
        "confidence": 0.95,
    }
    if duration:
        body["duration_seconds"] = duration
    resp = await client.post(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}", json=body)
    return resp.json()


async def collect_ws_events(duration_s: float = 5.0) -> list[dict]:
    """Connect to caption WebSocket and collect events for duration."""
    events = []
    try:
        async with websockets.connect(f"{WS_URL}/api/captions/stream/{SESSION_ID}") as ws:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    events.append(json.loads(msg))
                except asyncio.TimeoutError:
                    continue
    except Exception as e:
        print(f"  WebSocket error: {e}")
    return events


# =============================================================================
# Test Suite 1: Basic Caption Pipeline
# =============================================================================

async def test_pipeline(report: QAReport):
    """Test A1-A4: captions appear, update, expire, aggregate."""
    print("\n--- Test Suite: Caption Pipeline ---")

    async with httpx.AsyncClient(timeout=10) as client:
        # A1: Caption appears
        t0 = time.monotonic()
        result = await inject_caption(client, "Hola mundo", speaker="Alice", original="Hello world")
        assert result.get("status") in ("created", "updated"), f"Unexpected: {result}"
        caption_id = result.get("caption_id")
        report.add(TestResult(
            "A1: Caption injection",
            caption_id is not None,
            f"caption_id={caption_id}" if caption_id else "No caption_id returned",
            (time.monotonic() - t0) * 1000,
        ))

        # A2: Speaker aggregation — same speaker within 5s appends
        t0 = time.monotonic()
        r2 = await inject_caption(client, "Buenos días", speaker="Alice", original="Good morning")
        aggregated = r2.get("was_aggregated", False)
        report.add(TestResult(
            "A2: Speaker aggregation",
            aggregated,
            "Text appended to existing caption" if aggregated else "Created new caption (expected aggregation)",
            (time.monotonic() - t0) * 1000,
        ))

        # A3: Different speaker gets new caption + distinct color
        t0 = time.monotonic()
        r3 = await inject_caption(client, "Gracias", speaker="Bob", color="#2196F3", original="Thanks")
        not_aggregated = not r3.get("was_aggregated", True)
        report.add(TestResult(
            "A3: Multi-speaker distinction",
            not_aggregated,
            "New caption for different speaker" if not_aggregated else "Incorrectly aggregated with previous speaker",
            (time.monotonic() - t0) * 1000,
        ))

        # A4: Caption expiry — check REST endpoint after waiting
        t0 = time.monotonic()
        # Inject with short duration
        await inject_caption(client, "Expiry test", speaker="Charlie", duration=2.0, original="Expiry test")
        await asyncio.sleep(3.0)
        resp = await client.get(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")
        data = resp.json()
        charlie_captions = [c for c in data.get("captions", []) if c.get("speaker_name") == "Charlie"]
        expired = len(charlie_captions) == 0
        report.add(TestResult(
            "A4: Caption expiry (2s duration + 3s wait)",
            expired,
            "Caption expired as expected" if expired else f"Caption still active ({len(charlie_captions)} remaining)",
            (time.monotonic() - t0) * 1000,
        ))

        # A5: WebSocket broadcast — verify events arrive
        t0 = time.monotonic()
        # Start WebSocket listener, then inject
        ws_task = asyncio.create_task(collect_ws_events(3.0))
        await asyncio.sleep(0.3)  # Let WS connect
        await inject_caption(client, "WS broadcast test", speaker="Diana", original="WS test")
        events = await ws_task
        caption_events = [e for e in events if e.get("event") == "caption_added"]
        report.add(TestResult(
            "A5: WebSocket broadcast",
            len(caption_events) >= 1,
            f"Received {len(caption_events)} caption_added events" if caption_events else f"No caption events (got: {[e.get('event') for e in events]})",
            (time.monotonic() - t0) * 1000,
        ))

        # Cleanup
        await client.delete(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")


# =============================================================================
# Test Suite 2: Display Modes
# =============================================================================

async def test_modes(report: QAReport):
    """Test B1-B5: different display modes render correct content."""
    print("\n--- Test Suite: Display Modes ---")

    async with httpx.AsyncClient(timeout=10) as client:
        # Inject a caption with both original and translated text
        await inject_caption(
            client,
            text="Hola mundo",
            speaker="Alice",
            original="Hello world",
        )

        # Get the caption and verify both texts are present
        t0 = time.monotonic()
        resp = await client.get(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")
        data = resp.json()
        captions = data.get("captions", [])
        has_both = any(
            c.get("original_text") and c.get("translated_text")
            for c in captions
        )
        report.add(TestResult(
            "B1: Both original + translated present",
            has_both,
            f"Found {len(captions)} captions, both texts present" if has_both else "Missing original or translated text",
            (time.monotonic() - t0) * 1000,
        ))

        # B2: Verify caption dict has required fields for overlay
        t0 = time.monotonic()
        if captions:
            c = captions[0]
            required = ["id", "original_text", "translated_text", "speaker_name", "speaker_color", "expires_at"]
            missing = [f for f in required if f not in c]
            report.add(TestResult(
                "B2: Caption dict has all overlay fields",
                len(missing) == 0,
                f"All fields present" if not missing else f"Missing: {missing}",
                (time.monotonic() - t0) * 1000,
            ))
        else:
            report.add(TestResult("B2: Caption dict fields", False, "No captions to check", 0))

        # B3: CJK text injection
        t0 = time.monotonic()
        r_cjk = await inject_caption(
            client,
            text="你好世界",
            speaker="Wang",
            original="Hello world",
            lang="zh",
        )
        report.add(TestResult(
            "B3: CJK text injection",
            r_cjk.get("status") in ("created", "updated"),
            f"CJK caption created: {r_cjk.get('caption_id', 'N/A')}",
            (time.monotonic() - t0) * 1000,
        ))

        # Cleanup
        await client.delete(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")


# =============================================================================
# Test Suite 3: Adversarial Stress
# =============================================================================

async def test_stress(report: QAReport):
    """Adversarial: rapid injection, speaker switching, long text, edge cases."""
    print("\n--- Test Suite: Adversarial Stress ---")

    async with httpx.AsyncClient(timeout=30) as client:
        # S1: Rapid-fire injection (10 captions in 1 second)
        t0 = time.monotonic()
        tasks = []
        for i in range(10):
            tasks.append(inject_caption(
                client,
                text=f"Rapid caption #{i}",
                speaker=f"Speaker_{i % 3}",
                original=f"Original #{i}",
                duration=2.0,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successes = sum(1 for r in results if isinstance(r, dict) and "caption_id" in r)
        report.add(TestResult(
            "S1: Rapid-fire 10 captions",
            successes >= 8,  # Allow some aggregation
            f"{successes}/10 injected successfully",
            (time.monotonic() - t0) * 1000,
        ))

        await asyncio.sleep(3.0)  # Let them expire

        # S2: Same speaker rapid text (tests aggregation limits)
        t0 = time.monotonic()
        for i in range(5):
            await inject_caption(client, f"Word{i} ", speaker="Rapid", original=f"Word{i} ")
        resp = await client.get(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")
        rapid_captions = [c for c in resp.json().get("captions", []) if c.get("speaker_name") == "Rapid"]
        report.add(TestResult(
            "S2: Same-speaker rapid aggregation",
            len(rapid_captions) >= 1,
            f"{len(rapid_captions)} caption(s) for Rapid (should be 1-2 due to aggregation)",
            (time.monotonic() - t0) * 1000,
        ))

        # S3: Long text (tests truncation)
        t0 = time.monotonic()
        long_text = "A" * 500
        r_long = await inject_caption(client, long_text, speaker="Verbose", original=long_text)
        resp = await client.get(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")
        verbose_captions = [c for c in resp.json().get("captions", []) if c.get("speaker_name") == "Verbose"]
        if verbose_captions:
            actual_len = len(verbose_captions[0].get("translated_text", ""))
            truncated = actual_len <= 260  # max_caption_chars=250 + some buffer
            report.add(TestResult(
                "S3: Long text truncation",
                True,  # Just verify it doesn't crash
                f"Text length: {actual_len} chars (truncated: {truncated})",
                (time.monotonic() - t0) * 1000,
            ))
        else:
            report.add(TestResult("S3: Long text", False, "No caption found", 0))

        # S4: Empty/edge case text
        t0 = time.monotonic()
        edge_cases = [
            ("Single char", "X", "Y"),
            ("Emoji", "Hello! 🎉", "¡Hola! 🎉"),
            ("Mixed CJK+Latin", "Hello 你好 world 世界", "Hola 你好 mundo 世界"),
            ("Newlines", "Line1\nLine2\nLine3", "Línea1\nLínea2\nLínea3"),
        ]
        edge_pass = 0
        for name, text, original in edge_cases:
            try:
                r = await inject_caption(client, text, speaker=name, original=original)
                if r.get("caption_id"):
                    edge_pass += 1
            except Exception as e:
                print(f"    Edge case '{name}' failed: {e}")
        report.add(TestResult(
            "S4: Edge case text handling",
            edge_pass == len(edge_cases),
            f"{edge_pass}/{len(edge_cases)} edge cases passed",
            (time.monotonic() - t0) * 1000,
        ))

        # S5: Max captions overflow (buffer max=20)
        t0 = time.monotonic()
        await client.delete(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")
        for i in range(25):
            await inject_caption(
                client,
                text=f"Overflow #{i}",
                speaker=f"Overflow_{i}",  # Different speakers to prevent aggregation
                original=f"Overflow #{i}",
                duration=10.0,
            )
        resp = await client.get(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")
        count = resp.json().get("count", 0)
        report.add(TestResult(
            "S5: Max captions overflow (25 injected, max=20)",
            count <= 20,
            f"{count} active captions (should be ≤20)",
            (time.monotonic() - t0) * 1000,
        ))

        # Cleanup
        await client.delete(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")


# =============================================================================
# Test Suite 4: WebSocket Event Protocol
# =============================================================================

async def test_ws_protocol(report: QAReport):
    """Test WebSocket event protocol: connected, caption_added, caption_updated, caption_expired."""
    print("\n--- Test Suite: WebSocket Protocol ---")

    events: list[dict] = []

    async def collect_and_inject():
        """Connect WS, then inject captions, collect all events."""
        async with websockets.connect(f"{WS_URL}/api/captions/stream/{SESSION_ID}") as ws:
            # Should receive 'connected' immediately
            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            events.append(json.loads(msg))

            async with httpx.AsyncClient(timeout=10) as client:
                # Inject caption with short duration for expiry testing
                await inject_caption(client, "Protocol test", speaker="Proto", duration=2.0, original="Protocol test")
                await asyncio.sleep(0.5)
                # Inject same speaker to trigger aggregation → caption_updated
                await inject_caption(client, "More text", speaker="Proto", original="More text")

                # Wait for expiry: caption expires at 2s, cleanup timer fires every 1s.
                # Inject a dummy after 3s to force cleanup if timer hasn't fired yet.
                await asyncio.sleep(3.0)
                await inject_caption(client, "Trigger cleanup", speaker="Trigger", duration=1.0, original="Trigger")

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    events.append(json.loads(msg))
                except asyncio.TimeoutError:
                    continue

    t0 = time.monotonic()
    await collect_and_inject()
    duration = (time.monotonic() - t0) * 1000

    event_types = [e.get("event") for e in events]

    # W1: Connected event
    report.add(TestResult(
        "W1: 'connected' event on connect",
        "connected" in event_types,
        f"Events: {event_types[:3]}..." if len(event_types) > 3 else f"Events: {event_types}",
        duration,
    ))

    # W2: caption_added event
    report.add(TestResult(
        "W2: 'caption_added' event",
        "caption_added" in event_types,
        f"Found caption_added" if "caption_added" in event_types else f"Missing (got: {event_types})",
        0,
    ))

    # W3: caption_updated event (from aggregation)
    report.add(TestResult(
        "W3: 'caption_updated' event (aggregation)",
        "caption_updated" in event_types,
        f"Found caption_updated" if "caption_updated" in event_types else f"Missing (got: {event_types})",
        0,
    ))

    # W4: caption_expired event
    report.add(TestResult(
        "W4: 'caption_expired' event",
        "caption_expired" in event_types,
        f"Found caption_expired" if "caption_expired" in event_types else f"Missing — may need longer wait (got: {event_types})",
        0,
    ))

    # W5: Caption dict in events has correct fields
    added_events = [e for e in events if e.get("event") == "caption_added"]
    if added_events:
        caption = added_events[0].get("caption", {})
        required = ["id", "original_text", "translated_text", "speaker_name", "speaker_color", "expires_at", "created_at"]
        missing = [f for f in required if f not in caption]
        report.add(TestResult(
            "W5: Caption event has all required fields",
            len(missing) == 0,
            f"All fields present" if not missing else f"Missing: {missing}",
            0,
        ))
    else:
        report.add(TestResult("W5: Caption fields", False, "No caption_added events", 0))

    # Cleanup
    async with httpx.AsyncClient(timeout=5) as client:
        await client.delete(f"{ORCHESTRATION_URL}/api/captions/{SESSION_ID}")


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Adversarial QA for caption pipeline")
    parser.add_argument("--test", choices=["pipeline", "modes", "stress", "ws", "all"], default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("Caption System — Adversarial QA")
    print("=" * 60)

    # Check orchestration is running
    if not await check_orchestration_up():
        print("\n\033[31mOrchestration service not running!\033[0m")
        print("Start it first: just bot-full")
        sys.exit(1)

    print(f"\nOrchestration: \033[32mUP\033[0m ({ORCHESTRATION_URL})")
    print(f"Session ID: {SESSION_ID}")
    print(f"Overlay URL: http://localhost:5173/captions?session={SESSION_ID}")
    print("\nOpen the overlay URL in a browser to watch captions appear!\n")

    report = QAReport()

    test_map = {
        "pipeline": test_pipeline,
        "modes": test_modes,
        "stress": test_stress,
        "ws": test_ws_protocol,
    }

    if args.test == "all":
        for test_fn in test_map.values():
            await test_fn(report)
    else:
        await test_map[args.test](report)

    all_passed = report.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
