#!/usr/bin/env python3
"""
Test Multiple Meeting Codes with Mock Fireflies Server

Tests the Fireflies integration with different meeting codes/scenarios:
1. Standard tech meeting with API discussions
2. Sales call with product demo
3. Multi-language support meeting
4. Fast-paced standup meeting

Each meeting has different characteristics and tests different aspects
of the translation pipeline.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import aiohttp

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tests.fireflies.mocks.fireflies_mock_server import (
    FirefliesMockServer,
    MockChunk,
    MockMeeting,
    MockTranscriptScenario,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MOCK_SERVER_HOST = "localhost"
MOCK_SERVER_PORT = 8090
ORCHESTRATION_URL = "http://localhost:3000"
TRANSLATION_URL = "http://localhost:5003"
TEST_API_KEY = "test-multi-meeting-key"


# =============================================================================
# Meeting Scenarios
# =============================================================================

def create_tech_meeting_scenario() -> tuple:
    """Tech meeting discussing API changes and deployments."""
    meeting = MockMeeting(
        id="tech-meeting-001",
        title="API Architecture Review",
        organizer_email="tech-lead@example.com",
        state="active"
    )

    chunks = [
        MockChunk(chunk_id="tech-001", text="Good morning team. Let's discuss the API changes.",
                  speaker_name="Alice", start_time=0.0, end_time=2.5),
        MockChunk(chunk_id="tech-002", text="I've reviewed the endpoint documentation.",
                  speaker_name="Bob", start_time=3.0, end_time=5.0),
        MockChunk(chunk_id="tech-003", text="The microservice architecture needs refactoring.",
                  speaker_name="Alice", start_time=6.0, end_time=8.5),
        MockChunk(chunk_id="tech-004", text="We should add better error handling for the deployment pipeline.",
                  speaker_name="Charlie", start_time=9.0, end_time=12.0),
        MockChunk(chunk_id="tech-005", text="Let's schedule the production rollout for next week.",
                  speaker_name="Alice", start_time=13.0, end_time=15.5),
    ]

    scenario = MockTranscriptScenario(
        transcript_id=meeting.id,
        chunks=chunks,
        chunk_delay_ms=200.0,
        stream_mode="chunks"
    )

    return meeting, scenario


def create_sales_meeting_scenario() -> tuple:
    """Sales call with product demo and pricing discussion."""
    meeting = MockMeeting(
        id="sales-meeting-002",
        title="Enterprise Product Demo",
        organizer_email="sales@example.com",
        state="active"
    )

    chunks = [
        MockChunk(chunk_id="sales-001", text="Thank you for joining today's demo.",
                  speaker_name="SalesRep", start_time=0.0, end_time=2.0),
        MockChunk(chunk_id="sales-002", text="I'd like to show you our enterprise features.",
                  speaker_name="SalesRep", start_time=2.5, end_time=5.0),
        MockChunk(chunk_id="sales-003", text="How does the pricing model work?",
                  speaker_name="Customer", start_time=6.0, end_time=7.5),
        MockChunk(chunk_id="sales-004", text="We offer flexible licensing based on user count.",
                  speaker_name="SalesRep", start_time=8.0, end_time=11.0),
        MockChunk(chunk_id="sales-005", text="What about integration with our existing systems?",
                  speaker_name="Customer", start_time=12.0, end_time=14.5),
        MockChunk(chunk_id="sales-006", text="We have REST APIs and webhooks for seamless integration.",
                  speaker_name="SalesRep", start_time=15.0, end_time=18.0),
    ]

    scenario = MockTranscriptScenario(
        transcript_id=meeting.id,
        chunks=chunks,
        chunk_delay_ms=250.0,
        stream_mode="chunks"
    )

    return meeting, scenario


def create_multilang_meeting_scenario() -> tuple:
    """Multi-language support meeting with translations."""
    meeting = MockMeeting(
        id="multilang-meeting-003",
        title="International Team Sync",
        organizer_email="global@example.com",
        state="active"
    )

    chunks = [
        MockChunk(chunk_id="ml-001", text="Welcome everyone from around the world.",
                  speaker_name="Manager", start_time=0.0, end_time=2.5),
        MockChunk(chunk_id="ml-002", text="Let's start with the European team update.",
                  speaker_name="Manager", start_time=3.0, end_time=5.0),
        MockChunk(chunk_id="ml-003", text="Our Berlin office completed the project milestone.",
                  speaker_name="Hans", start_time=6.0, end_time=8.5),
        MockChunk(chunk_id="ml-004", text="Tokyo team has shipped the new mobile features.",
                  speaker_name="Yuki", start_time=9.0, end_time=11.5),
        MockChunk(chunk_id="ml-005", text="The São Paulo office needs additional resources.",
                  speaker_name="Carlos", start_time=12.0, end_time=14.5),
        MockChunk(chunk_id="ml-006", text="Let's coordinate the handoff between time zones.",
                  speaker_name="Manager", start_time=15.0, end_time=17.5),
    ]

    scenario = MockTranscriptScenario(
        transcript_id=meeting.id,
        chunks=chunks,
        chunk_delay_ms=200.0,
        stream_mode="chunks"
    )

    return meeting, scenario


def create_standup_meeting_scenario() -> tuple:
    """Fast-paced standup with quick updates."""
    meeting = MockMeeting(
        id="standup-meeting-004",
        title="Daily Standup",
        organizer_email="scrum@example.com",
        state="active"
    )

    chunks = [
        MockChunk(chunk_id="su-001", text="Let's start the standup. Dev one?",
                  speaker_name="ScrumMaster", start_time=0.0, end_time=1.5),
        MockChunk(chunk_id="su-002", text="Fixed three bugs yesterday. Working on auth today.",
                  speaker_name="Dev1", start_time=2.0, end_time=4.0),
        MockChunk(chunk_id="su-003", text="Completed the database migration. No blockers.",
                  speaker_name="Dev2", start_time=4.5, end_time=6.5),
        MockChunk(chunk_id="su-004", text="UI tests are passing. Starting integration tests.",
                  speaker_name="QA1", start_time=7.0, end_time=9.0),
        MockChunk(chunk_id="su-005", text="Waiting on API specs from product.",
                  speaker_name="Dev3", start_time=9.5, end_time=11.0),
        MockChunk(chunk_id="su-006", text="I'll follow up with product after this.",
                  speaker_name="ScrumMaster", start_time=11.5, end_time=13.0),
    ]

    scenario = MockTranscriptScenario(
        transcript_id=meeting.id,
        chunks=chunks,
        chunk_delay_ms=100.0,  # Faster delivery for standup
        stream_mode="chunks"
    )

    return meeting, scenario


# =============================================================================
# Test Runner
# =============================================================================

async def check_services():
    """Check if required services are running."""
    services_ok = True

    async with aiohttp.ClientSession() as session:
        # Check orchestration
        try:
            async with session.get(f"{ORCHESTRATION_URL}/api/health", timeout=5) as resp:
                if resp.status == 200:
                    logger.info("✓ Orchestration service healthy")
                else:
                    logger.error(f"✗ Orchestration service unhealthy: {resp.status}")
                    services_ok = False
        except Exception as e:
            logger.error(f"✗ Orchestration service not available: {e}")
            services_ok = False

        # Check translation
        try:
            async with session.get(f"{TRANSLATION_URL}/api/health", timeout=5) as resp:
                if resp.status == 200:
                    logger.info("✓ Translation service healthy")
                else:
                    logger.error(f"✗ Translation service unhealthy: {resp.status}")
                    services_ok = False
        except Exception as e:
            logger.error(f"✗ Translation service not available: {e}")
            services_ok = False

    return services_ok


async def test_translation(text: str, target_lang: str = "Spanish") -> Optional[str]:
    """Test translation endpoint."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{TRANSLATION_URL}/api/translate",
                json={"text": text, "target_language": target_lang},
                timeout=30
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("translated_text")
                else:
                    logger.error(f"Translation failed: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None


async def run_meeting_test(meeting: MockMeeting, scenario: MockTranscriptScenario, server: FirefliesMockServer):
    """Run a single meeting test."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {meeting.title}")
    logger.info(f"Meeting ID: {meeting.id}")
    logger.info(f"Chunks: {len(scenario.chunks)}")
    logger.info(f"{'='*60}")

    # Add scenario to server
    server.add_scenario(scenario)
    server.add_meeting(meeting)

    # Test each chunk with translation
    results = []
    for chunk in scenario.chunks:
        logger.info(f"\n[{chunk.speaker_name}]: {chunk.text}")

        # Translate to Spanish
        translated = await test_translation(chunk.text)
        if translated:
            logger.info(f"  → ES: {translated}")
            results.append({
                "speaker": chunk.speaker_name,
                "original": chunk.text,
                "translated": translated,
                "success": True
            })
        else:
            logger.warning(f"  → Translation failed")
            results.append({
                "speaker": chunk.speaker_name,
                "original": chunk.text,
                "translated": None,
                "success": False
            })

    # Calculate success rate
    success_count = sum(1 for r in results if r["success"])
    success_rate = (success_count / len(results)) * 100 if results else 0

    logger.info(f"\nResults: {success_count}/{len(results)} translations successful ({success_rate:.1f}%)")

    return {
        "meeting_id": meeting.id,
        "meeting_title": meeting.title,
        "total_chunks": len(results),
        "successful_translations": success_count,
        "success_rate": success_rate,
        "results": results
    }


async def main():
    """Main test runner."""
    print("=" * 70)
    print("Multiple Meeting Codes Test")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check services
    print("\n[1] Checking services...")
    if not await check_services():
        print("\n✗ Services not available. Please start them first.")
        print("  - Orchestration: cd modules/orchestration-service && python -m uvicorn src.main_fastapi:app --port 3000")
        print("  - Translation: cd modules/translation-service/src && python -m uvicorn api_server_fastapi:app --port 5003")
        return

    # Create mock server
    print("\n[2] Starting mock Fireflies server...")
    server = FirefliesMockServer(
        host=MOCK_SERVER_HOST,
        port=MOCK_SERVER_PORT,
        valid_api_keys={TEST_API_KEY}
    )

    try:
        await server.start()
        logger.info(f"Mock server running at http://{MOCK_SERVER_HOST}:{MOCK_SERVER_PORT}")

        # Create meeting scenarios
        meetings = [
            create_tech_meeting_scenario(),
            create_sales_meeting_scenario(),
            create_multilang_meeting_scenario(),
            create_standup_meeting_scenario(),
        ]

        print(f"\n[3] Testing {len(meetings)} different meeting scenarios...")

        all_results = []
        for meeting, scenario in meetings:
            result = await run_meeting_test(meeting, scenario, server)
            all_results.append(result)
            await asyncio.sleep(0.5)  # Brief pause between meetings

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        total_chunks = sum(r["total_chunks"] for r in all_results)
        total_success = sum(r["successful_translations"] for r in all_results)
        overall_rate = (total_success / total_chunks) * 100 if total_chunks else 0

        for r in all_results:
            status = "✓" if r["success_rate"] >= 80 else "✗"
            print(f"{status} {r['meeting_title']}: {r['successful_translations']}/{r['total_chunks']} ({r['success_rate']:.1f}%)")

        print("-" * 70)
        print(f"Overall: {total_success}/{total_chunks} translations ({overall_rate:.1f}%)")
        print("=" * 70)

        # Write results to log file
        output_dir = Path(__file__).parent.parent.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_test_multiple_meetings.log"

        with open(output_file, "w") as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "overall_success_rate": overall_rate,
                "total_chunks": total_chunks,
                "total_successful": total_success,
                "meetings": all_results
            }, indent=2))

        print(f"\nResults saved to: {output_file}")

    finally:
        await server.stop()
        logger.info("Mock server stopped")


if __name__ == "__main__":
    asyncio.run(main())
