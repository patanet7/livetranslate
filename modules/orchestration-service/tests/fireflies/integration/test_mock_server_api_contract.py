#!/usr/bin/env python3
"""
Test Mock Server and V3 API Contract

This test verifies:
1. Fireflies mock server works correctly
2. V3 translation API contract works
3. The integration between mock server -> translation service

Run with:
    cd modules/orchestration-service
    poetry run python tests/fireflies/integration/test_mock_server_api_contract.py
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add paths for imports
TEST_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(TEST_DIR))

import aiohttp

# Import fixtures
from fireflies.fixtures.meeting_transcript_5min import (
    MEETING_TRANSCRIPT,
)

# Import mock server
from fireflies.mocks.fireflies_mock_server import (
    FirefliesMockServer,
    MockChunk,
    MockTranscriptScenario,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Test Results Logging
# =============================================================================


class TestResultsLogger:
    """Log test results to file."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.start_time = datetime.now()

    def log(self, test_name: str, passed: bool, details: str = ""):
        """Log a test result."""
        status = "PASS" if passed else "FAIL"
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
        self.results.append(result)
        logger.info(f"[{status}] {test_name}: {details[:100] if details else 'OK'}")

    def write_results(self):
        """Write results to file."""
        filename = (
            f"{datetime.now().strftime('%Y-%m-%d')}_test_mock_server_api_contract_results.log"
        )
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            f.write("# Mock Server and V3 API Contract Test Results\n")
            f.write(f"# Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            passed = sum(1 for r in self.results if r["status"] == "PASS")
            failed = sum(1 for r in self.results if r["status"] == "FAIL")

            f.write("## Summary\n")
            f.write(f"- Total: {len(self.results)}\n")
            f.write(f"- Passed: {passed}\n")
            f.write(f"- Failed: {failed}\n\n")

            f.write("## Test Results\n\n")
            for result in self.results:
                f.write(f"### {result['test']}\n")
                f.write(f"Status: {result['status']}\n")
                if result["details"]:
                    f.write(f"Details:\n```\n{result['details']}\n```\n")
                f.write("\n")

        logger.info(f"Results written to: {filepath}")
        return filepath


# =============================================================================
# Mock Server Tests
# =============================================================================


async def test_mock_server_health():
    """Test that the mock server starts and health check works."""
    server = FirefliesMockServer(host="localhost", port=8090)

    try:
        await server.start()

        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8090/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                return True, f"Health check returned: {data}"
    except Exception as e:
        return False, str(e)
    finally:
        await server.stop()


async def test_mock_server_graphql():
    """Test GraphQL endpoint returns meetings."""
    server = FirefliesMockServer(host="localhost", port=8091)

    # Add a scenario which adds a meeting
    scenario = MockTranscriptScenario.conversation_scenario(
        speakers=["Alice", "Bob"],
        num_exchanges=3,
        chunk_delay_ms=100,
    )
    server.add_scenario(scenario)

    try:
        await server.start()

        async with aiohttp.ClientSession() as session:
            payload = {"query": "query { active_meetings { id title } }", "variables": {}}
            headers = {"Authorization": "Bearer test-api-key"}

            async with session.post(
                "http://localhost:8091/graphql", json=payload, headers=headers
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                meetings = data.get("data", {}).get("active_meetings", [])
                assert len(meetings) == 1
                return True, f"GraphQL returned {len(meetings)} meetings: {meetings[0]['title']}"
    except Exception as e:
        return False, str(e)
    finally:
        await server.stop()


async def test_mock_server_websocket():
    """Test WebSocket endpoint streams chunks using REAL Fireflies auth contract."""
    server = FirefliesMockServer(host="localhost", port=8092)

    # Create a short scenario
    chunks = [
        MockChunk(text="Hello world", speaker_name="Alice", start_time=0, end_time=0.5),
        MockChunk(text="How are you", speaker_name="Bob", start_time=0.6, end_time=1.0),
    ]
    scenario = MockTranscriptScenario(chunks=chunks, chunk_delay_ms=100)
    transcript_id = server.add_scenario(scenario)

    received_chunks = []

    try:
        await server.start()

        async with aiohttp.ClientSession() as session:
            # REAL Fireflies API: Auth via URL query parameters
            ws_url = (
                f"ws://localhost:8092/realtime?token=test-api-key&transcript_id={transcript_id}"
            )
            async with session.ws_connect(ws_url) as ws:
                # Receive messages (auth happens automatically via URL params)
                timeout = asyncio.get_event_loop().time() + 3.0
                while asyncio.get_event_loop().time() < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
                        if msg.get("type") == "transcription.broadcast":
                            received_chunks.append(msg["data"]["text"])
                        if len(received_chunks) >= 2:
                            break
                    except TimeoutError:
                        continue

        if len(received_chunks) == 2:
            return True, f"Received {len(received_chunks)} chunks: {received_chunks}"
        else:
            return False, f"Only received {len(received_chunks)} chunks (expected 2)"

    except Exception as e:
        return False, str(e)
    finally:
        await server.stop()


async def test_mock_server_with_transcript_fixture():
    """Test mock server with our meeting transcript fixture using REAL Fireflies auth."""
    server = FirefliesMockServer(host="localhost", port=8093)

    # Convert first 5 entries from our fixture to MockChunks
    chunks = []
    for entry in MEETING_TRANSCRIPT[:5]:
        chunk = MockChunk(
            text=entry.text,
            speaker_name=entry.speaker,
            start_time=entry.timestamp_ms / 1000.0,
            end_time=(entry.timestamp_ms + 2000) / 1000.0,
        )
        chunks.append(chunk)

    scenario = MockTranscriptScenario(chunks=chunks, chunk_delay_ms=50)
    transcript_id = server.add_scenario(scenario)

    received_chunks = []

    try:
        await server.start()

        async with aiohttp.ClientSession() as session:
            # REAL Fireflies API: Auth via URL query parameters
            ws_url = (
                f"ws://localhost:8093/realtime?token=test-api-key&transcript_id={transcript_id}"
            )
            async with session.ws_connect(ws_url) as ws:
                timeout = asyncio.get_event_loop().time() + 3.0
                while asyncio.get_event_loop().time() < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
                        if msg.get("type") == "transcription.broadcast":
                            received_chunks.append(
                                {
                                    "text": msg["data"]["text"],
                                    "speaker": msg["data"]["speaker_name"],
                                }
                            )
                        if len(received_chunks) >= 5:
                            break
                    except TimeoutError:
                        continue

        if len(received_chunks) >= 5:
            return (
                True,
                f"Received {len(received_chunks)} chunks from fixture. First: {received_chunks[0]}",
            )
        else:
            return False, f"Only received {len(received_chunks)} chunks (expected 5)"

    except Exception as e:
        return False, str(e)
    finally:
        await server.stop()


# =============================================================================
# V3 API Contract Tests
# =============================================================================


async def test_v3_translation_health():
    """Test V3 translation service is available."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:5003/api/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return True, f"Translation service healthy: {data}"
                else:
                    return False, f"Health check returned {resp.status}"
    except aiohttp.ClientError as e:
        return False, f"Cannot connect to translation service: {e}"


async def test_v3_translate_simple():
    """Test V3 translate endpoint with simple prompt."""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": "Translate to Spanish: Hello world",
                "backend": "ollama",
                "max_tokens": 64,
                "temperature": 0.3,
                "system_prompt": "You are a translator. Return ONLY the translation, nothing else.",
            }

            async with session.post("http://localhost:5003/api/v3/translate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    translation = data.get("text", "")
                    return (
                        True,
                        f"Translation: {translation}, Time: {data.get('processing_time_ms', 0):.0f}ms",
                    )
                else:
                    error = await resp.text()
                    return False, f"V3 translate returned {resp.status}: {error}"
    except aiohttp.ClientError as e:
        return False, f"Cannot connect to translation service: {e}"


async def test_v3_translate_with_context():
    """Test V3 translate with context window included in prompt."""
    try:
        async with aiohttp.ClientSession() as session:
            # Build a prompt with context (like TranslationPromptBuilder does)
            prompt = """You are translating a meeting transcript from English to Spanish.

Previous context:
- "Good morning everyone. Let's discuss the API changes."
- "I've reviewed the endpoint documentation."

Glossary (use these exact translations):
- endpoint: punto de acceso
- API: API

Now translate this sentence:
"The endpoint returns JSON data."

Provide ONLY the Spanish translation, no explanations."""

            payload = {
                "prompt": prompt,
                "backend": "ollama",
                "max_tokens": 128,
                "temperature": 0.3,
            }

            async with session.post("http://localhost:5003/api/v3/translate", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    translation = data.get("text", "")
                    # Check if glossary term was applied
                    has_glossary = "punto de acceso" in translation.lower()
                    return True, f"Translation: {translation}, Glossary applied: {has_glossary}"
                else:
                    error = await resp.text()
                    return False, f"V3 translate returned {resp.status}: {error}"
    except aiohttp.ClientError as e:
        return False, f"Cannot connect to translation service: {e}"


async def test_v3_translate_with_glossary_terms():
    """Test each glossary term is applied correctly."""
    results = []

    # Test a subset of glossary terms
    test_cases = [
        ("The endpoint is ready", "endpoint", "punto de acceso"),
        ("Check the deployment status", "deployment", "despliegue"),
        ("The microservice handles requests", "microservice", "microservicio"),
    ]

    try:
        async with aiohttp.ClientSession() as session:
            for english, term, expected_spanish in test_cases:
                prompt = f"""You are translating from English to Spanish.

Glossary (use these exact translations):
- {term}: {expected_spanish}

Translate this sentence:
"{english}"

Provide ONLY the Spanish translation."""

                payload = {
                    "prompt": prompt,
                    "backend": "ollama",
                    "max_tokens": 128,
                    "temperature": 0.3,
                }

                async with session.post(
                    "http://localhost:5003/api/v3/translate", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        translation = data.get("text", "").lower()
                        has_term = expected_spanish.lower() in translation
                        results.append(
                            {
                                "term": term,
                                "expected": expected_spanish,
                                "translation": data.get("text"),
                                "found": has_term,
                            }
                        )
                    else:
                        results.append({"term": term, "error": f"HTTP {resp.status}"})

        successful = sum(1 for r in results if r.get("found", False))
        details = "\n".join(
            [
                f"  - {r['term']}: {'OK' if r.get('found') else 'MISS'} -> {r.get('translation', r.get('error'))}"
                for r in results
            ]
        )

        if successful == len(test_cases):
            return True, f"All {len(test_cases)} glossary terms applied:\n{details}"
        else:
            return False, f"Only {successful}/{len(test_cases)} terms applied:\n{details}"

    except aiohttp.ClientError as e:
        return False, f"Cannot connect to translation service: {e}"


# =============================================================================
# Integration Test
# =============================================================================


async def test_mock_to_translation_integration():
    """Test full flow: Mock server -> (simulate) -> V3 Translation using REAL Fireflies auth."""
    server = FirefliesMockServer(host="localhost", port=8094)

    # Use first 3 entries from our fixture
    chunks = []
    for entry in MEETING_TRANSCRIPT[:3]:
        chunk = MockChunk(
            text=entry.text,
            speaker_name=entry.speaker,
            start_time=entry.timestamp_ms / 1000.0,
            end_time=(entry.timestamp_ms + 2000) / 1000.0,
        )
        chunks.append(chunk)

    scenario = MockTranscriptScenario(chunks=chunks, chunk_delay_ms=50)
    transcript_id = server.add_scenario(scenario)

    received_chunks = []
    translations = []

    try:
        await server.start()

        async with aiohttp.ClientSession() as session:
            # REAL Fireflies API: Auth via URL query parameters
            ws_url = (
                f"ws://localhost:8094/realtime?token=test-api-key&transcript_id={transcript_id}"
            )
            async with session.ws_connect(ws_url) as ws:
                timeout = asyncio.get_event_loop().time() + 3.0
                while asyncio.get_event_loop().time() < timeout:
                    try:
                        msg = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
                        if msg.get("type") == "transcription.broadcast":
                            received_chunks.append(msg["data"])
                        if len(received_chunks) >= 3:
                            break
                    except TimeoutError:
                        continue

            # Now translate each received chunk via V3 API
            for chunk in received_chunks:
                prompt = f"""Translate to Spanish: "{chunk['text']}"

Glossary:
- API: API
- endpoint: punto de acceso

Provide ONLY the Spanish translation."""

                payload = {
                    "prompt": prompt,
                    "backend": "ollama",
                    "max_tokens": 128,
                    "temperature": 0.3,
                }

                async with session.post(
                    "http://localhost:5003/api/v3/translate", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        translations.append(
                            {
                                "original": chunk["text"],
                                "speaker": chunk["speaker_name"],
                                "translation": data.get("text"),
                                "time_ms": data.get("processing_time_ms", 0),
                            }
                        )
                    else:
                        error = await resp.text()
                        translations.append(
                            {"original": chunk["text"], "error": f"HTTP {resp.status}: {error}"}
                        )

        # Check results
        successful = sum(1 for t in translations if "translation" in t)
        details = "\n".join(
            [
                f"  [{t.get('speaker', '?')}] {t['original'][:40]}... -> {t.get('translation', t.get('error'))[:50]}..."
                for t in translations
            ]
        )

        if successful == len(received_chunks):
            return (
                True,
                f"Full integration: {len(received_chunks)} chunks received, all translated:\n{details}",
            )
        else:
            return False, f"Only {successful}/{len(received_chunks)} translated:\n{details}"

    except Exception as e:
        return False, str(e)
    finally:
        await server.stop()


async def test_mock_with_real_fireflies_client():
    """
    Test that our mock server works with the ACTUAL FirefliesRealtimeClient.

    This is the ultimate contract verification - if the real client can
    connect to and receive data from our mock, the contracts match.
    """
    from clients.fireflies_client import FirefliesRealtimeClient

    server = FirefliesMockServer(host="localhost", port=8095)

    # Create a scenario
    chunks = [
        MockChunk(text="Hello from mock", speaker_name="MockSpeaker", start_time=0, end_time=0.5),
        MockChunk(
            text="Testing the contract", speaker_name="MockSpeaker", start_time=0.6, end_time=1.2
        ),
    ]
    scenario = MockTranscriptScenario(chunks=chunks, chunk_delay_ms=200)
    transcript_id = server.add_scenario(scenario)

    received_chunks = []

    async def on_transcript(chunk):
        received_chunks.append(chunk)

    try:
        await server.start()

        # Use the REAL Fireflies client to connect to our mock
        client = FirefliesRealtimeClient(
            api_key="test-api-key",
            transcript_id=transcript_id,
            endpoint="ws://localhost:8095/realtime",
            on_transcript=on_transcript,
            auto_reconnect=False,
        )

        connected = await client.connect()
        if not connected:
            return False, f"Real Fireflies client failed to connect. Status: {client.status}"

        # Wait for chunks
        for _ in range(30):  # Wait up to 3 seconds
            await asyncio.sleep(0.1)
            if len(received_chunks) >= 2:
                break

        await client.disconnect()

        if len(received_chunks) >= 2:
            texts = [c.text for c in received_chunks]
            return (
                True,
                f"REAL FirefliesRealtimeClient received {len(received_chunks)} chunks from mock: {texts}",
            )
        else:
            return False, f"Only received {len(received_chunks)} chunks (expected 2)"

    except Exception as e:
        return False, f"Exception: {e}"
    finally:
        await server.stop()


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run all tests."""
    output_dir = Path(__file__).parent.parent.parent / "output"
    results_logger = TestResultsLogger(output_dir)

    print("=" * 60)
    print("Mock Server and V3 API Contract Tests")
    print("=" * 60)
    print("\nContract Reference: https://docs.fireflies.ai/realtime-api/overview")

    # Mock Server Tests
    print("\n--- Mock Server Tests ---")

    passed, details = await test_mock_server_health()
    results_logger.log("Mock Server Health Check", passed, details)

    passed, details = await test_mock_server_graphql()
    results_logger.log("Mock Server GraphQL Endpoint", passed, details)

    passed, details = await test_mock_server_websocket()
    results_logger.log("Mock Server WebSocket Streaming (URL Auth)", passed, details)

    passed, details = await test_mock_server_with_transcript_fixture()
    results_logger.log("Mock Server with Transcript Fixture", passed, details)

    # Contract Verification with REAL Client
    print("\n--- REAL Fireflies Client Contract Test ---")

    passed, details = await test_mock_with_real_fireflies_client()
    results_logger.log("Mock Server <-> REAL FirefliesRealtimeClient", passed, details)

    # V3 API Contract Tests
    print("\n--- V3 API Contract Tests ---")

    passed, details = await test_v3_translation_health()
    results_logger.log("V3 Translation Service Health", passed, details)

    if not passed:
        print("\n⚠️  Translation service not available. Skipping V3 API tests.")
        print("    Start translation service with:")
        print("    cd modules/translation-service/src")
        print(
            "    OLLAMA_ENABLE=true OLLAMA_BASE_URL=http://localhost:11434/v1 poetry run uvicorn api_server_fastapi:app --port 5003"
        )
    else:
        passed, details = await test_v3_translate_simple()
        results_logger.log("V3 Simple Translation", passed, details)

        passed, details = await test_v3_translate_with_context()
        results_logger.log("V3 Translation with Context", passed, details)

        passed, details = await test_v3_translate_with_glossary_terms()
        results_logger.log("V3 Glossary Term Application", passed, details)

        # Integration Test
        print("\n--- Full Integration Tests ---")

        passed, details = await test_mock_to_translation_integration()
        results_logger.log("Mock Server -> V3 Translation Integration", passed, details)

    # Write results
    print("\n" + "=" * 60)
    filepath = results_logger.write_results()

    # Summary
    passed_count = sum(1 for r in results_logger.results if r["status"] == "PASS")
    total = len(results_logger.results)
    print(f"\nResults: {passed_count}/{total} tests passed")
    print(f"Log file: {filepath}")

    return passed_count == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
