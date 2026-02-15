"""
Browser E2E Test Fixtures

Provides:
- orchestration_server: Real uvicorn serving the FastAPI app on port 3001
- mock_fireflies: FirefliesMockServer with Spanish conversation scenario
- browser: AgentBrowser instance (headed or streaming based on env)
- test_output_dir: Path for screenshots and logs

Inherits from root conftest: PostgreSQL testcontainer, Redis, Alembic migrations.
"""

import asyncio
import logging
import multiprocessing
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import uvicorn

# Add src + tests to path
orchestration_root = Path(__file__).parent.parent.parent.parent
src_path = orchestration_root / "src"
tests_root = orchestration_root / "tests"
sys.path.insert(0, str(orchestration_root))
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_root))

from fireflies.e2e.browser.browser_helpers import AgentBrowser
from fireflies.mocks.fireflies_mock_server import (
    FirefliesMockServer,
    MockTranscriptScenario,
)

logger = logging.getLogger(__name__)

# Port configuration — avoid clashing with dev services
ORCHESTRATION_PORT = 3001
MOCK_FIREFLIES_PORT = 8090
STREAM_PORT = 9223

# Test API key recognized by the mock server
TEST_API_KEY = "test-api-key"  # pragma: allowlist secret


# =============================================================================
# Utilities
# =============================================================================


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a TCP port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout: float = 30.0, poll: float = 0.3) -> bool:
    """Block until a port is open or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _port_is_open(host, port):
            return True
        time.sleep(poll)
    return False


def _run_uvicorn(port: int):
    """
    Target function for the uvicorn subprocess.

    Imports the FastAPI app and runs uvicorn. Runs in a separate process
    so the test process can control its lifecycle.
    """
    # Re-add src to path in the child process
    child_src = Path(__file__).parent.parent.parent.parent / "src"
    sys.path.insert(0, str(child_src))

    from main_fastapi import app

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_output_dir():
    """Create and return the test output directory."""
    output_dir = orchestration_root / "tests" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture(scope="session")
def orchestration_server():
    """
    Start a real uvicorn server serving the FastAPI app on ORCHESTRATION_PORT.

    Runs in a child process. Waits for the port to become available.
    Tears down on session end.
    """
    if _port_is_open("localhost", ORCHESTRATION_PORT):
        logger.info(f"Orchestration server already running on port {ORCHESTRATION_PORT}")
        yield f"http://localhost:{ORCHESTRATION_PORT}"
        return

    proc = multiprocessing.Process(
        target=_run_uvicorn,
        args=(ORCHESTRATION_PORT,),
        daemon=True,
    )
    proc.start()

    if not _wait_for_port("localhost", ORCHESTRATION_PORT, timeout=30):
        proc.terminate()
        pytest.fail(f"Orchestration server failed to start on port {ORCHESTRATION_PORT}")

    logger.info(f"Orchestration server started on port {ORCHESTRATION_PORT} (pid={proc.pid})")
    yield f"http://localhost:{ORCHESTRATION_PORT}"

    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
    logger.info("Orchestration server stopped")


@pytest.fixture(scope="session")
def mock_fireflies_server():
    """
    Start the FirefliesMockServer with a Spanish conversation scenario.

    The scenario has 2 speakers (Alice, Bob) with 20 exchanges.
    Uses asyncio.run() for start/stop since this is a session-scoped sync fixture
    (avoids event_loop fixture conflicts with pytest-asyncio).
    """
    server = FirefliesMockServer(
        host="localhost",
        port=MOCK_FIREFLIES_PORT,
        valid_api_keys={TEST_API_KEY},
    )

    scenario = MockTranscriptScenario.conversation_scenario(
        speakers=["Alice", "Bob"],
        num_exchanges=20,
        chunk_delay_ms=500,
    )
    transcript_id = server.add_scenario(scenario)

    # Start server — use a dedicated event loop for the session-scoped fixture
    loop = asyncio.new_event_loop()
    loop.run_until_complete(server.start())

    logger.info(
        f"Mock Fireflies server started on port {MOCK_FIREFLIES_PORT}, "
        f"transcript_id={transcript_id}"
    )

    yield {
        "server": server,
        "url": f"http://localhost:{MOCK_FIREFLIES_PORT}",
        "transcript_id": transcript_id,
        "scenario": scenario,
        "api_key": TEST_API_KEY,
    }

    loop.run_until_complete(server.stop())
    loop.close()
    logger.info("Mock Fireflies server stopped")


@pytest.fixture(scope="session")
def base_url(orchestration_server):
    """Base URL for the orchestration service."""
    return orchestration_server


@pytest.fixture
def browser(test_output_dir):
    """
    Launch agent-browser instance.

    Headed mode by default. Set env BROWSER_STREAM=1 for headless + streaming.
    Closes browser on teardown.
    """
    stream_mode = os.environ.get("BROWSER_STREAM") == "1"

    b = AgentBrowser(
        headed=not stream_mode,
        stream_port=STREAM_PORT if stream_mode else None,
        timeout=30,
    )

    yield b

    b.close()


@pytest.fixture
def dashboard_url(base_url):
    """URL for the Fireflies dashboard."""
    return f"{base_url}/static/fireflies-dashboard.html"


@pytest.fixture
def captions_url(base_url):
    """URL builder for the captions overlay."""

    def _build(session_id: str, **params) -> str:
        url = f"{base_url}/static/captions.html?session={session_id}"
        # Default to showing speaker names and original text for testing
        params.setdefault("showSpeaker", "true")
        params.setdefault("showOriginal", "true")
        params.setdefault("showStatus", "true")
        for key, value in params.items():
            url += f"&{key}={value}"
        return url

    return _build


@pytest.fixture
def timestamp():
    """Current timestamp string for output file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
