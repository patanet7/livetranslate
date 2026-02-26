"""
Integrated E2E test fixtures for the SvelteKit dashboard.

Boots the FULL stack for behavioral testing:
1. Mock Fireflies server (port 8090) — streams real captured transcript data
2. Orchestration service (uvicorn on port 3001) — real FastAPI backend
3. SvelteKit dev server (port 5180) — the dashboard under test

Browser tests hit the real SvelteKit app, which proxies to the real orchestration
service, which connects to the real mock Fireflies server. No mocks in the
test code — the entire pipeline is exercised end-to-end.

Screenshots are saved to tests/browser/screenshots/ as visual evidence.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# =============================================================================
# Path setup — import from orchestration-service's test infrastructure
# =============================================================================

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
ORCH_ROOT = REPO_ROOT / "modules" / "orchestration-service"
ORCH_SRC = ORCH_ROOT / "src"
ORCH_TESTS = ORCH_ROOT / "tests"

# Add orchestration src + tests to path for imports
sys.path.insert(0, str(ORCH_ROOT))
sys.path.insert(0, str(ORCH_SRC))
sys.path.insert(0, str(ORCH_TESTS))
sys.path.insert(0, str(ORCH_TESTS / "fireflies" / "e2e" / "browser"))

from browser_helpers import AgentBrowser  # noqa: E402
from fireflies.mocks.fireflies_mock_server import (  # noqa: E402
    FirefliesMockServer,
    MockTranscriptScenario,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Port configuration — avoid clashing with dev services
# =============================================================================

ORCHESTRATION_PORT = 3001
MOCK_FIREFLIES_PORT = 8090
SVELTEKIT_PORT = 5180

SVELTEKIT_URL = f"http://localhost:{SVELTEKIT_PORT}"
ORCHESTRATION_URL = f"http://localhost:{ORCHESTRATION_PORT}"

# Test API key recognized by the mock server
TEST_API_KEY = "test-api-key"  # pragma: allowlist secret

DASHBOARD_DIR = Path(__file__).parent.parent.parent  # modules/dashboard-service
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"


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
    """Target function for the uvicorn subprocess."""
    import uvicorn

    child_src = Path(__file__).parent.parent.parent.parent.parent / "modules" / "orchestration-service" / "src"
    sys.path.insert(0, str(child_src))

    from main_fastapi import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# =============================================================================
# Session-scoped fixtures — one full stack per test session
# =============================================================================


@pytest.fixture(scope="session")
def mock_fireflies_server():
    """
    Start the FirefliesMockServer with a Spanish conversation scenario.

    Real captured transcript data — 2 speakers (Alice, Bob), 20 exchanges,
    streamed with realistic ASR chunk timing.
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
def orchestration_server():
    """
    Start a real uvicorn server serving the FastAPI orchestration app.

    This is the same orchestration service that handles /fireflies/connect,
    /api/captions/stream, /fireflies/sessions, etc.
    """
    if _port_is_open("localhost", ORCHESTRATION_PORT):
        logger.warning(
            f"Port {ORCHESTRATION_PORT} already in use — reusing existing server."
        )
        yield ORCHESTRATION_URL
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
    yield ORCHESTRATION_URL

    proc.terminate()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.kill()
    logger.info("Orchestration server stopped")


@pytest.fixture(scope="session")
def sveltekit_server(orchestration_server):
    """
    Start SvelteKit dev server pointed at the test orchestration service.

    The key env vars override the .env defaults so the dashboard talks to
    our test orchestration on port 3001 instead of the dev server on 3000.
    """
    SCREENSHOT_DIR.mkdir(exist_ok=True)

    env = {
        **os.environ,
        "PORT": str(SVELTEKIT_PORT),
        "ORCHESTRATION_URL": orchestration_server,
        "PUBLIC_WS_URL": orchestration_server.replace("http://", "ws://"),
        "ORIGIN": SVELTEKIT_URL,
    }

    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(DASHBOARD_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for SvelteKit to be ready (up to 30 seconds)
    for attempt in range(30):
        try:
            resp = httpx.get(SVELTEKIT_URL, timeout=2, follow_redirects=True)
            if resp.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("SvelteKit dev server did not start within 30 seconds")

    logger.info(f"SvelteKit dev server started on port {SVELTEKIT_PORT}")
    yield SVELTEKIT_URL

    proc.terminate()
    proc.wait(timeout=5)
    logger.info("SvelteKit dev server stopped")


# =============================================================================
# Function-scoped fixtures
# =============================================================================


@pytest.fixture
def browser(sveltekit_server):
    """Fresh AgentBrowser instance for each test."""
    stream_mode = os.environ.get("BROWSER_STREAM") == "1"
    b = AgentBrowser(
        headed=not stream_mode,
        stream_port=9224 if stream_mode else None,
        timeout=30,
    )
    b.open(sveltekit_server)
    yield b
    b.close()


@pytest.fixture
def screenshot_path():
    """Returns a function that generates timestamped screenshot paths."""
    def _path(name: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(SCREENSHOT_DIR / f"{ts}_{name}.png")
    return _path


@pytest.fixture
def live_session(orchestration_server, mock_fireflies_server):
    """
    Connect a live Fireflies session through the real pipeline.

    POSTs to /fireflies/connect with the mock server's API key and transcript ID.
    The orchestration service connects to the mock Fireflies server and starts
    streaming real transcript data through the translation pipeline.

    Yields session info. Disconnects on teardown.
    """
    mock = mock_fireflies_server

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
    resp.raise_for_status()
    data = resp.json()
    session_id = data["session_id"]

    logger.info(f"Live session connected: {session_id}")

    yield {
        "session_id": session_id,
        "transcript_id": mock["transcript_id"],
        "orchestration_url": orchestration_server,
        "ws_url": orchestration_server.replace("http://", "ws://"),
    }

    # Teardown: disconnect
    try:
        httpx.post(
            f"{orchestration_server}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=10,
        )
    except Exception:
        pass  # Best-effort cleanup


@pytest.fixture
def base_url(sveltekit_server):
    """Base URL for the SvelteKit dashboard."""
    return sveltekit_server
