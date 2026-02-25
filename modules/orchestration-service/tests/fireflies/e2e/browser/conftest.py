"""
Browser E2E Test Fixtures

Provides:
- orchestration_server: Real uvicorn serving the FastAPI app on port 3001
- mock_fireflies: FirefliesMockServer with Spanish conversation scenario
- browser: AgentBrowser instance (headed or streaming based on env)
- browser_output_dir: Path for screenshots and logs
- live_session: Connect a live Fireflies session through the real pipeline
- ws_caption_messages: Collect WebSocket caption events for assertions

Inherits from root conftest: PostgreSQL testcontainer, Redis, Alembic migrations.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import httpx
import pytest
import uvicorn
import websockets

# Add src + tests to path
orchestration_root = Path(__file__).parent.parent.parent.parent.parent
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
    child_src = Path(__file__).parent.parent.parent.parent.parent / "src"
    sys.path.insert(0, str(child_src))

    from main_fastapi import app

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def browser_output_dir():
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
        logger.warning(
            f"Port {ORCHESTRATION_PORT} already in use — reusing existing server. "
            "Tests may behave differently if this is not the expected test server."
        )
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
def browser(browser_output_dir):
    """
    Launch agent-browser instance.

    Headless by default. Set env BROWSER_HEADED=1 for visible browser.
    Closes browser on teardown.
    """
    headed = os.environ.get("BROWSER_HEADED") == "1"
    stream_mode = os.environ.get("BROWSER_STREAM") == "1"

    b = AgentBrowser(
        headed=headed,
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
def setup_api_key(mock_fireflies_server):
    """
    Returns a callable that configures the Fireflies API key in the browser.

    Uses json.dumps for safe JS string escaping. Must be called after
    browser.open(dashboard_url).
    """

    def _setup(browser):
        api_key = mock_fireflies_server["api_key"]
        safe_key = json.dumps(api_key)
        browser.eval_js(f"""
            apiKey = {safe_key};
            localStorage.setItem('fireflies_api_key', {safe_key});
            updateApiStatus(true);
        """)
        browser.wait("300")

    return _setup


@pytest.fixture
def timestamp():
    """Current timestamp string for output file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# Live Pipeline Fixtures
# =============================================================================


@pytest.fixture
def live_session(orchestration_server, mock_fireflies_server):
    """
    Connect a live Fireflies session through the real pipeline.

    POSTs to /fireflies/connect with the mock server's URL as api_base_url,
    then yields session info. On teardown, disconnects the session.
    """
    base = orchestration_server
    mock = mock_fireflies_server

    resp = httpx.post(
        f"{base}/fireflies/connect",
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

    yield {
        "session_id": session_id,
        "transcript_id": mock["transcript_id"],
    }

    # Teardown: disconnect the session
    try:
        httpx.post(
            f"{base}/fireflies/disconnect",
            json={"session_id": session_id},
            timeout=10,
        )
    except Exception:
        logger.warning("live_session teardown: disconnect failed", exc_info=True)


@pytest.fixture
def ws_caption_messages():
    """
    Collect WebSocket caption messages for assertions.

    Yields (messages, connect_fn, close_fn):
    - messages: list that accumulates parsed JSON messages from the WebSocket
    - connect_fn(base_url, session_id): opens the WebSocket in a background thread
    - close_fn(): closes the WebSocket connection

    Uses the ``websockets`` library (already a project dependency) running
    in a dedicated asyncio event loop on a daemon thread.
    """
    messages: list[dict] = []
    _ws_ref: list = [None]  # mutable ref for the websocket connection
    _loop_ref: list = [None]  # mutable ref for the background event loop
    _thread_ref: list = [None]  # mutable ref for the background thread

    def _run_ws_loop(ws_url: str, loop: asyncio.AbstractEventLoop, ready: threading.Event):
        """Background thread target: run an asyncio loop that listens on the WebSocket."""
        asyncio.set_event_loop(loop)

        async def _listen():
            try:
                async with websockets.connect(ws_url) as ws:
                    _ws_ref[0] = ws
                    ready.set()
                    async for raw in ws:
                        try:
                            messages.append(json.loads(raw))
                        except json.JSONDecodeError:
                            logger.warning("ws_caption_messages: non-JSON message: %s", raw)
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("ws_caption_messages: WebSocket closed", exc_info=True)
            finally:
                ready.set()  # ensure we unblock even on early failure

        loop.run_until_complete(_listen())

    def connect(base_url: str, session_id: str):
        ws_url = base_url.replace("http://", "ws://") + f"/api/captions/stream/{session_id}"
        loop = asyncio.new_event_loop()
        _loop_ref[0] = loop
        ready = threading.Event()
        t = threading.Thread(target=_run_ws_loop, args=(ws_url, loop, ready), daemon=True)
        t.start()
        _thread_ref[0] = t
        # Wait up to 5 seconds for the connection to be established
        if not ready.wait(timeout=5):
            logger.warning("ws_caption_messages: timed out waiting for WebSocket connection")

    def close():
        ws = _ws_ref[0]
        loop = _loop_ref[0]
        if ws and loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(ws.close(), loop)
        t = _thread_ref[0]
        if t:
            t.join(timeout=3)

    yield messages, connect, close

    # Cleanup on fixture teardown
    close()
