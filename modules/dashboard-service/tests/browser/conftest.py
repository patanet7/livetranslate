"""
Integrated E2E test fixtures for the SvelteKit dashboard.

Boots the FULL stack for behavioral testing:
1. Postgres + Redis containers (testcontainers) — real databases on dynamic ports
2. Mock Fireflies server (port 8090) — streams real captured transcript data
3. Orchestration service (uvicorn on port 3001) — real FastAPI backend
4. SvelteKit dev server (port 5180) — the dashboard under test

Browser tests hit the real SvelteKit app, which proxies to the real orchestration
service, which connects to the real mock Fireflies server. No mocks in the
test code — the entire pipeline is exercised end-to-end.

Screenshots are saved to tests/browser/screenshots/ as visual evidence.
"""

import asyncio
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

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

from browser_helpers import AgentBrowser, AgentBrowserError  # noqa: E402
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


# =============================================================================
# Infrastructure fixtures — Postgres and Redis testcontainers
# =============================================================================


@pytest.fixture(scope="session")
def postgres_container():
    """
    Start a PostgreSQL 16 container for the test session.

    Uses testcontainers for automatic lifecycle management and dynamic port
    allocation. The container is shared across all tests in the session.
    """
    with PostgresContainer(
        image="postgres:16-alpine",
        username="livetranslate",
        password="test_password",  # pragma: allowlist secret
        dbname="livetranslate_test",
    ) as pg:
        # Enable extensions required by the orchestration service schema
        from sqlalchemy import create_engine, text

        url = pg.get_connection_url()
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            conn.commit()
        engine.dispose()
        logger.info("Postgres testcontainer started with extensions enabled")
        yield pg


@pytest.fixture(scope="session")
def redis_container():
    """
    Start a Redis 7 container for the test session.

    Used by the orchestration service for session state, pub/sub, and the
    EventPublisher Redis Streams backend.
    """
    with RedisContainer(image="redis:7-alpine") as redis_c:
        host = redis_c.get_container_host_ip()
        port = redis_c.get_exposed_port(6379)
        redis_url = f"redis://{host}:{port}/0"
        logger.info(f"Redis testcontainer started: {redis_url}")
        yield redis_c


@pytest.fixture(scope="session")
def database_url(postgres_container):
    """
    Extract the database URL from the Postgres container and set env vars.

    Normalises the URL to plain ``postgresql://`` so that downstream code
    (Alembic env.py, DatabaseSettings, etc.) can convert it to the async
    driver (``postgresql+asyncpg://``) as expected.

    Also sets individual POSTGRES_*/DB_* env vars for code that reads
    them directly.
    """
    url = postgres_container.get_connection_url()
    # testcontainers returns postgresql+psycopg2:// -- strip the driver suffix
    url = url.replace("postgresql+psycopg2://", "postgresql://", 1)

    os.environ["DATABASE_URL"] = url
    os.environ["TEST_DATABASE_URL"] = url

    # Parse URL and set individual env vars for direct-connect code
    parsed = urlparse(url)
    os.environ["POSTGRES_HOST"] = parsed.hostname or "localhost"
    os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
    os.environ["POSTGRES_DB"] = (parsed.path or "/livetranslate_test").lstrip("/")
    os.environ["POSTGRES_USER"] = parsed.username or "livetranslate"
    os.environ["POSTGRES_PASSWORD"] = parsed.password or ""

    logger.info(f"Database URL configured: {parsed.hostname}:{parsed.port}")
    return url


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """
    Extract the Redis URL from the container and set the REDIS_URL env var.
    """
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    url = f"redis://{host}:{port}/0"
    os.environ["REDIS_URL"] = url
    return url


@pytest.fixture(scope="session")
def run_migrations(database_url):
    """
    Run Alembic migrations against the test database.

    Applies all migrations from the orchestration service's alembic directory
    to create the full schema (tables, indexes, constraints). This mirrors
    production deployment and catches migration issues early.
    """
    from alembic import command
    from alembic.config import Config

    alembic_ini = ORCH_ROOT / "alembic.ini"
    cfg = Config(str(alembic_ini))

    # Override script_location to absolute path — alembic.ini uses relative
    # "alembic" which would resolve against CWD (dashboard-service), not ORCH_ROOT
    cfg.set_main_option("script_location", str(ORCH_ROOT / "alembic"))

    # Convert to async URL for the asyncpg-based migrations
    async_url = database_url
    if database_url.startswith("postgresql://"):
        async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    cfg.set_main_option("sqlalchemy.url", async_url)

    command.upgrade(cfg, "head")
    logger.info("Alembic migrations applied successfully")


@pytest.fixture(scope="session")
def bot_sessions_schema(database_url, run_migrations):
    """
    Create the bot_sessions schema from raw SQL.

    The orchestration service uses both the Alembic-managed public schema
    and a separate bot_sessions schema created from scripts/bot-sessions-schema.sql.
    This fixture runs that SQL script if it exists.
    """
    import psycopg2

    scripts_dir = REPO_ROOT / "scripts"
    base_sql_file = scripts_dir / "bot-sessions-schema.sql"

    if not base_sql_file.exists():
        logger.warning("bot-sessions-schema.sql not found -- skipping bot_sessions schema setup")
        yield
        return

    conn = psycopg2.connect(database_url)
    conn.autocommit = True
    with conn.cursor() as cur:
        # Run base schema, stripping GRANT lines that reference roles
        # not present in the testcontainer
        base_sql = "\n".join(
            line
            for line in base_sql_file.read_text().splitlines()
            if not line.strip().upper().startswith("GRANT ")
        )
        cur.execute(base_sql)
        logger.info("bot_sessions schema created from SQL script")
    conn.close()
    yield


@pytest.fixture(scope="session")
def db_ready(database_url, redis_url, run_migrations, bot_sessions_schema):
    """
    Composite fixture that ensures all database infrastructure is ready.

    Depends on:
    - database_url: Postgres container running, URL set in env
    - redis_url: Redis container running, URL set in env
    - run_migrations: Alembic migrations applied
    - bot_sessions_schema: bot_sessions SQL schema created

    The orchestration_server fixture depends on this to guarantee
    Postgres and Redis are available before spawning the FastAPI subprocess.
    """
    return {"database_url": database_url, "redis_url": redis_url}


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

    # Run the event loop in a background thread so the aiohttp server
    # can accept and process Socket.IO connections. Without this, the
    # TCP port is open (kernel handles SYN/ACK) but the application
    # never runs — Socket.IO handshakes stall until wait_timeout.
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

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

    # Gracefully stop: schedule stop on the loop, then shut down the thread
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.run_until_complete(server.stop())
    loop.close()
    logger.info("Mock Fireflies server stopped")


@pytest.fixture(scope="session")
def orchestration_server(db_ready):
    """
    Start a real uvicorn server serving the FastAPI orchestration app.

    This is the same orchestration service that handles /fireflies/connect,
    /api/captions/stream, /fireflies/sessions, etc.

    Depends on db_ready to ensure Postgres and Redis containers are running
    and the schema is initialized before the FastAPI subprocess starts.

    Uses subprocess.Popen instead of multiprocessing.Process for reliability
    on macOS (spawn mode in Python 3.12+ can cause issues with forked processes).
    """
    if _port_is_open("localhost", ORCHESTRATION_PORT):
        logger.warning(
            f"Port {ORCHESTRATION_PORT} already in use -- reusing existing server."
        )
        yield ORCHESTRATION_URL
        return

    orch_src = str(ORCH_SRC)

    # Pass the testcontainer-provided database + redis URLs to the subprocess.
    # These come from the db_ready fixture which already set os.environ.
    orch_env = {
        **os.environ,
        "DATABASE_URL": db_ready["database_url"],
        "REDIS_URL": db_ready["redis_url"],
    }

    # Write server logs to a file so we can inspect them on failure
    orch_log_path = SCREENSHOT_DIR / "orchestration_server.log"
    SCREENSHOT_DIR.mkdir(exist_ok=True)
    orch_log_file = open(orch_log_path, "w")

    try:
        proc = subprocess.Popen(
            [
                sys.executable, "-c",
                f"import sys; sys.path.insert(0, '{orch_src}'); "
                f"import uvicorn; from main_fastapi import app; "
                f"uvicorn.run(app, host='127.0.0.1', port={ORCHESTRATION_PORT}, log_level='info')"
            ],
            stdout=orch_log_file,
            stderr=orch_log_file,
            cwd=orch_src,
            env=orch_env,
        )

        if not _wait_for_port("localhost", ORCHESTRATION_PORT, timeout=60):
            proc.terminate()
            proc.wait(timeout=5)
            log_tail = orch_log_path.read_text()[-1000:] if orch_log_path.exists() else ""
            pytest.fail(
                f"Orchestration server failed to start on port {ORCHESTRATION_PORT}. "
                f"Log tail: {log_tail}"
            )

        logger.info(f"Orchestration server started on port {ORCHESTRATION_PORT} (pid={proc.pid})")
        yield ORCHESTRATION_URL

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        logger.info("Orchestration server stopped")
    finally:
        orch_log_file.close()


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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    logger.info("SvelteKit dev server stopped")


# =============================================================================
# Browser fixture — session-scoped with auto-recovery
# =============================================================================


def _force_kill_browser():
    """Force-kill any lingering agent-browser / Chromium processes."""
    for proc_pattern in ["agent-browser", "chromium"]:
        try:
            subprocess.run(
                ["pkill", "-9", "-f", proc_pattern],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
    time.sleep(0.5)


class ResilientBrowser:
    """
    Wrapper around AgentBrowser that auto-recovers from daemon crashes.

    When the Chromium daemon becomes unresponsive (e.g., after a page fails
    with net::ERR_ABORTED), this wrapper force-kills the daemon and creates
    a fresh AgentBrowser instance transparently. Without this, a single bad
    page load causes every subsequent test to cascade-fail with 30s timeouts.
    """

    def __init__(self, headed: bool = False, timeout: int = 30, fallback_url: str = ""):
        self.headed = headed
        self.timeout = timeout
        self.fallback_url = fallback_url
        self._inner: AgentBrowser | None = None
        self._create_browser()

    def _create_browser(self) -> AgentBrowser:
        """Create a fresh AgentBrowser instance."""
        self._inner = AgentBrowser(headed=self.headed, timeout=self.timeout)
        return self._inner

    def _restart(self) -> None:
        """Force-kill daemon and create a fresh browser."""
        logger.warning("Restarting browser daemon after failure")
        try:
            if self._inner:
                self._inner.close()
        except Exception:
            pass
        _force_kill_browser()
        self._create_browser()

    def open(self, url: str) -> str:
        """Navigate to URL with auto-recovery on failure."""
        try:
            return self._inner.open(url)
        except AgentBrowserError as e:
            logger.warning(f"browser.open() failed: {e} — restarting daemon")
            self._restart()
            return self._inner.open(url)

    def close(self) -> None:
        if self._inner:
            self._inner.close()

    # Delegate all other methods to the inner browser
    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="session")
def browser(sveltekit_server):
    """
    Session-scoped resilient browser — one Chromium instance, auto-recovers.

    agent-browser runs as a daemon; creating/destroying per-test causes race
    conditions where the daemon hasn't fully exited before the next test starts.
    Session-scoping avoids this. The ResilientBrowser wrapper adds auto-recovery
    so a single page failure doesn't cascade to all subsequent tests.

    Headless by default. Set AGENT_BROWSER_HEADED=1 for visible browser.
    """
    # Clean slate — kill any leftover browser processes
    _force_kill_browser()

    headed = os.environ.get("AGENT_BROWSER_HEADED") == "1"
    b = ResilientBrowser(
        headed=headed,
        timeout=30,
        fallback_url=sveltekit_server,
    )
    b.open(sveltekit_server)
    yield b

    # Teardown
    b.close()
    _force_kill_browser()


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
