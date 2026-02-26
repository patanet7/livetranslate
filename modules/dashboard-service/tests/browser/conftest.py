"""
Agent-browser fixtures for SvelteKit dashboard visual verification.

Starts the SvelteKit dev server and provides an AgentBrowser instance
for each test. Screenshots are saved to tests/browser/screenshots/.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

# Import AgentBrowser from orchestration-service's test helpers
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
ORCH_TESTS = REPO_ROOT / "modules" / "orchestration-service" / "tests"
sys.path.insert(0, str(ORCH_TESTS / "fireflies" / "e2e" / "browser"))

from browser_helpers import AgentBrowser  # noqa: E402

DASHBOARD_DIR = Path(__file__).parent.parent.parent  # modules/dashboard-service
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
DASHBOARD_URL = "http://localhost:5180"


@pytest.fixture(scope="session")
def sveltekit_server():
    """Start SvelteKit dev server for the full test session."""
    SCREENSHOT_DIR.mkdir(exist_ok=True)

    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(DASHBOARD_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PORT": "5180"},
    )

    # Wait for server ready (up to 30 seconds)
    for attempt in range(30):
        try:
            resp = httpx.get(DASHBOARD_URL, timeout=2, follow_redirects=True)
            if resp.status_code == 200:
                break
        except (httpx.ConnectError, httpx.ReadTimeout):
            time.sleep(1)
    else:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.fail("SvelteKit dev server did not start within 30 seconds")

    yield DASHBOARD_URL

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture
def browser(sveltekit_server):
    """Fresh AgentBrowser instance for each test."""
    b = AgentBrowser(headed=True)
    b.open(sveltekit_server)
    yield b
    b.close()


@pytest.fixture
def screenshot_path():
    """Returns a function that generates screenshot paths."""
    def _path(name: str) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return str(SCREENSHOT_DIR / f"{ts}_{name}.png")
    return _path
