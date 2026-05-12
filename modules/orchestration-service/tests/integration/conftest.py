"""
Pytest configuration for integration tests
"""

import gc
import os
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Make `tests/` importable so integration tests can do `from integration.fakes import ...`
_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (real backend, no mocks)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "websocket: mark test as WebSocket-related")


def _empty_device_cache():
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


@pytest_asyncio.fixture(scope="function", autouse=True)
async def reset_singletons():
    """
    Reset all dependency singletons before each test.

    This prevents event loop binding issues where singletons
    (EventPublisher, RedisClient, etc.) created in one test's
    event loop cause "Event loop is closed" errors in subsequent tests.
    """
    # Reset before test
    try:
        from src.dependencies import reset_dependencies, shutdown_dependencies

        await shutdown_dependencies()
        reset_dependencies()
    except ImportError:
        pass  # Not all tests may have src in path

    yield

    # Reset after test (cleanup)
    try:
        from src.dependencies import reset_dependencies, shutdown_dependencies

        await shutdown_dependencies()
        reset_dependencies()
    except ImportError:
        pass

    gc.collect()
    _empty_device_cache()


# ---------------------------------------------------------------------------
# Fake LLM HTTP servers — function-scoped real aiohttp servers (no mocks).
# Each test gets a fresh server so request capture lists don't bleed across
# tests; OS-assigned port keeps parallel runs collision-free.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="function")
async def fake_ollama_server() -> AsyncIterator[Any]:
    from integration.fakes import FakeOllamaServer

    server = FakeOllamaServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest_asyncio.fixture(loop_scope="function")
async def fake_openai_server() -> AsyncIterator[Any]:
    from integration.fakes import FakeOpenAIServer

    server = FakeOpenAIServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest_asyncio.fixture(loop_scope="function")
async def fake_anthropic_server() -> AsyncIterator[Any]:
    from integration.fakes import FakeAnthropicServer

    server = FakeAnthropicServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest.fixture
def llm_connection_factory() -> Callable[..., Any]:
    """Build LLMConnection instances quickly with sensible defaults.

    Tests pass only the fields they care about (engine, base_url, model);
    everything else gets a reasonable default. Returns a factory rather than
    a fixture instance so the same test can create multiple connections.
    """
    from livetranslate_common.models.llm import LLMConnection

    def _make(**overrides: Any) -> Any:
        defaults: dict[str, Any] = {
            "engine": "openai_compatible",
            "base_url": "http://localhost:8006/v1",
            "model": "test-model",
            "api_key": "",
            "temperature": 0.3,
            "max_tokens": 256,
            "timeout_s": 5.0,
            "max_retries": 0,
        }
        defaults.update(overrides)
        return LLMConnection(**defaults)

    return _make


@pytest.fixture(scope="session")
def verify_backend_running():
    """Verify backend is running (only used by tests that need HTTP endpoints).

    NOT autouse -- database-only tests (TestDatabaseIntegration, TestAudioChunking,
    TestSessionMetrics) do not need the backend running.  Tests that hit the live
    HTTP API should request this fixture explicitly.
    """
    import time

    import httpx

    backend_url = "http://localhost:3000/api/health"
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = httpx.get(backend_url, timeout=5.0)
            if response.status_code == 200:
                print("\nBackend is running at http://localhost:3000")
                return
        except Exception:
            if attempt < max_retries - 1:
                print(f"Waiting for backend... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                pytest.skip(
                    "Backend not running at http://localhost:3000 -- "
                    "skipping HTTP-dependent tests"
                )
