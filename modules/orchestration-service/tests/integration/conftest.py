"""
Pytest configuration for integration tests
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (real backend, no mocks)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "websocket: mark test as WebSocket-related"
    )


@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for each test function"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def verify_backend_running():
    """Verify backend is running before tests"""
    import httpx
    import time

    backend_url = "http://localhost:3000/api/health"
    max_retries = 5
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = httpx.get(backend_url, timeout=5.0)
            if response.status_code == 200:
                print(f"\n✅ Backend is running at http://localhost:3000")
                return
        except:
            if attempt < max_retries - 1:
                print(f"⏳ Waiting for backend... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                pytest.exit(
                    "❌ Backend not running at http://localhost:3000\n"
                    "Start backend: cd modules/orchestration-service && python src/main_fastapi.py"
                )
