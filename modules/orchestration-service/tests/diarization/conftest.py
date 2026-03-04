"""
Diarization test conftest.

Overrides the heavy session-level infrastructure fixtures from the global
conftest (postgres container, Alembic migrations, Redis, etc.) so that
pure unit/model tests can run without any external services.
"""

import pytest


# ---------------------------------------------------------------------------
# Override session-scoped infrastructure fixtures from the global conftest.
# These stubs prevent testcontainers + Alembic from being invoked for tests
# that only exercise Pydantic models and need no database connectivity.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_container():
    """No-op: diarization model tests require no database container."""
    return None


@pytest.fixture(scope="session")
def redis_container():
    """No-op: diarization model tests require no Redis container."""
    return None


@pytest.fixture(scope="session")
def database_url():
    """Return a placeholder URL; no real connection is made."""
    return "postgresql://placeholder:placeholder@localhost:5432/placeholder"


@pytest.fixture(scope="session")
def run_migrations(database_url):
    """No-op: diarization model tests do not require schema migrations."""
    return None


@pytest.fixture(scope="session")
def verify_database_connection(database_url, run_migrations):
    """No-op: always returns True for model-only tests."""
    return True


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(verify_database_connection, database_url, redis_container):
    """Lightweight session setup for diarization model tests."""
    yield
