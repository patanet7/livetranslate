"""Conftest for unit tests — overrides Docker-dependent session fixtures.

The parent conftest.py has autouse session fixtures that require Docker
(testcontainers for Postgres and Redis). Unit tests don't need databases,
so this conftest overrides those fixture dependencies with lightweight stubs
to allow unit tests to run without Docker.
"""
import pytest


@pytest.fixture(scope="session")
def postgres_container():
    """Override: unit tests don't need a real Postgres container."""
    return None


@pytest.fixture(scope="session")
def redis_container():
    """Override: unit tests don't need a real Redis container."""

    class _Stub:
        def get_container_host_ip(self):
            return "localhost"

        def get_exposed_port(self, port):
            return port

    return _Stub()


@pytest.fixture(scope="session")
def database_url(postgres_container):
    """Override: return a placeholder URL — unit tests don't hit the DB."""
    return "postgresql://livetranslate:livetranslate_dev_password@localhost:5433/livetranslate_test"


@pytest.fixture(scope="session")
def run_migrations(database_url):
    """Override: unit tests skip migrations."""
    return None


@pytest.fixture(scope="session")
def verify_database_connection(database_url, run_migrations):
    """Override: unit tests skip DB connectivity check."""
    return False
