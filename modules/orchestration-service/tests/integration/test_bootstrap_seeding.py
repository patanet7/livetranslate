"""`bootstrap_llm_connections` seeds ai_connections from env on first run.

Idempotent: no-op when DB already has rows or when env vars are unset.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from services.llm_resolver import bootstrap_llm_connections


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_seeds_when_env_set_and_db_empty(db_session, monkeypatch) -> None:
    # Migration 012 seeds default ai_connections rows; clear them to simulate
    # a true first-run install.
    await db_session.execute(text("DELETE FROM ai_connections"))
    await db_session.commit()
    monkeypatch.setenv("LLM_BASE_URL", "http://test-host:8089/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    seeded = await bootstrap_llm_connections(db_session)
    assert seeded is True
    row = (
        await db_session.execute(
            text(
                "SELECT url, api_key, engine FROM ai_connections WHERE id = 'bootstrap-env'"
            )
        )
    ).fetchone()
    assert row is not None
    assert row[0] == "http://test-host:8089/v1"
    assert row[1] == "dummy-key"


async def test_no_seed_when_db_nonempty(db_session, monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://test:1/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    # Seed an existing row by hand
    await db_session.execute(
        text(
            "INSERT INTO ai_connections (id, name, engine, url, api_key, prefix, enabled) "
            "VALUES ('preexisting', 'pre', 'ollama', 'http://pre:1', '', 'pre', true)"
        )
    )
    await db_session.commit()
    seeded = await bootstrap_llm_connections(db_session)
    assert seeded is False
    # Bootstrap row was NOT inserted
    row = (
        await db_session.execute(
            text("SELECT id FROM ai_connections WHERE id = 'bootstrap-env'")
        )
    ).fetchone()
    assert row is None


async def test_no_seed_when_env_missing(db_session, monkeypatch) -> None:
    await db_session.execute(text("DELETE FROM ai_connections"))
    await db_session.commit()
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    seeded = await bootstrap_llm_connections(db_session)
    assert seeded is False
