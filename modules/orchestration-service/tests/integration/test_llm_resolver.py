"""Integration tests for the purpose-keyed LLM resolver.

Uses a real PostgreSQL database via the `db_session` testcontainer fixture
(no mocks — see repo CLAUDE.md). Verifies the 5-step priority chain:

    1. explicit overrides.connection_id
    2. system_config[{purpose}_model_preference]
    3. default enabled ai_connections row
    4. env-var bootstrap
    5. hardcoded last-resort default
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import pytest
from sqlalchemy import text

from livetranslate_common.models.llm import LLMParameterOverrides
from services.llm_resolver import (
    reset_warn_state_for_tests,
    resolve_llm_client,
    resolve_llm_connection,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _insert_ai_connection(
    db: Any,
    *,
    id_: str | None = None,
    name: str | None = None,
    engine: str = "ollama",
    url: str = "http://x:11434",
    api_key: str = "",
    prefix: str = "",
    enabled: bool = True,
    timeout_ms: int = 30000,
    priority: int = 0,
) -> str:
    conn_id = id_ or f"conn-{uuid.uuid4().hex[:8]}"
    await db.execute(
        text(
            "INSERT INTO ai_connections "
            "(id, name, engine, url, api_key, prefix, enabled, timeout_ms, priority) "
            "VALUES (:id, :name, :engine, :url, :api_key, :prefix, :enabled, :timeout_ms, :priority)"
        ),
        {
            "id": conn_id,
            "name": name or conn_id,
            "engine": engine,
            "url": url,
            "api_key": api_key,
            "prefix": prefix,
            "enabled": enabled,
            "timeout_ms": timeout_ms,
            "priority": priority,
        },
    )
    await db.commit()
    return conn_id


async def _set_preference(db: Any, purpose: str, active_model: str, **extras: Any) -> None:
    payload = {"active_model": active_model, **extras}
    await db.execute(
        text(
            "INSERT INTO system_config (key, value) VALUES (:k, :v) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
        ),
        {"k": f"{purpose}_model_preference", "v": json.dumps(payload)},
    )
    await db.commit()


@pytest.fixture(autouse=True)
def _reset_warn() -> None:
    reset_warn_state_for_tests()


# ---------------------------------------------------------------------------
# Step 1 — explicit connection_id via overrides
# ---------------------------------------------------------------------------


async def test_step1_explicit_connection_id(db_session, monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    conn_id = await _insert_ai_connection(
        db_session, id_="explicit-1", prefix="explicit", engine="ollama", url="http://h:11434"
    )
    resolved = await resolve_llm_connection(
        "translation",
        db_session,
        overrides=LLMParameterOverrides(connection_id=conn_id, model="qwen3:14b"),
    )
    assert resolved.connection_id == conn_id
    assert resolved.engine == "ollama"
    assert resolved.model == "qwen3:14b"


async def test_step1_unknown_connection_id_falls_through(
    db_session, monkeypatch
) -> None:
    # No DB row matches; with no env var fallback, step 5 kicks in
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    resolved = await resolve_llm_connection(
        "translation",
        db_session,
        overrides=LLMParameterOverrides(connection_id="nope", model="x"),
    )
    # step 5 default
    assert resolved.engine == "ollama"
    assert resolved.base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Step 2 — purpose preference
# ---------------------------------------------------------------------------


async def test_step2_translation_preference(db_session) -> None:
    await _insert_ai_connection(db_session, id_="c1", prefix="ollama", url="http://o:11434", engine="ollama")
    await _set_preference(db_session, "translation", "ollama/qwen3:14b", temperature=0.2)
    resolved = await resolve_llm_connection("translation", db_session)
    assert resolved.model == "qwen3:14b"
    assert resolved.engine == "ollama"
    assert resolved.temperature == 0.2


async def test_step2_per_purpose_isolation(db_session) -> None:
    await _insert_ai_connection(
        db_session, id_="cA", prefix="alpha", url="http://a:11434", engine="ollama"
    )
    await _insert_ai_connection(
        db_session,
        id_="cB",
        prefix="beta",
        url="https://api.openai.com/v1",
        engine="openai",
    )
    await _set_preference(db_session, "translation", "alpha/qwen3:14b")
    await _set_preference(db_session, "intelligence", "beta/gpt-4o-mini")

    t = await resolve_llm_connection("translation", db_session)
    i = await resolve_llm_connection("intelligence", db_session)
    assert t.connection_id == "cA"
    assert i.connection_id == "cB"
    assert t.model == "qwen3:14b"
    assert i.model == "gpt-4o-mini"


async def test_step2_disabled_connection_falls_through(db_session, monkeypatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    await _insert_ai_connection(
        db_session,
        id_="disabled",
        prefix="x",
        url="http://nope:1",
        engine="ollama",
        enabled=False,
    )
    await _set_preference(db_session, "translation", "x/qwen3:14b")
    # Step 2 returns None for disabled rows → step 3 is also empty → step 5
    resolved = await resolve_llm_connection("translation", db_session)
    assert resolved.base_url == "http://localhost:11434"  # step 5 default


# ---------------------------------------------------------------------------
# Step 3 — default enabled connection
# ---------------------------------------------------------------------------


async def test_step3_default_enabled_connection_priority_wins(
    db_session, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    await _insert_ai_connection(
        db_session, id_="lo", prefix="lo", url="http://lo:1", engine="ollama", priority=1
    )
    await _insert_ai_connection(
        db_session, id_="hi", prefix="hi", url="http://hi:1", engine="ollama", priority=99
    )
    resolved = await resolve_llm_connection("translation", db_session)
    assert resolved.connection_id == "hi"


# ---------------------------------------------------------------------------
# Step 4 — env-var bootstrap + one-shot WARN
# ---------------------------------------------------------------------------


async def test_step4_env_bootstrap(db_session, monkeypatch, caplog) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://100.64.0.2:8089/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3-4b-vllm-dmr")
    monkeypatch.setenv("LLM_API_KEY", "dummy")
    with caplog.at_level(logging.WARNING):
        resolved = await resolve_llm_connection("translation", db_session)
    assert resolved.base_url == "http://100.64.0.2:8089/v1"
    assert resolved.model == "qwen3-4b-vllm-dmr"
    assert resolved.api_key == "dummy"
    # WARN log was emitted
    assert any("env_fallback" in (r.message or "") for r in caplog.records)


async def test_step4_warn_only_once_per_process(
    db_session, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://x:1/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    with caplog.at_level(logging.WARNING):
        await resolve_llm_connection("translation", db_session)
        await resolve_llm_connection("translation", db_session)
    warns = [r for r in caplog.records if "env_fallback" in (r.message or "")]
    assert len(warns) == 1


# ---------------------------------------------------------------------------
# Step 5 — hardcoded default
# ---------------------------------------------------------------------------


async def test_step5_hardcoded_default(db_session, monkeypatch, caplog) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with caplog.at_level(logging.ERROR):
        resolved = await resolve_llm_connection("translation", db_session)
    assert resolved.engine == "ollama"
    assert resolved.base_url == "http://localhost:11434"
    assert any("hardcoded_default" in (r.message or "") for r in caplog.records)


# ---------------------------------------------------------------------------
# timeout_ms → timeout_s normalization
# ---------------------------------------------------------------------------


async def test_timeout_ms_to_timeout_s_conversion(db_session) -> None:
    await _insert_ai_connection(
        db_session,
        id_="t",
        prefix="t",
        url="http://t:11434",
        engine="ollama",
        timeout_ms=45000,
    )
    await _set_preference(db_session, "translation", "t/qwen3:14b")
    resolved = await resolve_llm_connection("translation", db_session)
    assert resolved.timeout_s == 45.0


# ---------------------------------------------------------------------------
# Overrides applied last
# ---------------------------------------------------------------------------


async def test_overrides_merge_after_resolution(db_session) -> None:
    await _insert_ai_connection(
        db_session, id_="o", prefix="o", url="http://o:11434", engine="ollama"
    )
    await _set_preference(db_session, "translation", "o/qwen3:14b", temperature=0.7)
    resolved = await resolve_llm_connection(
        "translation",
        db_session,
        overrides=LLMParameterOverrides(temperature=0.1, top_p=0.5),
    )
    assert resolved.temperature == 0.1
    assert resolved.top_p == 0.5


# ---------------------------------------------------------------------------
# resolve_llm_client returns a usable client
# ---------------------------------------------------------------------------


async def test_resolve_llm_client_builds_real_client(db_session, monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://example:8089/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    client = await resolve_llm_client("translation", db_session)
    try:
        assert client.connection.model == "test-model"
        assert client.connection.base_url.startswith("http://example:8089")
    finally:
        await client.aclose()
