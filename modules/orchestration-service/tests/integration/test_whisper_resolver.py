"""Integration tests for the purpose-keyed Whisper resolver.

Mirror of test_llm_resolver.py for the Whisper side. Real PostgreSQL via the
`db_session` testcontainer fixture (no mocks, per repo CLAUDE.md).
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import pytest
from sqlalchemy import text

from livetranslate_common.models.whisper import WhisperParameterOverrides
from services.whisper_resolver import (
    reset_warn_state_for_tests,
    resolve_whisper_connection,
)


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _insert_whisper_connection(
    db: Any,
    *,
    id_: str | None = None,
    name: str | None = None,
    engine: str = "openai_compatible",
    url: str = "http://localhost:8005",
    api_key: str = "",
    prefix: str = "",
    default_model: str = "mlx-community/whisper-large-v3-turbo",
    enabled: bool = True,
    timeout_ms: int = 30000,
    priority: int = 0,
) -> str:
    conn_id = id_ or f"w-{uuid.uuid4().hex[:8]}"
    await db.execute(
        text(
            "INSERT INTO whisper_connections "
            "(id, name, engine, url, api_key, prefix, default_model, "
            "enabled, timeout_ms, priority) VALUES "
            "(:id, :name, :engine, :url, :api_key, :prefix, :default_model, "
            ":enabled, :timeout_ms, :priority)"
        ),
        {
            "id": conn_id,
            "name": name or conn_id,
            "engine": engine,
            "url": url,
            "api_key": api_key,
            "prefix": prefix,
            "default_model": default_model,
            "enabled": enabled,
            "timeout_ms": timeout_ms,
            "priority": priority,
        },
    )
    await db.commit()
    return conn_id


async def _set_transcription_preference(
    db: Any, active_model: str, **extras: Any
) -> None:
    payload = {"active_model": active_model, **extras}
    await db.execute(
        text(
            "INSERT INTO system_config (key, value) "
            "VALUES ('transcription_model_preference', :v) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
        ),
        {"v": json.dumps(payload)},
    )
    await db.commit()


@pytest.fixture(autouse=True)
def _reset_warn() -> None:
    reset_warn_state_for_tests()


@pytest.fixture(autouse=True)
def _clear_whisper_env(monkeypatch) -> None:
    """Strip env vars so step 4 doesn't accidentally hide step 5."""
    for k in ("VLLM_MLX_URL", "VLLM_MLX_API_KEY", "VLLM_MLX_MODEL", "WHISPER_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# Step 1 — explicit connection_id via overrides
# ---------------------------------------------------------------------------


async def test_step1_explicit_connection_id(db_session) -> None:
    conn_id = await _insert_whisper_connection(
        db_session, id_="w-explicit-1", url="http://h:8005", api_key="k"
    )
    resolved = await resolve_whisper_connection(
        "transcription",
        db_session,
        overrides=WhisperParameterOverrides(
            connection_id=conn_id, model="mlx-community/whisper-medium"
        ),
    )
    assert resolved.connection_id == conn_id
    assert resolved.engine == "openai_compatible"
    assert resolved.base_url == "http://h:8005"
    assert resolved.model == "mlx-community/whisper-medium"
    assert resolved.api_key == "k"


async def test_step1_unknown_connection_id_falls_through(db_session) -> None:
    resolved = await resolve_whisper_connection(
        "transcription",
        db_session,
        overrides=WhisperParameterOverrides(connection_id="nope", model="x"),
    )
    # No env, no DB rows → step 5 default. base_url/api_key come from step 5;
    # `model="x"` is then applied via overrides.merge() — that's correct behavior:
    # overrides win even when the connection_id lookup falls through.
    assert resolved.base_url == "http://localhost:8005"
    assert resolved.model == "x"
    assert resolved.connection_id is None  # step 5 fallback has no id


# ---------------------------------------------------------------------------
# Step 2 — transcription_model_preference
# ---------------------------------------------------------------------------


async def test_step2_transcription_preference(db_session) -> None:
    await _insert_whisper_connection(
        db_session,
        id_="c1",
        prefix="vllm",
        url="http://vllm:8005",
        api_key="dummy",
    )
    await _set_transcription_preference(
        db_session, "vllm/mlx-community/whisper-medium", beam_size=5
    )
    resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.connection_id == "c1"
    assert resolved.model == "mlx-community/whisper-medium"
    assert resolved.api_key == "dummy"
    assert resolved.beam_size == 5


async def test_step2_pref_with_initial_prompt_passes_through(db_session) -> None:
    await _insert_whisper_connection(
        db_session, id_="c2", prefix="p2", url="http://p2:8005"
    )
    await _set_transcription_preference(
        db_session,
        "p2/mlx-community/whisper-large-v3-turbo",
        initial_prompt="A live meeting.",
        no_speech_threshold=0.4,
    )
    resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.initial_prompt == "A live meeting."
    assert resolved.no_speech_threshold == 0.4


# ---------------------------------------------------------------------------
# Step 3 — default enabled connection (no preference set)
# ---------------------------------------------------------------------------


async def test_step3_default_enabled_priority_wins(db_session) -> None:
    await _insert_whisper_connection(
        db_session,
        id_="lo",
        prefix="lo",
        url="http://lo:8005",
        default_model="mlx-community/whisper-small",
        priority=1,
    )
    await _insert_whisper_connection(
        db_session,
        id_="hi",
        prefix="hi",
        url="http://hi:8005",
        default_model="mlx-community/whisper-large-v3-turbo",
        priority=99,
    )
    resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.connection_id == "hi"
    assert resolved.model == "mlx-community/whisper-large-v3-turbo"


async def test_step3_disabled_skipped(db_session) -> None:
    await _insert_whisper_connection(
        db_session,
        id_="disabled",
        prefix="x",
        url="http://no:1",
        enabled=False,
    )
    resolved = await resolve_whisper_connection("transcription", db_session)
    # Disabled rows skipped; no env; falls to step 5
    assert resolved.base_url == "http://localhost:8005"


# ---------------------------------------------------------------------------
# Step 4 — env-var bootstrap + one-shot WARN
# ---------------------------------------------------------------------------


async def test_step4_env_bootstrap(db_session, monkeypatch, caplog) -> None:
    monkeypatch.setenv("VLLM_MLX_URL", "http://100.64.0.2:8089")
    monkeypatch.setenv("VLLM_MLX_API_KEY", "dummy")
    monkeypatch.setenv("VLLM_MLX_MODEL", "mlx-community/whisper-large-v3-turbo")
    with caplog.at_level(logging.WARNING):
        resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.base_url == "http://100.64.0.2:8089"
    assert resolved.api_key == "dummy"
    assert resolved.model == "mlx-community/whisper-large-v3-turbo"
    assert any("env_fallback" in (r.message or "") for r in caplog.records)


async def test_step4_shares_llm_api_key(db_session, monkeypatch) -> None:
    """When VLLM_MLX_API_KEY is absent, LLM_API_KEY is used (same Tailscale box)."""
    monkeypatch.setenv("VLLM_MLX_URL", "http://100.64.0.2:8089")
    monkeypatch.setenv("LLM_API_KEY", "shared-bearer")
    monkeypatch.delenv("VLLM_MLX_API_KEY", raising=False)
    resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.api_key == "shared-bearer"


async def test_step4_warn_only_once_per_process(
    db_session, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("VLLM_MLX_URL", "http://x:1")
    with caplog.at_level(logging.WARNING):
        await resolve_whisper_connection("transcription", db_session)
        await resolve_whisper_connection("transcription", db_session)
    warns = [r for r in caplog.records if "env_fallback" in (r.message or "")]
    assert len(warns) == 1


# ---------------------------------------------------------------------------
# Step 5 — hardcoded default + ERROR log
# ---------------------------------------------------------------------------


async def test_step5_hardcoded_default(db_session, caplog) -> None:
    with caplog.at_level(logging.ERROR):
        resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.engine == "openai_compatible"
    assert resolved.base_url == "http://localhost:8005"
    assert resolved.model == "mlx-community/whisper-large-v3-turbo"
    assert any("hardcoded_default" in (r.message or "") for r in caplog.records)


# ---------------------------------------------------------------------------
# timeout_ms → timeout_s normalization
# ---------------------------------------------------------------------------


async def test_timeout_ms_to_timeout_s(db_session) -> None:
    await _insert_whisper_connection(
        db_session, id_="t", prefix="t", url="http://t:8005", timeout_ms=45000
    )
    await _set_transcription_preference(db_session, "t/mlx-community/whisper-medium")
    resolved = await resolve_whisper_connection("transcription", db_session)
    assert resolved.timeout_s == 45.0


# ---------------------------------------------------------------------------
# Overrides applied last
# ---------------------------------------------------------------------------


async def test_overrides_merge_after_resolution(db_session) -> None:
    await _insert_whisper_connection(
        db_session, id_="o", prefix="o", url="http://o:8005"
    )
    await _set_transcription_preference(
        db_session, "o/mlx-community/whisper-large-v3-turbo", temperature=0.0
    )
    resolved = await resolve_whisper_connection(
        "transcription",
        db_session,
        overrides=WhisperParameterOverrides(beam_size=5, language_hint="zh"),
    )
    assert resolved.beam_size == 5
    assert resolved.language_hint == "zh"


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------


async def test_bootstrap_seeds_when_empty_and_env_set(
    db_session, monkeypatch
) -> None:
    """bootstrap_whisper_connections inserts a row when table is empty + env set."""
    from services.whisper_resolver import bootstrap_whisper_connections

    monkeypatch.setenv("VLLM_MLX_URL", "http://100.64.0.2:8089")
    monkeypatch.setenv("LLM_API_KEY", "dummy")

    # Ensure table is empty (db_session truncates between tests so this should
    # already be true, but be explicit)
    await db_session.execute(text("DELETE FROM whisper_connections"))
    await db_session.commit()

    inserted = await bootstrap_whisper_connections(db_session)
    assert inserted is True

    row = (
        await db_session.execute(
            text("SELECT url, api_key, default_model FROM whisper_connections LIMIT 1")
        )
    ).fetchone()
    assert row is not None
    assert row[0] == "http://100.64.0.2:8089"
    assert row[1] == "dummy"


async def test_bootstrap_noop_when_not_empty(db_session, monkeypatch) -> None:
    from services.whisper_resolver import bootstrap_whisper_connections

    monkeypatch.setenv("VLLM_MLX_URL", "http://100.64.0.2:8089")
    await _insert_whisper_connection(
        db_session, id_="pre-existing", url="http://x:1"
    )

    inserted = await bootstrap_whisper_connections(db_session)
    assert inserted is False


async def test_bootstrap_noop_when_env_unset(db_session) -> None:
    from services.whisper_resolver import bootstrap_whisper_connections

    await db_session.execute(text("DELETE FROM whisper_connections"))
    await db_session.commit()

    inserted = await bootstrap_whisper_connections(db_session)
    assert inserted is False
