"""Purpose-keyed resolver: produce an `LLMConnection` for any caller.

Single source of truth for "which LLM should I call for this purpose?".
Replaces the three old config systems (TranslationConfig env vars,
MeetingIntelligenceSettings.direct_llm_*, raw os.getenv) with one chain:

    1. explicit overrides.connection_id
    2. system_config[f"{purpose}_model_preference"]  (JSON active_model)
    3. default enabled connection in ai_connections (priority desc)
    4. env-var bootstrap (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL, then
       OLLAMA_BASE_URL / OLLAMA_MODEL). Logs a one-shot WARN per process.
    5. hardcoded last-resort default. Logs an ERROR.

Apply `LLMParameterOverrides.merge()` last; return.

Reuses the prefix-join SQL from the old `dependencies_connections.py:24-60`.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from livetranslate_common.llm.client import LLMClient
from livetranslate_common.logging import get_logger
from livetranslate_common.models.llm import (
    LLMConnection,
    LLMEngine,
    LLMParameterOverrides,
)

logger = get_logger()


Purpose = Literal["translation", "intelligence", "fireflies", "meetings", "chat"]


_ENGINE_BY_HEURISTIC: dict[str, LLMEngine] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    ":11434": "ollama",  # standard Ollama port
}


_ENV_WARN_EMITTED = False
_HARDCODED_WARN_EMITTED = False


def _infer_engine(url: str) -> LLMEngine:
    """Best-effort engine guess from a URL. Conservative — defaults to openai_compatible."""
    lowered = url.lower()
    for needle, engine in _ENGINE_BY_HEURISTIC.items():
        if needle in lowered:
            return engine
    return "openai_compatible"


def _normalize_base_url(url: str, engine: LLMEngine) -> str:
    """Ensure the URL ends in /v1 for openai-shaped engines.

    Ollama and Anthropic native APIs need bare hostnames (no /v1) — they
    have their own root paths (/api/generate, /v1/messages respectively).
    """
    url = url.rstrip("/")
    if engine in ("openai", "openai_compatible", "vllm") and not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _ms_to_s(ms: int | None, default: float = 30.0) -> float:
    if ms is None:
        return default
    return max(1.0, ms / 1000.0)


def _build_from_ai_connection_row(
    *,
    url: str,
    api_key: str,
    engine: str,
    model: str,
    timeout_ms: int | None,
    context_length: int | None,
    connection_id: str | None,
    pref: dict | None = None,
) -> LLMConnection:
    """Construct an LLMConnection from an ai_connections row + optional preference JSON.

    `timeout_ms` (DB) → `timeout_s` (model) happens here — single normalization point.
    """
    engine_lit: LLMEngine = engine if engine in {"ollama", "openai", "openai_compatible", "anthropic", "vllm"} else "openai_compatible"  # type: ignore[assignment]
    norm_url = _normalize_base_url(url, engine_lit)
    pref = pref or {}
    return LLMConnection(
        engine=engine_lit,
        base_url=norm_url,
        api_key=api_key or "",
        model=model,
        temperature=pref.get("temperature", 0.3),
        max_tokens=pref.get("max_tokens", 1024),
        top_p=pref.get("top_p", 0.8),
        top_k=pref.get("top_k", 20),
        repetition_penalty=pref.get("repetition_penalty", 1.05),
        presence_penalty=pref.get("presence_penalty", 1.5),
        timeout_s=_ms_to_s(timeout_ms),
        max_retries=pref.get("max_retries", 1),
        context_length=context_length,
        connection_id=connection_id,
    )


async def _lookup_by_id(db: AsyncSession, connection_id: str) -> LLMConnection | None:
    result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms, context_length "
            "FROM ai_connections "
            "WHERE id = :id AND enabled = true"
        ),
        {"id": connection_id},
    )
    row = result.fetchone()
    if not row:
        return None
    # When looking up by ID alone we don't know the model; caller (resolver)
    # must inject one via overrides.model or this returns None to fall through.
    return None  # signalled — see resolve() which composes with overrides.model


async def _step1_explicit_connection_id(
    db: AsyncSession, overrides: LLMParameterOverrides | None
) -> LLMConnection | None:
    if not overrides or not overrides.connection_id:
        return None
    result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms, context_length "
            "FROM ai_connections "
            "WHERE id = :id AND enabled = true"
        ),
        {"id": overrides.connection_id},
    )
    row = result.fetchone()
    if not row:
        return None
    # Model: prefer override; otherwise fall through (no model available).
    model = overrides.model
    if not model:
        return None
    return _build_from_ai_connection_row(
        connection_id=row[0],
        url=row[1],
        api_key=row[2],
        engine=row[3],
        timeout_ms=row[4],
        context_length=row[5],
        model=model,
    )


async def _step2_purpose_preference(
    db: AsyncSession, purpose: Purpose
) -> LLMConnection | None:
    """Look up system_config[{purpose}_model_preference] and join ai_connections."""
    pref_key = f"{purpose}_model_preference"
    result = await db.execute(
        text("SELECT value FROM system_config WHERE key = :key"),
        {"key": pref_key},
    )
    row = result.fetchone()
    if not row or not row[0]:
        return None
    try:
        pref = json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        logger.warning("llm_resolver_bad_pref_json", key=pref_key)
        return None
    active_model = pref.get("active_model") or ""
    if not active_model or "/" not in active_model:
        return None
    prefix, model_name = active_model.split("/", 1)
    conn_result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms, context_length "
            "FROM ai_connections "
            "WHERE prefix = :prefix AND enabled = true"
        ),
        {"prefix": prefix},
    )
    conn_row = conn_result.fetchone()
    if not conn_row:
        return None
    return _build_from_ai_connection_row(
        connection_id=conn_row[0],
        url=conn_row[1],
        api_key=conn_row[2],
        engine=conn_row[3],
        timeout_ms=conn_row[4],
        context_length=conn_row[5],
        model=model_name,
        pref=pref,
    )


async def _step3_default_enabled_connection(
    db: AsyncSession,
) -> LLMConnection | None:
    """Pick the highest-priority enabled ai_connections row.

    The row alone doesn't tell us which model to use. We use the first
    available model_name reachable: a stored 'default_model' system_config
    key, otherwise the connection's prefix (treating the prefix as model
    name is wrong but acceptable as a last default). In practice users
    should set a preference; this is a safety net.
    """
    result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms, context_length, prefix "
            "FROM ai_connections "
            "WHERE enabled = true "
            "ORDER BY priority DESC, name ASC LIMIT 1"
        )
    )
    row = result.fetchone()
    if not row:
        return None
    # Look for a system-wide default model
    dmodel_result = await db.execute(
        text("SELECT value FROM system_config WHERE key = 'default_model'")
    )
    dmodel_row = dmodel_result.fetchone()
    model = dmodel_row[0] if dmodel_row and dmodel_row[0] else (row[6] or "default")
    return _build_from_ai_connection_row(
        connection_id=row[0],
        url=row[1],
        api_key=row[2],
        engine=row[3],
        timeout_ms=row[4],
        context_length=row[5],
        model=model,
    )


def _step4_env_bootstrap() -> LLMConnection | None:
    """Synthesize a connection from LLM_* / OLLAMA_* env vars. One-shot WARN."""
    global _ENV_WARN_EMITTED
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    if not base_url:
        # Try OLLAMA_* fallback for one cycle
        base_url = os.getenv("OLLAMA_BASE_URL")
        model = model or os.getenv("OLLAMA_MODEL")
    if not base_url or not model:
        return None
    if not _ENV_WARN_EMITTED:
        logger.warning(
            "llm_resolver_env_fallback",
            base_url=base_url,
            model=model,
            message=(
                "LLM_BASE_URL / LLM_MODEL env vars used as bootstrap fallback. "
                "Configure ai_connections in the dashboard instead."
            ),
        )
        _ENV_WARN_EMITTED = True
    engine = _infer_engine(base_url)
    norm_url = _normalize_base_url(base_url, engine)
    api_key = os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    try:
        timeout_s = float(os.getenv("LLM_TIMEOUT_S", "30"))
    except ValueError:
        timeout_s = 30.0
    return LLMConnection(
        engine=engine,
        base_url=norm_url,
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        connection_id=None,
    )


def _step5_hardcoded_default() -> LLMConnection:
    global _HARDCODED_WARN_EMITTED
    if not _HARDCODED_WARN_EMITTED:
        logger.error(
            "llm_resolver_hardcoded_default",
            message=(
                "No LLM connection configured anywhere — falling back to local "
                "Ollama default. Set up ai_connections or LLM_* env vars."
            ),
        )
        _HARDCODED_WARN_EMITTED = True
    return LLMConnection(
        engine="ollama",
        base_url="http://localhost:11434",
        api_key="",
        model="qwen2.5:3b",
        timeout_s=30.0,
    )


async def resolve_llm_connection(
    purpose: Purpose,
    db: AsyncSession,
    *,
    overrides: LLMParameterOverrides | None = None,
) -> LLMConnection:
    """Resolve an LLM connection for `purpose`, applying per-call overrides last.

    Priority chain — first non-None wins:
        1. explicit `overrides.connection_id` (requires `overrides.model`)
        2. system_config[f"{purpose}_model_preference"] active_model
        3. default enabled ai_connections row
        4. LLM_BASE_URL / LLM_MODEL env vars (one-shot WARN)
        5. hardcoded ollama localhost default (ERROR)
    """
    resolved: LLMConnection | None = None
    try:
        resolved = await _step1_explicit_connection_id(db, overrides)
        if resolved is None:
            resolved = await _step2_purpose_preference(db, purpose)
        if resolved is None:
            resolved = await _step3_default_enabled_connection(db)
    except Exception as e:
        # Any DB error: degrade to env-var path. Don't take down the request.
        logger.warning("llm_resolver_db_error", error=str(e))

    if resolved is None:
        resolved = _step4_env_bootstrap()
    if resolved is None:
        resolved = _step5_hardcoded_default()

    if overrides is not None:
        resolved = resolved.merge(overrides)
    return resolved


async def resolve_llm_client(
    purpose: Purpose,
    db: AsyncSession,
    *,
    overrides: LLMParameterOverrides | None = None,
    proxy_mode: bool = False,
) -> LLMClient:
    """Convenience: resolve a connection + construct an LLMClient on it."""
    conn = await resolve_llm_connection(purpose, db, overrides=overrides)
    return LLMClient(conn, proxy_mode=proxy_mode)


def reset_warn_state_for_tests() -> None:
    """Test seam — reset the once-per-process WARN flags."""
    global _ENV_WARN_EMITTED, _HARDCODED_WARN_EMITTED
    _ENV_WARN_EMITTED = False
    _HARDCODED_WARN_EMITTED = False


async def bootstrap_llm_connections(db: AsyncSession) -> bool:
    """Seed an ai_connections row from env when the table is empty.

    Called from FastAPI lifespan startup. If `ai_connections` has zero rows
    AND `LLM_BASE_URL` is set, inserts a single row using env values so the
    dashboard Connections UI shows the env-configured backend rather than
    an empty list. Idempotent — no-op when the table already has rows or
    when env is unset. Returns True if a row was inserted.
    """
    count_result = await db.execute(text("SELECT COUNT(*) FROM ai_connections"))
    if (count_result.scalar() or 0) > 0:
        return False
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL")
    if not base_url or not model:
        logger.warning(
            "llm_bootstrap_skipped",
            reason="empty ai_connections + no LLM_BASE_URL/LLM_MODEL env",
        )
        return False
    engine = _infer_engine(base_url)
    api_key = os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    await db.execute(
        text(
            "INSERT INTO ai_connections "
            "(id, name, engine, url, api_key, prefix, enabled, timeout_ms, priority) "
            "VALUES (:id, :name, :engine, :url, :api_key, :prefix, true, 30000, 100)"
        ),
        {
            "id": "bootstrap-env",
            "name": "Bootstrap (from env)",
            "engine": engine,
            "url": base_url,
            "api_key": api_key,
            "prefix": "env",
        },
    )
    await db.commit()
    logger.info(
        "llm_bootstrap_seeded",
        base_url=base_url,
        model=model,
        engine=engine,
    )
    return True
