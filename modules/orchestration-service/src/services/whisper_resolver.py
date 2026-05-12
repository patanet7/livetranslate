"""Purpose-keyed resolver: produce a `WhisperConnection` for any caller.

Mirror of `llm_resolver.py` for the Whisper inference side. Single source of
truth for "which Whisper backend should we use?". Priority chain — first
non-None wins:

    1. explicit overrides.connection_id (requires overrides.model)
    2. system_config["transcription_model_preference"] active_model
    3. default enabled whisper_connections row (priority desc)
    4. env-var bootstrap (VLLM_MLX_URL / VLLM_MLX_MODEL, falls back to
       LLM_API_KEY for shared Tailscale boxes). One-shot WARN per process.
    5. hardcoded last-resort default (localhost:8005, no auth). ERROR log.

Apply WhisperParameterOverrides.merge() last; return.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from livetranslate_common.logging import get_logger
from livetranslate_common.models.whisper import (
    WhisperConnection,
    WhisperEngine,
    WhisperParameterOverrides,
)
from livetranslate_common.tracing import base_url_host

logger = get_logger()


WhisperPurpose = Literal["transcription"]


_ENV_WARN_EMITTED = False
_HARDCODED_WARN_EMITTED = False


def _infer_engine(url: str) -> WhisperEngine:
    """Engines that go over HTTP all use 'openai_compatible'. Local-only
    engines (mlx_local, faster_whisper_local) cannot be inferred from a URL
    — those are explicit user choices set via the dashboard."""
    return "openai_compatible"


def _normalize_base_url(url: str) -> str:
    """Strip trailing slash. Whisper API uses /v1/audio/transcriptions which the
    client appends; the base_url is the server root."""
    return url.rstrip("/")


def _ms_to_s(ms: int | None, default: float = 30.0) -> float:
    if ms is None:
        return default
    return max(1.0, ms / 1000.0)


_VALID_ENGINES: frozenset[str] = frozenset(
    {"openai_compatible", "mlx_local", "faster_whisper_local"}
)


def _coerce_engine(engine: str) -> WhisperEngine:
    """Strict engine validation.

    The DB has a CHECK constraint that rejects unknown engines at INSERT time,
    so a row with an out-of-set engine indicates database corruption — failing
    loud here surfaces the bug instead of silently routing to a default backend.
    """
    if engine not in _VALID_ENGINES:
        raise ValueError(
            f"whisper_connections.engine={engine!r} not in {sorted(_VALID_ENGINES)}; "
            "DB row is corrupt — CHECK constraint should have prevented this"
        )
    return engine  # type: ignore[return-value]


def _build_from_whisper_row(
    *,
    row_mapping: dict[str, object],
    model: str,
    connection_id: str | None,
    pref: dict | None = None,
) -> WhisperConnection:
    """Construct WhisperConnection from a whisper_connections row mapping + pref JSON.

    `row_mapping` is `sqlalchemy.Row._mapping` (or any dict-like with column-name
    keys). Reading by name decouples this builder from SELECT column order —
    adding a column to the SELECT won't silently shift fields the way positional
    `row[0], row[1]` would.
    """
    engine_lit = _coerce_engine(str(row_mapping["engine"]))
    raw_url = str(row_mapping["url"]) if row_mapping.get("url") is not None else ""
    api_key = str(row_mapping.get("api_key") or "")
    timeout_ms_val = row_mapping.get("timeout_ms")
    timeout_ms: int | None = (
        int(timeout_ms_val) if isinstance(timeout_ms_val, (int, float)) else None
    )
    norm_url = _normalize_base_url(raw_url) if engine_lit == "openai_compatible" else ""
    pref = pref or {}
    return WhisperConnection(
        engine=engine_lit,
        base_url=norm_url,
        api_key=api_key,
        model=model,
        temperature=pref.get("temperature", 0.0),
        beam_size=pref.get("beam_size", 1),
        no_speech_threshold=pref.get("no_speech_threshold", 0.6),
        compression_ratio_threshold=pref.get("compression_ratio_threshold", 2.4),
        language_hint=pref.get("language_hint"),
        initial_prompt=pref.get("initial_prompt"),
        compute_type=pref.get("compute_type", "float16"),
        timeout_s=_ms_to_s(timeout_ms),
        max_retries=pref.get("max_retries", 1),
        connection_id=connection_id,
    )


async def _step1_explicit_connection_id(
    db: AsyncSession, overrides: WhisperParameterOverrides | None
) -> WhisperConnection | None:
    if not overrides or not overrides.connection_id:
        return None
    result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms "
            "FROM whisper_connections "
            "WHERE id = :id AND enabled = true"
        ),
        {"id": overrides.connection_id},
    )
    row = result.fetchone()
    if not row:
        return None
    model = overrides.model
    if not model:
        # Caller pinned a connection but didn't say which model; fall through.
        return None
    return _build_from_whisper_row(
        row_mapping=dict(row._mapping),
        connection_id=str(row._mapping["id"]),
        model=model,
    )


async def _step2_purpose_preference(
    db: AsyncSession, purpose: WhisperPurpose
) -> WhisperConnection | None:
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
        logger.warning("whisper_resolver_bad_pref_json", key=pref_key)
        return None
    active_model = pref.get("active_model") or ""
    if not active_model or "/" not in active_model:
        return None
    prefix, model_name = active_model.split("/", 1)
    conn_result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms "
            "FROM whisper_connections "
            "WHERE prefix = :prefix AND enabled = true"
        ),
        {"prefix": prefix},
    )
    conn_row = conn_result.fetchone()
    if not conn_row:
        return None
    return _build_from_whisper_row(
        row_mapping=dict(conn_row._mapping),
        connection_id=str(conn_row._mapping["id"]),
        model=model_name,
        pref=pref,
    )


async def _step3_default_enabled_connection(
    db: AsyncSession,
) -> WhisperConnection | None:
    result = await db.execute(
        text(
            "SELECT id, url, api_key, engine, timeout_ms, prefix, default_model "
            "FROM whisper_connections "
            "WHERE enabled = true "
            "ORDER BY priority DESC, name ASC LIMIT 1"
        )
    )
    row = result.fetchone()
    if not row:
        return None
    model = (
        str(row._mapping["default_model"])
        if row._mapping.get("default_model")
        else "mlx-community/whisper-large-v3-turbo"
    )
    return _build_from_whisper_row(
        row_mapping=dict(row._mapping),
        connection_id=str(row._mapping["id"]),
        model=model,
    )


def _step4_env_bootstrap() -> WhisperConnection | None:
    """Synthesize a connection from VLLM_MLX_URL etc. One-shot WARN."""
    global _ENV_WARN_EMITTED
    base_url = os.getenv("VLLM_MLX_URL")
    model = os.getenv("VLLM_MLX_MODEL") or "mlx-community/whisper-large-v3-turbo"
    if not base_url:
        return None
    if not _ENV_WARN_EMITTED:
        logger.warning(
            "whisper_resolver_env_fallback",
            base_url=base_url,
            model=model,
            message=(
                "VLLM_MLX_URL env var used as bootstrap fallback. "
                "Configure whisper_connections in the dashboard instead."
            ),
        )
        _ENV_WARN_EMITTED = True
    # Share LLM_API_KEY for fleets where one Bearer covers both endpoints
    # (Tailscale-hosted vllm boxes commonly do this).
    api_key = os.getenv("VLLM_MLX_API_KEY") or os.getenv("LLM_API_KEY") or ""
    try:
        timeout_s = float(os.getenv("WHISPER_TIMEOUT_S", "30"))
    except ValueError:
        timeout_s = 30.0
    return WhisperConnection(
        engine="openai_compatible",
        base_url=_normalize_base_url(base_url),
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        connection_id=None,
    )


def _step5_hardcoded_default() -> WhisperConnection:
    global _HARDCODED_WARN_EMITTED
    if not _HARDCODED_WARN_EMITTED:
        logger.error(
            "whisper_resolver_hardcoded_default",
            message=(
                "No Whisper connection configured anywhere — falling back to "
                "localhost:8005. Configure whisper_connections or set VLLM_MLX_URL."
            ),
        )
        _HARDCODED_WARN_EMITTED = True
    return WhisperConnection(
        engine="openai_compatible",
        base_url="http://localhost:8005",
        api_key="",
        model="mlx-community/whisper-large-v3-turbo",
        timeout_s=30.0,
    )


async def resolve_whisper_connection(
    purpose: WhisperPurpose,
    db: AsyncSession,
    *,
    overrides: WhisperParameterOverrides | None = None,
) -> WhisperConnection:
    """Resolve a Whisper connection for `purpose`, applying overrides last.

    Emits whisper.connection.resolved with the source step + chosen
    connection_id + engine + base_url_host so operators can trace
    "which Whisper backend is this session actually using?".
    """
    resolved: WhisperConnection | None = None
    source: str = "step5"  # default; overwritten below
    try:
        resolved = await _step1_explicit_connection_id(db, overrides)
        if resolved is not None:
            source = "step1"
        if resolved is None:
            resolved = await _step2_purpose_preference(db, purpose)
            if resolved is not None:
                source = "step2"
        if resolved is None:
            resolved = await _step3_default_enabled_connection(db)
            if resolved is not None:
                source = "step3"
    except Exception as e:
        # DB error: degrade gracefully to env path.
        logger.warning("whisper_resolver_db_error", error=str(e))

    if resolved is None:
        resolved = _step4_env_bootstrap()
        if resolved is not None:
            source = "step4"
    if resolved is None:
        resolved = _step5_hardcoded_default()
        source = "step5"

    if overrides is not None:
        resolved = resolved.merge(overrides)

    logger.info(
        "whisper.connection.resolved",
        purpose=purpose,
        source=source,
        connection_id=resolved.connection_id,
        engine=resolved.engine,
        base_url_host=base_url_host(resolved.base_url),
        model=resolved.model,
        has_api_key=bool(resolved.api_key),
        had_overrides=overrides is not None,
    )
    return resolved


def reset_warn_state_for_tests() -> None:
    """Test seam — reset the once-per-process WARN flags."""
    global _ENV_WARN_EMITTED, _HARDCODED_WARN_EMITTED
    _ENV_WARN_EMITTED = False
    _HARDCODED_WARN_EMITTED = False


async def bootstrap_whisper_connections(db: AsyncSession) -> bool:
    """Seed a whisper_connections row from env when the table is empty.

    Called from FastAPI lifespan startup. If `whisper_connections` has zero rows
    AND `VLLM_MLX_URL` is set, inserts a single row using env values. Idempotent.
    Returns True if a row was inserted.
    """
    count_result = await db.execute(text("SELECT COUNT(*) FROM whisper_connections"))
    if (count_result.scalar() or 0) > 0:
        return False
    base_url = os.getenv("VLLM_MLX_URL")
    model = os.getenv("VLLM_MLX_MODEL") or "mlx-community/whisper-large-v3-turbo"
    if not base_url:
        logger.warning(
            "whisper_bootstrap_skipped",
            reason="empty whisper_connections + no VLLM_MLX_URL env",
        )
        return False
    api_key = os.getenv("VLLM_MLX_API_KEY") or os.getenv("LLM_API_KEY") or ""
    await db.execute(
        text(
            "INSERT INTO whisper_connections "
            "(id, name, engine, url, api_key, prefix, default_model, "
            "enabled, timeout_ms, priority) "
            "VALUES (:id, :name, :engine, :url, :api_key, :prefix, :model, "
            "true, 30000, 100)"
        ),
        {
            "id": "bootstrap-whisper-env",
            "name": "Bootstrap Whisper (from env)",
            "engine": "openai_compatible",
            "url": base_url,
            "api_key": api_key,
            "prefix": "env",
            "model": model,
        },
    )
    await db.commit()
    logger.info(
        "whisper_bootstrap_seeded",
        base_url=base_url,
        model=model,
    )
    return True
