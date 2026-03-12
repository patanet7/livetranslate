"""Unified AI connection management service.

Replaces ProviderRegistry and JSON config file with DB-backed connections.
All features (Chat, Translation, Intelligence) share this pool.
"""

import ipaddress
import time
from datetime import UTC, datetime
from urllib.parse import urlparse

import aiohttp
from fastapi import HTTPException
from livetranslate_common.logging import get_logger
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database.ai_connection import AIConnection
from models.connections import (
    AIConnectionCreate,
    AIConnectionResponse,
    AIConnectionUpdate,
    AggregatedModel,
    AggregateModelsResponse,
    VerifyResult,
    slugify,
)
from services.llm.adapter import LLMAdapter
from services.llm.providers.anthropic_provider import AnthropicAdapter
from services.llm.providers.ollama import OllamaAdapter
from services.llm.providers.openai_compat import OpenAICompatAdapter
from services.llm.providers.openai_provider import OpenAIAdapter

logger = get_logger()

# SSRF protection
_BLOCKED_METADATA_IPS = {
    "169.254.169.254",  # AWS/GCP
    "100.100.100.200",  # Alibaba
    "fd00::1",
}


def _validate_connection_url(url: str) -> None:
    """Block SSRF-prone URLs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed")
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL: no hostname")
    if hostname in _BLOCKED_METADATA_IPS:
        raise HTTPException(status_code=400, detail="Blocked: cloud metadata endpoint")
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_link_local:
            raise HTTPException(status_code=400, detail="Link-local addresses are not allowed")
    except ValueError:
        pass  # Hostname, not IP — fine


class ConnectionService:
    """Manages AI backend connections stored in PostgreSQL."""

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self._adapter_cache: dict[tuple[str, datetime | None], LLMAdapter] = {}

    # ── CRUD ────────────────────────────────────────────────────────────

    async def list_connections(self, *, enabled_only: bool = False) -> list[AIConnection]:
        stmt = select(AIConnection).order_by(AIConnection.priority, AIConnection.name)
        if enabled_only:
            stmt = stmt.where(AIConnection.enabled.is_(True))
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_connection(self, connection_id: str) -> AIConnection:
        result = await self.db.execute(
            select(AIConnection).where(AIConnection.id == connection_id)
        )
        conn = result.scalar_one_or_none()
        if not conn:
            raise HTTPException(status_code=404, detail=f"Connection '{connection_id}' not found")
        return conn

    async def create_connection(self, data: AIConnectionCreate) -> AIConnection:
        conn_id = data.id or slugify(data.name)
        if not conn_id:
            raise HTTPException(status_code=400, detail="Cannot generate ID from name")

        _validate_connection_url(data.url)

        existing = await self.db.execute(
            select(AIConnection).where(AIConnection.id == conn_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail=f"Connection '{conn_id}' already exists")

        conn = AIConnection(
            id=conn_id,
            name=data.name,
            engine=data.engine,
            url=data.url,
            api_key=data.api_key,
            prefix=data.prefix or conn_id,
            enabled=data.enabled,
            context_length=data.context_length,
            timeout_ms=data.timeout_ms,
            max_retries=data.max_retries,
            priority=data.priority,
        )
        self.db.add(conn)
        await self.db.commit()
        await self.db.refresh(conn)
        logger.info("connection_created", connection_id=conn_id, engine=data.engine)
        return conn

    async def update_connection(self, connection_id: str, data: AIConnectionUpdate) -> AIConnection:
        conn = await self.get_connection(connection_id)
        updates = data.model_dump(exclude_none=True)
        if "url" in updates:
            _validate_connection_url(updates["url"])
        for field, value in updates.items():
            setattr(conn, field, value)
        conn.updated_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(conn)

        # Invalidate adapter cache for this connection
        self._adapter_cache = {
            k: v for k, v in self._adapter_cache.items() if k[0] != connection_id
        }
        logger.info("connection_updated", connection_id=connection_id)
        return conn

    async def delete_connection(self, connection_id: str) -> None:
        await self.get_connection(connection_id)  # 404 if missing
        await self.db.execute(
            delete(AIConnection).where(AIConnection.id == connection_id)
        )
        await self.db.commit()
        self._adapter_cache = {
            k: v for k, v in self._adapter_cache.items() if k[0] != connection_id
        }
        logger.info("connection_deleted", connection_id=connection_id)

    # ── Verify ──────────────────────────────────────────────────────────

    async def verify_connection(self, connection_id: str) -> VerifyResult:
        """Probe a connection and return status + discovered models."""
        conn = await self.get_connection(connection_id)
        _validate_connection_url(conn.url)

        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                if conn.engine == "ollama":
                    result = await self._verify_ollama(session, conn)
                elif conn.engine in ("openai", "openai_compatible"):
                    result = await self._verify_openai_compat(session, conn)
                elif conn.engine == "anthropic":
                    result = await self._verify_anthropic(session, conn)
                else:
                    return VerifyResult(status="error", message=f"Unknown engine: {conn.engine}")

            result.latency_ms = round((time.time() - start) * 1000, 1)
            return result
        except aiohttp.ClientError as e:
            return VerifyResult(
                status="error",
                message=f"Connection failed: {e}",
                latency_ms=round((time.time() - start) * 1000, 1),
            )

    async def _verify_ollama(self, session: aiohttp.ClientSession, conn: AIConnection) -> VerifyResult:
        url = conn.url.rstrip("/")
        headers = {}
        if conn.api_key:
            headers["Authorization"] = f"Bearer {conn.api_key}"

        async with session.get(f"{url}/api/tags", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return VerifyResult(status="error", message=f"HTTP {resp.status}")
            data = await resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return VerifyResult(status="connected", message=f"{len(models)} models available", models=models)

    async def _verify_openai_compat(self, session: aiohttp.ClientSession, conn: AIConnection) -> VerifyResult:
        url = conn.url.rstrip("/")
        if conn.engine == "openai":
            url = "https://api.openai.com"
        headers = {"Authorization": f"Bearer {conn.api_key}"} if conn.api_key else {}

        async with session.get(f"{url}/v1/models", headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                text = await resp.text()
                return VerifyResult(status="error", message=f"HTTP {resp.status}: {text[:200]}")
            data = await resp.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            return VerifyResult(status="connected", message=f"{len(models)} models available", models=models)

    async def _verify_anthropic(self, session: aiohttp.ClientSession, conn: AIConnection) -> VerifyResult:
        if not conn.api_key:
            return VerifyResult(status="error", message="API key required for Anthropic")
        headers = {
            "x-api-key": conn.api_key,
            "anthropic-version": "2023-06-01",
        }
        url = "https://api.anthropic.com/v1/models"
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                return VerifyResult(status="connected", message=f"{len(models)} models", models=models)
            elif resp.status == 401:
                return VerifyResult(status="error", message="Invalid API key")
            else:
                return VerifyResult(status="connected", message="API key accepted (model list unavailable)", models=[])

    # ── Aggregate Models ────────────────────────────────────────────────

    async def aggregate_models(self, *, enabled_only: bool = True) -> AggregateModelsResponse:
        """Probe all enabled connections and return prefixed model list."""
        connections = await self.list_connections(enabled_only=enabled_only)
        all_models: list[AggregatedModel] = []
        errors: list[dict] = []

        for conn in connections:
            try:
                result = await self.verify_connection(conn.id)
                if result.status == "connected":
                    for model_name in result.models:
                        prefixed_id = f"{conn.prefix}/{model_name}" if conn.prefix else model_name
                        all_models.append(AggregatedModel(
                            id=prefixed_id,
                            name=model_name,
                            connection_id=conn.id,
                            connection_name=conn.name,
                            prefix=conn.prefix,
                            engine=conn.engine,
                        ))
                else:
                    errors.append({
                        "connection_id": conn.id,
                        "connection_name": conn.name,
                        "message": result.message,
                    })
            except Exception as e:
                errors.append({
                    "connection_id": conn.id,
                    "connection_name": conn.name,
                    "message": f"Verify failed: {e}",
                })

        return AggregateModelsResponse(models=all_models, errors=errors)

    # ── Adapter Factory ─────────────────────────────────────────────────

    async def get_adapter(self, connection_id: str) -> LLMAdapter:
        """Get or create a cached LLMAdapter for a connection."""
        conn = await self.get_connection(connection_id)
        cache_key = (connection_id, conn.updated_at)

        if cache_key in self._adapter_cache:
            return self._adapter_cache[cache_key]

        adapter = self._build_adapter(conn)

        # Evict stale entries for this connection
        self._adapter_cache = {
            k: v for k, v in self._adapter_cache.items() if k[0] != connection_id
        }
        self._adapter_cache[cache_key] = adapter
        return adapter

    def _build_adapter(self, conn: AIConnection) -> LLMAdapter:
        if conn.engine == "ollama":
            return OllamaAdapter(base_url=conn.url)
        elif conn.engine == "openai":
            return OpenAIAdapter(api_key=conn.api_key)
        elif conn.engine == "anthropic":
            return AnthropicAdapter(api_key=conn.api_key)
        elif conn.engine == "openai_compatible":
            return OpenAICompatAdapter(base_url=conn.url, api_key=conn.api_key)
        else:
            raise ValueError(f"Unknown engine: {conn.engine}")

    # ── Model Resolution ────────────────────────────────────────────────

    async def resolve_model(self, model_id: str) -> tuple[AIConnection, str]:
        """Parse 'prefix/model_name' and return (connection, model_name).

        Falls back to matching by connection ID if no prefix match.
        """
        if "/" in model_id:
            prefix, model_name = model_id.split("/", 1)
            result = await self.db.execute(
                select(AIConnection).where(AIConnection.prefix == prefix)
            )
            conn = result.scalar_one_or_none()
            if conn:
                return conn, model_name

            # Try matching by ID
            result = await self.db.execute(
                select(AIConnection).where(AIConnection.id == prefix)
            )
            conn = result.scalar_one_or_none()
            if conn:
                return conn, model_name

        # No prefix — use first enabled connection
        connections = await self.list_connections(enabled_only=True)
        if not connections:
            raise HTTPException(status_code=404, detail="No enabled connections available")
        return connections[0], model_id

    # ── Helpers ──────────────────────────────────────────────────────────

    def to_response(self, conn: AIConnection) -> AIConnectionResponse:
        return AIConnectionResponse(
            id=conn.id,
            name=conn.name,
            engine=conn.engine,
            url=conn.url,
            has_api_key=bool(conn.api_key),
            prefix=conn.prefix,
            enabled=conn.enabled,
            context_length=conn.context_length,
            timeout_ms=conn.timeout_ms,
            max_retries=conn.max_retries,
            priority=conn.priority,
        )
