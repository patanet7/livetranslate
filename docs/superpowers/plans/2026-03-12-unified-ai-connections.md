# Unified AI Connections Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate three separate AI backend config systems into a single `ai_connections` DB table with shared connection pool and per-feature model preferences.

**Architecture:** New `ai_connections` PostgreSQL table + `ConnectionService` replaces `ProviderRegistry`, JSON config files, and `MeetingIntelligenceSettings` LLM fields. Features (Chat, Translation, Intelligence) select models from the shared pool via `system_config` preference rows. Dashboard gets a new Config > Connections page.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Pydantic v2, SvelteKit 5, shadcn-svelte, Playwright

**Spec:** `docs/superpowers/specs/2026-03-11-unified-ai-connections-design.md`

---

## File Structure

### New Files
| Path | Responsibility |
|------|---------------|
| `modules/orchestration-service/alembic/versions/012_ai_connections.py` | Alembic migration: create `ai_connections` table, migrate existing data |
| `modules/orchestration-service/src/database/ai_connection.py` | SQLAlchemy `AIConnection` model |
| `modules/orchestration-service/src/services/connections.py` | `ConnectionService` — CRUD, verify, aggregate, adapter cache |
| `modules/orchestration-service/src/routers/connections.py` | `/api/connections` REST endpoints |
| `modules/orchestration-service/src/models/connections.py` | Pydantic request/response models for connections API |
| `modules/dashboard-service/src/routes/(app)/config/connections/+page.svelte` | Connections manager UI page |
| `modules/dashboard-service/src/routes/(app)/config/connections/+page.server.ts` | Server-side data loading for connections page |
| `modules/dashboard-service/src/lib/api/connections.ts` | API client for connections endpoints |
| `modules/dashboard-service/tests/e2e/connections.spec.ts` | Playwright E2E tests for connections page |

### Modified Files
| Path | What Changes |
|------|-------------|
| `modules/orchestration-service/src/database/models.py` | Import new `AIConnection` model so Alembic sees it |
| `modules/orchestration-service/src/main_fastapi.py` | Register `/api/connections` router |
| `modules/orchestration-service/src/dependencies.py` | Add `get_connection_service()` factory; update `get_meeting_intelligence_service()` to use it |
| `modules/orchestration-service/src/routers/chat.py` | Replace `get_registry()` calls with `ConnectionService` |
| `modules/orchestration-service/src/models/chat.py` | Update `ChatSettingsRequest/Response` to use `active_model` instead of `provider` |
| `modules/orchestration-service/src/routers/settings/translation.py` | Remove `verify-connection` and `aggregate-models` endpoints; update GET/POST to use `system_config` |
| `modules/orchestration-service/src/routers/settings/_shared.py` | Remove `connections`, `active_model`, `fallback_model` from `TranslationConfig` |
| `modules/dashboard-service/src/lib/types/config.ts` | Update `TranslationConnection` engine types; add `context_length` |
| `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte` | Add "Connections" nav item under Config |
| `modules/dashboard-service/src/lib/components/ConnectionCard.svelte` | Update engine badge for new types (openai, anthropic) |
| `modules/dashboard-service/src/lib/components/ConnectionDialog.svelte` | Add engine options (openai, anthropic); add context_length field; engine-specific URL behavior |
| `modules/dashboard-service/src/routes/(app)/config/translation/+page.svelte` | Remove connections section; add link to Config > Connections; model selector from shared pool |
| `modules/dashboard-service/src/routes/(app)/config/translation/+page.server.ts` | Remove `fullConfig` fetch; load models from `/api/connections/aggregate-models` |
| `modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte` | Replace provider/base_url/api_key with model selector from shared pool |

### Removed Files/Code
| What | Why |
|------|-----|
| `ProviderRegistry` class in `services/llm/registry.py` | Replaced by `ConnectionService` |
| `PROVIDER_FACTORIES` dict in `services/llm/registry.py` | Adapter construction moves into `ConnectionService` |
| `TranslationConfig.connections[]` field | Migrated to `ai_connections` table |
| `verify-connection` and `aggregate-models` endpoints in `settings/translation.py` | Moved to `/api/connections` router |
| `config/translation.json` file dependency | Settings move to `system_config` rows |

---

## Chunk 1: Database Layer

### Task 1: Create AIConnection SQLAlchemy Model

**Files:**
- Create: `modules/orchestration-service/src/database/ai_connection.py`
- Modify: `modules/orchestration-service/src/database/models.py`

- [ ] **Step 1: Create the AIConnection model file**

```python
# modules/orchestration-service/src/database/ai_connection.py
"""SQLAlchemy model for unified AI connections."""

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, Integer, Text, func

from .models import Base


class AIConnection(Base):
    """A configured AI backend connection (Ollama, OpenAI, Anthropic, etc.)."""

    __tablename__ = "ai_connections"
    __table_args__ = (
        CheckConstraint(
            "engine IN ('ollama', 'openai', 'anthropic', 'openai_compatible')",
            name="ck_ai_connections_engine",
        ),
    )

    id = Column(Text, primary_key=True)
    name = Column(Text, nullable=False)
    engine = Column(Text, nullable=False)
    url = Column(Text, nullable=False)
    api_key = Column(Text, nullable=False, server_default="")
    prefix = Column(Text, nullable=False, server_default="")
    enabled = Column(Boolean, nullable=False, server_default="true")
    context_length = Column(Integer, nullable=True)
    timeout_ms = Column(Integer, nullable=False, server_default="30000")
    max_retries = Column(Integer, nullable=False, server_default="3")
    priority = Column(Integer, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
```

- [ ] **Step 2: Import the model in models.py so Alembic discovers it**

At the end of `modules/orchestration-service/src/database/models.py`, add:

```python
# Unified AI Connections
from database.ai_connection import AIConnection  # noqa: F401, E402
```

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/database/ai_connection.py modules/orchestration-service/src/database/models.py
git commit -m "feat: add AIConnection SQLAlchemy model"
```

### Task 2: Create Alembic Migration

**Files:**
- Create: `modules/orchestration-service/alembic/versions/012_ai_connections.py`

**Before writing:** Run `grep -n "^revision\|^down_revision" modules/orchestration-service/alembic/versions/*.py` to verify the current chain head revision string. The `down_revision` below assumes `"011_chat_tables"` but must match the actual head.

- [ ] **Step 1: Check migration chain**

Run: `grep -n "^revision\|^down_revision" modules/orchestration-service/alembic/versions/*.py | tail -6`

Verify the current head revision string.

- [ ] **Step 2: Create migration file**

```python
# modules/orchestration-service/alembic/versions/012_ai_connections.py
"""Add ai_connections table and migrate connection data.

Revision ID: 012_ai_connections
Revises: <HEAD_REVISION_FROM_STEP_1>
Create Date: 2026-03-12
"""

import json
from pathlib import Path

from alembic import op
import sqlalchemy as sa

revision = "012_ai_connections"
down_revision = "<HEAD_REVISION_FROM_STEP_1>"  # REPLACE with actual head
branch_labels = None
depends_on = None

# Path to old translation config (relative to where alembic runs)
_TRANSLATION_CONFIG = Path("config/translation.json")


def upgrade():
    # 1. Create ai_connections table
    op.create_table(
        "ai_connections",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column(
            "engine",
            sa.Text,
            sa.CheckConstraint(
                "engine IN ('ollama', 'openai', 'anthropic', 'openai_compatible')",
                name="ck_ai_connections_engine",
            ),
            nullable=False,
        ),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("api_key", sa.Text, nullable=False, server_default=""),
        sa.Column("prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("context_length", sa.Integer, nullable=True),
        sa.Column("timeout_ms", sa.Integer, nullable=False, server_default="30000"),
        sa.Column("max_retries", sa.Integer, nullable=False, server_default="3"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # 2. Migrate existing data (best-effort, inside the same transaction)
    conn = op.get_bind()
    migrated_ids = set()

    # 2a. Migrate from system_config.chat_settings
    row = conn.execute(
        sa.text("SELECT value FROM system_config WHERE key = 'chat_settings'")
    ).fetchone()
    if row:
        try:
            chat_data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            provider = chat_data.get("provider", "ollama")
            base_url = chat_data.get("base_url", "http://localhost:11434")
            api_key = chat_data.get("api_key", "")

            conn_id = f"chat-{provider}"
            engine = provider if provider in ("ollama", "openai", "anthropic") else "openai_compatible"

            # Set URL based on provider
            if provider == "openai":
                url = "https://api.openai.com"
            elif provider == "anthropic":
                url = "https://api.anthropic.com"
            else:
                url = base_url or "http://localhost:11434"

            conn.execute(
                sa.text(
                    "INSERT INTO ai_connections (id, name, engine, url, api_key, prefix, enabled) "
                    "VALUES (:id, :name, :engine, :url, :api_key, :prefix, true) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                {
                    "id": conn_id,
                    "name": f"Chat {provider.title()}",
                    "engine": engine,
                    "url": url,
                    "api_key": api_key or "",
                    "prefix": provider,
                },
            )
            migrated_ids.add(conn_id)

            # Write chat model preference
            model = chat_data.get("model", "")
            pref = json.dumps({
                "active_model": f"{provider}/{model}" if model else "",
                "temperature": chat_data.get("temperature", 0.7),
                "max_tokens": chat_data.get("max_tokens", 4096),
            })
            conn.execute(
                sa.text(
                    "INSERT INTO system_config (key, value) VALUES ('chat_model_preference', :val) "
                    "ON CONFLICT (key) DO UPDATE SET value = :val"
                ),
                {"val": pref},
            )
        except Exception:
            pass  # Best-effort migration

    # 2b. Migrate from config/translation.json
    if _TRANSLATION_CONFIG.exists():
        try:
            translation_data = json.loads(_TRANSLATION_CONFIG.read_text())
            connections = translation_data.get("connections", [])
            for i, tc in enumerate(connections):
                tc_id = tc.get("id", f"translation-{i}")
                engine = tc.get("engine", "openai_compatible")
                # Map vllm/triton to openai_compatible
                if engine in ("vllm", "triton"):
                    engine = "openai_compatible"

                if tc_id not in migrated_ids:
                    conn.execute(
                        sa.text(
                            "INSERT INTO ai_connections "
                            "(id, name, engine, url, api_key, prefix, enabled, timeout_ms, max_retries) "
                            "VALUES (:id, :name, :engine, :url, :api_key, :prefix, :enabled, :timeout_ms, :max_retries) "
                            "ON CONFLICT (id) DO NOTHING"
                        ),
                        {
                            "id": tc_id,
                            "name": tc.get("name", f"Translation {i}"),
                            "engine": engine,
                            "url": tc.get("url", "http://localhost:5003"),
                            "api_key": tc.get("api_key", ""),
                            "prefix": tc.get("prefix", ""),
                            "enabled": tc.get("enabled", True),
                            "timeout_ms": tc.get("timeout_ms", 30000),
                            "max_retries": tc.get("max_retries", 3),
                        },
                    )
                    migrated_ids.add(tc_id)

            # Write translation model preference
            active_model = translation_data.get("active_model", "")
            fallback_model = translation_data.get("fallback_model", "")
            pref = json.dumps({
                "active_model": active_model,
                "fallback_model": fallback_model,
                "temperature": translation_data.get("quality", {}).get("temperature", 0.3),
                "max_tokens": translation_data.get("quality", {}).get("max_tokens", 512),
            })
            conn.execute(
                sa.text(
                    "INSERT INTO system_config (key, value) VALUES ('translation_model_preference', :val) "
                    "ON CONFLICT (key) DO UPDATE SET value = :val"
                ),
                {"val": pref},
            )

            # Migrate translation sub-configs to system_config rows
            for sub_key in ("languages", "quality", "service", "caching", "realtime"):
                sub_val = translation_data.get(sub_key)
                if sub_val:
                    config_key = f"translation_{sub_key}"
                    conn.execute(
                        sa.text(
                            "INSERT INTO system_config (key, value) VALUES (:key, :val) "
                            "ON CONFLICT (key) DO UPDATE SET value = :val"
                        ),
                        {"key": config_key, "val": json.dumps(sub_val)},
                    )
        except Exception:
            pass  # Best-effort migration

    # 2c. Write intelligence model preference (defaults)
    intel_pref = json.dumps({
        "active_model": "",
        "temperature": 0.3,
        "max_tokens": 1024,
    })
    conn.execute(
        sa.text(
            "INSERT INTO system_config (key, value) VALUES ('intelligence_model_preference', :val) "
            "ON CONFLICT (key) DO NOTHING"
        ),
        {"val": intel_pref},
    )

    # 2d. Seed default if no connections migrated
    if not migrated_ids:
        conn.execute(
            sa.text(
                "INSERT INTO ai_connections (id, name, engine, url, prefix, enabled) "
                "VALUES ('local', 'Local Ollama', 'ollama', 'http://localhost:11434', 'local', true) "
                "ON CONFLICT (id) DO NOTHING"
            )
        )


def downgrade():
    op.drop_table("ai_connections")
    conn = op.get_bind()
    for key in (
        "chat_model_preference",
        "translation_model_preference",
        "intelligence_model_preference",
        "translation_languages",
        "translation_quality",
        "translation_service",
        "translation_caching",
        "translation_realtime",
    ):
        conn.execute(sa.text("DELETE FROM system_config WHERE key = :key"), {"key": key})
```

- [ ] **Step 3: Run migration**

Run: `cd modules/orchestration-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic upgrade head`

Expected: Migration applies cleanly, `ai_connections` table created.

- [ ] **Step 4: Verify migration**

Run: `cd modules/orchestration-service && DATABASE_URL=postgresql://postgres:postgres@localhost:5432/livetranslate uv run alembic current`

Expected: Shows `012_ai_connections (head)`

- [ ] **Step 5: Commit**

```bash
git add modules/orchestration-service/alembic/versions/012_ai_connections.py
git commit -m "feat: add ai_connections migration with data migration from existing configs"
```

---

## Chunk 2: ConnectionService Backend

### Task 3: Create Pydantic Models for Connections API

**Files:**
- Create: `modules/orchestration-service/src/models/connections.py`

- [ ] **Step 1: Create models file**

```python
# modules/orchestration-service/src/models/connections.py
"""Pydantic models for the unified AI connections API."""

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")
_VALID_ENGINES = {"ollama", "openai", "anthropic", "openai_compatible"}


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug[:64]


class AIConnectionCreate(BaseModel):
    id: str | None = None  # Auto-generated from name if omitted
    name: str = Field(min_length=1, max_length=200)
    engine: str
    url: str
    api_key: str = ""
    prefix: str = ""
    enabled: bool = True
    context_length: int | None = None
    timeout_ms: int = Field(default=30000, ge=1000, le=300000)
    max_retries: int = Field(default=3, ge=0, le=10)
    priority: int = Field(default=0, ge=0)

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str) -> str:
        if v not in _VALID_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_ENGINES}")
        return v

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str | None) -> str | None:
        if v is not None and not _SLUG_RE.match(v):
            raise ValueError("id must be lowercase alphanumeric + hyphens, max 64 chars")
        return v


class AIConnectionUpdate(BaseModel):
    name: str | None = None
    engine: str | None = None
    url: str | None = None
    api_key: str | None = None
    prefix: str | None = None
    enabled: bool | None = None
    context_length: int | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    priority: int | None = None

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_ENGINES:
            raise ValueError(f"engine must be one of {_VALID_ENGINES}")
        return v


class AIConnectionResponse(BaseModel):
    id: str
    name: str
    engine: str
    url: str
    has_api_key: bool  # Never expose the actual key
    prefix: str
    enabled: bool
    context_length: int | None
    timeout_ms: int
    max_retries: int
    priority: int

    model_config = {"from_attributes": True}


class VerifyResult(BaseModel):
    status: str  # "connected" | "error"
    message: str
    version: str | None = None
    models: list[str] = Field(default_factory=list)
    latency_ms: float | None = None


class AggregatedModel(BaseModel):
    id: str  # "prefix/model_name"
    name: str
    connection_id: str
    connection_name: str
    prefix: str
    engine: str


class AggregateModelsResponse(BaseModel):
    models: list[AggregatedModel] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)


class FeaturePreference(BaseModel):
    active_model: str = ""
    fallback_model: str = ""
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/models/connections.py
git commit -m "feat: add Pydantic models for connections API"
```

### Task 4: Create ConnectionService

**Files:**
- Create: `modules/orchestration-service/src/services/connections.py`

**Key dependencies:** `database/ai_connection.py`, `models/connections.py`, `services/llm/adapter.py`, provider classes

- [ ] **Step 1: Create the service**

```python
# modules/orchestration-service/src/services/connections.py
"""Unified AI connection management service.

Replaces ProviderRegistry and JSON config file with DB-backed connections.
All features (Chat, Translation, Intelligence) share this pool.
"""

import ipaddress
import time
from datetime import datetime, UTC
from urllib.parse import urlparse

import aiohttp
from fastapi import HTTPException
from livetranslate_common.logging import get_logger
from sqlalchemy import select, update, delete
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
        # Anthropic doesn't have a list models endpoint; just verify API key works
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
```

- [ ] **Step 2: Commit**

```bash
git add modules/orchestration-service/src/services/connections.py
git commit -m "feat: add ConnectionService with CRUD, verify, aggregate, adapter cache"
```

### Task 5: Create Connections Router

**Files:**
- Create: `modules/orchestration-service/src/routers/connections.py`
- Modify: `modules/orchestration-service/src/main_fastapi.py` (add router)

- [ ] **Step 1: Create the router**

```python
# modules/orchestration-service/src/routers/connections.py
"""REST API for unified AI connections management."""

import json

from fastapi import APIRouter, Depends, HTTPException
from livetranslate_common.logging import get_logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db_session
from database.models import SystemConfig
from models.connections import (
    AIConnectionCreate,
    AIConnectionResponse,
    AIConnectionUpdate,
    AggregateModelsResponse,
    FeaturePreference,
    VerifyResult,
)
from services.connections import ConnectionService

logger = get_logger()
router = APIRouter(prefix="/api/connections", tags=["connections"])

_VALID_FEATURES = {"chat", "translation", "intelligence"}


def get_connection_service(db: AsyncSession = Depends(get_db_session)) -> ConnectionService:
    return ConnectionService(db)


# ── Fixed-path routes MUST come before {connection_id} to avoid path capture ──

@router.post("/aggregate-models", response_model=AggregateModelsResponse)
async def aggregate_models(
    svc: ConnectionService = Depends(get_connection_service),
):
    return await svc.aggregate_models()


@router.get("/preferences/all")
async def get_all_preferences(
    db: AsyncSession = Depends(get_db_session),
) -> dict[str, FeaturePreference]:
    result = {}
    for feature in _VALID_FEATURES:
        key = f"{feature}_model_preference"
        row = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
        config = row.scalar_one_or_none()
        if config and config.value:
            data = json.loads(config.value) if isinstance(config.value, str) else config.value
            result[feature] = FeaturePreference(**data)
        else:
            result[feature] = FeaturePreference()
    return result


@router.put("/preferences/{feature}", response_model=FeaturePreference)
async def update_feature_preference(
    feature: str,
    pref: FeaturePreference,
    db: AsyncSession = Depends(get_db_session),
):
    if feature not in _VALID_FEATURES:
        raise HTTPException(status_code=400, detail=f"feature must be one of {_VALID_FEATURES}")
    key = f"{feature}_model_preference"
    val = json.dumps(pref.model_dump())
    row = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
    config = row.scalar_one_or_none()
    if config:
        config.value = val
    else:
        db.add(SystemConfig(key=key, value=val))
    await db.commit()
    return pref


# ── Collection routes ──

@router.get("", response_model=list[AIConnectionResponse])
async def list_connections(
    enabled_only: bool = False,
    svc: ConnectionService = Depends(get_connection_service),
):
    connections = await svc.list_connections(enabled_only=enabled_only)
    return [svc.to_response(c) for c in connections]


@router.post("", response_model=AIConnectionResponse, status_code=201)
async def create_connection(
    data: AIConnectionCreate,
    svc: ConnectionService = Depends(get_connection_service),
):
    conn = await svc.create_connection(data)
    return svc.to_response(conn)


# ── Parameterized routes ──

@router.get("/{connection_id}", response_model=AIConnectionResponse)
async def get_connection(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
):
    conn = await svc.get_connection(connection_id)
    return svc.to_response(conn)


@router.put("/{connection_id}", response_model=AIConnectionResponse)
async def update_connection(
    connection_id: str,
    data: AIConnectionUpdate,
    svc: ConnectionService = Depends(get_connection_service),
):
    conn = await svc.update_connection(connection_id, data)
    return svc.to_response(conn)


@router.delete("/{connection_id}", status_code=204)
async def delete_connection(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
):
    await svc.delete_connection(connection_id)


@router.post("/{connection_id}/verify", response_model=VerifyResult)
async def verify_connection(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
):
    return await svc.verify_connection(connection_id)
```

- [ ] **Step 2: Register the router in main_fastapi.py**

In `modules/orchestration-service/src/main_fastapi.py`, add the import and include:

```python
# Near the other router imports:
from routers.connections import router as connections_router

# Near the other app.include_router calls:
app.include_router(connections_router)
```

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/connections.py modules/orchestration-service/src/main_fastapi.py
git commit -m "feat: add /api/connections REST router"
```

### ~~Task 6: Add Feature Preferences Endpoints~~

**Merged into Task 5** — preference endpoints are now defined in the connections router file (above `{connection_id}` routes to avoid path capture).

---

## Chunk 3: Feature Integration (Backend)

### Task 7: Update Chat Router to Use ConnectionService

**Files:**
- Modify: `modules/orchestration-service/src/routers/chat.py`
- Modify: `modules/orchestration-service/src/models/chat.py`

- [ ] **Step 1: Update ChatSettingsRequest/Response models**

In `modules/orchestration-service/src/models/chat.py`, replace the settings models:

```python
# Replace ChatSettingsRequest:
class ChatSettingsRequest(BaseModel):
    active_model: str = ""  # Prefixed model ID e.g. "home-gpu/qwen3.5:4b"
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=128000)

# Replace ChatSettingsResponse:
class ChatSettingsResponse(BaseModel):
    active_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
```

- [ ] **Step 2: Update chat.py imports and settings endpoints**

In `modules/orchestration-service/src/routers/chat.py`:

Replace the `from services.llm.registry import get_registry` import with:
```python
from services.connections import ConnectionService
from routers.connections import get_connection_service
```

Replace `get_chat_settings()` to read from `chat_model_preference`:
```python
@router.get("/settings")
async def get_chat_settings(
    db: AsyncSession = Depends(get_db_session),
) -> ChatSettingsResponse:
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "chat_model_preference")
    )
    row = result.scalar_one_or_none()
    if row and row.value:
        data = json.loads(row.value) if isinstance(row.value, str) else row.value
        return ChatSettingsResponse(**data)
    return ChatSettingsResponse()
```

Replace `update_chat_settings()`:
```python
@router.put("/settings")
async def update_chat_settings(
    request: ChatSettingsRequest,
    db: AsyncSession = Depends(get_db_session),
) -> ChatSettingsResponse:
    settings_data = request.model_dump()
    val = json.dumps(settings_data)
    result = await db.execute(
        select(SystemConfig).where(SystemConfig.key == "chat_model_preference")
    )
    row = result.scalar_one_or_none()
    if row:
        row.value = val
        row.updated_at = datetime.now(UTC)
    else:
        row = SystemConfig(key="chat_model_preference", value=val)
        db.add(row)
    await db.commit()
    return ChatSettingsResponse(**settings_data)
```

Replace `list_providers()` and `list_provider_models()` with connection-based equivalents:
```python
@router.get("/providers")
async def list_providers(
    svc: ConnectionService = Depends(get_connection_service),
) -> list[dict]:
    connections = await svc.list_connections()
    return [
        {"name": c.id, "engine": c.engine, "configured": True, "enabled": c.enabled}
        for c in connections
    ]

@router.get("/providers/{connection_id}/models")
async def list_provider_models(
    connection_id: str,
    svc: ConnectionService = Depends(get_connection_service),
) -> list[ModelInfoResponse]:
    result = await svc.verify_connection(connection_id)
    return [
        ModelInfoResponse(id=m, name=m, provider=connection_id)
        for m in result.models
    ]
```

Update `send_message()` and `stream_message()` to resolve model from shared pool:
```python
# In send_message and stream_message, replace:
#   registry = get_registry()
#   adapter = registry.get_adapter(provider_name)
# With:
    svc = ConnectionService(db)
    active_model = request.model or conv.model or ""
    if active_model and "/" in active_model:
        conn_obj, model_name = await svc.resolve_model(active_model)
        adapter = await svc.get_adapter(conn_obj.id)
    else:
        # Fallback: get settings preference
        pref_row = await db.execute(
            select(SystemConfig).where(SystemConfig.key == "chat_model_preference")
        )
        pref = pref_row.scalar_one_or_none()
        if pref and pref.value:
            pref_data = json.loads(pref.value)
            active_model = pref_data.get("active_model", "")
        if active_model and "/" in active_model:
            conn_obj, model_name = await svc.resolve_model(active_model)
            adapter = await svc.get_adapter(conn_obj.id)
        else:
            raise HTTPException(status_code=400, detail="No model configured. Set an active model in Chat settings.")
```

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/chat.py modules/orchestration-service/src/models/chat.py
git commit -m "refactor: chat router uses ConnectionService instead of ProviderRegistry"
```

### Task 8: Update Intelligence Service to Use ConnectionService

**Files:**
- Create: `modules/orchestration-service/src/dependencies_connections.py` (async helper)
- Modify: `modules/orchestration-service/src/routers/insights.py` (inject per-request LLM client)

The intelligence service uses `LLMClient` through `MeetingIntelligenceService.translation_client`. Instead of trying to resolve DB config in the sync `@lru_cache` singleton (which can't do async DB reads and would cache stale config), we create a per-request async dependency that resolves the connection from the DB at request time.

- [ ] **Step 1: Create async connection resolver for intelligence**

```python
# modules/orchestration-service/src/dependencies_connections.py
"""Async dependency helpers for resolving LLM clients from ai_connections."""

import json

from fastapi import Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from clients.llm_client import create_llm_client
from clients.protocol import LLMClientProtocol
from database import get_db_session
from livetranslate_common.logging import get_logger

logger = get_logger()


async def resolve_intelligence_llm_client(
    db: AsyncSession = Depends(get_db_session),
) -> LLMClientProtocol | None:
    """Resolve the intelligence LLM client from ai_connections at request time.

    Returns None if no preference is configured, allowing fallback behavior.
    """
    try:
        result = await db.execute(
            text("SELECT value FROM system_config WHERE key = 'intelligence_model_preference'")
        )
        row = result.fetchone()
        if not row or not row[0]:
            return None

        pref = json.loads(row[0])
        active_model = pref.get("active_model", "")
        if not active_model or "/" not in active_model:
            return None

        prefix, model_name = active_model.split("/", 1)
        conn_result = await db.execute(
            text(
                "SELECT url, api_key, engine FROM ai_connections "
                "WHERE prefix = :prefix AND enabled = true"
            ),
            {"prefix": prefix},
        )
        conn_row = conn_result.fetchone()
        if not conn_row:
            return None

        base_url = conn_row[0].rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        return create_llm_client(
            base_url=base_url,
            api_key=conn_row[1] or "",
            model=model_name,
            max_tokens=pref.get("max_tokens", 1024),
            temperature=pref.get("temperature", 0.3),
            proxy_mode=False,
        )
    except Exception as e:
        logger.warning("intelligence_llm_resolve_failed", error=str(e))
        return None
```

- [ ] **Step 2: Update insights router to inject per-request LLM client**

In `modules/orchestration-service/src/routers/insights.py`, update the endpoints that use the intelligence service to optionally override the translation_client with the DB-resolved one. Add to the dependency:

```python
from dependencies_connections import resolve_intelligence_llm_client

# For endpoints that call LLM (analyze_note, generate_insights, agent messages):
# Add the resolved client as a dependency parameter, then override on the service:
async def analyze_note(
    session_id: str,
    request: NoteAnalyzeRequest,
    service=Depends(get_intelligence_service),
    llm_client=Depends(resolve_intelligence_llm_client),
):
    if llm_client is not None:
        service.translation_client = llm_client
    # ... rest of handler unchanged
```

Apply the same pattern to `generate_insights`, `agent_message`, and `agent_message_stream` endpoints.

The existing `get_meeting_intelligence_service()` singleton in `dependencies.py` remains as-is for backwards compatibility — the per-request `resolve_intelligence_llm_client` just overrides the client when DB config is available, making the system pick up config changes without restart.

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/dependencies_connections.py modules/orchestration-service/src/routers/insights.py
git commit -m "refactor: intelligence resolves LLM client per-request from ai_connections"
```

### Task 9: Update Translation Settings Router

**Files:**
- Modify: `modules/orchestration-service/src/routers/settings/translation.py`
- Modify: `modules/orchestration-service/src/routers/settings/_shared.py`

- [ ] **Step 1: Remove connections fields from TranslationConfig**

In `modules/orchestration-service/src/routers/settings/_shared.py`, remove these fields from the `TranslationConfig` class:
- `connections`
- `active_model`
- `fallback_model`

These are now managed via `ai_connections` table and `system_config` preference rows.

- [ ] **Step 2: Remove verify-connection and aggregate-models endpoints from translation.py**

In `modules/orchestration-service/src/routers/settings/translation.py`, remove:
- The `verify_translation_connection()` endpoint and its `VerifyConnectionRequest` model
- The `aggregate_translation_models()` endpoint
- The `_validate_connection_url()` function (now lives in `services/connections.py`)

Keep the GET/POST `/translation` endpoints but update them to read/write `system_config` rows instead of JSON file. Replace `load_config(TRANSLATION_CONFIG_FILE, ...)` with DB reads:

```python
from database.models import SystemConfig
from sqlalchemy import select

@router.get("/translation")
async def get_translation_config(
    db: AsyncSession = Depends(get_db_session),
):
    """Get translation-specific settings from system_config."""
    config = {}
    for sub_key in ("languages", "quality", "service", "model", "caching", "realtime"):
        key = f"translation_{sub_key}"
        result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
        row = result.scalar_one_or_none()
        if row and row.value:
            config[sub_key] = json.loads(row.value) if isinstance(row.value, str) else row.value
        else:
            config[sub_key] = TranslationConfig().model_dump().get(sub_key, {})
    return config

@router.post("/translation")
async def save_translation_config(
    config: dict,
    db: AsyncSession = Depends(get_db_session),
):
    """Save translation-specific settings to system_config."""
    for sub_key in ("languages", "quality", "service", "model", "caching", "realtime"):
        if sub_key in config:
            key = f"translation_{sub_key}"
            val = json.dumps(config[sub_key])
            result = await db.execute(select(SystemConfig).where(SystemConfig.key == key))
            row = result.scalar_one_or_none()
            if row:
                row.value = val
            else:
                db.add(SystemConfig(key=key, value=val))
    await db.commit()
    return {"message": "Translation settings saved", "config": config}
```

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/settings/translation.py modules/orchestration-service/src/routers/settings/_shared.py
git commit -m "refactor: translation settings use system_config DB, remove connection endpoints"
```

---

## Chunk 4: Dashboard Frontend

### Task 10: Create Connections API Client

**Files:**
- Create: `modules/dashboard-service/src/lib/api/connections.ts`

- [ ] **Step 1: Create the API client**

```typescript
// modules/dashboard-service/src/lib/api/connections.ts
//
// NOTE: This uses createApi() which imports $env/static/private, so it can ONLY
// be used in +page.server.ts files, NOT in client-side +page.svelte.
// The +page.svelte uses direct fetch() calls to avoid this boundary.
//
import { createApi } from './orchestration';

export interface AIConnection {
    id: string;
    name: string;
    engine: 'ollama' | 'openai' | 'anthropic' | 'openai_compatible';
    url: string;
    has_api_key: boolean;
    prefix: string;
    enabled: boolean;
    context_length: number | null;
    timeout_ms: number;
    max_retries: number;
    priority: number;
}

export interface AggregatedModel {
    id: string;
    name: string;
    connection_id: string;
    connection_name: string;
    prefix: string;
    engine: string;
}

export interface AggregateModelsResponse {
    models: AggregatedModel[];
    errors: Array<{ connection_id: string; connection_name: string; message: string }>;
}

export interface FeaturePreference {
    active_model: string;
    fallback_model: string;
    temperature: number;
    max_tokens: number;
}

export function connectionsApi(fetch: typeof globalThis.fetch) {
    const api = createApi(fetch);
    return {
        list: (enabledOnly = false) =>
            api.get<AIConnection[]>(`/api/connections?enabled_only=${enabledOnly}`),
        aggregateModels: () =>
            api.post<AggregateModelsResponse>('/api/connections/aggregate-models'),
        getPreferences: () =>
            api.get<Record<string, FeaturePreference>>('/api/connections/preferences/all'),
    };
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/api/connections.ts
git commit -m "feat: add connections API client for dashboard"
```

### Task 11: Create Connections Config Page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/config/connections/+page.server.ts`
- Create: `modules/dashboard-service/src/routes/(app)/config/connections/+page.svelte`
- Modify: `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte`

- [ ] **Step 1: Create page server load**

```typescript
// modules/dashboard-service/src/routes/(app)/config/connections/+page.server.ts
import type { PageServerLoad } from './$types';
import { ORCHESTRATION_URL } from '$env/static/private';

export const load: PageServerLoad = async ({ fetch }) => {
    const base = ORCHESTRATION_URL || 'http://localhost:3000';

    const [connectionsRes, preferencesRes] = await Promise.all([
        fetch(`${base}/api/connections`).catch(() => null),
        fetch(`${base}/api/connections/preferences/all`).catch(() => null),
    ]);

    const connections = connectionsRes?.ok ? await connectionsRes.json() : [];
    const preferences = preferencesRes?.ok ? await preferencesRes.json() : {};

    return { connections, preferences };
};
```

- [ ] **Step 2: Create the connections page**

The page reuses the existing `ConnectionCard.svelte` and `ConnectionDialog.svelte` components (which will be updated in Task 12 for new engine types). The page uses direct `fetch()` calls to avoid the `$env/static/private` import boundary issue.

Create `modules/dashboard-service/src/routes/(app)/config/connections/+page.svelte` with:
- Connections Manager section (cards + add button)
- Delete confirmation dialog
- Aggregated models summary
- Feature preferences section (3 model selectors for chat/translation/intelligence)

The page structure follows the same pattern as the current translation config page's connections section, but promoted to its own page with the preferences section added.

**Key differences from the translation page version:**
- Engine options include `openai` and `anthropic` (not just `ollama`, `vllm`, `triton`, `openai_compatible`)
- Has a "Feature Preferences" card with 3 dropdowns (chat, translation, intelligence)
- No translation-specific settings (those stay on the translation config page)

```svelte
<!-- modules/dashboard-service/src/routes/(app)/config/connections/+page.svelte -->
<script lang="ts">
    import { onMount } from 'svelte';
    import PageHeader from '$lib/components/layout/PageHeader.svelte';
    import * as Card from '$lib/components/ui/card';
    import * as Dialog from '$lib/components/ui/dialog';
    import { Button } from '$lib/components/ui/button';
    import { Label } from '$lib/components/ui/label';
    import { Badge } from '$lib/components/ui/badge';
    import { toastStore } from '$lib/stores/toast.svelte';
    import ConnectionCard from '$lib/components/ConnectionCard.svelte';
    import ConnectionDialog from '$lib/components/ConnectionDialog.svelte';
    import PlusIcon from '@lucide/svelte/icons/plus';
    import type { AIConnection, AggregatedModel, FeaturePreference } from '$lib/api/connections';

    let { data } = $props();

    // ── Connections State ──────────────────────────────────────────────
    let connections: AIConnection[] = $state(data.connections ?? []);
    let connectionStatuses: Record<string, 'unknown' | 'connected' | 'error' | 'verifying'> = $state({});
    let connectionModelCounts: Record<string, number> = $state({});
    let aggregatedModels: AggregatedModel[] = $state([]);
    let dialogOpen = $state(false);
    let editingConnection: AIConnection | null = $state(null);
    let pendingDeleteId: string | null = $state(null);

    // ── Feature Preferences State ──────────────────────────────────────
    let preferences: Record<string, FeaturePreference> = $state(data.preferences ?? {});

    // ── Connection CRUD (direct fetch to avoid $env boundary) ──────────

    async function reloadConnections() {
        try {
            const res = await fetch('/api/connections');
            if (res.ok) connections = await res.json();
        } catch { /* ignore */ }
    }

    async function createConnection(conn: Record<string, unknown>) {
        try {
            const res = await fetch('/api/connections', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(conn),
            });
            if (res.ok) {
                toastStore.success('Connection created');
                await reloadConnections();
            } else {
                const err = await res.json().catch(() => ({}));
                toastStore.error(err.detail || 'Failed to create connection');
            }
        } catch {
            toastStore.error('Failed to create connection');
        }
    }

    async function updateConnection(id: string, data: Record<string, unknown>) {
        try {
            const res = await fetch(`/api/connections/${id}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (res.ok) {
                toastStore.success('Connection updated');
                await reloadConnections();
            }
        } catch {
            toastStore.error('Failed to update connection');
        }
    }

    async function deleteConnection(id: string) {
        pendingDeleteId = id;
    }

    async function confirmDelete() {
        if (!pendingDeleteId) return;
        try {
            await fetch(`/api/connections/${pendingDeleteId}`, { method: 'DELETE' });
            toastStore.success('Connection deleted');
            pendingDeleteId = null;
            await reloadConnections();
        } catch {
            toastStore.error('Failed to delete connection');
        }
    }

    function cancelDelete() {
        pendingDeleteId = null;
    }

    async function toggleConnection(id: string, enabled: boolean) {
        await updateConnection(id, { enabled });
    }

    async function verifyConnection(conn: AIConnection) {
        connectionStatuses[conn.id] = 'verifying';
        try {
            const res = await fetch(`/api/connections/${conn.id}/verify`, { method: 'POST' });
            const result = await res.json();
            if (result.status === 'connected') {
                connectionStatuses[conn.id] = 'connected';
                connectionModelCounts[conn.id] = result.models?.length ?? 0;
                toastStore.success(`${conn.name}: Connected (${result.models?.length ?? 0} models)`);
            } else {
                connectionStatuses[conn.id] = 'error';
                toastStore.error(`${conn.name}: ${result.message}`);
            }
        } catch {
            connectionStatuses[conn.id] = 'error';
            toastStore.error(`${conn.name}: Connection failed`);
        }
    }

    async function loadAggregatedModels() {
        try {
            const res = await fetch('/api/connections/aggregate-models', { method: 'POST' });
            const result = await res.json();
            aggregatedModels = result.models ?? [];
        } catch { /* ignore */ }
    }

    // ── Dialog handlers ────────────────────────────────────────────────

    function openAddDialog() {
        editingConnection = null;
        dialogOpen = true;
    }

    function openEditDialog(conn: AIConnection) {
        // ConnectionDialog expects api_key, but API returns has_api_key
        editingConnection = { ...conn, api_key: '' } as any;
        dialogOpen = true;
    }

    function handleSaveConnection(conn: Record<string, unknown>) {
        if (editingConnection) {
            const updates = { ...conn };
            delete updates.id;
            // Only send api_key if user provided one
            if (!updates.api_key) delete updates.api_key;
            updateConnection(editingConnection.id, updates);
        } else {
            createConnection(conn);
        }
    }

    // ── Feature Preference Save ────────────────────────────────────────

    async function savePreference(feature: string) {
        const pref = preferences[feature];
        if (!pref) return;
        try {
            await fetch(`/api/connections/preferences/${feature}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pref),
            });
            toastStore.success(`${feature} preference saved`);
        } catch {
            toastStore.error(`Failed to save ${feature} preference`);
        }
    }

    // ── Mount ──────────────────────────────────────────────────────────

    onMount(() => {
        for (const conn of connections) {
            if (conn.enabled) verifyConnection(conn);
        }
        loadAggregatedModels();
    });
</script>

<PageHeader
    title="AI Connections"
    description="Manage AI backend connections shared across Chat, Translation, and Intelligence"
/>

<div class="max-w-3xl space-y-6">
    <!-- Connections Manager -->
    <Card.Root>
        <Card.Header>
            <Card.Title>Connections</Card.Title>
            <Card.Action>
                <Button variant="outline" size="sm" onclick={openAddDialog}>
                    <PlusIcon class="mr-1 h-4 w-4" />
                    Add Connection
                </Button>
            </Card.Action>
        </Card.Header>
        <Card.Content>
            <div class="space-y-3">
                {#if connections.length === 0}
                    <p class="text-sm text-muted-foreground">
                        No connections configured. Add an AI backend to get started.
                    </p>
                {:else}
                    {#each connections as conn (conn.id)}
                        <ConnectionCard
                            connection={conn}
                            status={connectionStatuses[conn.id] ?? 'unknown'}
                            modelCount={connectionModelCounts[conn.id] ?? 0}
                            onverify={() => verifyConnection(conn)}
                            onconfigure={() => openEditDialog(conn)}
                            ondelete={() => deleteConnection(conn.id)}
                            ontoggle={(enabled: boolean) => toggleConnection(conn.id, enabled)}
                        />
                    {/each}
                {/if}
            </div>

            {#if aggregatedModels.length > 0}
                <div class="mt-4 rounded-md border bg-muted/50 p-3">
                    <p class="mb-2 text-xs font-medium text-muted-foreground">
                        Aggregated Models ({aggregatedModels.length})
                    </p>
                    <div class="flex flex-wrap gap-1.5">
                        {#each aggregatedModels as model}
                            <Badge variant="secondary" class="text-xs">{model.id}</Badge>
                        {/each}
                    </div>
                </div>
            {/if}
        </Card.Content>
    </Card.Root>

    <!-- Connection Dialog -->
    <ConnectionDialog
        bind:open={dialogOpen}
        connection={editingConnection}
        onsave={handleSaveConnection}
        onclose={() => { dialogOpen = false; }}
    />

    <!-- Delete Confirmation Dialog -->
    <Dialog.Root
        open={pendingDeleteId !== null}
        onOpenChange={(open) => { if (!open) cancelDelete(); }}
    >
        <Dialog.Content class="sm:max-w-md">
            <Dialog.Header>
                <Dialog.Title>Delete Connection</Dialog.Title>
                <Dialog.Description>
                    Are you sure you want to remove this connection? This action cannot be undone.
                </Dialog.Description>
            </Dialog.Header>
            <Dialog.Footer>
                <Button variant="outline" onclick={cancelDelete}>Cancel</Button>
                <Button variant="destructive" onclick={confirmDelete}>Delete</Button>
            </Dialog.Footer>
        </Dialog.Content>
    </Dialog.Root>

    <!-- Feature Preferences -->
    <Card.Root>
        <Card.Header>
            <Card.Title>Feature Model Preferences</Card.Title>
            <Card.Description>Select which model each feature uses from the shared pool</Card.Description>
        </Card.Header>
        <Card.Content>
            <div class="space-y-4">
                {#each ['chat', 'translation', 'intelligence'] as feature}
                    {@const pref = preferences[feature] ?? { active_model: '', fallback_model: '', temperature: 0.7, max_tokens: 4096 }}
                    <div class="space-y-2 rounded-md border p-3">
                        <Label class="text-sm font-medium capitalize">{feature}</Label>
                        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <div class="space-y-1">
                                <Label class="text-xs text-muted-foreground">Active Model</Label>
                                <select
                                    class="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                                    value={pref.active_model}
                                    onchange={(e) => {
                                        if (!preferences[feature]) {
                                            preferences[feature] = { active_model: '', fallback_model: '', temperature: 0.7, max_tokens: 4096 };
                                        }
                                        preferences[feature].active_model = (e.target as HTMLSelectElement).value;
                                    }}
                                >
                                    <option value="">None</option>
                                    {#each aggregatedModels as model}
                                        <option value={model.id}>{model.id} ({model.engine})</option>
                                    {/each}
                                </select>
                            </div>
                            <div class="space-y-1">
                                <Label class="text-xs text-muted-foreground">Fallback Model</Label>
                                <select
                                    class="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
                                    value={pref.fallback_model}
                                    onchange={(e) => {
                                        if (!preferences[feature]) {
                                            preferences[feature] = { active_model: '', fallback_model: '', temperature: 0.7, max_tokens: 4096 };
                                        }
                                        preferences[feature].fallback_model = (e.target as HTMLSelectElement).value;
                                    }}
                                >
                                    <option value="">None</option>
                                    {#each aggregatedModels as model}
                                        <option value={model.id}>{model.id} ({model.engine})</option>
                                    {/each}
                                </select>
                            </div>
                        </div>
                        <Button size="sm" variant="outline" onclick={() => savePreference(feature)}>
                            Save {feature} preference
                        </Button>
                    </div>
                {/each}
            </div>
        </Card.Content>
    </Card.Root>
</div>
```

- [ ] **Step 3: Add "Connections" to sidebar nav**

In `modules/dashboard-service/src/lib/components/layout/Sidebar.svelte`, add an import and nav entry:

```typescript
// Add import:
import PlugIcon from '@lucide/svelte/icons/plug';

// Update the Config children array:
{
    label: 'Config',
    href: '/config',
    icon: SettingsIcon,
    children: [
        { label: 'Connections', href: '/config/connections' },
        { label: 'Audio', href: '/config/audio' },
        { label: 'System', href: '/config/system' },
        { label: 'Settings', href: '/config/settings' }
    ]
},
```

- [ ] **Step 4: Build to verify**

Run: `cd modules/dashboard-service && npx vite build`

Expected: Build succeeds with no errors.

- [ ] **Step 5: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/config/connections/ modules/dashboard-service/src/lib/api/connections.ts modules/dashboard-service/src/lib/components/layout/Sidebar.svelte
git commit -m "feat: add Config > Connections page with shared model preferences"
```

### Task 12: Update ConnectionCard and ConnectionDialog for New Engines

**Files:**
- Modify: `modules/dashboard-service/src/lib/components/ConnectionCard.svelte`
- Modify: `modules/dashboard-service/src/lib/components/ConnectionDialog.svelte`

- [ ] **Step 1: Update ConnectionCard engine badges**

In `ConnectionCard.svelte`, update the engine color mapping to include `openai` and `anthropic`:

```typescript
const engineColors: Record<string, string> = {
    ollama: 'bg-green-500/10 text-green-700 dark:text-green-400',
    openai: 'bg-blue-500/10 text-blue-700 dark:text-blue-400',
    anthropic: 'bg-orange-500/10 text-orange-700 dark:text-orange-400',
    openai_compatible: 'bg-purple-500/10 text-purple-700 dark:text-purple-400',
};
```

Remove `vllm` and `triton` engine colors if present.

- [ ] **Step 2: Update ConnectionDialog engine options**

In `ConnectionDialog.svelte`, update the engine select options and URL defaults:

```typescript
const engineOptions = [
    { value: 'ollama', label: 'Ollama' },
    { value: 'openai', label: 'OpenAI' },
    { value: 'anthropic', label: 'Anthropic' },
    { value: 'openai_compatible', label: 'OpenAI Compatible (vLLM, Groq, etc.)' },
];

const engineDefaults: Record<string, { url: string; urlEditable: boolean; apiKeyRequired: boolean }> = {
    ollama: { url: 'http://localhost:11434', urlEditable: true, apiKeyRequired: false },
    openai: { url: 'https://api.openai.com', urlEditable: false, apiKeyRequired: true },
    anthropic: { url: 'https://api.anthropic.com', urlEditable: false, apiKeyRequired: true },
    openai_compatible: { url: '', urlEditable: true, apiKeyRequired: false },
};
```

Add a `context_length` field to the form (in the Advanced section):

```svelte
<div class="space-y-1">
    <Label for="context-length">Context Length (optional)</Label>
    <Input
        id="context-length"
        type="number"
        min="1024"
        max="2000000"
        step="1024"
        placeholder="Use model default"
        bind:value={formContextLength}
    />
    <p class="text-xs text-muted-foreground">Override the model's default context window</p>
</div>
```

When engine is `openai` or `anthropic`, hide the URL field and set it from `engineDefaults`.

- [ ] **Step 3: Build to verify**

Run: `cd modules/dashboard-service && npx vite build`

- [ ] **Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/components/ConnectionCard.svelte modules/dashboard-service/src/lib/components/ConnectionDialog.svelte
git commit -m "feat: update connection components for openai/anthropic engines and context_length"
```

### Task 13: Update Translation Config Page

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/config/translation/+page.svelte`
- Modify: `modules/dashboard-service/src/routes/(app)/config/translation/+page.server.ts`

- [ ] **Step 1: Remove connections section from translation page**

In `+page.svelte`, remove:
- All connection-related state (`connections`, `connectionStatuses`, `connectionModelCounts`, `aggregatedModels`, `dialogOpen`, `editingConnection`, `pendingDeleteId`)
- All connection-related functions (`verifyConnection`, `loadAggregatedModels`, `openAddDialog`, `openEditDialog`, `handleSaveConnection`, `deleteConnection`, `confirmDelete`, `cancelDelete`, `toggleConnection`, `saveConnections`)
- The `onMount` that auto-verifies connections
- The Connections Manager card in the template
- The ConnectionDialog and delete confirmation Dialog
- Imports for `ConnectionCard`, `ConnectionDialog`, `PlusIcon`, `Dialog`, connection types

Add a link card pointing to Config > Connections:

```svelte
<Card.Root>
    <Card.Header>
        <Card.Title>AI Connections</Card.Title>
    </Card.Header>
    <Card.Content>
        <p class="text-sm text-muted-foreground">
            Translation models are loaded from shared AI connections.
        </p>
        <Button variant="link" class="mt-2 px-0" onclick={() => window.location.href = '/config/connections'}>
            Manage Connections →
        </Button>
    </Card.Content>
</Card.Root>
```

- [ ] **Step 2: Update page server load**

In `+page.server.ts`, remove the `fullConfig` fetch. The page no longer needs connection data.

- [ ] **Step 3: Build to verify**

Run: `cd modules/dashboard-service && npx vite build`

- [ ] **Step 4: Commit**

```bash
git add modules/dashboard-service/src/routes/\(app\)/config/translation/
git commit -m "refactor: remove connections from translation page, add link to Config > Connections"
```

### Task 14: Update Chat SettingsDrawer

**Files:**
- Modify: `modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte`
- Modify: `modules/dashboard-service/src/lib/api/chat.ts` (update `ChatSettings` type)

- [ ] **Step 1: Update ChatSettings type in chat.ts**

In `modules/dashboard-service/src/lib/api/chat.ts`, replace the `ChatSettings` interface:

```typescript
// OLD:
// export interface ChatSettings {
//     provider: string;
//     model: string | null;
//     temperature: number;
//     max_tokens: number;
//     has_api_key: boolean;
//     base_url: string | null;
// }

// NEW:
export interface ChatSettings {
    active_model: string;
    temperature: number;
    max_tokens: number;
}
```

- [ ] **Step 2: Rewrite SettingsDrawer.svelte**

Replace the full content of `modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte`:

```svelte
<script lang="ts">
    import { Button } from '$lib/components/ui/button';
    import { Input } from '$lib/components/ui/input';
    import { Label } from '$lib/components/ui/label';
    import * as Dialog from '$lib/components/ui/dialog';
    import { toastStore } from '$lib/stores/toast.svelte';
    import type { AggregatedModel } from '$lib/api/connections';

    interface Props {
        open: boolean;
        onclose: () => void;
    }

    let { open, onclose }: Props = $props();

    let models = $state<AggregatedModel[]>([]);
    let loading = $state(false);
    let saving = $state(false);

    let activeModel = $state('');
    let temperature = $state(0.7);
    let maxTokens = $state(4096);

    $effect(() => {
        if (open) loadSettings();
    });

    async function loadSettings() {
        loading = true;
        try {
            const [modelsRes, prefRes] = await Promise.all([
                fetch('/api/connections/aggregate-models', { method: 'POST' }),
                fetch('/api/connections/preferences/all'),
            ]);
            if (modelsRes.ok) {
                const data = await modelsRes.json();
                models = data.models ?? [];
            }
            if (prefRes.ok) {
                const prefs = await prefRes.json();
                const chat = prefs.chat;
                if (chat) {
                    activeModel = chat.active_model ?? '';
                    temperature = chat.temperature ?? 0.7;
                    maxTokens = chat.max_tokens ?? 4096;
                }
            }
        } catch {
            toastStore.error('Failed to load chat settings');
        } finally {
            loading = false;
        }
    }

    async function save() {
        saving = true;
        try {
            const res = await fetch('/api/connections/preferences/chat', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    active_model: activeModel,
                    fallback_model: '',
                    temperature,
                    max_tokens: maxTokens,
                }),
            });
            if (res.ok) {
                toastStore.success('Chat settings saved');
                onclose();
            } else {
                const data = await res.json().catch(() => null);
                toastStore.error(data?.detail ?? 'Failed to save settings');
            }
        } catch {
            toastStore.error('Network error saving settings');
        } finally {
            saving = false;
        }
    }
</script>

<Dialog.Root bind:open>
    <Dialog.Content class="sm:max-w-md">
        <Dialog.Header>
            <Dialog.Title>Chat Settings</Dialog.Title>
            <Dialog.Description>
                Select a model from your shared AI connections.
            </Dialog.Description>
        </Dialog.Header>

        {#if loading}
            <div class="flex items-center justify-center py-8">
                <div class="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
                <span class="ml-2 text-sm text-muted-foreground">Loading...</span>
            </div>
        {:else}
            <div class="space-y-4 py-4">
                <!-- Model from shared pool -->
                <div class="space-y-2">
                    <Label>Model</Label>
                    <select
                        class="w-full rounded-md border bg-background px-3 py-2 text-sm"
                        bind:value={activeModel}
                    >
                        <option value="">Select a model...</option>
                        {#each models as model (model.id)}
                            <option value={model.id}>
                                {model.id} ({model.engine})
                            </option>
                        {/each}
                    </select>
                    {#if models.length === 0}
                        <p class="text-xs text-muted-foreground">
                            No models available. <a href="/config/connections" class="underline">Add a connection</a> first.
                        </p>
                    {/if}
                </div>

                <!-- Temperature -->
                <div class="space-y-2">
                    <Label>Temperature: {temperature.toFixed(1)}</Label>
                    <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.1"
                        bind:value={temperature}
                        class="w-full accent-primary"
                    />
                    <div class="flex justify-between text-xs text-muted-foreground">
                        <span>Precise</span>
                        <span>Creative</span>
                    </div>
                </div>

                <!-- Max Tokens -->
                <div class="space-y-2">
                    <Label>Max Tokens</Label>
                    <Input type="number" bind:value={maxTokens} min={256} max={128000} />
                </div>
            </div>

            <Dialog.Footer>
                <Button variant="outline" onclick={onclose}>Cancel</Button>
                <Button disabled={saving} onclick={save}>
                    {#if saving}Saving...{:else}Save Settings{/if}
                </Button>
            </Dialog.Footer>
        {/if}
    </Dialog.Content>
</Dialog.Root>
```

- [ ] **Step 3: Build to verify**

Run: `cd modules/dashboard-service && npx vite build`

- [ ] **Step 4: Commit**

```bash
git add modules/dashboard-service/src/lib/components/chat/SettingsDrawer.svelte modules/dashboard-service/src/lib/api/chat.ts
git commit -m "refactor: chat settings use shared connection pool model selector"
```

---

## Chunk 5: Testing and Cleanup

### Task 15: Write E2E Tests for Connections Page

**Files:**
- Create: `modules/dashboard-service/tests/e2e/connections.spec.ts`

- [ ] **Step 1: Create E2E test file**

Adapt from existing `translation-connections.spec.ts` but targeting `/config/connections`:

```typescript
// modules/dashboard-service/tests/e2e/connections.spec.ts
import { test, expect } from '@playwright/test';

test.describe('AI Connections', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/config/connections');
        await page.waitForTimeout(2000);
    });

    test('page loads and shows connections section', async ({ page }) => {
        await expect(page.getByText('AI Connections', { exact: true })).toBeVisible();
        await expect(page.getByText('Connections', { exact: true })).toBeVisible();
    });

    test('add connection dialog opens and submits', async ({ page }) => {
        await page.getByRole('button', { name: /add connection/i }).click();
        const dialog = page.getByRole('dialog');
        await expect(dialog).toBeVisible({ timeout: 3000 });

        await page.getByLabel('Name').fill('Test Ollama');
        await page.getByLabel('URL').clear();
        await page.getByLabel('URL').fill('http://test:11434');
        await page.getByLabel('Prefix ID').fill('test');
        await dialog.getByRole('button', { name: /add connection/i }).click();
    });

    test('delete connection shows confirmation', async ({ page }) => {
        // Add a connection first
        await page.getByRole('button', { name: /add connection/i }).click();
        const dialog = page.getByRole('dialog');
        await expect(dialog).toBeVisible({ timeout: 3000 });
        await page.getByLabel('Name').fill('Delete Me');
        await page.getByLabel('URL').clear();
        await page.getByLabel('URL').fill('http://deleteme:11434');
        await page.getByLabel('Prefix ID').fill('del');
        await dialog.getByRole('button', { name: /add connection/i }).click();
        await page.waitForTimeout(1000);

        // Click delete
        const cards = page.locator('[data-testid="connection-card"]');
        const lastCard = cards.last();
        await lastCard.locator('button').filter({ has: page.locator('svg.lucide-trash-2') }).click();

        // Expect confirmation dialog
        const confirmDialog = page.getByRole('dialog');
        await expect(confirmDialog).toBeVisible({ timeout: 3000 });
        await expect(page.getByText('Delete Connection')).toBeVisible();
    });

    test('feature preferences section is visible', async ({ page }) => {
        await expect(page.getByText('Feature Model Preferences')).toBeVisible();
        await expect(page.getByText('chat', { exact: false })).toBeVisible();
        await expect(page.getByText('translation', { exact: false })).toBeVisible();
        await expect(page.getByText('intelligence', { exact: false })).toBeVisible();
    });
});
```

- [ ] **Step 2: Run E2E tests**

Run: `cd modules/dashboard-service && npx playwright test tests/e2e/connections.spec.ts`

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/tests/e2e/connections.spec.ts
git commit -m "test: add Playwright E2E tests for Config > Connections page"
```

### Task 16: Clean Up Old Code

**Files:**
- Modify: `modules/orchestration-service/src/services/llm/registry.py` — deprecate or remove
- Modify: `modules/orchestration-service/src/routers/settings/_shared.py` — remove `TRANSLATION_CONFIG_FILE` if no longer used
- Modify: `modules/dashboard-service/src/lib/types/config.ts` — remove old `TranslationConnection` type
- Modify: `modules/dashboard-service/src/lib/api/config.ts` — remove `verifyConnection`, `aggregateModels`, `getFullTranslationConfig`, `saveFullTranslationConfig`

- [ ] **Step 1: Remove ProviderRegistry usage**

In `modules/orchestration-service/src/services/llm/registry.py`, add a deprecation comment at the top:

```python
"""DEPRECATED: Use services.connections.ConnectionService instead.

This module is retained temporarily for any code that hasn't been migrated yet.
"""
```

Grep the codebase for remaining `get_registry` or `ProviderRegistry` imports and update any stragglers.

- [ ] **Step 2: Clean up frontend types and API**

In `modules/dashboard-service/src/lib/types/config.ts`, remove:
- `TranslationConnection` interface
- `VerifyConnectionRequest` interface
- `VerifyConnectionResponse` interface
- `AggregatedModel` interface
- `AggregateModelsResponse` interface
- `FullTranslationConfig` interface

These are now in `connections.ts`.

In `modules/dashboard-service/src/lib/api/config.ts`, remove:
- `verifyConnection` method
- `aggregateModels` method
- `getFullTranslationConfig` method
- `saveFullTranslationConfig` method

- [ ] **Step 3: Build to verify nothing is broken**

Run: `cd modules/dashboard-service && npx vite build`

- [ ] **Step 4: Commit**

```bash
git add modules/orchestration-service/src/services/llm/registry.py modules/dashboard-service/src/lib/types/config.ts modules/dashboard-service/src/lib/api/config.ts
git commit -m "chore: remove deprecated ProviderRegistry and old connection types"
```

### Task 17: Final Verification

- [ ] **Step 1: Run backend tests**

Run: `cd modules/orchestration-service && uv run pytest tests/ -v --timeout=30 -x`

- [ ] **Step 2: Run frontend build**

Run: `cd modules/dashboard-service && npx vite build`

- [ ] **Step 3: Run all E2E tests**

Run: `cd modules/dashboard-service && npx playwright test`

- [ ] **Step 4: Verify against real Ollama**

If `thomas-pc` (100.79.188.56) is reachable:
1. Create a connection via the UI pointing to `http://100.79.188.56:11434` with engine `ollama`
2. Verify it — should show models
3. Set it as chat active model
4. Send a chat message — should get a response

- [ ] **Step 5: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "fix: final verification fixes for unified connections"
```
