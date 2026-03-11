# Unified AI Connections

## Problem

Three features (Chat, Translation, Intelligence) each maintain separate configuration for connecting to the same AI backends. Adding a new backend or rotating an API key requires updating three different systems:

- **Translation**: `connections[]` array in `config/translation.json`
- **Chat**: `system_config` DB row with provider/model/base_url/api_key
- **Intelligence**: `MeetingIntelligenceSettings` class with hardcoded `default_llm_backend`

All three ultimately talk to the same Ollama/vLLM/OpenAI/Anthropic backends.

## Solution

A single `ai_connections` table in PostgreSQL that all features share. Each feature selects which model to use from the shared pool via per-feature preference rows in `system_config`.

Pattern follows Open WebUI: connections are a system-level concern, features just pick models.

## Supported Engines

4 engine types (collapsed from the prior 6):

| Engine | URL | API Key | Probe Endpoint |
|--------|-----|---------|----------------|
| `ollama` | User-provided (default `localhost:11434`) | Optional | `GET /api/tags` |
| `openai` | Fixed `https://api.openai.com` | Required | `GET /v1/models` |
| `anthropic` | Fixed `https://api.anthropic.com` | Required | `GET /v1/models` |
| `openai_compatible` | User-provided (required) | Optional | `GET /v1/models` |

vLLM and Triton are handled as `openai_compatible` since they expose `/v1/` endpoints.

## Database Schema

### `ai_connections` table

```sql
CREATE TABLE ai_connections (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    engine          TEXT NOT NULL CHECK (engine IN ('ollama', 'openai', 'anthropic', 'openai_compatible')),
    url             TEXT NOT NULL,
    api_key         TEXT NOT NULL DEFAULT '',
    prefix          TEXT NOT NULL DEFAULT '',
    enabled         BOOLEAN NOT NULL DEFAULT TRUE,
    context_length  INTEGER,
    timeout_ms      INTEGER NOT NULL DEFAULT 30000,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    priority        INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
```

- `id`: Slug identifier (e.g., `"home-gpu"`, `"cloud-openai"`). Validated: lowercase alphanumeric + hyphens, max 64 chars. Auto-generated from `name` if not provided (e.g., "Home GPU" → `"home-gpu"`). Uniqueness violations return 409.
- `prefix`: Model namespace for disambiguation (e.g., `"home-gpu"` → `"home-gpu/llama2:7b"`)
- `context_length`: Optional override; NULL means use model default
- `priority`: UI ordering and fallback preference (lower = higher priority)
- `engine`: Constrained at DB level via CHECK constraint

### `system_config` rows for feature preferences

Each feature stores its model preference as a JSON value:

| Key | Value |
|-----|-------|
| `translation_model_preference` | `{"active_model": "home-gpu/qwen3.5:4b", "fallback_model": "...", "temperature": 0.3, "max_tokens": 512}` |
| `chat_model_preference` | `{"active_model": "cloud-openai/gpt-4o", "temperature": 0.7, "max_tokens": 4096}` |
| `intelligence_model_preference` | `{"active_model": "home-gpu/qwen3.5:4b", "temperature": 0.3, "max_tokens": 1024}` |

Translation-specific settings (languages, quality, prompt, caching, realtime) migrate from JSON file to individual `system_config` rows:

| Key | Value |
|-----|-------|
| `translation_languages` | `{...}` |
| `translation_quality` | `{...}` |
| `translation_service` | `{...}` |
| `translation_caching` | `{...}` |
| `translation_realtime` | `{...}` |

## Backend Architecture

### ConnectionService

**Location:** `modules/orchestration-service/src/services/connections.py`

Single service that replaces both `ProviderRegistry` and the JSON config file.

```python
class ConnectionService:
    async def list_connections(*, enabled_only: bool = False) -> list[AIConnection]
    async def get_connection(id: str) -> AIConnection
    async def create_connection(conn: AIConnectionCreate) -> AIConnection
    async def update_connection(id: str, conn: AIConnectionUpdate) -> AIConnection
    async def delete_connection(id: str) -> None
    async def verify_connection(id: str) -> VerifyResult
    async def aggregate_models(*, enabled_only: bool = True) -> list[AggregatedModel]
    async def get_adapter(connection_id: str) -> LLMAdapter
```

**Adapter caching:** `get_adapter()` caches adapter instances keyed by `(connection_id, updated_at)`. If a connection is updated, the next `get_adapter()` call constructs a fresh adapter. This avoids per-request overhead for providers with connection pools (OpenAI, Anthropic) while still picking up config changes.

### Model Resolution

Features reference models by prefixed ID (e.g., `"home-gpu/qwen3.5:4b"`). A shared resolver parses this:

```python
async def resolve_model(model_id: str) -> tuple[AIConnection, str]:
    """Parse 'prefix/model' → (connection, model_name).

    Falls back to matching by connection ID if no prefix match found.
    """
```

### Feature Adapter Integration

- **Chat**: `ConnectionService.get_adapter(connection_id)` returns an `LLMAdapter` — same interface Chat already uses. The chat router swaps `registry.get_adapter("ollama")` for `connection_service.get_adapter("home-gpu")`.
- **Intelligence**: `LLMClient.from_connection(conn: AIConnection)` constructs a client from the connection's url/api_key/engine. Always uses direct mode (`proxy_mode=False`) — the proxy mode path through Translation Service is retired.
- **Translation**: Verify/aggregate endpoints delegate to `ConnectionService` methods.

### What Gets Removed

- `ProviderRegistry` class → replaced by `ConnectionService`
- `TranslationConfig.connections[]` field → migrated to `ai_connections` table
- `TRANSLATION_CONFIG_FILE` / `config/translation.json` → settings move to `system_config` rows
- `PROVIDER_FACTORIES` dict → adapter construction moves into `ConnectionService.get_adapter()`
- Chat settings' `provider`, `base_url`, `has_api_key` fields → replaced by `active_model` referencing the shared pool

## API Endpoints

### New: `/api/connections` router

```
GET    /api/connections                    → List all connections
POST   /api/connections                    → Create connection
GET    /api/connections/{id}               → Get single connection
PUT    /api/connections/{id}               → Update connection
DELETE /api/connections/{id}               → Delete connection
POST   /api/connections/{id}/verify        → Verify connectivity + discover models
POST   /api/connections/aggregate-models   → Aggregate models from all enabled connections
```

### New: `/api/settings/feature-preferences` router

```
GET  /api/settings/feature-preferences                → All feature model preferences
PUT  /api/settings/feature-preferences/{feature}      → Update one feature's preference
```

`feature` is one of: `"chat"`, `"translation"`, `"intelligence"`.

### Modified existing routes

- `POST /api/settings/translation` — no longer accepts `connections[]`, `active_model`, `fallback_model`. Only handles translation-specific settings (languages, quality, prompt config). Reads/writes `system_config` rows instead of JSON file.
- `GET/PUT /api/chat/settings` — drops `provider`, `base_url`, `has_api_key`. Reads/writes `chat_model_preference` from `system_config`.
- Intelligence endpoints — `default_llm_backend` parameter reads from `intelligence_model_preference` instead of config class.

### Removed endpoints

- `POST /api/settings/translation/verify-connection` → use `POST /api/connections/{id}/verify`
- `POST /api/settings/translation/aggregate-models` → use `POST /api/connections/aggregate-models`

### Security

SSRF validation (`_validate_connection_url()`) applies to connection create, update, and verify. Blocks cloud metadata IPs, link-local addresses, and non-http(s) schemes.

## Dashboard UI

### New page: Config > Connections

**Route:** `/routes/(app)/config/connections/+page.svelte`

The connections manager currently on the translation page moves here. Same card-per-connection UX with verify, edit, delete (with confirmation), and enable/disable toggle.

Engine-specific dialog behavior:

| Field | Ollama | OpenAI | Anthropic | OpenAI-Compatible |
|-------|--------|--------|-----------|-------------------|
| URL | Editable (default `localhost:11434`) | Hidden (fixed) | Hidden (fixed) | Editable (required) |
| API Key | Optional | Required | Required | Optional |
| Context Length | Optional | Optional | Optional | Optional |
| Prefix | Required | Required | Required | Required |

### Modified: Config > Translation

- Connections section removed (link to Config > Connections instead)
- Model selector dropdown populated from aggregated models
- Translation-specific settings remain (temperature, max_tokens, prompt template, languages, quality)

### Modified: Chat SettingsDrawer

- Provider/base_url/api_key fields replaced by single model selector from shared pool
- Temperature/max_tokens stay

### Modified: Intelligence page

- `default_llm_backend` references replaced by model selector from shared pool

### Navigation

"Connections" added to config sidebar nav, above "Translation".

## Migration

### Alembic migration: `010_ai_connections`

1. Create `ai_connections` table
2. Read existing `system_config.chat_settings` — if it has provider/base_url/api_key, insert as a connection
3. Read `config/translation.json` — if it has `connections[]`, insert each as a connection. Map `engine: "vllm"` or `engine: "triton"` to `"openai_compatible"`.
4. Insert per-feature preference rows into `system_config`
5. Migrate translation settings (languages, quality, service, caching, realtime) from JSON to `system_config` rows
6. If no connections exist after steps 2-3, seed a default Ollama connection (`localhost:11434`, prefix `local`)

### Backward compatibility

None — clean cut. JSON config file becomes dead code and is removed. Features updated in the same changeset.

### Risk mitigation

Migration reads old data before writing new. If it fails partway, old config files are untouched. Alembic downgrade drops the table and preference rows.

## Testing

- E2E Playwright tests for the new Connections page (adapted from existing translation-connections tests)
- Behavioral tests for `ConnectionService` CRUD operations against real PostgreSQL
- Verify/aggregate tests against a real Ollama instance (thomas-pc at `100.79.188.56:11434`)
- Integration tests: Chat sends a message through a shared connection, Intelligence generates an insight through a shared connection
- Migration test: seed old-format data, run migration, verify connections and preferences are correct
