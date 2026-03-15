# Translation Service Connections — Open WebUI Style

**Date**: 2026-03-11
**Service**: dashboard-service (SvelteKit)
**Backend**: orchestration-service (FastAPI)

## Problem

The dashboard's translation config page (`/config/translation`) has no way to configure _where_ the translation service runs. Users running Ollama on a separate device, vLLM on a GPU server, or multiple backends simultaneously have no UI to manage these connections. The richer settings from the React frontend-service (language config, quality, caching, real-time) have never been migrated.

## Solution

An aggregated connection pool inspired by Open WebUI's Connections settings. Multiple backends can be added, each with a prefix ID for model disambiguation. All discovered models appear in one unified list.

## Data Model

```typescript
interface TranslationConnection {
  id: string;                // crypto.randomUUID()
  name: string;              // user label, e.g. "Home GPU"
  engine: 'ollama' | 'vllm' | 'triton' | 'openai_compatible';
  url: string;               // e.g. http://192.168.1.50:11434
  prefix: string;            // model prefix, e.g. "home-gpu"
  api_key: string;           // empty for local, token for remote
  enabled: boolean;
  timeout_ms: number;        // default 30000
  max_retries: number;       // default 3
}

interface TranslationConfig {
  connections: TranslationConnection[];
  active_model: string;      // prefixed: "home-gpu/llama2:7b"
  fallback_model: string;    // prefixed
  languages: { ... };        // migrated from React frontend
  quality: { ... };          // migrated
  model: { ... };            // temperature, max_tokens, etc.
  realtime: { ... };         // migrated
  caching: { ... };          // migrated
}
```

## UI Layout

### Page: `/config/translation`

**Section 1 — Connections Manager** (top of page)
- Header: "Translation Connections" + "+ Add Connection" button
- Each connection renders as a `ConnectionCard`:
  - Engine badge (color-coded)
  - URL display (read-only inline, editable via dialog)
  - Prefix ID shown inline
  - Enable/disable toggle (auto-saves)
  - Verify button → calls `POST /api/settings/translation/verify-connection`
  - Configure (gear) → opens `ConnectionDialog`
  - Delete (trash) → confirmation then remove
  - Status dot: green (connected), red (error), yellow (verifying), gray (unknown)
  - Model count badge after successful verify
- Below cards: "Aggregated Models" summary showing all prefixed models

**Section 2 — Active Model** (existing, enhanced)
- Model dropdown populated from aggregated models across all enabled connections
- Fallback model dropdown (same list)
- Current model health card (existing)

**Section 3 — Translation Settings** (collapsible cards)
- Language Configuration (target languages, auto-detect, confidence threshold)
- Model Parameters (temperature, top_p, max_tokens, repetition_penalty)
- Quality & Performance (quality threshold, validation toggles)
- Real-time Configuration (streaming, batching, delay)
- Caching (enable, duration, similarity threshold, memory limit)
- Prompt Templates (existing, preserved as-is)

### ConnectionDialog (shadcn Dialog)

Fields:
- Name (text input)
- Engine (select: vLLM / Ollama / Triton / OpenAI Compatible)
- URL (text input, placeholder changes per engine)
- Prefix ID (text input, auto-suggested from name)
- API Key (password input with show/hide, shown when engine is openai_compatible or toggled)
- Timeout (number, collapsed under "Advanced")
- Max Retries (number, collapsed under "Advanced")

## Backend Endpoints

| Method | Path | Status | Purpose |
|--------|------|--------|---------|
| GET | `/api/settings/translation` | exists | Returns full config including `connections[]` |
| POST | `/api/settings/translation` | exists | Saves full config |
| POST | `/api/settings/translation/verify-connection` | exists | Probes a single URL by engine type |
| POST | `/api/settings/translation/aggregate-models` | **new** | Iterates enabled connections, verifies each, returns prefixed model list |
| GET | `/api/settings/translation/stats` | exists | Translation statistics |
| POST | `/api/settings/translation/test` | exists | Real translation test |
| POST | `/api/settings/translation/clear-cache` | exists | Clear cache |

### `POST /aggregate-models` behavior
1. For each enabled connection in the config:
2. Call verify-connection logic to probe the backend
3. Prefix each discovered model with `{connection.prefix}/`
4. Return `{ models: [{id, name, connection_id, prefix}], errors: [{connection_id, message}] }`

## Backend Config Changes

Update `TranslationConfig` in `_shared.py`:
- Add `connections: list[dict]` field with default containing one local connection
- Keep backward compatibility: if `connections` is missing, fall back to legacy `service` block

## Files to Create/Modify

### Dashboard Service (frontend)
| File | Action |
|------|--------|
| `src/routes/(app)/config/translation/+page.svelte` | Rewrite with connections manager + migrated settings |
| `src/routes/(app)/config/translation/+page.server.ts` | Update load to fetch connections, add save action |
| `src/lib/api/config.ts` | Add verifyConnection, aggregateModels |
| `src/lib/components/ConnectionCard.svelte` | New component |
| `src/lib/components/ConnectionDialog.svelte` | New component |
| `src/lib/types.ts` | Add TranslationConnection, update TranslationConfig |

### Orchestration Service (backend)
| File | Action |
|------|--------|
| `src/routers/settings/translation.py` | Add aggregate-models endpoint |
| `src/routers/settings/_shared.py` | Add connections to TranslationConfig default |

### Tests
| File | Action |
|------|--------|
| `tests/e2e/translation-connections.spec.ts` | Playwright E2E tests |

## Playwright E2E Test Plan

1. **Page loads** — translation config page renders without errors
2. **Default connection visible** — at least one connection card shown
3. **Add connection** — click "+", fill dialog, save → new card appears
4. **Verify connection** — click Verify on a connection → status updates (green or red)
5. **Edit connection** — click gear → dialog opens pre-filled → edit URL → save
6. **Delete connection** — click delete → confirmation → card removed
7. **Prefix in models** — after verify, model dropdown shows prefixed models
8. **Toggle enable/disable** — toggle off → connection grayed, models removed from dropdown
9. **Settings save round-trip** — change language/quality settings → save → reload → values persist
10. **Engine-specific behavior** — switching engine updates URL placeholder and helper text

## Migration Checklist

Settings to migrate from React `TranslationSettings.tsx` → SvelteKit:
- [x] Service URL + engine → replaced by connections manager
- [ ] Language config (auto-detect, source lang, target languages, confidence threshold)
- [ ] Model parameters (temperature, top_p, max_tokens, repetition_penalty, context_window, batch_size)
- [ ] Quality settings (quality threshold, confidence scoring, translation validation, context preservation, speaker attribution)
- [ ] Real-time config (streaming, partial results, delay, batching, adaptive batching)
- [ ] Caching config (enabled, duration, similarity threshold, memory limit, cleanup interval)
- [x] Prompt templates → already in dashboard
- [ ] Translation statistics card
