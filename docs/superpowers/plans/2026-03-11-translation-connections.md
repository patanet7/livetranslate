# Translation Connections Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Open WebUI-style aggregated connection pool to the dashboard's translation config page, with multi-backend support, prefix IDs, and migrated settings from the React frontend.

**Architecture:** The dashboard-service (SvelteKit + shadcn-svelte) gets a Connections Manager UI backed by the orchestration-service's existing settings API. Connections are stored as an array in `config/translation.json`. A new `aggregate-models` endpoint iterates all enabled connections and returns a unified, prefixed model list.

**Tech Stack:** SvelteKit 5 (Svelte runes), shadcn-svelte (bits-ui), Tailwind CSS 4, Lucide icons, FastAPI (backend), Playwright (E2E tests)

---

## File Structure

### Dashboard Service (SvelteKit)
| File | Action | Responsibility |
|------|--------|---------------|
| `src/lib/types/config.ts` | Modify | Add `TranslationConnection`, `VerifyConnectionResponse`, update `TranslationConfig` |
| `src/lib/api/config.ts` | Modify | Add `verifyConnection()`, `aggregateModels()`, `saveTranslationConnections()` |
| `src/lib/components/ConnectionCard.svelte` | Create | Single connection row: URL, prefix, status, verify, toggle, configure, delete |
| `src/lib/components/ConnectionDialog.svelte` | Create | Add/edit connection dialog: name, engine, URL, prefix, API key, advanced |
| `src/routes/(app)/config/translation/+page.svelte` | Rewrite | Connections manager + model selector + migrated settings sections |
| `src/routes/(app)/config/translation/+page.server.ts` | Modify | Load connections, add save actions |
| `tests/e2e/translation-connections.spec.ts` | Create | Playwright E2E tests |

### Orchestration Service (FastAPI)
| File | Action | Responsibility |
|------|--------|---------------|
| `src/routers/settings/_shared.py` | Modify | Add `connections` default to `TranslationConfig` |
| `src/routers/settings/translation.py` | Modify | Add `POST /aggregate-models` endpoint |

---

## Chunk 1: Backend — Connections Data Model + Aggregate Endpoint

### Task 1: Update TranslationConfig with connections array

**Files:**
- Modify: `modules/orchestration-service/src/routers/settings/_shared.py` (TranslationConfig class, ~line 337)

- [ ] **Step 1: Add connections field to TranslationConfig**

In `_shared.py`, update the `TranslationConfig` class to include a `connections` list with one default local connection:

```python
class TranslationConfig(BaseModel):
    """Translation service configuration schema"""

    connections: list[dict[str, Any]] = [
        {
            "id": "default",
            "name": "Local Translation Service",
            "engine": "vllm",
            "url": "http://localhost:5003",
            "prefix": "local",
            "api_key": "",
            "enabled": True,
            "timeout_ms": 30000,
            "max_retries": 3,
        }
    ]
    # Keep legacy service block for backward compat
    service: dict[str, Any] = {
        "enabled": True,
        "service_url": "http://localhost:5003",
        "inference_engine": "vllm",
        "model_name": "llama2-7b-chat",
        "fallback_model": "orca-mini-3b",
        "timeout_ms": 30000,
        "max_retries": 3,
        "api_key": "",
    }
    active_model: str = ""
    fallback_model: str = ""
    # ... rest unchanged (languages, quality, model, etc.)
```

- [ ] **Step 2: Verify config loads with new field**

Run: `cd /Users/thomaspatane/GitHub/personal/livetranslate && curl -s http://localhost:3000/api/settings/translation | python3 -m json.tool | head -30`

Expected: JSON includes `"connections"` array with the default entry.

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/settings/_shared.py
git commit -m "feat: add connections array to TranslationConfig"
```

### Task 2: Add aggregate-models endpoint

**Files:**
- Modify: `modules/orchestration-service/src/routers/settings/translation.py`

- [ ] **Step 1: Add the aggregate-models endpoint**

After the `verify_translation_connection` function, add:

```python
@router.post("/translation/aggregate-models")
async def aggregate_translation_models():
    """
    Iterate all enabled connections, probe each for models,
    prefix them with the connection's prefix, and return a unified list.
    """
    default_config = TranslationConfig().dict()
    config = await load_config(TRANSLATION_CONFIG_FILE, default_config)
    connections = config.get("connections", [])

    all_models: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for conn in connections:
        if not conn.get("enabled", True):
            continue

        conn_id = conn.get("id", "unknown")
        prefix = conn.get("prefix", "")
        url = conn.get("url", "").rstrip("/")
        engine = conn.get("engine", "vllm")
        api_key = conn.get("api_key", "")

        if not url:
            continue

        # Reuse verify logic to probe
        req = VerifyConnectionRequest(url=url, engine=engine, api_key=api_key or None)
        result = await verify_translation_connection(req)

        if result.get("status") == "connected":
            raw_models = result.get("models", [])
            for model_name in raw_models:
                prefixed = f"{prefix}/{model_name}" if prefix else model_name
                all_models.append({
                    "id": prefixed,
                    "name": model_name,
                    "connection_id": conn_id,
                    "connection_name": conn.get("name", ""),
                    "prefix": prefix,
                    "engine": engine,
                })
        else:
            errors.append({
                "connection_id": conn_id,
                "connection_name": conn.get("name", ""),
                "message": result.get("message", "Unknown error"),
            })

    logger.info(
        "aggregate_models_complete",
        total_models=len(all_models),
        total_errors=len(errors),
    )
    return {"models": all_models, "errors": errors}
```

- [ ] **Step 2: Test the endpoint**

Run: `curl -s -X POST http://localhost:3000/api/settings/translation/aggregate-models | python3 -m json.tool`

Expected: JSON with `"models"` array (may be empty if no backend running) and `"errors"` array.

- [ ] **Step 3: Commit**

```bash
git add modules/orchestration-service/src/routers/settings/translation.py
git commit -m "feat: add aggregate-models endpoint for multi-connection model discovery"
```

---

## Chunk 2: Dashboard Types + API Layer

### Task 3: Add TypeScript types for connections

**Files:**
- Modify: `modules/dashboard-service/src/lib/types/config.ts`

- [ ] **Step 1: Add connection types**

Add these types after the existing `TranslationHealth` interface:

```typescript
export interface TranslationConnection {
  id: string;
  name: string;
  engine: 'ollama' | 'vllm' | 'triton' | 'openai_compatible';
  url: string;
  prefix: string;
  api_key: string;
  enabled: boolean;
  timeout_ms: number;
  max_retries: number;
}

export interface VerifyConnectionRequest {
  url: string;
  engine: string;
  api_key?: string;
}

export interface VerifyConnectionResponse {
  status: 'connected' | 'error';
  message: string;
  version?: string;
  models?: string[];
  latency_ms?: number;
}

export interface AggregatedModel {
  id: string;          // prefixed: "home-gpu/llama2:7b"
  name: string;        // raw: "llama2:7b"
  connection_id: string;
  connection_name: string;
  prefix: string;
  engine: string;
}

export interface AggregateModelsResponse {
  models: AggregatedModel[];
  errors: Array<{ connection_id: string; connection_name: string; message: string }>;
}

export interface FullTranslationConfig {
  connections: TranslationConnection[];
  active_model: string;
  fallback_model: string;
  service: Record<string, unknown>;
  languages: Record<string, unknown>;
  quality: Record<string, unknown>;
  model: Record<string, unknown>;
  realtime?: Record<string, unknown>;
  caching?: Record<string, unknown>;
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/types/config.ts
git commit -m "feat: add TypeScript types for translation connections"
```

### Task 4: Add API methods for connections

**Files:**
- Modify: `modules/dashboard-service/src/lib/api/config.ts`

- [ ] **Step 1: Add connection API methods**

Add these methods to the return object of `configApi()`:

```typescript
verifyConnection: (req: VerifyConnectionRequest) =>
  api.post<VerifyConnectionResponse>('/api/settings/translation/verify-connection', req),

aggregateModels: () =>
  api.post<AggregateModelsResponse>('/api/settings/translation/aggregate-models'),

getFullTranslationConfig: () =>
  api.get<FullTranslationConfig>('/api/settings/translation'),

saveFullTranslationConfig: (config: FullTranslationConfig) =>
  api.post<{ message: string; config: FullTranslationConfig }>(
    '/api/settings/translation',
    config
  ),
```

Also add the imports at the top:

```typescript
import type {
  UserSettings,
  TranslationSettings,
  UiConfig,
  SystemConfigUpdate,
  TranslationModelsResponse,
  TranslationHealth,
  VerifyConnectionRequest,
  VerifyConnectionResponse,
  AggregateModelsResponse,
  FullTranslationConfig
} from '$lib/types';
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/api/config.ts
git commit -m "feat: add connection verify and aggregate API methods"
```

---

## Chunk 3: ConnectionCard Component

### Task 5: Create ConnectionCard.svelte

**Files:**
- Create: `modules/dashboard-service/src/lib/components/ConnectionCard.svelte`

- [ ] **Step 1: Create the component**

```svelte
<script lang="ts">
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Badge } from '$lib/components/ui/badge';
  import { Input } from '$lib/components/ui/input';
  import {
    Settings,
    Trash2,
    Loader2,
    CheckCircle,
    XCircle,
    Circle,
    Power
  } from '@lucide/svelte';
  import type { TranslationConnection, VerifyConnectionResponse } from '$lib/types';

  interface Props {
    connection: TranslationConnection;
    status: 'unknown' | 'connected' | 'error' | 'verifying';
    modelCount: number;
    onverify: () => void;
    onconfigure: () => void;
    ondelete: () => void;
    ontoggle: (enabled: boolean) => void;
  }

  let {
    connection,
    status,
    modelCount,
    onverify,
    onconfigure,
    ondelete,
    ontoggle
  }: Props = $props();

  const engineColors: Record<string, string> = {
    ollama: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
    vllm: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300',
    triton: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
    openai_compatible: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300'
  };

  const engineLabels: Record<string, string> = {
    ollama: 'Ollama',
    vllm: 'vLLM',
    triton: 'Triton',
    openai_compatible: 'OpenAI'
  };

  const statusDot: Record<string, string> = {
    unknown: 'text-muted-foreground',
    connected: 'text-green-500',
    error: 'text-red-500',
    verifying: 'text-yellow-500'
  };
</script>

<div
  class="flex items-center gap-3 rounded-lg border p-3 transition-opacity {connection.enabled ? '' : 'opacity-50'}"
  class:border-green-500/30={status === 'connected'}
  class:border-red-500/30={status === 'error'}
>
  <!-- Status dot -->
  <div class="flex-shrink-0">
    {#if status === 'verifying'}
      <Loader2 class="h-4 w-4 animate-spin text-yellow-500" />
    {:else if status === 'connected'}
      <CheckCircle class="h-4 w-4 text-green-500" />
    {:else if status === 'error'}
      <XCircle class="h-4 w-4 text-red-500" />
    {:else}
      <Circle class="h-4 w-4 text-muted-foreground" />
    {/if}
  </div>

  <!-- Connection info -->
  <div class="flex-1 min-w-0 space-y-1">
    <div class="flex items-center gap-2">
      <span class="text-sm font-medium truncate">{connection.name}</span>
      <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {engineColors[connection.engine] ?? 'bg-muted text-muted-foreground'}">
        {engineLabels[connection.engine] ?? connection.engine}
      </span>
      {#if connection.prefix}
        <Badge variant="outline" class="text-xs">prefix: {connection.prefix}</Badge>
      {/if}
    </div>
    <div class="flex items-center gap-2">
      <p class="text-xs text-muted-foreground truncate">{connection.url}</p>
      {#if status === 'connected' && modelCount > 0}
        <Badge variant="secondary" class="text-xs">{modelCount} model{modelCount !== 1 ? 's' : ''}</Badge>
      {/if}
    </div>
  </div>

  <!-- Actions -->
  <div class="flex items-center gap-1 flex-shrink-0">
    <Button
      variant="outline"
      size="sm"
      onclick={onverify}
      disabled={!connection.enabled || status === 'verifying'}
    >
      {#if status === 'verifying'}
        <Loader2 class="h-3 w-3 animate-spin mr-1" />
      {/if}
      Verify
    </Button>
    <Button variant="ghost" size="icon" onclick={onconfigure} class="h-8 w-8">
      <Settings class="h-4 w-4" />
    </Button>
    <Button
      variant="ghost"
      size="icon"
      onclick={() => ontoggle(!connection.enabled)}
      class="h-8 w-8 {connection.enabled ? 'text-green-500' : 'text-muted-foreground'}"
    >
      <Power class="h-4 w-4" />
    </Button>
    <Button variant="ghost" size="icon" onclick={ondelete} class="h-8 w-8 text-destructive">
      <Trash2 class="h-4 w-4" />
    </Button>
  </div>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/ConnectionCard.svelte
git commit -m "feat: add ConnectionCard component"
```

---

## Chunk 4: ConnectionDialog Component

### Task 6: Create ConnectionDialog.svelte

**Files:**
- Create: `modules/dashboard-service/src/lib/components/ConnectionDialog.svelte`

- [ ] **Step 1: Create the dialog component**

```svelte
<script lang="ts">
  import * as Dialog from '$lib/components/ui/dialog';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Eye, EyeOff, ChevronDown } from '@lucide/svelte';
  import type { TranslationConnection } from '$lib/types';

  interface Props {
    open: boolean;
    connection: TranslationConnection | null;
    onsave: (connection: TranslationConnection) => void;
    onclose: () => void;
  }

  let { open = $bindable(), connection, onsave, onclose }: Props = $props();

  const engineDefaults: Record<string, { url: string; modelPlaceholder: string; helperText: string }> = {
    vllm: {
      url: 'http://localhost:8000',
      modelPlaceholder: 'meta-llama/Llama-2-7b-chat-hf',
      helperText: 'vLLM serves OpenAI-compatible endpoints at /v1/chat/completions'
    },
    ollama: {
      url: 'http://localhost:11434',
      modelPlaceholder: 'llama2:7b',
      helperText: 'Ollama API serves models at /api/chat'
    },
    triton: {
      url: 'http://localhost:8001',
      modelPlaceholder: 'ensemble_model',
      helperText: 'NVIDIA Triton Inference Server'
    },
    openai_compatible: {
      url: 'https://api.openai.com/v1',
      modelPlaceholder: 'gpt-4',
      helperText: 'Any OpenAI-compatible API endpoint'
    }
  };

  // Form state — reset when dialog opens
  let name = $state('');
  let engine = $state<TranslationConnection['engine']>('ollama');
  let url = $state('');
  let prefix = $state('');
  let api_key = $state('');
  let timeout_ms = $state(30000);
  let max_retries = $state(3);
  let showApiKey = $state(false);
  let showAdvanced = $state(false);

  // Sync form from connection prop when dialog opens
  $effect(() => {
    if (open && connection) {
      name = connection.name;
      engine = connection.engine;
      url = connection.url;
      prefix = connection.prefix;
      api_key = connection.api_key;
      timeout_ms = connection.timeout_ms;
      max_retries = connection.max_retries;
    } else if (open && !connection) {
      name = '';
      engine = 'ollama';
      url = engineDefaults.ollama.url;
      prefix = '';
      api_key = '';
      timeout_ms = 30000;
      max_retries = 3;
    }
    showApiKey = false;
    showAdvanced = false;
  });

  function handleEngineChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    engine = target.value as TranslationConnection['engine'];
    url = engineDefaults[engine]?.url ?? '';
  }

  function autoPrefix() {
    if (!prefix && name) {
      prefix = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-+$/, '');
    }
  }

  function handleSave() {
    const result: TranslationConnection = {
      id: connection?.id ?? crypto.randomUUID(),
      name: name || 'Unnamed Connection',
      engine,
      url: url.replace(/\/+$/, ''),
      prefix: prefix || name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
      api_key,
      enabled: connection?.enabled ?? true,
      timeout_ms,
      max_retries
    };
    onsave(result);
    open = false;
  }
</script>

<Dialog.Root bind:open>
  <Dialog.Content class="max-w-lg">
    <Dialog.Header>
      <Dialog.Title>{connection ? 'Edit Connection' : 'Add Connection'}</Dialog.Title>
      <Dialog.Description>
        {connection ? 'Update the connection settings' : 'Add a new translation backend'}
      </Dialog.Description>
    </Dialog.Header>

    <div class="space-y-4 py-4">
      <!-- Name -->
      <div class="space-y-2">
        <Label for="conn-name">Name</Label>
        <Input
          id="conn-name"
          bind:value={name}
          placeholder="Home GPU, Work Server, etc."
          onblur={autoPrefix}
        />
      </div>

      <!-- Engine -->
      <div class="space-y-2">
        <Label for="conn-engine">Engine</Label>
        <select
          id="conn-engine"
          class="w-full rounded-md border bg-background px-3 py-2 text-sm"
          value={engine}
          onchange={handleEngineChange}
        >
          <option value="ollama">Ollama</option>
          <option value="vllm">vLLM</option>
          <option value="triton">Triton</option>
          <option value="openai_compatible">OpenAI Compatible</option>
        </select>
        <p class="text-xs text-muted-foreground">{engineDefaults[engine]?.helperText ?? ''}</p>
      </div>

      <!-- URL -->
      <div class="space-y-2">
        <Label for="conn-url">URL</Label>
        <Input
          id="conn-url"
          bind:value={url}
          placeholder={engineDefaults[engine]?.url ?? 'http://localhost:8000'}
        />
      </div>

      <!-- Prefix -->
      <div class="space-y-2">
        <Label for="conn-prefix">Prefix ID</Label>
        <Input
          id="conn-prefix"
          bind:value={prefix}
          placeholder="e.g. home-gpu, work-server"
        />
        <p class="text-xs text-muted-foreground">Prepended to model names for disambiguation (e.g. home-gpu/llama2:7b)</p>
      </div>

      <!-- API Key -->
      {#if engine === 'openai_compatible' || api_key}
        <div class="space-y-2">
          <Label for="conn-apikey">API Key</Label>
          <div class="relative">
            <Input
              id="conn-apikey"
              type={showApiKey ? 'text' : 'password'}
              bind:value={api_key}
              placeholder="sk-... or Bearer token"
              class="pr-10"
            />
            <button
              type="button"
              class="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              onclick={() => showApiKey = !showApiKey}
            >
              {#if showApiKey}
                <EyeOff class="h-4 w-4" />
              {:else}
                <Eye class="h-4 w-4" />
              {/if}
            </button>
          </div>
        </div>
      {/if}

      <!-- Advanced toggle -->
      <button
        type="button"
        class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
        onclick={() => showAdvanced = !showAdvanced}
      >
        <ChevronDown class="h-3 w-3 transition-transform {showAdvanced ? 'rotate-180' : ''}" />
        {showAdvanced ? 'Hide advanced' : 'Show advanced'}
      </button>

      {#if showAdvanced}
        <div class="grid grid-cols-2 gap-3">
          <div class="space-y-2">
            <Label for="conn-timeout">Timeout (ms)</Label>
            <Input id="conn-timeout" type="number" bind:value={timeout_ms} />
          </div>
          <div class="space-y-2">
            <Label for="conn-retries">Max Retries</Label>
            <Input id="conn-retries" type="number" bind:value={max_retries} />
          </div>
        </div>
      {/if}
    </div>

    <Dialog.Footer>
      <Button variant="outline" onclick={() => { open = false; onclose(); }}>Cancel</Button>
      <Button onclick={handleSave}>
        {connection ? 'Save Changes' : 'Add Connection'}
      </Button>
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.Root>
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/lib/components/ConnectionDialog.svelte
git commit -m "feat: add ConnectionDialog component for add/edit connections"
```

---

## Chunk 5: Translation Config Page Rewrite

### Task 7: Update page server load + actions

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/config/translation/+page.server.ts`

- [ ] **Step 1: Rewrite page server**

Replace the file with:

```typescript
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { firefliesApi } from '$lib/api/fireflies';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
  const ff = firefliesApi(fetch);
  const cfg = configApi(fetch);
  const [
    translationConfig,
    fullConfig,
    uiConfig,
    translationModels,
    translationHealth
  ] = await Promise.all([
    ff.getTranslationConfig().catch(() => null),
    cfg.getFullTranslationConfig().catch(() => null),
    cfg.getUiConfig().catch(() => null),
    cfg.getTranslationModels().catch(() => null),
    cfg.getTranslationHealth().catch(() => null)
  ]);
  return { translationConfig, fullConfig, uiConfig, translationModels, translationHealth };
};

export const actions: Actions = {
  update: async ({ request, fetch }) => {
    const data = await request.formData();
    const target_language = data.get('target_language')?.toString() ?? 'es';
    const temperature = parseFloat(data.get('temperature')?.toString() ?? '0.3');
    const max_tokens = parseInt(data.get('max_tokens')?.toString() ?? '512', 10);

    const ff = firefliesApi(fetch);
    try {
      await ff.updateTranslationConfig({ target_language, temperature, max_tokens });
      return { success: true };
    } catch (err) {
      return fail(500, { errors: { form: `Update failed: ${err}` } });
    }
  }
};
```

- [ ] **Step 2: Commit**

```bash
git add modules/dashboard-service/src/routes/(app)/config/translation/+page.server.ts
git commit -m "feat: load full translation config with connections in page server"
```

### Task 8: Rewrite the translation config page

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/config/translation/+page.svelte`

- [ ] **Step 1: Rewrite the page**

This is the largest change. The new page has three sections:
1. **Connections Manager** — list of ConnectionCards + "Add Connection" button
2. **Active Model** — model selector from aggregated models
3. **Settings** — collapsible cards for language, quality, model params, real-time, caching, prompt templates

The complete file content is large (300+ lines). Write it with these key sections:

**Script section:**
- Import ConnectionCard, ConnectionDialog, all UI components
- State: `connections` array (from `data.fullConfig`), `connectionStatuses` map, `connectionModelCounts` map, `dialogOpen`, `editingConnection`, `aggregatedModels`, `deleteConfirmId`
- Functions: `verifyConnection(conn)`, `addConnection(conn)`, `updateConnection(conn)`, `deleteConnection(id)`, `toggleConnection(id, enabled)`, `saveConnections()`, `loadAggregatedModels()`
- On mount: auto-verify all enabled connections

**Template sections:**
1. Connections Manager card with `{#each connections}` → `<ConnectionCard>` + `<ConnectionDialog>`
2. Active Model card with select dropdown from aggregated models
3. Existing model health card (preserved)
4. Translation Settings form (preserved + enhanced with migrated settings)
5. Prompt Template editor (preserved as-is)

- [ ] **Step 2: Verify the page renders**

Start dashboard dev server: `cd modules/dashboard-service && npm run dev`
Navigate to: `http://localhost:5180/config/translation`
Expected: Page loads showing connections manager with at least one default connection

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/(app)/config/translation/+page.svelte
git commit -m "feat: rewrite translation config page with connections manager"
```

---

## Chunk 6: Playwright E2E Tests

### Task 9: Create E2E tests for translation connections

**Files:**
- Create: `modules/dashboard-service/tests/e2e/translation-connections.spec.ts`

- [ ] **Step 1: Write the test file**

```typescript
import { test, expect } from '@playwright/test';

test.describe('Translation Connections', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/config/translation');
  });

  test('page loads and shows connections section', async ({ page }) => {
    await expect(page.getByText('Translation Connections')).toBeVisible();
    await expect(page.getByRole('button', { name: /add connection/i })).toBeVisible();
  });

  test('default connection is visible', async ({ page }) => {
    // Should show at least one connection card
    await expect(page.locator('[data-testid="connection-card"]').first()).toBeVisible();
  });

  test('add connection dialog opens and submits', async ({ page }) => {
    await page.getByRole('button', { name: /add connection/i }).click();

    // Dialog should open
    await expect(page.getByText('Add Connection')).toBeVisible();

    // Fill form
    await page.getByLabel('Name').fill('Test Ollama');
    // Engine defaults to ollama
    await page.getByLabel('URL').clear();
    await page.getByLabel('URL').fill('http://192.168.1.100:11434');
    await page.getByLabel('Prefix ID').fill('test-ollama');

    // Submit
    await page.getByRole('button', { name: /add connection/i }).last().click();

    // New card should appear
    await expect(page.getByText('Test Ollama')).toBeVisible();
    await expect(page.getByText('http://192.168.1.100:11434')).toBeVisible();
  });

  test('verify connection shows status', async ({ page }) => {
    // Click verify on first connection
    const firstVerify = page.getByRole('button', { name: /verify/i }).first();
    await firstVerify.click();

    // Should show either connected or error status (depending on backend availability)
    await expect(
      page.locator('.text-green-500, .text-red-500').first()
    ).toBeVisible({ timeout: 10000 });
  });

  test('edit connection via configure button', async ({ page }) => {
    // Click gear icon on first connection
    await page.locator('button:has(svg.lucide-settings)').first().click();

    // Dialog should open with pre-filled values
    await expect(page.getByText('Edit Connection')).toBeVisible();

    // Modify name
    const nameInput = page.getByLabel('Name');
    await nameInput.clear();
    await nameInput.fill('Renamed Connection');

    await page.getByRole('button', { name: /save changes/i }).click();

    // Card should show updated name
    await expect(page.getByText('Renamed Connection')).toBeVisible();
  });

  test('delete connection with confirmation', async ({ page }) => {
    // First add a connection so we have something to delete
    await page.getByRole('button', { name: /add connection/i }).click();
    await page.getByLabel('Name').fill('Temp Connection');
    await page.getByLabel('URL').clear();
    await page.getByLabel('URL').fill('http://temp:11434');
    await page.getByLabel('Prefix ID').fill('temp');
    await page.getByRole('button', { name: /add connection/i }).last().click();
    await expect(page.getByText('Temp Connection')).toBeVisible();

    // Click delete on that connection
    const deleteButtons = page.locator('button:has(svg.lucide-trash-2)');
    await deleteButtons.last().click();

    // Should be removed
    await expect(page.getByText('Temp Connection')).not.toBeVisible();
  });

  test('toggle connection enable/disable', async ({ page }) => {
    // Click power toggle on first connection
    const powerButton = page.locator('button:has(svg.lucide-power)').first();
    await powerButton.click();

    // Card should show dimmed (opacity-50)
    const firstCard = page.locator('[data-testid="connection-card"]').first();
    await expect(firstCard).toHaveClass(/opacity-50/);

    // Click again to re-enable
    await powerButton.click();
    await expect(firstCard).not.toHaveClass(/opacity-50/);
  });

  test('engine selection changes URL placeholder', async ({ page }) => {
    await page.getByRole('button', { name: /add connection/i }).click();

    // Change engine to vLLM
    await page.getByLabel('Engine').selectOption('vllm');

    // URL should update to vLLM default
    const urlInput = page.getByLabel('URL');
    await expect(urlInput).toHaveValue('http://localhost:8000');

    // Change to Triton
    await page.getByLabel('Engine').selectOption('triton');
    await expect(urlInput).toHaveValue('http://localhost:8001');
  });

  test('settings sections are visible', async ({ page }) => {
    // The existing settings sections should still render
    await expect(page.getByText('Current Model')).toBeVisible();
    await expect(page.getByText('Translation Settings')).toBeVisible();
    await expect(page.getByText('Prompt Template')).toBeVisible();
  });
});
```

- [ ] **Step 2: Run tests**

Run: `cd modules/dashboard-service && npx playwright test tests/e2e/translation-connections.spec.ts --headed`

Expected: Tests run against the dev server. Some verify tests may show "error" status if no translation backend is running — that's correct behavior (the UI should show the error state).

- [ ] **Step 3: Commit**

```bash
git add modules/dashboard-service/tests/e2e/translation-connections.spec.ts
git commit -m "test: add Playwright E2E tests for translation connections"
```

---

## Chunk 7: Integration Verification

### Task 10: Full round-trip verification

- [ ] **Step 1: Start both services**

```bash
# Terminal 1: Backend
cd /Users/thomaspatane/GitHub/personal/livetranslate
uv run python modules/orchestration-service/src/main_fastapi.py

# Terminal 2: Dashboard
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npm run dev
```

- [ ] **Step 2: Manual verification checklist**

Open `http://localhost:5180/config/translation` and verify:

1. Connections section visible with default connection
2. Click "+ Add Connection" → dialog opens
3. Add Ollama connection with prefix "test" → card appears
4. Click Verify → status updates (red if no Ollama running, which is expected)
5. Click gear → edit dialog opens pre-filled
6. Click power → toggles enable/disable
7. Click delete → card removed
8. Save Settings → reload page → connections persist
9. Switch Model dropdown shows models (if backend available)
10. Prompt template editor still works

- [ ] **Step 3: Run Playwright tests**

```bash
cd modules/dashboard-service && npx playwright test --headed
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete translation connections manager with E2E tests"
```
