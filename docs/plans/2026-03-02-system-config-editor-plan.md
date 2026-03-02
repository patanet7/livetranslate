# System Config Editor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add an editable system configuration page that lets users enable/disable languages, CRUD domains, and override default config — persisted to a JSON overlay file.

**Architecture:** The existing `GET /api/system/ui-config` endpoint (in `system.py`) is extended with `PUT` and `POST /reset` endpoints. A JSON file (`./config/system.json`) stores user overrides that merge on top of the hardcoded `system_constants.py` values. The existing SvelteKit `/config/system` stub page is rebuilt with three tabbed sections (Languages, Domains, Defaults). The dashboard already has the sidebar link.

**Tech Stack:** FastAPI (Pydantic models, async JSON I/O), SvelteKit (Svelte 5 runes, Tabs/Card/Dialog UI components), existing `load_config`/`save_config` from `settings/_shared.py`

---

## Task 1: Add `SystemConfigUpdate` model and `PUT /api/system/ui-config` endpoint

**Files:**
- Modify: `src/routers/system.py:26-30` (add imports and model)
- Modify: `src/routers/system.py:350-421` (modify GET, add PUT and reset endpoints)

**Step 1: Add Pydantic model and imports to `system.py`**

At the top of `system.py` (after line 27 `router = APIRouter()`), add:

```python
from pathlib import Path

SYSTEM_CONFIG_PATH = Path("./config/system.json")
```

After the existing `# Request/Response Models` section (~line 35), add:

```python
class DomainItem(BaseModel):
    value: str
    label: str
    description: str = ""

class SystemConfigUpdate(BaseModel):
    enabled_languages: list[str] | None = None
    custom_domains: list[DomainItem] | None = None
    disabled_domains: list[str] | None = None
    defaults: dict[str, Any] | None = None
```

**Step 2: Import `load_config`/`save_config` from settings shared**

At the top imports section of `system.py`, add:

```python
from routers.settings._shared import load_config, save_config
```

**Step 3: Modify `GET /api/system/ui-config` to merge overrides**

Replace the return block at lines 406-421 with merge logic:

```python
    # Load user overrides
    overrides = await load_config(SYSTEM_CONFIG_PATH, {})

    # Filter languages if enabled_languages is set
    enabled_langs = overrides.get("enabled_languages")
    if enabled_langs:
        languages = [l for l in SUPPORTED_LANGUAGES if l["code"] in enabled_langs]
    else:
        languages = SUPPORTED_LANGUAGES

    # Merge domains: start with built-in, remove disabled, add custom
    domains = list(GLOSSARY_DOMAINS)
    disabled_domains = overrides.get("disabled_domains", [])
    domains = [d for d in domains if d["value"] not in disabled_domains]
    custom_domains = overrides.get("custom_domains", [])
    domains.extend(custom_domains)

    # Merge defaults
    defaults = {**DEFAULT_CONFIG, **overrides.get("defaults", {})}

    return {
        # Core configuration (merged)
        "languages": languages,
        "language_codes": [lang["code"] for lang in languages],
        "domains": domains,
        "defaults": defaults,
        "prompt_variables": PROMPT_TEMPLATE_VARIABLES,
        # Dynamic configuration (from services)
        "translation_models": translation_models,
        "translation_service_available": translation_service_available,
        "prompt_templates": prompt_templates,
        "prompts_available": prompts_available,
        # Override metadata
        "has_overrides": bool(overrides),
        "enabled_language_count": len(languages),
        "total_language_count": len(SUPPORTED_LANGUAGES),
        # Metadata
        "config_version": "1.0",
        "source": "orchestration-service",
    }
```

**Step 4: Add `PUT /api/system/ui-config` endpoint**

After the existing `get_ui_config` function, add:

```python
@router.put("/ui-config")
async def update_ui_config(config: SystemConfigUpdate):
    """
    Update system configuration overrides.

    Saves user customizations to ./config/system.json.
    These overrides merge on top of system_constants.py defaults.
    """
    from system_constants import VALID_LANGUAGE_CODES, VALID_DOMAINS

    # Validate language codes
    if config.enabled_languages is not None:
        invalid = [c for c in config.enabled_languages if c not in VALID_LANGUAGE_CODES]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid language codes: {invalid}",
            )

    # Validate disabled_domains reference existing built-in domains
    if config.disabled_domains is not None:
        invalid = [d for d in config.disabled_domains if d not in VALID_DOMAINS]
        if invalid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid domain values: {invalid}",
            )

    # Load existing overrides and merge
    existing = await load_config(SYSTEM_CONFIG_PATH, {})
    update_data = config.model_dump(exclude_none=True)

    # Convert custom_domains from Pydantic models to dicts
    if "custom_domains" in update_data:
        update_data["custom_domains"] = [d.model_dump() if hasattr(d, "model_dump") else d for d in update_data["custom_domains"]]

    merged = {**existing, **update_data}

    success = await save_config(SYSTEM_CONFIG_PATH, merged)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save configuration",
        )

    return {"status": "ok", "message": "Configuration saved"}
```

**Step 5: Add `POST /api/system/ui-config/reset` endpoint**

```python
@router.post("/ui-config/reset")
async def reset_ui_config():
    """
    Reset system configuration to factory defaults.

    Deletes the override file, restoring system_constants.py values.
    """
    SYSTEM_CONFIG_PATH.unlink(missing_ok=True)
    logger.info("System configuration reset to factory defaults")
    return {"status": "ok", "message": "Reset to factory defaults"}
```

**Step 6: Verify the server starts and endpoints work**

Run:
```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service
uv run python -c "from routers.system import router; print(f'Routes: {len(router.routes)}')"
```
Expected: No import errors, route count increased

**Step 7: Commit**

```bash
git add src/routers/system.py
git commit -m "feat: add PUT/reset endpoints for system config overrides"
```

---

## Task 2: Add `updateUiConfig` and `resetUiConfig` to dashboard API client

**Files:**
- Modify: `modules/dashboard-service/src/lib/api/config.ts:30` (add methods)
- Modify: `modules/dashboard-service/src/lib/types/config.ts:28-41` (add types)

**Step 1: Add `SystemConfigUpdate` type to `config.ts`**

In `modules/dashboard-service/src/lib/types/config.ts`, after the `UiConfig` interface (line 41), add:

```typescript
export interface DomainItem {
	value: string;
	label: string;
	description?: string;
}

export interface SystemConfigUpdate {
	enabled_languages?: string[];
	custom_domains?: DomainItem[];
	disabled_domains?: string[];
	defaults?: Record<string, unknown>;
}
```

**Step 2: Add API methods to `config.ts`**

In `modules/dashboard-service/src/lib/api/config.ts`, after the `getUiConfig` method (line 30), add:

```typescript
		updateUiConfig: (config: SystemConfigUpdate) =>
			api.put<{ status: string; message: string }>('/api/system/ui-config', config),

		resetUiConfig: () =>
			api.post<{ status: string; message: string }>('/api/system/ui-config/reset'),
```

Also add `SystemConfigUpdate` to the import at the top:

```typescript
import type {
	UserSettings,
	TranslationSettings,
	UiConfig,
	SystemConfigUpdate,
	TranslationModelsResponse,
	TranslationHealth
} from '$lib/types';
```

**Step 3: Export new type from types index**

In `modules/dashboard-service/src/lib/types/index.ts`, ensure `SystemConfigUpdate` and `DomainItem` are exported. Check the file and add if needed.

**Step 4: Verify TypeScript compiles**

Run:
```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npx svelte-check --threshold error 2>&1 | tail -5
```
Expected: 0 errors

**Step 5: Commit**

```bash
git add modules/dashboard-service/src/lib/types/config.ts modules/dashboard-service/src/lib/api/config.ts modules/dashboard-service/src/lib/types/index.ts
git commit -m "feat: add system config API client methods and types"
```

---

## Task 3: Create `+page.server.ts` for `/config/system` page

**Files:**
- Create: `modules/dashboard-service/src/routes/(app)/config/system/+page.server.ts`

**Step 1: Write the server load and actions**

```typescript
import { fail } from '@sveltejs/kit';
import type { Actions, PageServerLoad } from './$types';
import { configApi } from '$lib/api/config';

export const load: PageServerLoad = async ({ fetch }) => {
	const cfg = configApi(fetch);

	const [uiConfig] = await Promise.all([
		cfg.getUiConfig().catch(() => null)
	]);

	// Also fetch the raw overrides to know what the user has customized
	// (The uiConfig is already merged — we need the raw overrides for the form state)
	let overrides: Record<string, unknown> = {};
	try {
		const res = await fetch(
			`${process.env.ORCHESTRATION_URL || 'http://localhost:3001'}/api/system/ui-config`
		);
		if (res.ok) {
			const data = await res.json();
			// The GET response now includes has_overrides and total counts
			overrides = {
				has_overrides: data.has_overrides ?? false,
				enabled_language_count: data.enabled_language_count ?? 0,
				total_language_count: data.total_language_count ?? 0
			};
		}
	} catch {
		// ignore
	}

	return { uiConfig, overrides };
};

export const actions: Actions = {
	updateLanguages: async ({ request, fetch }) => {
		const data = await request.formData();
		const enabled = data.getAll('languages') as string[];

		const cfg = configApi(fetch);
		try {
			await cfg.updateUiConfig({ enabled_languages: enabled.length > 0 ? enabled : undefined });
			return { success: true, section: 'languages' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to save languages: ${err}` } });
		}
	},

	updateDomains: async ({ request, fetch }) => {
		const data = await request.formData();
		const customDomainsJson = data.get('custom_domains') as string;
		const disabledDomainsJson = data.get('disabled_domains') as string;

		const cfg = configApi(fetch);
		try {
			await cfg.updateUiConfig({
				custom_domains: customDomainsJson ? JSON.parse(customDomainsJson) : [],
				disabled_domains: disabledDomainsJson ? JSON.parse(disabledDomainsJson) : []
			});
			return { success: true, section: 'domains' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to save domains: ${err}` } });
		}
	},

	updateDefaults: async ({ request, fetch }) => {
		const data = await request.formData();
		const defaults: Record<string, unknown> = {};

		const source = data.get('default_source_language') as string;
		if (source) defaults.default_source_language = source;

		const targets = data.getAll('default_target_languages') as string[];
		if (targets.length > 0) defaults.default_target_languages = targets;

		defaults.auto_detect_language = data.get('auto_detect_language') === 'on';

		const threshold = data.get('confidence_threshold');
		if (threshold) defaults.confidence_threshold = parseFloat(threshold as string);

		const contextWindow = data.get('context_window_size');
		if (contextWindow) defaults.context_window_size = parseInt(contextWindow as string);

		const maxBuffer = data.get('max_buffer_words');
		if (maxBuffer) defaults.max_buffer_words = parseInt(maxBuffer as string);

		const pauseThreshold = data.get('pause_threshold_ms');
		if (pauseThreshold) defaults.pause_threshold_ms = parseInt(pauseThreshold as string);

		const cfg = configApi(fetch);
		try {
			await cfg.updateUiConfig({ defaults });
			return { success: true, section: 'defaults' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to save defaults: ${err}` } });
		}
	},

	reset: async ({ fetch }) => {
		const cfg = configApi(fetch);
		try {
			await cfg.resetUiConfig();
			return { success: true, section: 'reset' };
		} catch (err) {
			return fail(500, { errors: { form: `Failed to reset: ${err}` } });
		}
	}
};
```

**Step 2: Verify TypeScript compiles**

Run:
```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npx svelte-check --threshold error 2>&1 | tail -5
```
Expected: 0 errors

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/(app)/config/system/+page.server.ts
git commit -m "feat: add server load/actions for system config page"
```

---

## Task 4: Rebuild `/config/system` page — Languages tab

**Files:**
- Modify: `modules/dashboard-service/src/routes/(app)/config/system/+page.svelte`

This task replaces the existing stub with the full three-tabbed config editor. We'll implement the Languages tab first.

**Step 1: Write the page scaffold with Tabs and Languages section**

Replace the entire content of `+page.svelte` with:

```svelte
<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import * as Tabs from '$lib/components/ui/tabs';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import { Input } from '$lib/components/ui/input';
  import * as Dialog from '$lib/components/ui/dialog';
  import { toastStore } from '$lib/stores/toast.svelte';
  import type { DomainItem } from '$lib/types';

  let { data, form } = $props();

  // --- Languages state ---
  // All 51 languages from system_constants (GET returns the merged/filtered list,
  // but we need the FULL list to show checkboxes. We'll use total_language_count
  // from overrides to detect if filtering is active.)
  let allLanguages = $derived(data.uiConfig?.languages ?? []);
  let enabledCodes = $state<Set<string>>(new Set());

  // Initialize enabled codes from data
  $effect(() => {
    if (data.uiConfig?.language_codes) {
      enabledCodes = new Set(data.uiConfig.language_codes);
    }
  });

  function toggleLanguage(code: string) {
    const next = new Set(enabledCodes);
    if (next.has(code)) next.delete(code);
    else next.add(code);
    enabledCodes = next;
  }

  function selectAllLanguages() {
    enabledCodes = new Set(allLanguages.map((l: { code: string }) => l.code));
  }

  function deselectAllLanguages() {
    // Always keep English
    enabledCodes = new Set(['en']);
  }

  // --- Domains state ---
  let builtinDomains = $derived(
    (data.uiConfig?.domains ?? []).filter((d: DomainItem) => !d._custom)
  );
  let customDomains = $state<DomainItem[]>([]);
  let disabledDomains = $state<Set<string>>(new Set());
  let addDomainOpen = $state(false);
  let newDomain = $state<DomainItem>({ value: '', label: '', description: '' });

  function addCustomDomain() {
    if (!newDomain.value || !newDomain.label) return;
    customDomains = [...customDomains, { ...newDomain }];
    newDomain = { value: '', label: '', description: '' };
    addDomainOpen = false;
  }

  function removeCustomDomain(index: number) {
    customDomains = customDomains.filter((_, i) => i !== index);
  }

  function toggleBuiltinDomain(value: string) {
    const next = new Set(disabledDomains);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    disabledDomains = next;
  }

  // --- Defaults state ---
  let defaults = $state<Record<string, unknown>>({});

  $effect(() => {
    if (data.uiConfig?.defaults) {
      defaults = { ...data.uiConfig.defaults };
    }
  });

  // --- Form submission ---
  let submitting = $state(false);

  function handleSubmitResult(section: string) {
    return () => {
      submitting = true;
      return async ({ result, update }: { result: { type: string }; update: () => Promise<void> }) => {
        await update();
        submitting = false;
        if (result.type === 'success') {
          toastStore.success(`${section} saved successfully`);
        } else if (result.type === 'failure') {
          toastStore.error(`Failed to save ${section}`);
        }
      };
    };
  }
</script>

<PageHeader
  title="System Configuration"
  description="Manage languages, domains, and default settings"
/>

{#if !data.uiConfig}
  <Card.Root>
    <Card.Content>
      <p class="text-muted-foreground">Unable to load system configuration. Is the orchestration service running?</p>
    </Card.Content>
  </Card.Root>
{:else}
  <Tabs.Root value="languages" class="w-full">
    <Tabs.List class="mb-4">
      <Tabs.Trigger value="languages">Languages ({allLanguages.length})</Tabs.Trigger>
      <Tabs.Trigger value="domains">Domains</Tabs.Trigger>
      <Tabs.Trigger value="defaults">Defaults</Tabs.Trigger>
    </Tabs.List>

    <!-- ═══════════ LANGUAGES TAB ═══════════ -->
    <Tabs.Content value="languages">
      <Card.Root>
        <Card.Header>
          <Card.Title>Enabled Languages</Card.Title>
          <Card.Description>
            Select which languages are available for translation. Disabled languages won't appear in language selectors across the app.
          </Card.Description>
        </Card.Header>
        <Card.Content>
          <form method="POST" action="?/updateLanguages" use:enhance={handleSubmitResult('Languages')}>
            <div class="flex gap-2 mb-4">
              <Button type="button" variant="outline" size="sm" onclick={selectAllLanguages}>
                Select All
              </Button>
              <Button type="button" variant="outline" size="sm" onclick={deselectAllLanguages}>
                Deselect All
              </Button>
              <span class="text-sm text-muted-foreground self-center ml-auto">
                {enabledCodes.size} of {allLanguages.length} enabled
              </span>
            </div>

            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
              {#each allLanguages as lang (lang.code)}
                <label
                  class="flex items-center gap-2 p-2 rounded-md border cursor-pointer transition-colors
                    {enabledCodes.has(lang.code)
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:border-muted-foreground/50'}"
                >
                  <input
                    type="checkbox"
                    name="languages"
                    value={lang.code}
                    checked={enabledCodes.has(lang.code)}
                    onchange={() => toggleLanguage(lang.code)}
                    class="h-4 w-4"
                  />
                  <div class="min-w-0">
                    <div class="text-sm font-medium truncate">
                      {lang.name}
                      {#if lang.rtl}<span class="text-xs text-muted-foreground ml-1">RTL</span>{/if}
                    </div>
                    <div class="text-xs text-muted-foreground truncate">{lang.native}</div>
                  </div>
                  <span class="text-xs text-muted-foreground ml-auto font-mono">{lang.code}</span>
                </label>
              {/each}
            </div>

            {#if form?.errors?.form}
              <p class="text-sm text-destructive mt-4">{form.errors.form}</p>
            {/if}

            <div class="flex justify-end mt-4">
              <Button type="submit" disabled={submitting}>
                {#if submitting}Saving...{:else}Save Languages{/if}
              </Button>
            </div>
          </form>
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- ═══════════ DOMAINS TAB ═══════════ -->
    <Tabs.Content value="domains">
      <Card.Root>
        <Card.Header>
          <Card.Title>Glossary Domains</Card.Title>
          <Card.Description>
            Manage domain categories for glossary organization. Built-in domains can be disabled but not deleted.
          </Card.Description>
        </Card.Header>
        <Card.Content>
          <form method="POST" action="?/updateDomains" use:enhance={handleSubmitResult('Domains')}>
            <input type="hidden" name="custom_domains" value={JSON.stringify(customDomains)} />
            <input type="hidden" name="disabled_domains" value={JSON.stringify([...disabledDomains])} />

            <h3 class="text-sm font-medium mb-2">Built-in Domains</h3>
            <div class="space-y-1 mb-6">
              {#each data.uiConfig.domains.filter((d: DomainItem) => !customDomains.some(c => c.value === d.value)) as domain (domain.value)}
                <div class="flex items-center justify-between p-2 rounded-md border">
                  <div>
                    <span class="text-sm font-medium">{domain.label}</span>
                    {#if domain.description}
                      <span class="text-xs text-muted-foreground ml-2">{domain.description}</span>
                    {/if}
                  </div>
                  {#if domain.value !== ''}
                    <label class="flex items-center gap-2 cursor-pointer">
                      <span class="text-xs text-muted-foreground">
                        {disabledDomains.has(domain.value) ? 'Disabled' : 'Enabled'}
                      </span>
                      <input
                        type="checkbox"
                        checked={!disabledDomains.has(domain.value)}
                        onchange={() => toggleBuiltinDomain(domain.value)}
                        class="h-4 w-4"
                      />
                    </label>
                  {:else}
                    <span class="text-xs text-muted-foreground">Always enabled</span>
                  {/if}
                </div>
              {/each}
            </div>

            <div class="flex items-center justify-between mb-2">
              <h3 class="text-sm font-medium">Custom Domains</h3>
              <Button type="button" variant="outline" size="sm" onclick={() => (addDomainOpen = true)}>
                + Add Domain
              </Button>
            </div>
            {#if customDomains.length === 0}
              <p class="text-sm text-muted-foreground p-4 text-center border rounded-md border-dashed">
                No custom domains added yet.
              </p>
            {:else}
              <div class="space-y-1">
                {#each customDomains as domain, i (domain.value)}
                  <div class="flex items-center justify-between p-2 rounded-md border">
                    <div>
                      <span class="text-sm font-medium">{domain.label}</span>
                      <span class="text-xs text-muted-foreground ml-2">{domain.description}</span>
                      <span class="text-xs font-mono text-muted-foreground ml-2">({domain.value})</span>
                    </div>
                    <Button type="button" variant="ghost" size="sm" onclick={() => removeCustomDomain(i)}>
                      Remove
                    </Button>
                  </div>
                {/each}
              </div>
            {/if}

            {#if form?.errors?.form}
              <p class="text-sm text-destructive mt-4">{form.errors.form}</p>
            {/if}

            <div class="flex justify-end mt-4">
              <Button type="submit" disabled={submitting}>
                {#if submitting}Saving...{:else}Save Domains{/if}
              </Button>
            </div>
          </form>
        </Card.Content>
      </Card.Root>
    </Tabs.Content>

    <!-- ═══════════ DEFAULTS TAB ═══════════ -->
    <Tabs.Content value="defaults">
      <Card.Root>
        <Card.Header>
          <Card.Title>Default Settings</Card.Title>
          <Card.Description>
            Override default values for translation configuration. These apply to new sessions and when no explicit setting is provided.
          </Card.Description>
        </Card.Header>
        <Card.Content>
          <form method="POST" action="?/updateDefaults" use:enhance={handleSubmitResult('Defaults')} class="space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="space-y-2">
                <Label for="default_source_language">Source Language</Label>
                <select
                  id="default_source_language"
                  name="default_source_language"
                  class="w-full rounded-md border bg-background px-3 py-2 text-sm"
                >
                  {#each allLanguages as lang (lang.code)}
                    <option value={lang.code} selected={defaults.default_source_language === lang.code}>
                      {lang.name} ({lang.code})
                    </option>
                  {/each}
                </select>
              </div>

              <div class="space-y-2">
                <Label for="default_target_languages">Target Languages</Label>
                <select
                  id="default_target_languages"
                  name="default_target_languages"
                  multiple
                  class="w-full rounded-md border bg-background px-3 py-2 text-sm min-h-[100px]"
                >
                  {#each allLanguages as lang (lang.code)}
                    <option
                      value={lang.code}
                      selected={Array.isArray(defaults.default_target_languages)
                        && (defaults.default_target_languages as string[]).includes(lang.code)}
                    >
                      {lang.name}
                    </option>
                  {/each}
                </select>
                <p class="text-xs text-muted-foreground">Hold Ctrl/Cmd to select multiple</p>
              </div>

              <div class="flex items-center justify-between col-span-full">
                <Label for="auto_detect_language">Auto-detect source language</Label>
                <input
                  type="checkbox"
                  id="auto_detect_language"
                  name="auto_detect_language"
                  checked={defaults.auto_detect_language !== false}
                  class="h-4 w-4"
                />
              </div>

              <div class="space-y-2">
                <Label for="confidence_threshold">Confidence Threshold</Label>
                <div class="flex items-center gap-3">
                  <input
                    type="range"
                    id="confidence_threshold"
                    name="confidence_threshold"
                    min="0"
                    max="1"
                    step="0.05"
                    value={defaults.confidence_threshold ?? 0.8}
                    oninput={(e) => defaults = {...defaults, confidence_threshold: parseFloat((e.target as HTMLInputElement).value)}}
                    class="flex-1"
                  />
                  <span class="text-sm font-mono w-12 text-right">{defaults.confidence_threshold ?? 0.8}</span>
                </div>
              </div>

              <div class="space-y-2">
                <Label for="context_window_size">Context Window Size</Label>
                <Input
                  type="number"
                  id="context_window_size"
                  name="context_window_size"
                  value={String(defaults.context_window_size ?? 3)}
                  min="1"
                  max="20"
                />
                <p class="text-xs text-muted-foreground">Number of previous sentences for context</p>
              </div>

              <div class="space-y-2">
                <Label for="max_buffer_words">Max Buffer Words</Label>
                <Input
                  type="number"
                  id="max_buffer_words"
                  name="max_buffer_words"
                  value={String(defaults.max_buffer_words ?? 50)}
                  min="10"
                  max="200"
                />
              </div>

              <div class="space-y-2">
                <Label for="pause_threshold_ms">Pause Threshold (ms)</Label>
                <Input
                  type="number"
                  id="pause_threshold_ms"
                  name="pause_threshold_ms"
                  value={String(defaults.pause_threshold_ms ?? 500)}
                  min="100"
                  max="5000"
                  step="100"
                />
              </div>
            </div>

            {#if form?.errors?.form}
              <p class="text-sm text-destructive">{form.errors.form}</p>
            {/if}

            <div class="flex justify-end">
              <Button type="submit" disabled={submitting}>
                {#if submitting}Saving...{:else}Save Defaults{/if}
              </Button>
            </div>
          </form>
        </Card.Content>
      </Card.Root>

      <!-- Reset Section -->
      <Card.Root class="mt-4 border-destructive/50">
        <Card.Content class="pt-6">
          <form method="POST" action="?/reset" use:enhance={handleSubmitResult('Config reset')}>
            <div class="flex items-center justify-between">
              <div>
                <p class="text-sm font-medium">Reset to Factory Defaults</p>
                <p class="text-xs text-muted-foreground">
                  Removes all customizations and restores the built-in configuration.
                </p>
              </div>
              <Button type="submit" variant="destructive" size="sm" disabled={submitting}>
                Reset All
              </Button>
            </div>
          </form>
        </Card.Content>
      </Card.Root>
    </Tabs.Content>
  </Tabs.Root>

  <!-- Add Domain Dialog -->
  <Dialog.Root bind:open={addDomainOpen}>
    <Dialog.Content>
      <Dialog.Header>
        <Dialog.Title>Add Custom Domain</Dialog.Title>
        <Dialog.Description>
          Create a new glossary domain category for organizing terminology.
        </Dialog.Description>
      </Dialog.Header>
      <div class="space-y-3 py-4">
        <div class="space-y-2">
          <Label for="domain_value">Value (slug)</Label>
          <Input
            id="domain_value"
            placeholder="e.g. automotive"
            bind:value={newDomain.value}
            oninput={(e) => newDomain = {...newDomain, value: (e.target as HTMLInputElement).value.toLowerCase().replace(/[^a-z0-9_-]/g, '')}}
          />
          <p class="text-xs text-muted-foreground">Lowercase, no spaces. Used as identifier.</p>
        </div>
        <div class="space-y-2">
          <Label for="domain_label">Label</Label>
          <Input id="domain_label" placeholder="e.g. Automotive" bind:value={newDomain.label} />
        </div>
        <div class="space-y-2">
          <Label for="domain_desc">Description</Label>
          <Input id="domain_desc" placeholder="e.g. Vehicle and transportation terms" bind:value={newDomain.description} />
        </div>
      </div>
      <Dialog.Footer>
        <Button variant="outline" onclick={() => (addDomainOpen = false)}>Cancel</Button>
        <Button onclick={addCustomDomain} disabled={!newDomain.value || !newDomain.label}>
          Add Domain
        </Button>
      </Dialog.Footer>
    </Dialog.Content>
  </Dialog.Root>
{/if}
```

**Step 2: Verify the page renders**

Run:
```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npx svelte-check --threshold error 2>&1 | tail -5
```
Expected: 0 errors

**Step 3: Commit**

```bash
git add modules/dashboard-service/src/routes/(app)/config/system/+page.svelte
git commit -m "feat: rebuild system config page with languages, domains, and defaults tabs"
```

---

## Task 5: Integration test — verify end-to-end flow

**Files:**
- No new files — verification only

**Step 1: Start orchestration service and verify GET returns merged data**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service
curl -s http://localhost:3001/api/system/ui-config | python3 -m json.tool | head -20
```
Expected: JSON with `languages`, `domains`, `defaults`, `has_overrides: false`

**Step 2: Test PUT — enable only 5 languages**

```bash
curl -s -X PUT http://localhost:3001/api/system/ui-config \
  -H "Content-Type: application/json" \
  -d '{"enabled_languages": ["en", "es", "fr", "de", "zh"]}' | python3 -m json.tool
```
Expected: `{"status": "ok", "message": "Configuration saved"}`

**Step 3: Verify GET reflects the change**

```bash
curl -s http://localhost:3001/api/system/ui-config | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Languages: {len(d[\"languages\"])}, has_overrides: {d[\"has_overrides\"]}')"
```
Expected: `Languages: 5, has_overrides: True`

**Step 4: Test reset**

```bash
curl -s -X POST http://localhost:3001/api/system/ui-config/reset | python3 -m json.tool
```
Expected: `{"status": "ok", "message": "Reset to factory defaults"}`

**Step 5: Verify GET returns all languages again**

```bash
curl -s http://localhost:3001/api/system/ui-config | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Languages: {len(d[\"languages\"])}, has_overrides: {d[\"has_overrides\"]}')"
```
Expected: `Languages: 51, has_overrides: False`

**Step 6: Run unit tests to verify no regressions**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service
uv run pytest tests/fireflies/unit/ -x -q 2>&1 | tail -5
```
Expected: All pass

**Step 7: Run svelte-check**

```bash
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npx svelte-check --threshold error 2>&1 | tail -5
```
Expected: 0 errors

**Step 8: Commit (if any fixes were needed)**

```bash
git add -A
git commit -m "fix: integration fixes for system config editor"
```

---

## Verification

After all tasks, the full verification is:

```bash
# 1. Backend — endpoints work
curl -s http://localhost:3001/api/system/ui-config | python3 -c "import sys,json; d=json.load(sys.stdin); print('GET ok:', len(d['languages']), 'languages')"

# 2. Backend — PUT works
curl -s -X PUT http://localhost:3001/api/system/ui-config -H "Content-Type: application/json" -d '{"enabled_languages":["en","es"]}' | python3 -m json.tool

# 3. Backend — Reset works
curl -s -X POST http://localhost:3001/api/system/ui-config/reset | python3 -m json.tool

# 4. Unit tests
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/orchestration-service
uv run pytest tests/fireflies/unit/ -x -q

# 5. TypeScript compilation
cd /Users/thomaspatane/GitHub/personal/livetranslate/modules/dashboard-service
npx svelte-check --threshold error

# 6. Visual — open http://localhost:5173/config/system in browser
```

## Files Summary

| Action | File |
|--------|------|
| Modify | `modules/orchestration-service/src/routers/system.py` |
| Modify | `modules/dashboard-service/src/lib/types/config.ts` |
| Modify | `modules/dashboard-service/src/lib/api/config.ts` |
| Create | `modules/dashboard-service/src/routes/(app)/config/system/+page.server.ts` |
| Modify | `modules/dashboard-service/src/routes/(app)/config/system/+page.svelte` |
