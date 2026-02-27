<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';
  import { Badge } from '$lib/components/ui/badge';
  import { Textarea } from '$lib/components/ui/textarea';
  import { toastStore } from '$lib/stores/toast.svelte';
  import type { TranslationHealth, TranslationModel } from '$lib/types';

  let { data, form } = $props();

  let submitting = $state(false);

  // --- Section A: Current Model state ---
  let health: TranslationHealth | null = $state(null);
  let healthLoading = $state(false);
  let healthError: string | null = $state(null);

  // Sync SSR data into client state
  $effect(() => {
    if (data.translationHealth) {
      health = data.translationHealth;
    }
  });

  async function refreshHealth() {
    healthLoading = true;
    healthError = null;
    try {
      const res = await fetch('/api/translation/health');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      health = await res.json();
    } catch (err) {
      healthError = err instanceof Error && err.message === 'Failed to fetch'
        ? 'Connection error. Please check your network and try again.'
        : `Failed to fetch health: ${err instanceof Error ? err.message : 'Unknown error'}`;
    } finally {
      healthLoading = false;
    }
  }

  // --- Section B: Available Models state ---
  let models: TranslationModel[] = $derived(data.translationModels?.models ?? []);
  let selectedModelId = $state('');
  let switchLoading = $state(false);
  let switchMessage: string | null = $state(null);
  let switchError: string | null = $state(null);

  let selectedModelDisplay = $derived(
    models.find((m) => m.model === selectedModelId)
  );

  async function switchModel() {
    if (!selectedModelId) return;
    switchLoading = true;
    switchMessage = null;
    switchError = null;
    try {
      const res = await fetch('/api/translation/model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModelId })
      });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        const detail = body?.detail;
        throw new Error(detail ?? `HTTP ${res.status}`);
      }
      const result = await res.json();
      const msg = (result.message as string) ?? 'Model switched successfully';
      switchMessage = msg;
      toastStore.success(msg);
      await refreshHealth();
    } catch (err) {
      if (err instanceof TypeError && err.message === 'Failed to fetch') {
        switchError = 'Connection error. Please check your network and try again.';
        toastStore.error(switchError);
      } else {
        switchError = `Failed to switch model: ${err instanceof Error ? err.message : 'Unknown error'}`;
        toastStore.error(switchError);
      }
    } finally {
      switchLoading = false;
    }
  }

  // --- Section D: Prompt Template state ---
  const PROMPT_TEMPLATES: Record<string, string> = {
    simple: 'Translate the following to {target_language}:\n{current_sentence}',
    full: 'You are a professional translator. Translate to {target_language}.\n\nGlossary:\n{glossary_section}\n\nContext:\n{context_window}\n\nTranslate:\n{current_sentence}',
    minimal: '{target_language}: {current_sentence}'
  };

  const STORAGE_KEY = 'translation_prompt_template';
  const STYLE_STORAGE_KEY = 'translation_prompt_style';

  let promptStyle = $state('simple');
  let promptTemplate = $state(PROMPT_TEMPLATES.simple);
  let promptSaved = $state(false);

  $effect(() => {
    if (typeof window !== 'undefined') {
      const savedTemplate = localStorage.getItem(STORAGE_KEY);
      const savedStyle = localStorage.getItem(STYLE_STORAGE_KEY);
      if (savedStyle && savedStyle in PROMPT_TEMPLATES) {
        promptStyle = savedStyle;
      }
      if (savedTemplate) {
        promptTemplate = savedTemplate;
      } else {
        promptTemplate = PROMPT_TEMPLATES[promptStyle];
      }
    }
  });

  function onStyleChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    promptStyle = target.value;
    promptTemplate = PROMPT_TEMPLATES[promptStyle] ?? PROMPT_TEMPLATES.simple;
  }

  function savePrompt() {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY, promptTemplate);
      localStorage.setItem(STYLE_STORAGE_KEY, promptStyle);
      promptSaved = true;
      toastStore.success('Prompt template saved');
      setTimeout(() => { promptSaved = false; }, 2000);
    }
  }

  function resetPrompt() {
    promptTemplate = PROMPT_TEMPLATES[promptStyle] ?? PROMPT_TEMPLATES.simple;
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY);
      localStorage.removeItem(STYLE_STORAGE_KEY);
    }
    promptStyle = 'simple';
    promptTemplate = PROMPT_TEMPLATES.simple;
    toastStore.info('Prompt template reset to default');
  }

  // --- Status badge variant ---
  function statusVariant(status: string | undefined): 'default' | 'secondary' | 'destructive' | 'outline' {
    if (!status) return 'outline';
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'active':
      case 'ready':
        return 'default';
      case 'degraded':
      case 'loading':
        return 'secondary';
      case 'unavailable':
      case 'error':
      case 'down':
        return 'destructive';
      default:
        return 'outline';
    }
  }
</script>

<PageHeader title="Translation Configuration" description="Translation models, language settings, and prompt templates" />

<div class="space-y-6 max-w-3xl">

  <!-- Section A: Current Model Card -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Current Model</Card.Title>
      <Card.Action>
        <Button variant="outline" size="sm" onclick={refreshHealth} disabled={healthLoading}>
          {healthLoading ? 'Refreshing...' : 'Refresh'}
        </Button>
      </Card.Action>
    </Card.Header>
    <Card.Content>
      {#if healthError}
        <p class="text-sm text-destructive">{healthError}</p>
      {:else if health}
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div class="space-y-1">
            <p class="text-xs text-muted-foreground">Model</p>
            <p class="text-sm font-medium">{health.model}</p>
          </div>
          <div class="space-y-1">
            <p class="text-xs text-muted-foreground">Backend</p>
            <p class="text-sm font-medium">{health.backend}</p>
          </div>
          <div class="space-y-1">
            <p class="text-xs text-muted-foreground">Device</p>
            <p class="text-sm font-medium">{health.device}</p>
          </div>
          <div class="space-y-1">
            <p class="text-xs text-muted-foreground">Status</p>
            <Badge variant={statusVariant(health.status)}>{health.status}</Badge>
          </div>
        </div>
        {#if health.available_backends && health.available_backends.length > 0}
          <div class="mt-4 space-y-1">
            <p class="text-xs text-muted-foreground">Available Backends</p>
            <div class="flex gap-1.5 flex-wrap">
              {#each health.available_backends as backend}
                <Badge variant="outline">{backend}</Badge>
              {/each}
            </div>
          </div>
        {/if}
      {:else}
        <p class="text-sm text-muted-foreground">No health data available. Click Refresh to load.</p>
      {/if}
    </Card.Content>
  </Card.Root>

  <!-- Section B: Available Models -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Switch Model</Card.Title>
      <Card.Description>Select a model from the translation service to activate</Card.Description>
    </Card.Header>
    <Card.Content>
      {#if models.length > 0}
        <div class="space-y-4">
          <div class="space-y-2">
            <Label for="model-select">Available Models</Label>
            <select
              id="model-select"
              class="w-full rounded-md border bg-background px-3 py-2 text-sm"
              bind:value={selectedModelId}
            >
              <option value="" disabled>Select a model...</option>
              {#each models as model}
                <option value={model.model}>
                  {model.display_name} ({model.backend})
                </option>
              {/each}
            </select>
          </div>

          {#if selectedModelDisplay}
            <div class="rounded-md border bg-muted/50 p-3 text-sm space-y-1">
              <p><span class="text-muted-foreground">Name:</span> {selectedModelDisplay.display_name}</p>
              <p><span class="text-muted-foreground">Model ID:</span> {selectedModelDisplay.model}</p>
              <p><span class="text-muted-foreground">Backend:</span> {selectedModelDisplay.backend}</p>
            </div>
          {/if}

          <Button onclick={switchModel} disabled={!selectedModelId || switchLoading}>
            {switchLoading ? 'Switching...' : 'Switch Model'}
          </Button>

          {#if switchMessage}
            <p class="text-sm text-green-600">{switchMessage}</p>
          {/if}
          {#if switchError}
            <p class="text-sm text-destructive">{switchError}</p>
          {/if}
        </div>
      {:else}
        <p class="text-sm text-muted-foreground">
          No models available from the translation service. Ensure the service is running.
        </p>
      {/if}
    </Card.Content>
  </Card.Root>

  <!-- Section C: Translation Settings (form action) -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Translation Settings</Card.Title>
      <Card.Description>Default language, temperature, and token limits</Card.Description>
    </Card.Header>
    <Card.Content>
      <form method="POST" action="?/update" use:enhance={() => {
        submitting = true;
        return async ({ result, update }) => {
          await update();
          submitting = false;
          if (result.type === 'success') {
            toastStore.success('Translation settings saved');
          } else if (result.type === 'failure') {
            toastStore.error('Failed to save translation settings');
          }
        };
      }} class="space-y-4">
        <div class="space-y-2">
          <Label for="target_language">Default Target Language</Label>
          <select
            id="target_language"
            name="target_language"
            class="w-full rounded-md border bg-background px-3 py-2 text-sm"
          >
            {#if data.uiConfig?.languages}
              {#each data.uiConfig.languages as lang}
                <option
                  value={lang.code}
                  selected={data.translationConfig?.target_language === lang.code}
                >
                  {lang.name} ({lang.code})
                </option>
              {/each}
            {/if}
          </select>
        </div>

        <div class="space-y-2">
          <Label for="temperature">Temperature</Label>
          <div class="flex items-center gap-3">
            <input
              id="temperature"
              name="temperature"
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={data.translationConfig?.temperature ?? 0.3}
              class="flex-1"
              oninput={(e) => {
                const display = document.getElementById('temperature-display');
                if (display) display.textContent = (e.target as HTMLInputElement).value;
              }}
            />
            <span
              id="temperature-display"
              class="w-10 text-sm text-right tabular-nums"
            >
              {data.translationConfig?.temperature ?? 0.3}
            </span>
          </div>
        </div>

        <div class="space-y-2">
          <Label for="max_tokens">Max Tokens</Label>
          <Input
            id="max_tokens"
            name="max_tokens"
            type="number"
            min="64"
            max="4096"
            step="64"
            value={data.translationConfig?.max_tokens ?? 512}
          />
        </div>

        {#if form?.errors?.form}
          <p class="text-sm text-destructive">{form.errors.form}</p>
        {/if}
        {#if form?.success}
          <p class="text-sm text-green-600">Translation settings saved</p>
        {/if}

        <Button type="submit" disabled={submitting}>
          {#if submitting}Saving...{:else}Save Settings{/if}
        </Button>
      </form>
    </Card.Content>
  </Card.Root>

  <!-- Section D: Prompt Template Editor -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Prompt Template</Card.Title>
      <Card.Description>Customize the translation prompt sent to the model</Card.Description>
    </Card.Header>
    <Card.Content>
      <div class="space-y-4">
        <div class="space-y-2">
          <Label for="prompt-style">Template Style</Label>
          <select
            id="prompt-style"
            class="w-full rounded-md border bg-background px-3 py-2 text-sm"
            onchange={onStyleChange}
            value={promptStyle}
          >
            <option value="simple">Simple</option>
            <option value="full">Full</option>
            <option value="minimal">Minimal</option>
          </select>
        </div>

        <div class="space-y-2">
          <Label for="prompt-template">Template Content</Label>
          <Textarea
            id="prompt-template"
            bind:value={promptTemplate}
            rows={6}
            class="font-mono text-sm"
            placeholder="Enter your translation prompt template..."
          />
        </div>

        <div class="rounded-md border bg-muted/50 p-3">
          <p class="text-xs font-medium text-muted-foreground mb-1.5">Available Variables</p>
          <div class="flex flex-wrap gap-1.5">
            <Badge variant="secondary">{'{target_language}'}</Badge>
            <Badge variant="secondary">{'{current_sentence}'}</Badge>
            <Badge variant="secondary">{'{glossary_section}'}</Badge>
            <Badge variant="secondary">{'{context_window}'}</Badge>
          </div>
        </div>

        <div class="flex gap-2">
          <Button onclick={savePrompt}>Save Prompt</Button>
          <Button variant="outline" onclick={resetPrompt}>Reset to Default</Button>
        </div>

        {#if promptSaved}
          <p class="text-sm text-green-600">Prompt template saved to local storage</p>
        {/if}
      </div>
    </Card.Content>
  </Card.Root>
</div>
