<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import { Textarea } from '$lib/components/ui/textarea';
  import { toastStore } from '$lib/stores/toast.svelte';

  let { data, form } = $props();

  let submitting = $state(false);
</script>

<PageHeader title="Translation Test Bench" description="Test translation quality interactively" />

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
  <!-- Input -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Input</Card.Title>
    </Card.Header>
    <Card.Content>
      <form method="POST" action="?/translate" use:enhance={() => {
        submitting = true;
        return async ({ result, update }) => {
          await update();
          submitting = false;
          if (result.type === 'success') {
            toastStore.success('Translation complete');
          } else if (result.type === 'failure') {
            toastStore.error('Translation failed');
          }
        };
      }} class="space-y-4">
        <div class="space-y-2">
          <Label for="text">Text to translate</Label>
          <Textarea id="text" name="text" rows={6} placeholder="Enter text to translate..."
            value={form?.text ?? ''} required />
          {#if form?.errors?.text}
            <p class="text-sm text-destructive">{form.errors.text}</p>
          {/if}
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div class="space-y-2">
            <Label for="target_language">Target Language</Label>
            <select id="target_language" name="target_language" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
              {#if data.uiConfig?.languages}
                {#each data.uiConfig.languages as lang}
                  <option value={lang.code}>{lang.name}</option>
                {/each}
              {:else}
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              {/if}
            </select>
          </div>

          <div class="space-y-2">
            <Label for="service">Service</Label>
            <select id="service" name="service" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
              {#each data.models as model}
                <option value={model.backend_name ?? model.backend}>{model.backend_name ?? model.backend} — {model.name}</option>
              {/each}
              {#if data.models.length === 0}
                <option value="ollama">ollama</option>
              {/if}
            </select>
          </div>
        </div>

        {#if form?.errors?.form}
          <p class="text-sm text-destructive">{form.errors.form}</p>
        {/if}

        <Button type="submit" class="w-full" disabled={submitting}>
          {#if submitting}Translating...{:else}Translate{/if}
        </Button>
      </form>
    </Card.Content>
  </Card.Root>

  <!-- Output -->
  <Card.Root>
    <Card.Header>
      <Card.Title>Result</Card.Title>
    </Card.Header>
    <Card.Content>
      {#if form?.success && form?.result}
        <div class="space-y-4">
          <div class="p-4 bg-accent rounded-lg">
            <p class="text-sm font-medium">{form.result.translated_text}</p>
          </div>
          <div class="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
            <div>Confidence: <span class="text-foreground">{Math.round(form.result.confidence * 100)}%</span></div>
            <div>Time: <span class="text-foreground">{Math.round(form.result.processing_time * 1000)}ms</span></div>
            <div>Model: <span class="text-foreground">{form.result.model_used}</span></div>
            <div>Backend: <span class="text-foreground">{form.result.backend_used}</span></div>
            <div>Source: <span class="text-foreground">{form.result.source_language}</span></div>
            <div>Target: <span class="text-foreground">{form.result.target_language}</span></div>
          </div>
        </div>
      {:else}
        <p class="text-muted-foreground text-center py-8">Submit text to see translation results</p>
      {/if}
    </Card.Content>
  </Card.Root>
</div>
