<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Label } from '$lib/components/ui/label';

  let { data, form } = $props();
</script>

<PageHeader title="Translation Configuration" description="Translation models and language settings" />

<Card.Root class="max-w-2xl">
  <Card.Content>
    <form method="POST" action="?/update" use:enhance class="space-y-4">
      <div class="space-y-2">
        <Label for="backend">Backend</Label>
        <select id="backend" name="backend" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          {#each ['ollama', 'vllm', 'openai', 'groq'] as b}
            <option value={b} selected={data.translationConfig?.backend === b}>{b}</option>
          {/each}
        </select>
      </div>

      <div class="space-y-2">
        <Label for="model">Model</Label>
        <Input id="model" name="model" value={data.translationConfig?.model ?? 'qwen2.5:3b'} />
      </div>

      <div class="space-y-2">
        <Label for="target_language">Default Target Language</Label>
        <select id="target_language" name="target_language" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          {#if data.uiConfig?.languages}
            {#each data.uiConfig.languages as lang}
              <option value={lang.code} selected={data.translationConfig?.target_language === lang.code}>
                {lang.name} ({lang.code})
              </option>
            {/each}
          {/if}
        </select>
      </div>

      <div class="space-y-2">
        <Label for="temperature">Temperature</Label>
        <Input id="temperature" name="temperature" type="number" step="0.1" min="0" max="2"
          value={data.translationConfig?.temperature ?? 0.3} />
      </div>

      {#if form?.errors?.form}
        <p class="text-sm text-destructive">{form.errors.form}</p>
      {/if}
      {#if form?.success}
        <p class="text-sm text-green-600">Translation config saved</p>
      {/if}

      <Button type="submit">Save</Button>
    </form>
  </Card.Content>
</Card.Root>
