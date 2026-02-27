<script lang="ts">
  import { enhance } from '$app/forms';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import * as Card from '$lib/components/ui/card';
  import { Button } from '$lib/components/ui/button';
  import { Label } from '$lib/components/ui/label';
  import { toastStore } from '$lib/stores/toast.svelte';

  let { data, form } = $props();

  let submitting = $state(false);
</script>

<PageHeader title="Audio Configuration" description="Audio processing settings" />

<Card.Root class="max-w-2xl">
  <Card.Content>
    <form method="POST" action="?/update" use:enhance={() => {
      submitting = true;
      return async ({ result, update }) => {
        await update();
        submitting = false;
        if (result.type === 'success') {
          toastStore.success('Audio settings saved');
        } else if (result.type === 'failure') {
          toastStore.error('Failed to save audio settings');
        }
      };
    }} class="space-y-4">
      <div class="flex items-center justify-between">
        <Label for="audio_auto_start">Auto-start audio capture</Label>
        <input
          type="checkbox"
          id="audio_auto_start"
          name="audio_auto_start"
          checked={data.settings?.audio_auto_start ?? false}
          class="h-4 w-4"
        />
      </div>

      {#if form?.errors?.form}
        <p class="text-sm text-destructive">{form.errors.form}</p>
      {/if}
      {#if form?.success}
        <p class="text-sm text-green-600">Settings saved</p>
      {/if}

      <Button type="submit" disabled={submitting}>
        {#if submitting}Saving...{:else}Save{/if}
      </Button>
    </form>
  </Card.Content>
</Card.Root>
