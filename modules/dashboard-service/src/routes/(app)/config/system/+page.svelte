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

<PageHeader title="System Configuration" description="System preferences and feature flags" />

<Card.Root class="max-w-2xl">
  <Card.Content>
    <form method="POST" action="?/update" use:enhance={() => {
      submitting = true;
      return async ({ result, update }) => {
        await update();
        submitting = false;
        if (result.type === 'success') {
          toastStore.success('System settings saved');
        } else if (result.type === 'failure') {
          toastStore.error('Failed to save system settings');
        }
      };
    }} class="space-y-4">
      <div class="space-y-2">
        <Label for="theme">Theme</Label>
        <select id="theme" name="theme" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          <option value="dark" selected={data.settings?.theme === 'dark'}>Dark</option>
          <option value="light" selected={data.settings?.theme === 'light'}>Light</option>
        </select>
      </div>

      <div class="space-y-2">
        <Label for="language">Interface Language</Label>
        <select id="language" name="language" class="w-full rounded-md border bg-background px-3 py-2 text-sm">
          <option value="en" selected={data.settings?.language === 'en'}>English</option>
          <option value="es" selected={data.settings?.language === 'es'}>Spanish</option>
          <option value="fr" selected={data.settings?.language === 'fr'}>French</option>
        </select>
      </div>

      <div class="flex items-center justify-between">
        <Label for="notifications">Enable Notifications</Label>
        <input type="checkbox" id="notifications" name="notifications"
          checked={data.settings?.notifications ?? true} class="h-4 w-4" />
      </div>

      {#if form?.errors?.form}
        <p class="text-sm text-destructive">{form.errors.form}</p>
      {/if}
      {#if form?.success}
        <p class="text-sm text-green-600">System settings saved</p>
      {/if}

      <Button type="submit" disabled={submitting}>
        {#if submitting}Saving...{:else}Save{/if}
      </Button>
    </form>
  </Card.Content>
</Card.Root>
