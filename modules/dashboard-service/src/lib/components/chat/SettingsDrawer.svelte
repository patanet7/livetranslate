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
				fetch('/api/connections/preferences/all')
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
					max_tokens: maxTokens
				})
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
				<div
					class="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent"
				></div>
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
							No models available. <a href="/config/connections" class="underline"
								>Add a connection</a
							> first.
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
