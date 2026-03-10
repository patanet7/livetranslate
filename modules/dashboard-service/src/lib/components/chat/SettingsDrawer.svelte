<script lang="ts">
	import type { Provider, ModelInfo, ChatSettings } from '$lib/api/chat';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import * as Dialog from '$lib/components/ui/dialog';
	import { toastStore } from '$lib/stores/toast.svelte';

	interface Props {
		open: boolean;
		onclose: () => void;
	}

	let { open, onclose }: Props = $props();

	// ── State ──────────────────────────────────────────────────────

	let providers = $state<Provider[]>([]);
	let models = $state<ModelInfo[]>([]);
	let loading = $state(false);
	let saving = $state(false);

	let selectedProvider = $state('');
	let selectedModel = $state('');
	let temperature = $state(0.7);
	let maxTokens = $state(4096);
	let apiKey = $state('');
	let baseUrl = $state('');
	let hasApiKey = $state(false);

	// ── Derived ────────────────────────────────────────────────────

	let currentProvider = $derived(
		providers.find((p) => p.name === selectedProvider)
	);
	let needsApiKey = $derived(
		selectedProvider === 'openai' ||
			selectedProvider === 'anthropic' ||
			selectedProvider === 'openai_compatible'
	);
	let needsBaseUrl = $derived(selectedProvider === 'openai_compatible');

	// ── Effects ────────────────────────────────────────────────────

	$effect(() => {
		if (open) {
			loadSettings();
		}
	});

	$effect(() => {
		if (selectedProvider && open) {
			loadModels(selectedProvider);
		}
	});

	// ── Functions ──────────────────────────────────────────────────

	async function loadSettings() {
		loading = true;
		try {
			const [providersRes, settingsRes] = await Promise.all([
				fetch('/api/chat/providers'),
				fetch('/api/chat/settings')
			]);
			if (providersRes.ok) {
				providers = await providersRes.json();
			}
			if (settingsRes.ok) {
				const s: ChatSettings = await settingsRes.json();
				selectedProvider = s.provider;
				selectedModel = s.model ?? '';
				temperature = s.temperature;
				maxTokens = s.max_tokens;
				hasApiKey = s.has_api_key;
				baseUrl = s.base_url ?? '';
			}
		} catch {
			toastStore.error('Failed to load chat settings');
		} finally {
			loading = false;
		}
	}

	async function loadModels(provider: string) {
		try {
			const res = await fetch(`/api/chat/providers/${provider}/models`);
			if (res.ok) {
				models = await res.json();
			} else {
				models = [];
			}
		} catch {
			models = [];
		}
	}

	async function save() {
		saving = true;
		try {
			const body: Record<string, unknown> = {
				provider: selectedProvider,
				model: selectedModel || null,
				temperature,
				max_tokens: maxTokens
			};
			if (apiKey) body.api_key = apiKey;
			if (needsBaseUrl) body.base_url = baseUrl || null;

			const res = await fetch('/api/chat/settings', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});
			if (res.ok) {
				toastStore.success('Settings saved');
				apiKey = '';
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
				Configure the LLM provider and model for business insights chat.
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
				<!-- Provider -->
				<div class="space-y-2">
					<Label>Provider</Label>
					<Select.Root type="single" bind:value={selectedProvider}>
						<Select.Trigger class="w-full">
							{#if selectedProvider}
								{selectedProvider}
							{:else}
								Select provider
							{/if}
						</Select.Trigger>
						<Select.Content>
							{#each providers as p (p.name)}
								<Select.Item value={p.name} label={p.name}>
									<span class="flex items-center gap-2">
										<span
											class="inline-block size-2 rounded-full {p.configured
												? p.healthy
													? 'bg-green-500'
													: 'bg-yellow-500'
												: 'bg-gray-400'}"
										></span>
										{p.name}
									</span>
								</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				</div>

				<!-- Model -->
				<div class="space-y-2">
					<Label>Model</Label>
					<Select.Root type="single" bind:value={selectedModel}>
						<Select.Trigger class="w-full">
							{#if selectedModel}
								{selectedModel}
							{:else}
								Select model
							{/if}
						</Select.Trigger>
						<Select.Content>
							{#each models as m (m.id)}
								<Select.Item value={m.id} label={m.name}>
									<div>
										<div class="font-medium">{m.name}</div>
										{#if m.context_window}
											<div class="text-xs text-muted-foreground">
												{m.context_window.toLocaleString()} tokens
											</div>
										{/if}
									</div>
								</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
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
					<Input type="number" bind:value={maxTokens} min={256} max={32768} />
				</div>

				<!-- API Key (conditional) -->
				{#if needsApiKey}
					<div class="space-y-2">
						<Label>
							API Key
							{#if hasApiKey}
								<span class="text-xs text-muted-foreground ml-1">(configured)</span>
							{/if}
						</Label>
						<Input
							type="password"
							bind:value={apiKey}
							placeholder={hasApiKey ? 'Leave blank to keep current key' : 'Enter API key'}
						/>
					</div>
				{/if}

				<!-- Base URL (conditional) -->
				{#if needsBaseUrl}
					<div class="space-y-2">
						<Label>Base URL</Label>
						<Input
							bind:value={baseUrl}
							placeholder="https://api.example.com/v1"
						/>
					</div>
				{/if}
			</div>

			<Dialog.Footer>
				<Button variant="outline" onclick={onclose}>Cancel</Button>
				<Button disabled={saving || !selectedProvider} onclick={save}>
					{#if saving}
						Saving...
					{:else}
						Save Settings
					{/if}
				</Button>
			</Dialog.Footer>
		{/if}
	</Dialog.Content>
</Dialog.Root>
