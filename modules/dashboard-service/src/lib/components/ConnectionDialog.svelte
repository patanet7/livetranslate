<script lang="ts">
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import EyeIcon from '@lucide/svelte/icons/eye';
	import EyeOffIcon from '@lucide/svelte/icons/eye-off';
	import ChevronDownIcon from '@lucide/svelte/icons/chevron-down';
	import type { AIConnection } from '$lib/api/connections';

	type ConnectionFormData = AIConnection & { api_key: string };

	interface Props {
		open: boolean;
		connection: ConnectionFormData | null;
		onsave: (connection: Record<string, unknown>) => void;
		onclose: () => void;
	}

	let { open = $bindable(), connection, onsave, onclose }: Props = $props();

	const engineDefaults: Record<
		string,
		{ url: string; modelPlaceholder: string; helperText: string }
	> = {
		ollama: {
			url: 'http://localhost:11434',
			modelPlaceholder: 'llama2:7b',
			helperText: 'Ollama API serves models at /api/chat'
		},
		openai: {
			url: 'https://api.openai.com/v1',
			modelPlaceholder: 'gpt-4o',
			helperText: 'OpenAI API — requires API key'
		},
		anthropic: {
			url: 'https://api.anthropic.com',
			modelPlaceholder: 'claude-sonnet-4-20250514',
			helperText: 'Anthropic API — requires API key'
		},
		openai_compatible: {
			url: 'http://localhost:8000',
			modelPlaceholder: 'meta-llama/Llama-2-7b-chat-hf',
			helperText: 'Any OpenAI-compatible API (vLLM, Groq, etc.)'
		}
	};

	// Form state
	let name = $state('');
	let engine = $state<AIConnection['engine']>('ollama');
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
		engine = target.value as AIConnection['engine'];
		url = engineDefaults[engine]?.url ?? '';
	}

	function autoPrefix() {
		if (!prefix && name) {
			prefix = name
				.toLowerCase()
				.replace(/[^a-z0-9]+/g, '-')
				.replace(/-+$/, '');
		}
	}

	function handleSave() {
		const result: Record<string, unknown> = {
			name: name || 'Unnamed Connection',
			engine,
			url: url.replace(/\/+$/, ''),
			prefix: prefix || name.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
			enabled: connection?.enabled ?? true,
			timeout_ms,
			max_retries
		};
		if (api_key) result.api_key = api_key;
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
					<option value="openai">OpenAI</option>
					<option value="anthropic">Anthropic</option>
					<option value="openai_compatible">OpenAI Compatible (vLLM, Groq, etc.)</option>
				</select>
				<p class="text-xs text-muted-foreground">
					{engineDefaults[engine]?.helperText ?? ''}
				</p>
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
				<p class="text-xs text-muted-foreground">
					Prepended to model names for disambiguation (e.g. home-gpu/llama2:7b)
				</p>
			</div>

			<!-- API Key -->
			{#if engine === 'openai' || engine === 'anthropic' || engine === 'openai_compatible' || api_key}
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
							onclick={() => (showApiKey = !showApiKey)}
						>
							{#if showApiKey}
								<EyeOffIcon class="h-4 w-4" />
							{:else}
								<EyeIcon class="h-4 w-4" />
							{/if}
						</button>
					</div>
				</div>
			{/if}

			<!-- Advanced toggle -->
			<button
				type="button"
				class="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
				onclick={() => (showAdvanced = !showAdvanced)}
			>
				<ChevronDownIcon
					class="h-3 w-3 transition-transform {showAdvanced ? 'rotate-180' : ''}"
				/>
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
			<Button
				variant="outline"
				onclick={() => {
					open = false;
					onclose();
				}}>Cancel</Button
			>
			<Button onclick={handleSave}>
				{connection ? 'Save Changes' : 'Add Connection'}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
