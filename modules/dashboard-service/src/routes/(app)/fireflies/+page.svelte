<script lang="ts">
	import { enhance } from '$app/forms';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { toastStore } from '$lib/stores/toast.svelte';

	let { data, form } = $props();

	let submitting = $state(false);
	let selectedLanguages = $state<Set<string>>(new Set());
	let selectedModel = $state('');
	let initialized = $state(false);

	$effect(() => {
		if (initialized) return;
		const defaults = data.uiConfig?.defaults?.default_target_languages;
		if (Array.isArray(defaults)) {
			selectedLanguages = new Set(defaults as string[]);
		}
		const defaultModelEntry = data.uiConfig?.translation_models?.find((m) => m.default);
		if (defaultModelEntry) {
			selectedModel = defaultModelEntry.name;
		}
		initialized = true;
	});

	function toggleLanguage(code: string) {
		if (selectedLanguages.has(code)) {
			selectedLanguages.delete(code);
		} else {
			selectedLanguages.add(code);
		}
		selectedLanguages = new Set(selectedLanguages);
	}
</script>

<PageHeader
	title="Fireflies"
	description="Connect to a live Fireflies transcript for real-time translation"
/>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
	<!-- Connect Form -->
	<div class="lg:col-span-2">
		<Card.Root>
			<Card.Header>
				<Card.Title>Connect to Transcript</Card.Title>
			</Card.Header>
			<Card.Content>
				<form method="POST" action="?/connect" use:enhance={() => {
				submitting = true;
				return async ({ result, update }) => {
					await update();
					submitting = false;
					if (result.type === 'redirect') {
						toastStore.success('Connected to Fireflies transcript');
					} else if (result.type === 'failure') {
						toastStore.error('Failed to connect to transcript');
					}
				};
			}} class="space-y-4">
					<div class="space-y-2">
						<Label for="transcript_id">Transcript ID</Label>
						<Input
							id="transcript_id"
							name="transcript_id"
							placeholder="Enter Fireflies transcript ID"
							value={form?.transcript_id ?? ''}
							required
						/>
						{#if form?.errors?.transcript_id}
							<p class="text-sm text-destructive">{form.errors.transcript_id}</p>
						{/if}
					</div>

					<div class="space-y-2">
						<Label for="api_key">API Key (optional)</Label>
						<Input
							id="api_key"
							name="api_key"
							type="password"
							placeholder="Uses env default if blank"
						/>
					</div>

					<div class="space-y-2">
						<Label>Target Languages</Label>
						{#if data.uiConfig?.languages && data.uiConfig.languages.length > 0}
							<div
								class="grid grid-cols-2 sm:grid-cols-3 gap-2 rounded-md border border-input bg-background p-3 max-h-60 overflow-y-auto"
							>
								{#each data.uiConfig.languages as lang (lang.code)}
									<label
										class="flex items-center gap-2 rounded px-2 py-1.5 text-sm cursor-pointer hover:bg-accent transition-colors"
									>
										<input
											type="checkbox"
											name="target_languages"
											value={lang.code}
											checked={selectedLanguages.has(lang.code)}
											onchange={() => toggleLanguage(lang.code)}
											class="size-4 rounded border-input accent-primary"
										/>
										<span>{lang.name}</span>
										<span class="text-muted-foreground text-xs">({lang.code})</span>
									</label>
								{/each}
							</div>
							{#if selectedLanguages.size > 0}
								<p class="text-xs text-muted-foreground">
									{selectedLanguages.size} language{selectedLanguages.size === 1
										? ''
										: 's'} selected
								</p>
							{/if}
						{:else}
							<p class="text-sm text-muted-foreground">
								No languages available. Check backend configuration.
							</p>
						{/if}
					</div>

					<div class="space-y-2">
						<Label for="translation_model">Translation Model</Label>
						{#if data.uiConfig?.translation_models && data.uiConfig.translation_models.length > 0}
							<select
								id="translation_model"
								name="translation_model"
								class="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
								bind:value={selectedModel}
							>
								<option value="">Default</option>
								{#each data.uiConfig.translation_models as model}
									<option value={model.name}>
										{model.name} ({model.backend})
									</option>
								{/each}
							</select>
						{:else}
							<select
								id="translation_model"
								name="translation_model"
								class="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
								disabled
							>
								<option value="">No models available</option>
							</select>
						{/if}
					</div>

					<div class="space-y-2">
						<Label for="domain">Domain</Label>
						<select
							id="domain"
							name="domain"
							class="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
						>
							<option value="">General</option>
							{#if data.uiConfig?.domains}
								{#each data.uiConfig.domains as d}
									<option value={d}>{d}</option>
								{/each}
							{/if}
						</select>
					</div>

					{#if form?.errors?.form}
						<p class="text-sm text-destructive">{form.errors.form}</p>
					{/if}

					<Button type="submit" disabled={submitting}>
					{#if submitting}Connecting...{:else}Connect{/if}
				</Button>
				</form>
			</Card.Content>
		</Card.Root>
	</div>

	<!-- Active Sessions -->
	<div>
		<Card.Root>
			<Card.Header>
				<Card.Title>Active Sessions</Card.Title>
			</Card.Header>
			<Card.Content>
				{#if data.sessions.length === 0}
					<p class="text-sm text-muted-foreground">No active sessions</p>
				{:else}
					<ul class="space-y-2">
						{#each data.sessions as session}
							<li>
								<a
									href="/fireflies/connect?session={session.session_id}"
									class="block p-2 rounded border hover:bg-accent transition-colors"
								>
									<div class="flex items-center justify-between">
										<span class="text-sm font-mono truncate"
											>{session.session_id.slice(0, 16)}...</span
										>
										<StatusIndicator
											status={session.connection_status === 'CONNECTED'
												? 'connected'
												: 'disconnected'}
										/>
									</div>
									<p class="text-xs text-muted-foreground mt-1">
										{session.chunks_received} chunks · {session.translations_completed} translations
									</p>
								</a>
							</li>
						{/each}
					</ul>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>
</div>
