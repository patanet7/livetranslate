<script lang="ts">
	import { enhance } from '$app/forms';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Tabs from '$lib/components/ui/tabs';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button, buttonVariants } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import * as Select from '$lib/components/ui/select';
	import { toastStore } from '$lib/stores/toast.svelte';
	import type { DomainItem } from '$lib/types';

	// ── Data ───────────────────────────────────────────────────────────

	let { data, form } = $props();

	// ── State ──────────────────────────────────────────────────────────

	let submitting = $state(false);
	let activeTab = $state('languages');
	let enabledCodes = $state<Set<string>>(new Set());
	let customDomains = $state<DomainItem[]>([]);
	let disabledDomains = $state<Set<string>>(new Set());
	let defaults = $state<Record<string, unknown>>({});
	let addDomainOpen = $state(false);
	let newDomain = $state<DomainItem>({ value: '', label: '', description: '' });
	let defaultTargetLanguages = $state<Set<string>>(new Set());

	// ── Derived ────────────────────────────────────────────────────────

	interface LangInfo {
		code: string;
		name: string;
		native: string;
		rtl: boolean;
	}

	let allLanguages = $derived<LangInfo[]>(
		(data.uiConfig?.languages as LangInfo[] | undefined) ?? []
	);

	let builtinDomains = $derived<DomainItem[]>(
		(data.uiConfig?.domains as DomainItem[] | undefined) ?? []
	);

	let enabledCount = $derived(enabledCodes.size);

	// ── Effects ────────────────────────────────────────────────────────

	$effect(() => {
		if (data.uiConfig) {
			const codes = (data.uiConfig.language_codes as string[]) ?? [];
			enabledCodes = new Set(codes);

			defaults = { ...(data.uiConfig.defaults as Record<string, unknown>) };

			const targets = (defaults.default_target_languages as string[]) ?? [];
			defaultTargetLanguages = new Set(targets);
		}
	});

	// ── Helpers ─────────────────────────────────────────────────────────

	function toggleLanguage(code: string) {
		const next = new Set(enabledCodes);
		if (next.has(code)) {
			next.delete(code);
		} else {
			next.add(code);
		}
		enabledCodes = next;
	}

	function selectAll() {
		enabledCodes = new Set(allLanguages.map((l) => l.code));
	}

	function deselectAll() {
		enabledCodes = new Set();
	}

	function toggleBuiltinDomain(value: string) {
		const next = new Set(disabledDomains);
		if (next.has(value)) {
			next.delete(value);
		} else {
			next.add(value);
		}
		disabledDomains = next;
	}

	function addCustomDomain() {
		if (!newDomain.value.trim() || !newDomain.label.trim()) return;
		const slug = newDomain.value
			.trim()
			.toLowerCase()
			.replace(/[^a-z0-9_-]/g, '_');
		customDomains = [
			...customDomains,
			{ value: slug, label: newDomain.label.trim(), description: newDomain.description?.trim() }
		];
		newDomain = { value: '', label: '', description: '' };
		addDomainOpen = false;
	}

	function removeCustomDomain(index: number) {
		customDomains = customDomains.filter((_, i) => i !== index);
	}

	function toggleTargetLanguage(code: string) {
		const next = new Set(defaultTargetLanguages);
		if (next.has(code)) {
			next.delete(code);
		} else {
			next.add(code);
		}
		defaultTargetLanguages = next;
	}

	function handleSubmitResult() {
		return async ({ result, update }: { result: { type: string }; update: () => Promise<void> }) => {
			await update();
			submitting = false;
			if (result.type === 'success') {
				toastStore.success('Settings saved successfully');
			} else if (result.type === 'failure') {
				toastStore.error('Failed to save settings');
			}
		};
	}
</script>

<PageHeader
	title="System Configuration"
	description="Manage languages, translation domains, and default settings"
/>

<Tabs.Root bind:value={activeTab}>
	<Tabs.List>
		<Tabs.Trigger value="languages">Languages</Tabs.Trigger>
		<Tabs.Trigger value="domains">Domains</Tabs.Trigger>
		<Tabs.Trigger value="defaults">Defaults</Tabs.Trigger>
	</Tabs.List>

	<!-- ════════════════════════════════════════════════════════════════ -->
	<!-- Tab 1: Languages                                                -->
	<!-- ════════════════════════════════════════════════════════════════ -->
	<Tabs.Content value="languages">
		<form
			method="POST"
			action="?/updateLanguages"
			use:enhance={() => {
				submitting = true;
				return handleSubmitResult();
			}}
			class="mt-4 space-y-4"
		>
			<Card.Root>
				<Card.Header>
					<div class="flex items-center justify-between">
						<div>
							<Card.Title>Enabled Languages</Card.Title>
							<Card.Description>
								{enabledCount} of {allLanguages.length} languages enabled
							</Card.Description>
						</div>
						<div class="flex gap-2">
							<Button type="button" variant="outline" size="sm" onclick={selectAll}>
								Select All
							</Button>
							<Button type="button" variant="outline" size="sm" onclick={deselectAll}>
								Deselect All
							</Button>
						</div>
					</div>
				</Card.Header>
				<Card.Content>
					<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
						{#each allLanguages as lang (lang.code)}
							{@const checked = enabledCodes.has(lang.code)}
							<button
								type="button"
								class="flex items-start gap-3 rounded-lg border p-3 text-left transition-colors hover:bg-accent/50 {checked
									? 'border-primary bg-primary/5'
									: 'border-border'}"
								onclick={() => toggleLanguage(lang.code)}
							>
								<input
									type="checkbox"
									class="mt-0.5 h-4 w-4 shrink-0"
									checked={checked}
									tabindex={-1}
									onchange={() => toggleLanguage(lang.code)}
								/>
								<div class="min-w-0">
									<div class="flex items-center gap-1.5">
										<span class="text-sm font-medium">{lang.name}</span>
										{#if lang.rtl}
											<span
												class="inline-flex items-center rounded-full bg-orange-100 px-1.5 py-0.5 text-[10px] font-medium text-orange-700 dark:bg-orange-900 dark:text-orange-300"
											>
												RTL
											</span>
										{/if}
									</div>
									<span class="text-xs text-muted-foreground">{lang.native}</span>
									<span class="ml-1 text-xs text-muted-foreground/60">({lang.code})</span>
								</div>
							</button>
						{/each}
					</div>

					<!-- Hidden inputs for form submission -->
					{#each [...enabledCodes] as code}
						<input type="hidden" name="languages" value={code} />
					{/each}
				</Card.Content>
				<Card.Footer class="flex justify-end border-t pt-4">
					<Button type="submit" disabled={submitting}>
						{#if submitting}Saving...{:else}Save Languages{/if}
					</Button>
				</Card.Footer>
			</Card.Root>
		</form>
	</Tabs.Content>

	<!-- ════════════════════════════════════════════════════════════════ -->
	<!-- Tab 2: Domains                                                  -->
	<!-- ════════════════════════════════════════════════════════════════ -->
	<Tabs.Content value="domains">
		<form
			method="POST"
			action="?/updateDomains"
			use:enhance={() => {
				submitting = true;
				return handleSubmitResult();
			}}
			class="mt-4 space-y-6"
		>
			<!-- Built-in Domains -->
			<Card.Root>
				<Card.Header>
					<Card.Title>Built-in Domains</Card.Title>
					<Card.Description>
						Toggle domain categories for glossary organization. General is always enabled.
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<div class="space-y-2">
						{#each builtinDomains as domain (domain.value)}
							{@const isGeneral = domain.value === ''}
							{@const isDisabled = disabledDomains.has(domain.value)}
							<div
								class="flex items-center justify-between rounded-lg border p-3 {isDisabled
									? 'opacity-60'
									: ''}"
							>
								<div>
									<span class="text-sm font-medium">{domain.label}</span>
									{#if domain.description}
										<p class="text-xs text-muted-foreground">{domain.description}</p>
									{/if}
								</div>
								{#if isGeneral}
									<span
										class="inline-flex items-center rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700 dark:bg-green-900 dark:text-green-300"
									>
										Always On
									</span>
								{:else}
									<button
										type="button"
										class="relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors {isDisabled
											? 'bg-muted'
											: 'bg-primary'}"
										onclick={() => toggleBuiltinDomain(domain.value)}
										role="switch"
										aria-checked={!isDisabled}
										aria-label="Toggle {domain.label} domain"
									>
										<span
											class="pointer-events-none inline-block h-5 w-5 transform rounded-full bg-background shadow-lg ring-0 transition-transform {isDisabled
												? 'translate-x-0'
												: 'translate-x-5'}"
										></span>
									</button>
								{/if}
							</div>
						{/each}
					</div>
				</Card.Content>
			</Card.Root>

			<!-- Custom Domains -->
			<Card.Root>
				<Card.Header>
					<div class="flex items-center justify-between">
						<div>
							<Card.Title>Custom Domains</Card.Title>
							<Card.Description>
								Add your own domain categories for glossary organization.
							</Card.Description>
						</div>
						<Dialog.Dialog bind:open={addDomainOpen}>
							<Dialog.DialogTrigger class={buttonVariants({ variant: 'outline', size: 'sm' })}>
								Add Domain
							</Dialog.DialogTrigger>
							<Dialog.DialogContent>
								<Dialog.DialogHeader>
									<Dialog.DialogTitle>Add Custom Domain</Dialog.DialogTitle>
									<Dialog.DialogDescription>
										Create a new domain category for organizing glossary terms.
									</Dialog.DialogDescription>
								</Dialog.DialogHeader>
								<div class="space-y-4 py-4">
									<div class="space-y-2">
										<Label for="domain-value">Slug</Label>
										<Input
											id="domain-value"
											placeholder="e.g. healthcare"
											bind:value={newDomain.value}
										/>
										<p class="text-xs text-muted-foreground">
											Lowercase identifier used internally.
										</p>
									</div>
									<div class="space-y-2">
										<Label for="domain-label">Label</Label>
										<Input
											id="domain-label"
											placeholder="e.g. Healthcare"
											bind:value={newDomain.label}
										/>
									</div>
									<div class="space-y-2">
										<Label for="domain-desc">Description (optional)</Label>
										<Input
											id="domain-desc"
											placeholder="e.g. Healthcare and wellness terminology"
											bind:value={newDomain.description}
										/>
									</div>
								</div>
								<Dialog.DialogFooter>
									<Button variant="outline" onclick={() => (addDomainOpen = false)}>
										Cancel
									</Button>
									<Button
										onclick={addCustomDomain}
										disabled={!newDomain.value.trim() || !newDomain.label.trim()}
									>
										Add Domain
									</Button>
								</Dialog.DialogFooter>
							</Dialog.DialogContent>
						</Dialog.Dialog>
					</div>
				</Card.Header>
				<Card.Content>
					{#if customDomains.length === 0}
						<p class="text-sm text-muted-foreground py-4 text-center">
							No custom domains added yet.
						</p>
					{:else}
						<div class="space-y-2">
							{#each customDomains as domain, i (domain.value)}
								<div class="flex items-center justify-between rounded-lg border p-3">
									<div>
										<span class="text-sm font-medium">{domain.label}</span>
										<span class="ml-1 text-xs text-muted-foreground/60">({domain.value})</span>
										{#if domain.description}
											<p class="text-xs text-muted-foreground">{domain.description}</p>
										{/if}
									</div>
									<Button
										type="button"
										variant="destructive"
										size="sm"
										onclick={() => removeCustomDomain(i)}
									>
										Remove
									</Button>
								</div>
							{/each}
						</div>
					{/if}
				</Card.Content>
			</Card.Root>

			<!-- Hidden inputs for form submission -->
			<input type="hidden" name="custom_domains" value={JSON.stringify(customDomains)} />
			<input type="hidden" name="disabled_domains" value={JSON.stringify([...disabledDomains])} />

			<div class="flex justify-end">
				<Button type="submit" disabled={submitting}>
					{#if submitting}Saving...{:else}Save Domains{/if}
				</Button>
			</div>
		</form>
	</Tabs.Content>

	<!-- ════════════════════════════════════════════════════════════════ -->
	<!-- Tab 3: Defaults                                                 -->
	<!-- ════════════════════════════════════════════════════════════════ -->
	<Tabs.Content value="defaults">
		<div class="mt-4 space-y-6">
			<form
				method="POST"
				action="?/updateDefaults"
				use:enhance={() => {
					submitting = true;
					return handleSubmitResult();
				}}
				class="space-y-6"
			>
				<Card.Root>
					<Card.Header>
						<Card.Title>Translation Defaults</Card.Title>
						<Card.Description>
							Default settings applied to new translation sessions.
						</Card.Description>
					</Card.Header>
					<Card.Content class="space-y-6">
						<!-- Source Language -->
						<div class="space-y-2">
							<Label for="source-lang">Default Source Language</Label>
							<select
								id="source-lang"
								name="default_source_language"
								class="w-full rounded-md border bg-background px-3 py-2 text-sm"
								value={String(defaults.default_source_language ?? 'en')}
							>
								{#each allLanguages as lang (lang.code)}
									<option value={lang.code}>{lang.name} ({lang.code})</option>
								{/each}
							</select>
						</div>

						<!-- Target Languages -->
						<div class="space-y-2">
							<Label>Default Target Languages</Label>
							<div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
								{#each allLanguages as lang (lang.code)}
									{@const checked = defaultTargetLanguages.has(lang.code)}
									<label
										class="flex items-center gap-2 rounded-md border p-2 text-sm cursor-pointer transition-colors hover:bg-accent/50 {checked
											? 'border-primary bg-primary/5'
											: 'border-border'}"
									>
										<input
											type="checkbox"
											name="default_target_languages"
											value={lang.code}
											checked={checked}
											class="h-3.5 w-3.5"
											onchange={() => toggleTargetLanguage(lang.code)}
										/>
										<span>{lang.name}</span>
									</label>
								{/each}
							</div>
						</div>

						<!-- Auto-detect Toggle -->
						<div class="flex items-center justify-between rounded-lg border p-3">
							<div>
								<Label for="auto-detect" class="text-sm font-medium">Auto-detect Language</Label>
								<p class="text-xs text-muted-foreground">
									Automatically detect the source language from audio input.
								</p>
							</div>
							<input
								type="checkbox"
								id="auto-detect"
								name="auto_detect_language"
								checked={defaults.auto_detect_language === true}
								class="h-4 w-4"
							/>
						</div>

						<!-- Confidence Threshold -->
						<div class="space-y-2">
							<div class="flex items-center justify-between">
								<Label for="confidence">Confidence Threshold</Label>
								<span class="text-sm font-mono text-muted-foreground">
									{Number(defaults.confidence_threshold ?? 0.8).toFixed(2)}
								</span>
							</div>
							<input
								type="range"
								id="confidence"
								name="confidence_threshold"
								min="0"
								max="1"
								step="0.05"
								value={Number(defaults.confidence_threshold ?? 0.8)}
								oninput={(e) => {
									defaults = {
										...defaults,
										confidence_threshold: parseFloat(e.currentTarget.value)
									};
								}}
								class="w-full accent-primary"
							/>
							<div class="flex justify-between text-xs text-muted-foreground">
								<span>0.00 (accept all)</span>
								<span>1.00 (strictest)</span>
							</div>
						</div>

						<!-- Context Window Size -->
						<div class="space-y-2">
							<Label for="context-window">Context Window Size</Label>
							<Input
								id="context-window"
								type="number"
								name="context_window_size"
								value={String(defaults.context_window_size ?? 3)}
								min="1"
								max="20"
							/>
							<p class="text-xs text-muted-foreground">
								Number of previous sentences used as translation context.
							</p>
						</div>

						<!-- Max Buffer Words -->
						<div class="space-y-2">
							<Label for="max-buffer">Max Buffer Words</Label>
							<Input
								id="max-buffer"
								type="number"
								name="max_buffer_words"
								value={String(defaults.max_buffer_words ?? 50)}
								min="10"
								max="500"
							/>
							<p class="text-xs text-muted-foreground">
								Maximum words buffered before forcing a translation flush.
							</p>
						</div>

						<!-- Pause Threshold -->
						<div class="space-y-2">
							<Label for="pause-threshold">Pause Threshold (ms)</Label>
							<Input
								id="pause-threshold"
								type="number"
								name="pause_threshold_ms"
								value={String(defaults.pause_threshold_ms ?? 500)}
								min="100"
								max="5000"
								step="100"
							/>
							<p class="text-xs text-muted-foreground">
								Silence duration that triggers a translation flush.
							</p>
						</div>
					</Card.Content>
					<Card.Footer class="flex justify-end border-t pt-4">
						<Button type="submit" disabled={submitting}>
							{#if submitting}Saving...{:else}Save Defaults{/if}
						</Button>
					</Card.Footer>
				</Card.Root>
			</form>

			<!-- Reset to Factory Defaults -->
			<Card.Root class="border-destructive/30">
				<Card.Header>
					<Card.Title class="text-destructive">Reset to Factory Defaults</Card.Title>
					<Card.Description>
						This will remove all customizations and restore the original system configuration.
						This action cannot be undone.
					</Card.Description>
				</Card.Header>
				<Card.Footer class="border-t pt-4">
					<form
						method="POST"
						action="?/reset"
						use:enhance={() => {
							submitting = true;
							return async ({ result, update }: { result: { type: string }; update: () => Promise<void> }) => {
								await update();
								submitting = false;
								if (result.type === 'success') {
									toastStore.success('Configuration reset to factory defaults');
								} else if (result.type === 'failure') {
									toastStore.error('Failed to reset configuration');
								}
							};
						}}
					>
						<Button type="submit" variant="destructive" disabled={submitting}>
							{#if submitting}Resetting...{:else}Reset All Settings{/if}
						</Button>
					</form>
				</Card.Footer>
			</Card.Root>
		</div>
	</Tabs.Content>
</Tabs.Root>

{#if form?.errors?.form}
	<div class="mt-4 rounded-md border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
		{form.errors.form}
	</div>
{/if}
