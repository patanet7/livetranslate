<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Table from '$lib/components/ui/table';
	import * as Dialog from '$lib/components/ui/dialog';
	import * as Select from '$lib/components/ui/select';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { Separator } from '$lib/components/ui/separator';
	import { toastStore } from '$lib/stores/toast.svelte';
	import type { Glossary, GlossaryEntry, UiConfig } from '$lib/types';

	let { data } = $props();

	// --- Reactive state ---
	let glossaries = $state<Glossary[]>(data.glossaries);
	let entries = $state<GlossaryEntry[]>(data.entries);
	let selectedGlossaryId = $state<string | null>(data.activeGlossaryId);
	let uiConfig = $state<UiConfig>(data.uiConfig);

	// Dialog visibility states
	let newGlossaryOpen = $state(false);
	let addTermOpen = $state(false);
	let editTermOpen = $state(false);
	let deleteGlossaryOpen = $state(false);
	let deleteEntryOpen = $state(false);

	// New glossary form
	let newGlossaryName = $state('');

	// Glossary details form
	let detailName = $state('');
	let detailDomain = $state('');
	let detailSourceLanguage = $state('');

	// Add/Edit term form
	let termSourceTerm = $state('');
	let termTranslationEs = $state('');
	let termTranslationFr = $state('');
	let termTranslationDe = $state('');
	let termPriority = $state(5);
	let editingEntryId = $state<string | null>(null);

	// Delete confirmation state
	let deletingEntryId = $state<string | null>(null);

	// Loading/error feedback
	let loading = $state(false);
	let errorMessage = $state('');
	let successMessage = $state('');

	// --- Derived ---
	let selectedGlossary = $derived(glossaries.find((g) => g.glossary_id === selectedGlossaryId) ?? null);

	// Sync detail form when selection changes
	$effect(() => {
		if (selectedGlossary) {
			detailName = selectedGlossary.name;
			detailDomain = selectedGlossary.domain;
			detailSourceLanguage = selectedGlossary.source_language;
		} else {
			detailName = '';
			detailDomain = '';
			detailSourceLanguage = '';
		}
	});

	// --- Helpers ---
	function clearMessages() {
		errorMessage = '';
		successMessage = '';
	}

	function clearTermForm() {
		termSourceTerm = '';
		termTranslationEs = '';
		termTranslationFr = '';
		termTranslationDe = '';
		termPriority = 5;
		editingEntryId = null;
	}

	// --- Helpers: API error extraction ---
	async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
		try {
			const body = await res.json();
			return body?.detail ?? body?.error ?? fallback;
		} catch {
			return fallback;
		}
	}

	function networkErrorMessage(err: unknown): string {
		if (err instanceof TypeError && err.message === 'Failed to fetch') {
			return 'Connection error. Please check your network and try again.';
		}
		return err instanceof Error ? err.message : 'Unknown error';
	}

	// --- API calls ---
	async function fetchEntries(glossaryId: string) {
		try {
			const res = await fetch(`/api/glossaries/${glossaryId}/entries`);
			if (res.ok) {
				entries = await res.json();
			}
		} catch {
			entries = [];
		}
	}

	async function fetchGlossaries() {
		try {
			const res = await fetch('/api/glossaries');
			if (res.ok) {
				glossaries = await res.json();
			}
		} catch {
			/* keep existing */
		}
	}

	async function selectGlossary(glossaryId: string) {
		clearMessages();
		selectedGlossaryId = glossaryId;
		await fetchEntries(glossaryId);
	}

	async function createGlossary() {
		if (!newGlossaryName.trim()) return;
		clearMessages();
		loading = true;
		try {
			const res = await fetch('/api/glossaries', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: newGlossaryName.trim(),
					description: '',
					domain: 'general',
					source_language: 'en',
					target_languages: ['es', 'fr', 'de'],
					is_default: glossaries.length === 0
				})
			});
			if (res.ok) {
				const created: Glossary = await res.json();
				await fetchGlossaries();
				selectedGlossaryId = created.glossary_id;
				await fetchEntries(created.glossary_id);
				newGlossaryName = '';
				newGlossaryOpen = false;
				successMessage = 'Glossary created';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, 'Failed to create glossary');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
		}
	}

	async function saveGlossaryDetails() {
		if (!selectedGlossaryId) return;
		clearMessages();
		loading = true;
		try {
			const res = await fetch(`/api/glossaries/${selectedGlossaryId}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					name: detailName,
					domain: detailDomain,
					source_language: detailSourceLanguage
				})
			});
			if (res.ok) {
				await fetchGlossaries();
				successMessage = 'Glossary updated';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, 'Failed to update glossary');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
		}
	}

	async function setAsDefault() {
		if (!selectedGlossaryId) return;
		clearMessages();
		loading = true;
		try {
			const res = await fetch(`/api/glossaries/${selectedGlossaryId}`, {
				method: 'PATCH',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ is_default: true })
			});
			if (res.ok) {
				await fetchGlossaries();
				successMessage = 'Set as default glossary';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, 'Failed to set default');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
		}
	}

	async function deleteGlossary() {
		if (!selectedGlossaryId) return;
		clearMessages();
		loading = true;
		try {
			const res = await fetch(`/api/glossaries/${selectedGlossaryId}`, {
				method: 'DELETE'
			});
			if (res.ok) {
				await fetchGlossaries();
				selectedGlossaryId = glossaries[0]?.glossary_id ?? null;
				if (selectedGlossaryId) {
					await fetchEntries(selectedGlossaryId);
				} else {
					entries = [];
				}
				deleteGlossaryOpen = false;
				successMessage = 'Glossary deleted';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, 'Failed to delete glossary');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
		}
	}

	async function addOrUpdateTerm() {
		if (!selectedGlossaryId || !termSourceTerm.trim()) return;
		clearMessages();
		loading = true;

		const translations: Record<string, string> = {};
		if (termTranslationEs.trim()) translations['es'] = termTranslationEs.trim();
		if (termTranslationFr.trim()) translations['fr'] = termTranslationFr.trim();
		if (termTranslationDe.trim()) translations['de'] = termTranslationDe.trim();

		if (Object.keys(translations).length === 0) {
			errorMessage = 'At least one translation is required';
			toastStore.error(errorMessage);
			loading = false;
			return;
		}

		try {
			const isEdit = editingEntryId !== null;
			const url = isEdit
				? `/api/glossaries/${selectedGlossaryId}/entries/${editingEntryId}`
				: `/api/glossaries/${selectedGlossaryId}/entries`;
			const method = isEdit ? 'PATCH' : 'POST';

			const body: Record<string, unknown> = {
				source_term: termSourceTerm.trim(),
				translations,
				priority: termPriority
			};

			if (!isEdit) {
				body.context = '';
				body.notes = '';
				body.case_sensitive = false;
				body.match_whole_word = true;
			}

			const res = await fetch(url, {
				method,
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body)
			});
			if (res.ok) {
				await fetchEntries(selectedGlossaryId);
				await fetchGlossaries();
				clearTermForm();
				addTermOpen = false;
				editTermOpen = false;
				successMessage = isEdit ? 'Term updated' : 'Term added';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, isEdit ? 'Failed to update term' : 'Failed to add term');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
		}
	}

	function openEditTerm(entry: GlossaryEntry) {
		editingEntryId = entry.entry_id;
		termSourceTerm = entry.source_term;
		termTranslationEs = entry.translations['es'] ?? '';
		termTranslationFr = entry.translations['fr'] ?? '';
		termTranslationDe = entry.translations['de'] ?? '';
		termPriority = entry.priority;
		editTermOpen = true;
	}

	async function deleteEntry() {
		if (!selectedGlossaryId || !deletingEntryId) return;
		clearMessages();
		loading = true;
		try {
			const res = await fetch(
				`/api/glossaries/${selectedGlossaryId}/entries/${deletingEntryId}`,
				{ method: 'DELETE' }
			);
			if (res.ok) {
				await fetchEntries(selectedGlossaryId);
				await fetchGlossaries();
				deletingEntryId = null;
				deleteEntryOpen = false;
				successMessage = 'Entry deleted';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, 'Failed to delete entry');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
		}
	}

	async function importCsv(event: Event) {
		if (!selectedGlossaryId) return;
		const target = event.target as HTMLInputElement;
		const file = target.files?.[0];
		if (!file) return;

		clearMessages();
		loading = true;
		const formData = new FormData();
		formData.append('file', file);

		try {
			const res = await fetch(`/api/glossaries/${selectedGlossaryId}/import`, {
				method: 'POST',
				body: formData
			});
			if (res.ok) {
				await fetchEntries(selectedGlossaryId);
				await fetchGlossaries();
				successMessage = 'CSV imported successfully';
				toastStore.success(successMessage);
			} else {
				errorMessage = await extractErrorMessage(res, 'Failed to import CSV');
				toastStore.error(errorMessage);
			}
		} catch (err) {
			errorMessage = networkErrorMessage(err);
			toastStore.error(errorMessage);
		} finally {
			loading = false;
			target.value = '';
		}
	}

	function exportCsv() {
		if (entries.length === 0) return;

		const languages = ['es', 'fr', 'de'];
		const headers = ['source_term', ...languages.map((l) => `translation_${l}`), 'priority'];
		const rows = entries.map((e) => [
			e.source_term,
			...languages.map((l) => e.translations[l] ?? ''),
			String(e.priority)
		]);

		const csv = [headers.join(','), ...rows.map((r) => r.map((c) => `"${c.replace(/"/g, '""')}"`).join(','))].join('\n');

		const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
		const url = URL.createObjectURL(blob);
		const link = document.createElement('a');
		link.href = url;
		link.download = `${selectedGlossary?.name ?? 'glossary'}_export.csv`;
		link.click();
		URL.revokeObjectURL(url);
		toastStore.success('Glossary exported as CSV');
	}
</script>

<PageHeader title="Glossary" description="Manage translation glossary terms and dictionaries">
	{#snippet actions()}
		<Button variant="outline" onclick={() => (newGlossaryOpen = true)}>New Glossary</Button>
	{/snippet}
</PageHeader>

<!-- Feedback messages -->
{#if errorMessage}
	<div class="mb-4 rounded-md border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
		{errorMessage}
		<button class="ml-2 underline" onclick={() => (errorMessage = '')}>dismiss</button>
	</div>
{/if}
{#if successMessage}
	<div class="mb-4 rounded-md border border-green-500/50 bg-green-500/10 px-4 py-3 text-sm text-green-700 dark:text-green-400">
		{successMessage}
		<button class="ml-2 underline" onclick={() => (successMessage = '')}>dismiss</button>
	</div>
{/if}

<div class="grid grid-cols-1 lg:grid-cols-5 gap-6">
	<!-- Left Column: Glossary Management (40%) -->
	<div class="lg:col-span-2 space-y-4">
		<!-- Glossary List -->
		<Card.Root>
			<Card.Header>
				<Card.Title>Glossaries</Card.Title>
			</Card.Header>
			<Card.Content class="p-0">
				{#if glossaries.length === 0}
					<div class="p-6 text-center text-muted-foreground">
						No glossaries yet. Create one to get started.
					</div>
				{:else}
					<div class="divide-y">
						{#each glossaries as glossary (glossary.glossary_id)}
							<button
								class="flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-accent/50 {selectedGlossaryId === glossary.glossary_id ? 'bg-accent' : ''}"
								onclick={() => selectGlossary(glossary.glossary_id)}
							>
								<div class="flex-1 min-w-0">
									<div class="flex items-center gap-2">
										{#if glossary.is_default}
											<span class="text-amber-500" title="Default glossary">&#11088;</span>
										{/if}
										<span class="font-medium truncate">{glossary.name}</span>
									</div>
									{#if glossary.domain}
										<span class="text-xs text-muted-foreground">{glossary.domain}</span>
									{/if}
								</div>
								<Badge variant="secondary">{glossary.entry_count}</Badge>
							</button>
						{/each}
					</div>
				{/if}
			</Card.Content>
		</Card.Root>

		<!-- Glossary Details -->
		{#if selectedGlossary}
			<Card.Root>
				<Card.Header>
					<Card.Title>Glossary Details</Card.Title>
				</Card.Header>
				<Card.Content class="space-y-4">
					<div class="space-y-1.5">
						<Label for="detail-name">Name</Label>
						<Input id="detail-name" bind:value={detailName} placeholder="Glossary name" />
					</div>

					<div class="space-y-1.5">
						<Label for="detail-domain">Domain</Label>
						<Select.Root type="single" bind:value={detailDomain}>
							<Select.Trigger id="detail-domain" class="w-full">
								{detailDomain || 'Select domain'}
							</Select.Trigger>
							<Select.Content>
								{#each uiConfig.domains as domain (domain.value)}
									<Select.Item value={domain.value} label={domain.label} />
								{/each}
								{#if !uiConfig.domains.some((d) => d.value === 'general')}
									<Select.Item value="general" label="General" />
								{/if}
							</Select.Content>
						</Select.Root>
					</div>

					<div class="space-y-1.5">
						<Label for="detail-source-lang">Source Language</Label>
						<Select.Root type="single" bind:value={detailSourceLanguage}>
							<Select.Trigger id="detail-source-lang" class="w-full">
								{(uiConfig.languages.find((l) => l.code === detailSourceLanguage)?.name ?? detailSourceLanguage) || 'Select language'}
							</Select.Trigger>
							<Select.Content>
								{#each uiConfig.languages as lang (lang.code)}
									<Select.Item value={lang.code} label={lang.name} />
								{/each}
								{#if uiConfig.languages.length === 0}
									<Select.Item value="en" label="English" />
								{/if}
							</Select.Content>
						</Select.Root>
					</div>

					<Separator />

					<div class="flex flex-wrap gap-2">
						<Button onclick={saveGlossaryDetails} disabled={loading}>Save</Button>
						<Button
							variant="outline"
							onclick={setAsDefault}
							disabled={loading || selectedGlossary.is_default}
						>
							{selectedGlossary.is_default ? 'Is Default' : 'Set as Default'}
						</Button>
						<Button
							variant="destructive"
							onclick={() => (deleteGlossaryOpen = true)}
							disabled={loading}
						>
							Delete
						</Button>
					</div>
				</Card.Content>
			</Card.Root>
		{/if}
	</div>

	<!-- Right Column: Entries (60%) -->
	<div class="lg:col-span-3 space-y-4">
		<Card.Root>
			<Card.Header>
				<div class="flex items-center justify-between">
					<Card.Title>
						{#if selectedGlossary}
							{selectedGlossary.name}
							<Badge variant="secondary" class="ml-2">{entries.length} terms</Badge>
						{:else}
							Glossary Entries
						{/if}
					</Card.Title>
					{#if selectedGlossary}
						<div class="flex flex-wrap items-center gap-2">
							<Button size="sm" onclick={() => { clearTermForm(); addTermOpen = true; }}>
								Add Term
							</Button>
							<label class="inline-flex">
								<Button size="sm" variant="outline" onclick={() => { const el = document.getElementById('csv-import') as HTMLInputElement; el?.click(); }}>
									Import CSV
								</Button>
								<input
									id="csv-import"
									type="file"
									accept=".csv"
									class="hidden"
									onchange={importCsv}
								/>
							</label>
							<Button size="sm" variant="outline" onclick={exportCsv} disabled={entries.length === 0}>
								Export CSV
							</Button>
						</div>
					{/if}
				</div>
			</Card.Header>
			<Card.Content class="p-0">
				{#if !selectedGlossary}
					<div class="p-6 text-center text-muted-foreground">
						Select a glossary to view its entries
					</div>
				{:else if entries.length === 0}
					<div class="p-6 text-center text-muted-foreground">
						No entries yet. Add a term to get started.
					</div>
				{:else}
					<div class="overflow-x-auto">
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Source Term</Table.Head>
								<Table.Head>ES</Table.Head>
								<Table.Head>FR</Table.Head>
								<Table.Head>DE</Table.Head>
								<Table.Head class="w-20">Priority</Table.Head>
								<Table.Head class="w-24">Actions</Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							{#each entries as entry (entry.entry_id)}
								<Table.Row>
									<Table.Cell class="font-medium">{entry.source_term}</Table.Cell>
									<Table.Cell class="text-sm">{entry.translations['es'] ?? '-'}</Table.Cell>
									<Table.Cell class="text-sm">{entry.translations['fr'] ?? '-'}</Table.Cell>
									<Table.Cell class="text-sm">{entry.translations['de'] ?? '-'}</Table.Cell>
									<Table.Cell>
										<Badge variant="outline">{entry.priority}</Badge>
									</Table.Cell>
									<Table.Cell>
										<div class="flex items-center gap-1">
											<Button
												variant="ghost"
												size="sm"
												onclick={() => openEditTerm(entry)}
											>
												Edit
											</Button>
											<Button
												variant="ghost"
												size="sm"
												class="text-destructive hover:text-destructive"
												onclick={() => { deletingEntryId = entry.entry_id; deleteEntryOpen = true; }}
											>
												Del
											</Button>
										</div>
									</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
					</div>
				{/if}
			</Card.Content>
		</Card.Root>
	</div>
</div>

<!-- New Glossary Dialog -->
<Dialog.Root bind:open={newGlossaryOpen}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Create New Glossary</Dialog.Title>
			<Dialog.Description>Enter a name for the new glossary.</Dialog.Description>
		</Dialog.Header>
		<div class="space-y-4 py-4">
			<div class="space-y-1.5">
				<Label for="new-glossary-name">Name</Label>
				<Input
					id="new-glossary-name"
					bind:value={newGlossaryName}
					placeholder="e.g. Medical Terminology"
				/>
			</div>
		</div>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => (newGlossaryOpen = false)}>Cancel</Button>
			<Button onclick={createGlossary} disabled={loading || !newGlossaryName.trim()}>Create</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Add Term Dialog -->
<Dialog.Root bind:open={addTermOpen}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Add Term</Dialog.Title>
			<Dialog.Description>Add a new glossary entry with translations.</Dialog.Description>
		</Dialog.Header>
		<div class="space-y-4 py-4">
			<div class="space-y-1.5">
				<Label for="add-source-term">Source Term</Label>
				<Input id="add-source-term" bind:value={termSourceTerm} placeholder="e.g. heart attack" />
			</div>
			<div class="space-y-1.5">
				<Label for="add-translation-es">Spanish (ES)</Label>
				<Input id="add-translation-es" bind:value={termTranslationEs} placeholder="e.g. infarto de miocardio" />
			</div>
			<div class="space-y-1.5">
				<Label for="add-translation-fr">French (FR)</Label>
				<Input id="add-translation-fr" bind:value={termTranslationFr} placeholder="e.g. crise cardiaque" />
			</div>
			<div class="space-y-1.5">
				<Label for="add-translation-de">German (DE)</Label>
				<Input id="add-translation-de" bind:value={termTranslationDe} placeholder="e.g. Herzinfarkt" />
			</div>
			<div class="space-y-1.5">
				<Label for="add-priority">Priority</Label>
				<Input id="add-priority" type="number" bind:value={termPriority} min={1} max={10} />
			</div>
		</div>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => (addTermOpen = false)}>Cancel</Button>
			<Button onclick={addOrUpdateTerm} disabled={loading || !termSourceTerm.trim()}>Add</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Edit Term Dialog -->
<Dialog.Root bind:open={editTermOpen}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Edit Term</Dialog.Title>
			<Dialog.Description>Update the glossary entry translations.</Dialog.Description>
		</Dialog.Header>
		<div class="space-y-4 py-4">
			<div class="space-y-1.5">
				<Label for="edit-source-term">Source Term</Label>
				<Input id="edit-source-term" bind:value={termSourceTerm} placeholder="Source term" />
			</div>
			<div class="space-y-1.5">
				<Label for="edit-translation-es">Spanish (ES)</Label>
				<Input id="edit-translation-es" bind:value={termTranslationEs} placeholder="Spanish translation" />
			</div>
			<div class="space-y-1.5">
				<Label for="edit-translation-fr">French (FR)</Label>
				<Input id="edit-translation-fr" bind:value={termTranslationFr} placeholder="French translation" />
			</div>
			<div class="space-y-1.5">
				<Label for="edit-translation-de">German (DE)</Label>
				<Input id="edit-translation-de" bind:value={termTranslationDe} placeholder="German translation" />
			</div>
			<div class="space-y-1.5">
				<Label for="edit-priority">Priority</Label>
				<Input id="edit-priority" type="number" bind:value={termPriority} min={1} max={10} />
			</div>
		</div>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => { editTermOpen = false; clearTermForm(); }}>Cancel</Button>
			<Button onclick={addOrUpdateTerm} disabled={loading || !termSourceTerm.trim()}>Save</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Delete Glossary Confirmation Dialog -->
<Dialog.Root bind:open={deleteGlossaryOpen}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Delete Glossary</Dialog.Title>
			<Dialog.Description>
				Are you sure you want to delete "{selectedGlossary?.name}"? This will remove all entries and cannot be undone.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => (deleteGlossaryOpen = false)}>Cancel</Button>
			<Button variant="destructive" onclick={deleteGlossary} disabled={loading}>Delete</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>

<!-- Delete Entry Confirmation Dialog -->
<Dialog.Root bind:open={deleteEntryOpen}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Delete Entry</Dialog.Title>
			<Dialog.Description>
				Are you sure you want to delete this glossary entry? This cannot be undone.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" onclick={() => { deleteEntryOpen = false; deletingEntryId = null; }}>Cancel</Button>
			<Button variant="destructive" onclick={deleteEntry} disabled={loading}>Delete</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
