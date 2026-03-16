<script lang="ts">
	import { onMount } from 'svelte';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Button } from '$lib/components/ui/button';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { toastStore } from '$lib/stores/toast.svelte';
	import ConnectionCard from '$lib/components/ConnectionCard.svelte';
	import ConnectionDialog from '$lib/components/ConnectionDialog.svelte';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import type { AIConnection, AggregatedModel, FeaturePreference } from '$lib/api/connections';

	let { data } = $props();

	// ── Connections State ──────────────────────────────────────────────
	let connections: AIConnection[] = $state([]);
	let connectionStatuses: Record<string, 'unknown' | 'connected' | 'error' | 'verifying'> =
		$state({});
	let connectionModelCounts: Record<string, number> = $state({});
	let aggregatedModels: AggregatedModel[] = $state([]);
	let dialogOpen = $state(false);
	let editingConnection: AIConnection | null = $state(null);
	let pendingDeleteId: string | null = $state(null);

	// ── Feature Preferences State ──────────────────────────────────────
	let preferences: Record<string, FeaturePreference> = $state({});

	// Sync from server load data
	$effect(() => {
		connections = data.connections ?? [];
	});
	$effect(() => {
		preferences = data.preferences ?? {};
	});

	// ── Connection CRUD (direct fetch to avoid $env boundary) ──────────

	async function reloadConnections() {
		try {
			const res = await fetch('/api/connections');
			if (res.ok) connections = await res.json();
		} catch {
			/* ignore */
		}
	}

	async function createConnection(conn: Record<string, unknown>) {
		try {
			const res = await fetch('/api/connections', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(conn)
			});
			if (res.ok) {
				toastStore.success('Connection created');
				await reloadConnections();
			} else {
				const err = await res.json().catch(() => ({}));
				toastStore.error(err.detail || 'Failed to create connection');
			}
		} catch {
			toastStore.error('Failed to create connection');
		}
	}

	async function updateConnection(id: string, updates: Record<string, unknown>) {
		try {
			const res = await fetch(`/api/connections/${id}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(updates)
			});
			if (res.ok) {
				toastStore.success('Connection updated');
				await reloadConnections();
			}
		} catch {
			toastStore.error('Failed to update connection');
		}
	}

	async function deleteConnection(id: string) {
		pendingDeleteId = id;
	}

	async function confirmDelete() {
		if (!pendingDeleteId) return;
		try {
			await fetch(`/api/connections/${pendingDeleteId}`, { method: 'DELETE' });
			toastStore.success('Connection deleted');
			pendingDeleteId = null;
			await reloadConnections();
		} catch {
			toastStore.error('Failed to delete connection');
		}
	}

	function cancelDelete() {
		pendingDeleteId = null;
	}

	async function toggleConnection(id: string, enabled: boolean) {
		await updateConnection(id, { enabled });
	}

	async function verifyConnection(conn: AIConnection) {
		connectionStatuses[conn.id] = 'verifying';
		try {
			const res = await fetch(`/api/connections/${conn.id}/verify`, { method: 'POST' });
			const result = await res.json();
			if (result.status === 'connected') {
				connectionStatuses[conn.id] = 'connected';
				connectionModelCounts[conn.id] = result.models?.length ?? 0;
				toastStore.success(`${conn.name}: Connected (${result.models?.length ?? 0} models)`);
			} else {
				connectionStatuses[conn.id] = 'error';
				toastStore.error(`${conn.name}: ${result.message}`);
			}
		} catch {
			connectionStatuses[conn.id] = 'error';
			toastStore.error(`${conn.name}: Connection failed`);
		}
	}

	async function loadAggregatedModels() {
		try {
			const res = await fetch('/api/connections/aggregate-models', { method: 'POST' });
			const result = await res.json();
			aggregatedModels = result.models ?? [];
		} catch {
			/* ignore */
		}
	}

	// ── Dialog handlers ────────────────────────────────────────────────

	function openAddDialog() {
		editingConnection = null;
		dialogOpen = true;
	}

	function openEditDialog(conn: AIConnection) {
		editingConnection = { ...conn, api_key: '' } as any;
		dialogOpen = true;
	}

	function handleSaveConnection(conn: Record<string, unknown>) {
		if (editingConnection) {
			const updates = { ...conn };
			delete updates.id;
			if (!updates.api_key) delete updates.api_key;
			updateConnection(editingConnection.id, updates);
		} else {
			createConnection(conn);
		}
	}

	// ── Feature Preference Save ────────────────────────────────────────

	async function savePreference(feature: string) {
		const pref = preferences[feature];
		if (!pref) return;
		try {
			await fetch(`/api/connections/preferences/${feature}`, {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(pref)
			});
			toastStore.success(`${feature} preference saved`);
		} catch {
			toastStore.error(`Failed to save ${feature} preference`);
		}
	}

	// ── Mount ──────────────────────────────────────────────────────────

	onMount(() => {
		for (const conn of connections) {
			if (conn.enabled) verifyConnection(conn);
		}
		loadAggregatedModels();
	});
</script>

<PageHeader
	title="AI Connections"
	description="Manage AI backend connections shared across Chat, Translation, and Intelligence"
/>

<div class="max-w-3xl space-y-6">
	<!-- Connections Manager -->
	<Card.Root>
		<Card.Header>
			<Card.Title>Connections</Card.Title>
			<Card.Action>
				<Button variant="outline" size="sm" onclick={openAddDialog}>
					<PlusIcon class="mr-1 h-4 w-4" />
					Add Connection
				</Button>
			</Card.Action>
		</Card.Header>
		<Card.Content>
			<div class="space-y-3">
				{#if connections.length === 0}
					<p class="text-sm text-muted-foreground">
						No connections configured. Add an AI backend to get started.
					</p>
				{:else}
					{#each connections as conn (conn.id)}
						<ConnectionCard
							connection={conn as any}
							status={connectionStatuses[conn.id] ?? 'unknown'}
							modelCount={connectionModelCounts[conn.id] ?? 0}
							onverify={() => verifyConnection(conn)}
							onconfigure={() => openEditDialog(conn)}
							ondelete={() => deleteConnection(conn.id)}
							ontoggle={(enabled) => toggleConnection(conn.id, enabled)}
						/>
					{/each}
				{/if}
			</div>

			{#if aggregatedModels.length > 0}
				<div class="mt-4 rounded-md border bg-muted/50 p-3">
					<p class="mb-2 text-xs font-medium text-muted-foreground">
						Aggregated Models ({aggregatedModels.length})
					</p>
					<div class="flex flex-wrap gap-1.5">
						{#each aggregatedModels as model}
							<Badge variant="secondary" class="text-xs">{model.id}</Badge>
						{/each}
					</div>
				</div>
			{/if}
		</Card.Content>
	</Card.Root>

	<!-- Connection Dialog -->
	<ConnectionDialog
		bind:open={dialogOpen}
		connection={editingConnection as any}
		onsave={handleSaveConnection}
		onclose={() => {
			dialogOpen = false;
		}}
	/>

	<!-- Delete Confirmation Dialog -->
	<Dialog.Root
		open={pendingDeleteId !== null}
		onOpenChange={(open) => {
			if (!open) cancelDelete();
		}}
	>
		<Dialog.Content class="sm:max-w-md">
			<Dialog.Header>
				<Dialog.Title>Delete Connection</Dialog.Title>
				<Dialog.Description>
					Are you sure you want to remove this connection? This action cannot be undone.
				</Dialog.Description>
			</Dialog.Header>
			<Dialog.Footer>
				<Button variant="outline" onclick={cancelDelete}>Cancel</Button>
				<Button variant="destructive" onclick={confirmDelete}>Delete</Button>
			</Dialog.Footer>
		</Dialog.Content>
	</Dialog.Root>

	<!-- Feature Preferences -->
	<Card.Root>
		<Card.Header>
			<Card.Title>Feature Model Preferences</Card.Title>
			<Card.Description
				>Select which model each feature uses from the shared pool</Card.Description
			>
		</Card.Header>
		<Card.Content>
			<div class="space-y-4">
				{#each ['chat', 'translation', 'intelligence'] as feature}
					{@const pref = preferences[feature] ?? {
						active_model: '',
						fallback_model: '',
						temperature: 0.7,
						max_tokens: 4096
					}}
					<div class="space-y-2 rounded-md border p-3">
						<Label class="text-sm font-medium capitalize">{feature}</Label>
						<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
							<div class="space-y-1">
								<Label class="text-xs text-muted-foreground">Active Model</Label>
								<select
									class="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
									value={pref.active_model}
									onchange={(e) => {
										if (!preferences[feature]) {
											preferences[feature] = {
												active_model: '',
												fallback_model: '',
												temperature: 0.7,
												max_tokens: 4096
											};
										}
										preferences[feature].active_model = (
											e.target as HTMLSelectElement
										).value;
									}}
								>
									<option value="">None</option>
									{#each aggregatedModels as model}
										<option value={model.id}>{model.id} ({model.engine})</option>
									{/each}
								</select>
							</div>
							<div class="space-y-1">
								<Label class="text-xs text-muted-foreground">Fallback Model</Label>
								<select
									class="w-full rounded-md border bg-background px-3 py-1.5 text-sm"
									value={pref.fallback_model}
									onchange={(e) => {
										if (!preferences[feature]) {
											preferences[feature] = {
												active_model: '',
												fallback_model: '',
												temperature: 0.7,
												max_tokens: 4096
											};
										}
										preferences[feature].fallback_model = (
											e.target as HTMLSelectElement
										).value;
									}}
								>
									<option value="">None</option>
									{#each aggregatedModels as model}
										<option value={model.id}>{model.id} ({model.engine})</option>
									{/each}
								</select>
							</div>
						</div>
						<Button
							size="sm"
							variant="outline"
							onclick={() => savePreference(feature)}
						>
							Save {feature} preference
						</Button>
					</div>
				{/each}
			</div>
		</Card.Content>
	</Card.Root>
</div>
