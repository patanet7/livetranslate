<script lang="ts">
	import { invalidateAll } from '$app/navigation';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import * as Dialog from '$lib/components/ui/dialog';
	import { toastStore } from '$lib/stores/toast.svelte';
	import type { FirefliesSession } from '$lib/types';

	let { data } = $props();

	// Track which sessions are being disconnected
	let disconnecting = $state<Record<string, boolean>>({});

	// Confirmation dialog state
	let confirmDialogOpen = $state(false);
	let sessionToDisconnect = $state<FirefliesSession | null>(null);

	// Derived stats
	let totalSessions = $derived(data.sessions.length);
	let connectedCount = $derived(
		data.sessions.filter((s) => s.connection_status === 'CONNECTED').length
	);
	let totalChunks = $derived(data.sessions.reduce((sum, s) => sum + s.chunks_received, 0));
	let totalTranslations = $derived(
		data.sessions.reduce((sum, s) => sum + s.translations_completed, 0)
	);

	function statusVariant(
		status: FirefliesSession['connection_status']
	): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (status) {
			case 'CONNECTED':
				return 'default';
			case 'CONNECTING':
				return 'secondary';
			case 'ERROR':
			case 'DISCONNECTED':
				return 'destructive';
			default:
				return 'outline';
		}
	}

	function statusDotClass(status: FirefliesSession['connection_status']): string {
		switch (status) {
			case 'CONNECTED':
				return 'bg-green-500';
			case 'CONNECTING':
				return 'bg-yellow-500';
			case 'ERROR':
				return 'bg-red-500';
			case 'DISCONNECTED':
				return 'bg-gray-400';
			default:
				return 'bg-gray-400';
		}
	}

	function relativeTime(isoDate: string): string {
		const now = Date.now();
		const then = new Date(isoDate).getTime();
		const diffMs = now - then;

		if (isNaN(then)) return 'unknown';

		const seconds = Math.floor(diffMs / 1000);
		if (seconds < 60) return `${seconds}s ago`;

		const minutes = Math.floor(seconds / 60);
		if (minutes < 60) return `${minutes}m ago`;

		const hours = Math.floor(minutes / 60);
		if (hours < 24) return `${hours}h ago`;

		const days = Math.floor(hours / 24);
		return `${days}d ago`;
	}

	function truncateId(id: string): string {
		if (id.length <= 12) return id;
		return id.slice(0, 12) + '...';
	}

	function promptDisconnect(session: FirefliesSession) {
		sessionToDisconnect = session;
		confirmDialogOpen = true;
	}

	async function confirmDisconnect() {
		if (!sessionToDisconnect) return;
		const sessionId = sessionToDisconnect.session_id;
		confirmDialogOpen = false;

		disconnecting[sessionId] = true;
		try {
			const res = await fetch('/api/fireflies/disconnect', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ session_id: sessionId })
			});
			if (res.ok) {
				toastStore.success('Session disconnected');
				invalidateAll();
			} else {
				toastStore.error('Failed to disconnect session');
			}
		} catch {
			toastStore.error('Network error disconnecting session');
		} finally {
			disconnecting[sessionId] = false;
			sessionToDisconnect = null;
		}
	}

	function cancelDisconnect() {
		confirmDialogOpen = false;
		sessionToDisconnect = null;
	}
</script>

<PageHeader title="Session Management" description="Browse and manage active Fireflies sessions">
	{#snippet actions()}
		<Button variant="outline" size="sm" onclick={() => invalidateAll()}>Refresh</Button>
	{/snippet}
</PageHeader>

<!-- Stats Grid -->
<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Description>Total Sessions</Card.Description>
		</Card.Header>
		<Card.Content>
			<p class="text-2xl font-bold">{totalSessions}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Description>Connected</Card.Description>
		</Card.Header>
		<Card.Content>
			<p class="text-2xl font-bold text-green-600 dark:text-green-400">{connectedCount}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Description>Total Chunks</Card.Description>
		</Card.Header>
		<Card.Content>
			<p class="text-2xl font-bold">{totalChunks}</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Description>Total Translations</Card.Description>
		</Card.Header>
		<Card.Content>
			<p class="text-2xl font-bold">{totalTranslations}</p>
		</Card.Content>
	</Card.Root>
</div>

<!-- Session Cards -->
{#if data.sessions.length === 0}
	<Card.Root>
		<Card.Content class="py-12">
			<div class="text-center">
				<p class="text-muted-foreground mb-2">No active sessions.</p>
				<p class="text-sm text-muted-foreground">
					Connect to Fireflies to start a session.
				</p>
				<Button variant="outline" size="sm" class="mt-4" href="/fireflies">
					Go to Fireflies
				</Button>
			</div>
		</Card.Content>
	</Card.Root>
{:else}
	<div class="space-y-4">
		{#each data.sessions as session (session.session_id)}
			<Card.Root>
				<Card.Header>
					<div class="flex items-center justify-between flex-wrap gap-2">
						<div class="flex items-center gap-3">
							<span
								class="inline-block size-2.5 rounded-full {statusDotClass(session.connection_status)}"
							></span>
							<Card.Title class="font-mono text-sm">
								{truncateId(session.session_id)}
							</Card.Title>
							<span class="text-sm text-muted-foreground">
								transcript: {session.transcript_id}
							</span>
						</div>
						<Badge variant={statusVariant(session.connection_status)}>
							{session.connection_status}
						</Badge>
					</div>
				</Card.Header>
				<Card.Content>
					<div class="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted-foreground mb-4">
						<span>Chunks: <span class="text-foreground font-medium">{session.chunks_received}</span></span>
						<span>Translations: <span class="text-foreground font-medium">{session.translations_completed}</span></span>
						<span>Speakers: <span class="text-foreground font-medium">{session.speakers_detected.length}</span></span>
						{#if session.error_count > 0}
							<span class="text-destructive">Errors: {session.error_count}</span>
						{/if}
					</div>
					<div class="flex items-center justify-between flex-wrap gap-2">
						<span class="text-xs text-muted-foreground">
							Connected {relativeTime(session.connected_at)}
						</span>
						<div class="flex items-center gap-2">
							<Button
								variant="outline"
								size="sm"
								href="/captions?session={session.session_id}"
								target="_blank"
							>
								Captions
							</Button>
							<Button
								variant="outline"
								size="sm"
								href="/fireflies/live-feed?session={session.session_id}"
							>
								Live Feed
							</Button>
							<Button
								variant="outline"
								size="sm"
								href="/data?session={session.session_id}"
							>
								Data
							</Button>
							<Button
								variant="destructive"
								size="sm"
								disabled={disconnecting[session.session_id] ?? false}
								onclick={() => promptDisconnect(session)}
							>
								{#if disconnecting[session.session_id]}
									Disconnecting...
								{:else}
									Disconnect
								{/if}
							</Button>
						</div>
					</div>
				</Card.Content>
			</Card.Root>
		{/each}
	</div>
{/if}

<!-- Disconnect Confirmation Dialog -->
<Dialog.Root bind:open={confirmDialogOpen}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Disconnect Session</Dialog.Title>
			<Dialog.Description>
				Are you sure you want to disconnect session
				{#if sessionToDisconnect}
					<span class="font-mono">{truncateId(sessionToDisconnect.session_id)}</span>?
				{/if}
				This action cannot be undone.
			</Dialog.Description>
		</Dialog.Header>
		<Dialog.Footer>
			<Button variant="outline" onclick={cancelDisconnect}>Cancel</Button>
			<Button variant="destructive" onclick={confirmDisconnect}>Disconnect</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
