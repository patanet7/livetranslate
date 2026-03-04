<script lang="ts">
	import { goto, invalidateAll } from '$app/navigation';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Badge } from '$lib/components/ui/badge';
	import SyncBadge from '$lib/components/meetings/SyncBadge.svelte';
	import MeetingSkeleton from '$lib/components/meetings/MeetingSkeleton.svelte';
	import type { Meeting } from '$lib/types';

	let { data } = $props();

	let searchQuery = $state(data.query ?? '');
	let statusFilter = $state<'all' | 'live' | 'completed'>('all');

	// Sync All state
	let syncing = $state(false);
	let syncMessage = $state('');

	// Sync status
	let lastSyncAt = $state<string | null>(null);

	// Invite Bot dialog state
	let showInviteDialog = $state(false);
	let inviteLink = $state('');
	let inviteTitle = $state('');
	let inviteDuration = $state(60);
	let inviting = $state(false);
	let inviteMessage = $state('');

	// Fetch sync status on mount
	$effect(() => {
		fetch('/api/fireflies/sync-status')
			.then((r) => r.json())
			.then((d) => {
				lastSyncAt = d.last_sync_at ?? null;
			})
			.catch(() => {});
	});

	const filtered = $derived(
		statusFilter === 'all'
			? data.meetings
			: data.meetings.filter((m: Meeting) => m.status === statusFilter)
	);

	function handleSearch(e: Event) {
		e.preventDefault();
		const params = new URLSearchParams();
		if (searchQuery.trim()) params.set('q', searchQuery.trim());
		goto(`/meetings?${params.toString()}`);
	}

	async function handleSyncAll() {
		syncing = true;
		syncMessage = '';
		try {
			const res = await fetch('/api/fireflies/sync-all', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({})
			});
			const result = await res.json();
			if (res.ok) {
				const parts = [`Synced ${result.synced} meetings`];
				if (result.skipped) parts.push(`${result.skipped} already up-to-date`);
				if (result.errors) parts.push(`${result.errors} errors`);
				parts.push(`${result.api_calls_used ?? '?'} API calls used`);
				syncMessage = parts.join(' · ');
				lastSyncAt = new Date().toISOString();
				await invalidateAll();
			} else if (res.status === 429) {
				const detail = result.detail;
				const retryAfter =
					typeof detail === 'object' ? detail?.retry_after : null;
				syncMessage = retryAfter
					? `Rate limited — resets ${retryAfter}`
					: `Rate limited — try again later`;
			} else {
				const msg =
					typeof result.detail === 'string'
						? result.detail
						: result.detail?.message ?? result.error ?? 'Unknown error';
				syncMessage = `Sync failed: ${msg}`;
			}
		} catch (err) {
			syncMessage = `Sync failed: ${err instanceof Error ? err.message : 'Network error'}`;
		} finally {
			syncing = false;
			setTimeout(() => (syncMessage = ''), 5000);
		}
	}

	async function handleInviteBot() {
		if (!inviteLink.trim()) return;
		inviting = true;
		inviteMessage = '';
		try {
			const res = await fetch('/api/fireflies/invite-bot', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					meeting_link: inviteLink.trim(),
					title: inviteTitle.trim() || undefined,
					duration: inviteDuration
				})
			});
			const result = await res.json();
			if (res.ok && result.success) {
				inviteMessage = 'Fireflies bot invited. Will auto-connect when ready.';
				showInviteDialog = false;
				inviteLink = '';
				inviteTitle = '';
				inviteDuration = 60;
			} else {
				inviteMessage = `Failed: ${result.detail ?? result.error ?? 'Unknown error'}`;
			}
		} catch (err) {
			inviteMessage = `Failed: ${err instanceof Error ? err.message : 'Network error'}`;
		} finally {
			inviting = false;
			setTimeout(() => (inviteMessage = ''), 5000);
		}
	}

	function formatDate(iso: string | null): string {
		if (!iso) return '--';
		return new Date(iso).toLocaleDateString(undefined, {
			month: 'short',
			day: 'numeric',
			year: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	function formatDuration(seconds: number | null): string {
		if (!seconds) return '--';
		const m = Math.floor(seconds / 60);
		const s = seconds % 60;
		return m > 0 ? `${m}m ${s}s` : `${s}s`;
	}
</script>

<PageHeader title="Meetings" description="Browse all meeting transcripts, translations, and insights" />

<!-- Action Buttons -->
<div class="mb-4 flex flex-wrap items-center gap-2">
	<Button variant="outline" onclick={handleSyncAll} disabled={syncing}>
		{syncing ? 'Syncing...' : 'Sync from Fireflies'}
	</Button>
	<Button variant="outline" onclick={() => (showInviteDialog = !showInviteDialog)}>
		Invite Bot
	</Button>
	{#if lastSyncAt}
		<span class="text-xs text-muted-foreground">
			Last synced: {formatDate(lastSyncAt)}
		</span>
	{/if}
</div>

<!-- Toast messages -->
{#if syncMessage}
	<div class="mb-4 rounded-md border px-4 py-2 text-sm" role="status">{syncMessage}</div>
{/if}
{#if inviteMessage}
	<div class="mb-4 rounded-md border px-4 py-2 text-sm" role="status">{inviteMessage}</div>
{/if}

<!-- Invite Bot Dialog -->
{#if showInviteDialog}
	<Card.Root class="mb-6">
		<Card.Header>
			<Card.Title>Invite Fireflies Bot</Card.Title>
			<Card.Description>Paste a meeting link to invite the Fireflies bot. Rate limit: 3 invites per 20 minutes.</Card.Description>
		</Card.Header>
		<Card.Content>
			<form
				onsubmit={(e) => { e.preventDefault(); handleInviteBot(); }}
				class="space-y-4"
			>
				<div>
					<label for="invite-link" class="text-sm font-medium">Meeting Link *</label>
					<Input id="invite-link" placeholder="https://meet.google.com/..." bind:value={inviteLink} required />
				</div>
				<div>
					<label for="invite-title" class="text-sm font-medium">Title (optional)</label>
					<Input id="invite-title" placeholder="Weekly Standup" bind:value={inviteTitle} />
				</div>
				<div>
					<label for="invite-duration" class="text-sm font-medium">Duration: {inviteDuration} min</label>
					<input
						id="invite-duration"
						type="range"
						min="15"
						max="120"
						step="5"
						bind:value={inviteDuration}
						class="w-full"
					/>
				</div>
				<div class="flex gap-2">
					<Button type="submit" disabled={inviting || !inviteLink.trim()}>
						{inviting ? 'Inviting...' : 'Send Invite'}
					</Button>
					<Button variant="ghost" type="button" onclick={() => (showInviteDialog = false)}>
						Cancel
					</Button>
				</div>
			</form>
		</Card.Content>
	</Card.Root>
{/if}

<!-- Search & Filters -->
<div class="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center">
	<form onsubmit={handleSearch} class="flex flex-1 gap-2">
		<Input
			placeholder="Search meetings..."
			aria-label="Search meetings"
			bind:value={searchQuery}
			class="max-w-sm"
		/>
		<Button type="submit" variant="outline">Search</Button>
		{#if data.query}
			<Button variant="ghost" onclick={() => goto('/meetings')}>Clear</Button>
		{/if}
	</form>

	<div class="flex gap-1" role="group" aria-label="Filter meetings by status">
		{#each ['all', 'live', 'completed'] as filter}
			<Button
				variant={statusFilter === filter ? 'default' : 'outline'}
				size="sm"
				aria-pressed={statusFilter === filter}
				onclick={() => (statusFilter = filter as typeof statusFilter)}
			>
				{filter === 'all' ? 'All' : filter.charAt(0).toUpperCase() + filter.slice(1)}
			</Button>
		{/each}
	</div>
</div>

<!-- Search results info -->
{#if data.query}
	<p class="mb-4 text-sm text-muted-foreground">
		{data.total} result{data.total === 1 ? '' : 's'} for "{data.query}"
	</p>
{/if}

<!-- Meeting List -->
{#if filtered.length === 0}
	<Card.Root>
		<Card.Content class="py-16">
			<div class="text-center space-y-4">
				<div class="text-5xl">&#x1F4CB;</div>
				<h3 class="text-lg font-semibold">No meetings yet</h3>
				<p class="text-muted-foreground max-w-md mx-auto">
					Connect to a Fireflies transcript to start capturing meeting data, or upload a transcript file.
				</p>
				<div class="flex gap-2 justify-center">
					<Button href="/fireflies">Connect to Fireflies</Button>
				</div>
			</div>
		</Card.Content>
	</Card.Root>
{:else}
	<div class="space-y-3">
		{#each filtered as meeting (meeting.id)}
			<a href="/meetings/{meeting.id}" class="block">
				<Card.Root class="transition-colors hover:bg-accent/50">
					<Card.Content class="flex items-center gap-4 py-4">
						<div class="flex-1 min-w-0">
							<div class="flex items-center gap-2 mb-1">
								<h3 class="font-medium truncate">
									{meeting.title ?? 'Untitled Meeting'}
								</h3>
								<SyncBadge status={meeting.sync_status ?? 'none'} compact />
								<Badge variant="outline" class="text-xs">{meeting.source}</Badge>
							</div>
							<div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
								<span>{formatDate(meeting.created_at)}</span>
								<span>{formatDuration(meeting.duration)}</span>
								<span>{meeting.sentence_count} sentences</span>
								{#if meeting.chunk_count}
									<span>{meeting.chunk_count} chunks</span>
								{/if}
							</div>
						</div>
						<Badge
							variant={meeting.status === 'live' ? 'default' : 'secondary'}
							class={meeting.status === 'live' ? 'bg-green-600 animate-pulse' : ''}
						>
							{meeting.status}
						</Badge>
					</Card.Content>
				</Card.Root>
			</a>
		{/each}
	</div>

	<!-- Pagination -->
	{#if data.total >= data.limit}
		<div class="mt-6 flex justify-center gap-2">
			{#if data.offset > 0}
				<Button
					variant="outline"
					href="/meetings?{data.query ? `q=${encodeURIComponent(data.query)}&` : ''}offset={Math.max(0, data.offset - data.limit)}"
				>
					Previous
				</Button>
			{/if}
			<Button variant="outline" href="/meetings?{data.query ? `q=${encodeURIComponent(data.query)}&` : ''}offset={data.offset + data.limit}">
				Next
			</Button>
		</div>
	{/if}
{/if}
