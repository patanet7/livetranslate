<script lang="ts">
	import { goto } from '$app/navigation';
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

<!-- Search & Filters -->
<div class="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center">
	<form onsubmit={handleSearch} class="flex flex-1 gap-2">
		<Input
			placeholder="Search meetings..."
			bind:value={searchQuery}
			class="max-w-sm"
		/>
		<Button type="submit" variant="outline">Search</Button>
		{#if data.query}
			<Button variant="ghost" onclick={() => goto('/meetings')}>Clear</Button>
		{/if}
	</form>

	<div class="flex gap-1">
		{#each ['all', 'live', 'completed'] as filter}
			<Button
				variant={statusFilter === filter ? 'default' : 'outline'}
				size="sm"
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
