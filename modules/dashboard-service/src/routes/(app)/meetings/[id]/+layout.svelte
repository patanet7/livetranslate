<script lang="ts">
	import { invalidateAll } from '$app/navigation';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import SyncBadge from '$lib/components/meetings/SyncBadge.svelte';

	let { data, children } = $props();

	const meeting = $derived(data.meeting);

	let syncingMeeting = $state(false);
	let syncMsg = $state('');

	const needsSync = $derived(
		meeting.source === 'fireflies' &&
		(meeting.insight_count === 0 || meeting.sync_status === 'none' || !meeting.sync_status)
	);

	async function handleSyncMeeting() {
		syncingMeeting = true;
		syncMsg = '';
		try {
			const res = await fetch(`/api/meetings/${meeting.id}/sync`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' }
			});
			const result = await res.json();
			if (res.ok && result.success) {
				syncMsg = 'Sync started — intelligence data will appear shortly.';
				// Poll for completion
				setTimeout(async () => {
					await invalidateAll();
					syncMsg = '';
				}, 5000);
			} else {
				syncMsg = `Sync failed: ${result.detail ?? result.error ?? 'Unknown error'}`;
			}
		} catch (err) {
			syncMsg = `Sync failed: ${err instanceof Error ? err.message : 'Network error'}`;
		} finally {
			syncingMeeting = false;
		}
	}

	function formatDate(iso: string | null): string {
		if (!iso) return '--';
		return new Date(iso).toLocaleDateString(undefined, {
			month: 'short', day: 'numeric', year: 'numeric',
			hour: '2-digit', minute: '2-digit'
		});
	}

	function formatDuration(seconds: number | null): string {
		if (!seconds) return '--';
		const m = Math.floor(seconds / 60);
		return m > 0 ? `${m} min` : `${seconds}s`;
	}
</script>

<!-- Meeting Header -->
<div class="mb-6">
	<div class="mb-2">
		<a href="/meetings" class="text-sm text-muted-foreground hover:text-foreground transition-colors">
			← Back to Meetings
		</a>
	</div>

	<div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
		<div>
			<h1 class="text-2xl font-bold">
				{meeting.title ?? 'Untitled Meeting'}
			</h1>
			<div class="flex flex-wrap items-center gap-2 mt-1 text-sm text-muted-foreground">
				<Badge variant={meeting.status === 'live' ? 'default' : 'secondary'}
					class={meeting.status === 'live' ? 'bg-green-600 animate-pulse' : ''}>
					{meeting.status}
				</Badge>
				<SyncBadge status={meeting.sync_status ?? 'none'} />
				<span>{formatDate(meeting.created_at)}</span>
				{#if meeting.duration}
					<span>· {formatDuration(meeting.duration)}</span>
				{/if}
				{#if meeting.sentence_count}
					<span>· {meeting.sentence_count} sentences</span>
				{/if}
			</div>
		</div>

		<div class="flex gap-2">
			{#if needsSync}
				<Button variant="outline" onclick={handleSyncMeeting} disabled={syncingMeeting}>
					{syncingMeeting ? 'Syncing...' : 'Sync from Fireflies'}
				</Button>
			{/if}
			{#if meeting.status === 'live'}
				<Button href="/meetings/{meeting.id}/live" variant="default">
					View Live
				</Button>
			{/if}
		</div>
	</div>
	{#if syncMsg}
		<div class="mt-2 rounded-md border px-4 py-2 text-sm" role="status">{syncMsg}</div>
	{/if}
</div>

{@render children()}
