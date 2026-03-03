<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import SyncBadge from '$lib/components/meetings/SyncBadge.svelte';

	let { data, children } = $props();

	const meeting = $derived(data.meeting);

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
			{#if meeting.status === 'live'}
				<Button href="/meetings/{meeting.id}/live" variant="default">
					View Live
				</Button>
			{/if}
		</div>
	</div>
</div>

{@render children()}
