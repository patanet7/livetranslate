<script lang="ts">
	import { invalidateAll } from '$app/navigation';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import * as Select from '$lib/components/ui/select';
	import SyncBadge from '$lib/components/meetings/SyncBadge.svelte';

	let { data, children } = $props();

	const meeting = $derived(data.meeting);

	let syncingMeeting = $state(false);
	let syncMsg = $state('');

	// ── Bot state ──────────────────────────────────────────────────────
	let sendingBot = $state(false);
	let botConnectionId = $state<string | null>(null);
	let botStatus = $state<string | null>(null);
	let botStatusInterval = $state<ReturnType<typeof setInterval> | null>(null);
	let selectedLanguage = $state<string>('auto');

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

	// ── Bot management ─────────────────────────────────────────────────
	async function handleSendBot() {
		if (!meeting.meeting_link) return;
		sendingBot = true;
		try {
			const res = await fetch('/api/bot/start', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					meeting_url: meeting.meeting_link,
					user_token: 'dashboard',
					user_id: 'dashboard-user',
					language: selectedLanguage === 'auto' ? 'en' : selectedLanguage,
					task: 'transcribe',
					enable_virtual_webcam: false,
					metadata: {
						meeting_id: meeting.id,
						meeting_title: meeting.title,
						source: 'dashboard',
						auto_language: selectedLanguage === 'auto'
					}
				})
			});
			const result = await res.json();
			if (res.ok && result.connection_id) {
				botConnectionId = result.connection_id;
				botStatus = 'spawning';
				window.open(meeting.meeting_link, '_blank');
				startBotStatusPolling(result.connection_id);
			} else {
				botStatus = 'error';
				syncMsg = `Bot failed: ${result.detail ?? result.error ?? 'Unknown error'}`;
			}
		} catch (err) {
			botStatus = 'error';
			syncMsg = `Bot failed: ${err instanceof Error ? err.message : 'Network error'}`;
		} finally {
			sendingBot = false;
		}
	}

	function startBotStatusPolling(connectionId: string) {
		if (botStatusInterval) clearInterval(botStatusInterval);
		let polls = 0;
		botStatusInterval = setInterval(async () => {
			if (++polls > 100) { stopBotPolling(); return; }
			try {
				const res = await fetch(`/api/bot/${connectionId}/status`);
				if (!res.ok) return;
				const data = await res.json();
				botStatus = data.status;
				if (['completed', 'failed', 'stopped', 'error'].includes(data.status)) {
					stopBotPolling();
				}
			} catch { /* ignore polling errors */ }
		}, 3000);
	}

	function stopBotPolling() {
		if (botStatusInterval) { clearInterval(botStatusInterval); botStatusInterval = null; }
	}

	$effect(() => { return () => stopBotPolling(); });
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

			{#if meeting.meeting_link}
				<Select.Root type="single" bind:value={selectedLanguage}>
					<Select.Trigger size="sm" class="w-28 text-xs">
						{selectedLanguage === 'auto' ? 'Auto Lang' : selectedLanguage.toUpperCase()}
					</Select.Trigger>
					<Select.Content>
						<Select.Item value="auto" label="Auto (detect)">Auto (detect)</Select.Item>
						<Select.Item value="en" label="English">English</Select.Item>
						<Select.Item value="zh" label="Chinese">Chinese</Select.Item>
						<Select.Item value="es" label="Spanish">Spanish</Select.Item>
						<Select.Item value="fr" label="French">French</Select.Item>
						<Select.Item value="de" label="German">German</Select.Item>
						<Select.Item value="ja" label="Japanese">Japanese</Select.Item>
						<Select.Item value="ko" label="Korean">Korean</Select.Item>
					</Select.Content>
				</Select.Root>

				<Button
					variant="default"
					size="sm"
					onclick={handleSendBot}
					disabled={sendingBot || botStatus === 'spawning' || botStatus === 'active'}
				>
					{#if sendingBot}
						Sending...
					{:else if botStatus === 'spawning'}
						Bot Joining...
					{:else if botStatus === 'active'}
						Bot Active
					{:else}
						Send Bot
					{/if}
				</Button>
			{/if}
		</div>
	</div>
	{#if syncMsg}
		<div class="mt-2 rounded-md border px-4 py-2 text-sm" role="status">{syncMsg}</div>
	{/if}
	{#if botStatus && botStatus !== 'error'}
		<div class="mt-2 flex items-center gap-2 rounded-md border px-4 py-2 text-sm">
			<Badge variant={botStatus === 'active' ? 'default' : 'secondary'}
				class={botStatus === 'active' ? 'bg-green-600' : botStatus === 'spawning' ? 'animate-pulse bg-yellow-600' : ''}>
				{botStatus}
			</Badge>
			<span>
				{#if botStatus === 'spawning'}Bot starting up and joining the meeting...
				{:else if botStatus === 'active'}Bot is active in the meeting.
				{:else if botStatus === 'completed'}Bot has left the meeting.
				{:else}Bot: {botStatus}{/if}
			</span>
		</div>
	{/if}
</div>

{@render children()}
