<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { onMount } from 'svelte';
	import RefreshCwIcon from '@lucide/svelte/icons/refresh-cw';

	interface ServiceStatus {
		name: string;
		healthy: boolean;
		latency_ms?: number;
		last_check?: string;
		error?: string;
	}

	interface ActiveMeeting {
		id: string;
		source: string;
		status: string;
		title?: string;
		started_at?: string;
		duration_seconds?: number;
		chunks_count: number;
		translations_count: number;
	}

	interface DailyActivity {
		date: string;
		meetings: number;
		chunks: number;
		translations: number;
		audio_minutes: number;
	}

	interface DashboardStats {
		total_meetings: number;
		active_meetings: number;
		total_chunks: number;
		total_translations: number;
		total_audio_minutes: number;
		by_source: {
			fireflies: number;
			loopback: number;
			gmeet: number;
			other: number;
		};
		by_status: {
			ephemeral: number;
			active: number;
			completed: number;
			interrupted: number;
		};
		active_meeting_list: ActiveMeeting[];
		daily_activity: DailyActivity[];
		services: ServiceStatus[];
		generated_at: string;
		database_connected: boolean;
	}

	let stats = $state<DashboardStats | null>(null);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let lastRefresh = $state<string>('');

	async function loadStats() {
		loading = true;
		error = null;
		try {
			const res = await fetch('/api/system/dashboard-stats');
			if (!res.ok) throw new Error(`HTTP ${res.status}`);
			stats = await res.json();
			lastRefresh = new Date().toLocaleTimeString();
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to load stats';
		} finally {
			loading = false;
		}
	}

	onMount(() => {
		loadStats();
		// Refresh every 30 seconds
		const interval = setInterval(loadStats, 30000);
		return () => clearInterval(interval);
	});

	function formatDuration(seconds: number): string {
		const h = Math.floor(seconds / 3600);
		const m = Math.floor((seconds % 3600) / 60);
		const s = seconds % 60;
		if (h > 0) return `${h}h ${m}m`;
		if (m > 0) return `${m}m ${s}s`;
		return `${s}s`;
	}

	function formatTimeAgo(dateStr: string): string {
		const diff = Date.now() - new Date(dateStr).getTime();
		const mins = Math.floor(diff / 60000);
		if (mins < 1) return 'just now';
		if (mins < 60) return `${mins}m ago`;
		const hours = Math.floor(mins / 60);
		if (hours < 24) return `${hours}h ago`;
		return `${Math.floor(hours / 24)}d ago`;
	}

	function sourceBadgeVariant(source: string): 'default' | 'secondary' | 'outline' {
		switch (source) {
			case 'fireflies': return 'default';
			case 'loopback': return 'secondary';
			default: return 'outline';
		}
	}

	// Chart helpers
	function maxChartValue(data: DailyActivity[], key: keyof DailyActivity): number {
		const max = Math.max(...data.map(d => Number(d[key]) || 0));
		return max > 0 ? max : 1;
	}

	const quickActions = [
		{ label: 'Loopback Capture', href: '/loopback' },
		{ label: 'Fireflies Connect', href: '/fireflies' },
		{ label: 'Live Feed', href: '/fireflies/live-feed' },
		{ label: 'Meetings', href: '/meetings' },
		{ label: 'Translation Test', href: '/translation/test' },
		{ label: 'Glossary', href: '/fireflies/glossary' },
		{ label: 'Configuration', href: '/config' },
		{ label: 'Data & Logs', href: '/data' }
	];
</script>

<svelte:head>
	<title>Overview — LiveTranslate</title>
</svelte:head>

<PageHeader
	eyebrow="overview · the desk"
	title="Today's Edition"
	description="A live read of the orchestration service, the speakers it's heard, and the conversations in flight."
>
	{#snippet actions()}
		<Button variant="ghost" size="sm" onclick={loadStats} disabled={loading}>
			<RefreshCwIcon class="size-4 mr-1.5 {loading ? 'animate-spin' : ''}" />
			Refresh
		</Button>
	{/snippet}
</PageHeader>

{#if error}
	<div class="alert-banner" role="alert">
		<span class="alert-mark" aria-hidden="true"></span>
		<span class="alert-body">{error}</span>
	</div>
{/if}

<!-- Row 1: Stats Grid — magazine "by the numbers" callouts -->
<section class="stats-grid">
	{#each [
		{ label: 'meetings, total', value: stats?.total_meetings ?? null, accent: false },
		{ label: 'active now', value: stats?.active_meetings ?? 0, accent: true },
		{ label: 'transcriptions', value: stats?.total_chunks ?? null, accent: false },
		{ label: 'translations', value: stats?.total_translations ?? null, accent: false },
		{ label: 'audio, minutes', value: stats?.total_audio_minutes !== undefined ? Math.round(stats.total_audio_minutes) : null, accent: false },
	] as stat}
		<article class="stat" class:stat-accent={stat.accent && (stat.value as number) > 0}>
			<p class="stat-value font-display">
				{stat.value === null ? '—' : stat.value}
			</p>
			<p class="eyebrow stat-label">{stat.label}</p>
		</article>
	{/each}
</section>

<!-- Row 2: Active Meetings + Services -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
	<!-- Active Meetings -->
	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium">Active Meetings</Card.Title>
		</Card.Header>
		<Card.Content>
			{#if stats?.active_meeting_list && stats.active_meeting_list.length > 0}
				<div class="space-y-3">
					{#each stats.active_meeting_list as meeting (meeting.id)}
						<div class="flex items-center justify-between p-2 rounded-md bg-paper-cream">
							<div class="flex items-center gap-2">
								<span class="relative flex h-2 w-2">
									<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-peach opacity-75"></span>
									<span class="relative inline-flex rounded-full h-2 w-2 bg-peach-deep"></span>
								</span>
								<span class="font-mono text-xs">{meeting.id.slice(0, 8)}...</span>
								<Badge variant={sourceBadgeVariant(meeting.source)}>{meeting.source}</Badge>
							</div>
							<div class="flex items-center gap-3 text-xs text-muted-foreground">
								<span>{meeting.chunks_count} chunks</span>
								<span>{meeting.translations_count} xlat</span>
								{#if meeting.duration_seconds}
									<span class="font-mono">{formatDuration(meeting.duration_seconds)}</span>
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{:else}
				<p class="text-sm text-muted-foreground py-4 text-center">
					No active meetings. <a href="/loopback" class="text-primary hover:underline">Start capture</a> to begin.
				</p>
			{/if}
		</Card.Content>
	</Card.Root>

	<!-- Services Health -->
	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium">Services</Card.Title>
		</Card.Header>
		<Card.Content>
			{#if stats?.services}
				<ul class="space-y-2">
					{#each stats.services as service (service.name)}
						<li class="flex items-center justify-between text-sm p-2 rounded-md bg-paper-cream/60">
							<span class="capitalize">{service.name.replace(/-/g, ' ')}</span>
							<div class="flex items-center gap-2">
								{#if service.latency_ms}
									<span class="text-xs text-muted-foreground">{service.latency_ms}ms</span>
								{/if}
								<StatusIndicator
									status={service.healthy ? 'healthy' : 'down'}
									label={service.healthy ? 'Healthy' : 'Down'}
								/>
							</div>
						</li>
					{/each}
				</ul>
			{:else}
				<p class="text-sm text-muted-foreground py-4 text-center">Loading services...</p>
			{/if}
		</Card.Content>
	</Card.Root>
</div>

<!-- Row 3: Charts -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
	<!-- Activity Chart -->
	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium">7-Day Activity</Card.Title>
		</Card.Header>
		<Card.Content>
			{#if stats?.daily_activity && stats.daily_activity.length > 0}
				<div class="h-40 flex items-end gap-1">
					{#each stats.daily_activity as day, i (day.date)}
						{@const maxMeetings = maxChartValue(stats.daily_activity, 'meetings')}
						{@const height = (day.meetings / maxMeetings) * 100}
						<div class="flex-1 flex flex-col items-center gap-1">
							<div
								class="w-full bg-peach-deep/85 rounded-t transition-all"
								style="height: {Math.max(height, 4)}%"
								title="{day.meetings} meetings"
							></div>
							<span class="text-[10px] text-muted-foreground">
								{new Date(day.date).toLocaleDateString('en-US', { weekday: 'short' })}
							</span>
						</div>
					{/each}
				</div>
				<div class="flex justify-between mt-2 text-xs text-muted-foreground">
					<span>Meetings per day</span>
					<span>Max: {maxChartValue(stats.daily_activity, 'meetings')}</span>
				</div>
			{:else}
				<p class="text-sm text-muted-foreground py-8 text-center">No activity data</p>
			{/if}
		</Card.Content>
	</Card.Root>

	<!-- Source Breakdown -->
	<Card.Root>
		<Card.Header class="pb-2">
			<Card.Title class="text-sm font-medium">Meetings by Source</Card.Title>
		</Card.Header>
		<Card.Content>
			{#if stats?.by_source}
				{@const total = stats.by_source.fireflies + stats.by_source.loopback + stats.by_source.gmeet + stats.by_source.other}
				{#if total > 0}
					<div class="space-y-3">
						{#each [
							{ label: 'Fireflies', count: stats.by_source.fireflies, color: 'bg-slate-blue' },
							{ label: 'Loopback', count: stats.by_source.loopback, color: 'bg-sage' },
							{ label: 'Google Meet', count: stats.by_source.gmeet, color: 'bg-ochre' },
							{ label: 'Other', count: stats.by_source.other, color: 'bg-plum' },
						] as item}
							{#if item.count > 0}
								<div>
									<div class="flex justify-between text-sm mb-1">
										<span>{item.label}</span>
										<span class="text-muted-foreground">{item.count} ({Math.round(item.count / total * 100)}%)</span>
									</div>
									<div class="h-2 bg-muted rounded-full overflow-hidden">
										<div
											class="{item.color} h-full rounded-full transition-all"
											style="width: {(item.count / total) * 100}%"
										></div>
									</div>
								</div>
							{/if}
						{/each}
					</div>
				{:else}
					<p class="text-sm text-muted-foreground py-8 text-center">No meetings yet</p>
				{/if}
			{/if}
		</Card.Content>
	</Card.Root>
</div>

<!-- Row 4: Quick Actions -->
<Card.Root>
	<Card.Header class="pb-2">
		<Card.Title class="text-sm font-medium">Quick Actions</Card.Title>
	</Card.Header>
	<Card.Content>
		<div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
			{#each quickActions as action (action.href)}
				<Button variant="outline" href={action.href} class="justify-start h-9 px-3">
					{action.label}
				</Button>
			{/each}
		</div>
	</Card.Content>
</Card.Root>

<!-- Footer — colophon -->
{#if lastRefresh}
	<p class="font-mono text-xs text-ink-faint text-center mt-6">
		printed {lastRefresh}
		{#if stats?.database_connected}
			<span class="ml-2 text-sage">· db connected</span>
		{:else if stats}
			<span class="ml-2 text-ochre">· db unavailable</span>
		{/if}
	</p>
{/if}

<style>
	/* ── Editorial home callouts — D5.1 ───────────────────────── */
	.stats-grid {
		display: grid;
		grid-template-columns: repeat(2, minmax(0, 1fr));
		gap: 1px;
		background: var(--rule);
		border: 1px solid var(--rule);
		border-radius: 0.375rem;
		overflow: hidden;
		margin-bottom: 1.5rem;
	}
	@media (min-width: 1024px) {
		.stats-grid { grid-template-columns: repeat(5, minmax(0, 1fr)); }
	}
	.stat {
		background: var(--paper);
		padding: 1.25rem 1.25rem 1rem;
		min-height: 6.5rem;
	}
	.stat-accent {
		background: color-mix(in srgb, var(--peach) 10%, var(--paper));
	}
	.stat-value {
		margin: 0;
		font-size: 2.75rem;
		line-height: 1;
		color: var(--ink);
		font-variation-settings: "opsz" 96, "SOFT" 50, "WONK" 1;
		letter-spacing: -0.04em;
		font-feature-settings: "lnum"; /* lining figures for big stat numbers */
	}
	.stat-accent .stat-value { color: var(--peach-deep); }
	.stat-label {
		margin: 0.5rem 0 0;
	}

	/* ── Alert banner ─────────────────────────────────────────── */
	.alert-banner {
		display: flex;
		align-items: center;
		gap: 0.875rem;
		padding: 0.75rem 1.25rem;
		background: color-mix(in srgb, var(--oxblood) 12%, var(--paper));
		border-left: 3px solid var(--oxblood);
		color: var(--ink);
		margin-bottom: 1rem;
		border-radius: 0.25rem;
	}
	.alert-mark {
		width: 0.5rem;
		height: 0.5rem;
		border-radius: 9999px;
		background: var(--oxblood);
	}
</style>
