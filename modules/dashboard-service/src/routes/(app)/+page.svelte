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

<PageHeader title="Dashboard" description="LiveTranslate system overview">
	{#snippet actions()}
		<Button variant="ghost" size="sm" onclick={loadStats} disabled={loading}>
			<RefreshCwIcon class="size-4 mr-1 {loading ? 'animate-spin' : ''}" />
			Refresh
		</Button>
	{/snippet}
</PageHeader>

{#if error}
	<div class="bg-destructive/10 text-destructive px-4 py-3 rounded-md mb-4">
		{error}
	</div>
{/if}

<!-- Row 1: Stats Grid -->
<div class="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{stats?.total_meetings ?? '—'}</p>
			<p class="text-sm text-muted-foreground mt-1">Total Meetings</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight text-green-500">{stats?.active_meetings ?? 0}</p>
			<p class="text-sm text-muted-foreground mt-1">Active Now</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{stats?.total_chunks ?? '—'}</p>
			<p class="text-sm text-muted-foreground mt-1">Transcriptions</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{stats?.total_translations ?? '—'}</p>
			<p class="text-sm text-muted-foreground mt-1">Translations</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{stats?.total_audio_minutes?.toFixed(0) ?? '—'}</p>
			<p class="text-sm text-muted-foreground mt-1">Audio Minutes</p>
		</Card.Content>
	</Card.Root>
</div>

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
						<div class="flex items-center justify-between p-2 rounded-md bg-muted/50">
							<div class="flex items-center gap-2">
								<span class="relative flex h-2 w-2">
									<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
									<span class="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
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
						<li class="flex items-center justify-between text-sm p-2 rounded-md bg-muted/30">
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
								class="w-full bg-primary/80 rounded-t transition-all"
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
							{ label: 'Fireflies', count: stats.by_source.fireflies, color: 'bg-blue-500' },
							{ label: 'Loopback', count: stats.by_source.loopback, color: 'bg-green-500' },
							{ label: 'Google Meet', count: stats.by_source.gmeet, color: 'bg-yellow-500' },
							{ label: 'Other', count: stats.by_source.other, color: 'bg-gray-500' },
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
			{#each quickActions as action}
				<Button variant="outline" href={action.href} class="justify-start h-9 px-3">
					{action.label}
				</Button>
			{/each}
		</div>
	</Card.Content>
</Card.Root>

<!-- Footer -->
{#if lastRefresh}
	<p class="text-xs text-muted-foreground text-center mt-4">
		Last updated: {lastRefresh}
		{#if stats?.database_connected}
			<span class="ml-2 text-green-500">DB connected</span>
		{:else if stats}
			<span class="ml-2 text-yellow-500">DB unavailable</span>
		{/if}
	</p>
{/if}
