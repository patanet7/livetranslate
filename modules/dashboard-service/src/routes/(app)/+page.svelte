<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import { healthStore } from '$lib/stores/health.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import type { FirefliesSession } from '$lib/types';

	let { data } = $props();

	let sessions = $derived<FirefliesSession[]>(data.sessions ?? []);

	let totalSessions = $derived(sessions.length);
	let connectedCount = $derived(
		sessions.filter((s) => s.connection_status === 'CONNECTED').length
	);
	let totalChunks = $derived(sessions.reduce((sum, s) => sum + (s.chunks_received ?? 0), 0));
	let totalTranslations = $derived(
		sessions.reduce((sum, s) => sum + (s.translations_completed ?? 0), 0)
	);
	let recentSessions = $derived(sessions.slice(0, 5));

	// Derive service health from both server data and the client-side health store
	let systemHealth = $derived(data.systemHealth as Record<string, unknown> | null);
	let translationHealth = $derived(data.translationHealth as Record<string, unknown> | null);

	function serviceStatus(
		healthy: boolean | undefined
	): 'healthy' | 'down' | 'unknown' {
		if (healthy === undefined) return 'unknown';
		return healthy ? 'healthy' : 'down';
	}

	function connectionBadgeVariant(
		status: string
	): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (status) {
			case 'CONNECTED':
				return 'default';
			case 'CONNECTING':
				return 'secondary';
			case 'ERROR':
				return 'destructive';
			default:
				return 'outline';
		}
	}

	function truncateId(id: string): string {
		if (id.length <= 12) return id;
		return id.slice(0, 8) + '...';
	}

	const quickActions = [
		{ label: 'Connect to Fireflies', href: '/fireflies' },
		{ label: 'Live Feed', href: '/fireflies/live-feed' },
		{ label: 'Session Manager', href: '/sessions' },
		{ label: 'Translation Test', href: '/translation/test' },
		{ label: 'Glossary', href: '/fireflies/glossary' },
		{ label: 'History', href: '/fireflies/history' },
		{ label: 'Data & Logs', href: '/data' },
		{ label: 'Intelligence', href: '/intelligence' },
		{ label: 'Configuration', href: '/config' }
	];
</script>

<PageHeader title="Dashboard" description="LiveTranslate system overview" />

<!-- Row 1: Stats Grid -->
<div class="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{totalSessions}</p>
			<p class="text-sm text-muted-foreground mt-1">Total Sessions</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight text-green-500">{connectedCount}</p>
			<p class="text-sm text-muted-foreground mt-1">Connected</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{totalChunks}</p>
			<p class="text-sm text-muted-foreground mt-1">Total Chunks</p>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Content class="pt-6">
			<p class="text-3xl font-bold tracking-tight">{totalTranslations}</p>
			<p class="text-sm text-muted-foreground mt-1">Total Translations</p>
		</Card.Content>
	</Card.Root>
</div>

<!-- Row 2: Quick Actions + Services -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
	<!-- Quick Actions -->
	<Card.Root>
		<Card.Header>
			<Card.Title class="text-sm font-medium">Quick Actions</Card.Title>
		</Card.Header>
		<Card.Content>
			<div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
				{#each quickActions as action}
					<Button variant="ghost" href={action.href} class="justify-start h-9 px-3">
						{action.label}
					</Button>
				{/each}
			</div>
		</Card.Content>
	</Card.Root>

	<!-- Services Health -->
	<Card.Root>
		<Card.Header>
			<Card.Title class="text-sm font-medium">Services</Card.Title>
		</Card.Header>
		<Card.Content>
			<ul class="space-y-3">
				<li class="flex items-center justify-between text-sm">
					<span>Orchestration</span>
					<StatusIndicator
						status={systemHealth ? 'healthy' : 'down'}
						label={systemHealth ? 'Healthy' : 'Down'}
					/>
				</li>
				<li class="flex items-center justify-between text-sm">
					<span>Translation</span>
					<StatusIndicator
						status={translationHealth ? 'healthy' : 'down'}
						label={translationHealth ? 'Healthy' : 'Down'}
					/>
				</li>
				{#if Object.keys(healthStore.services).length > 0}
					{#each Object.entries(healthStore.services) as [name, healthy]}
						{#if name !== 'orchestration' && name !== 'translation'}
							<li class="flex items-center justify-between text-sm">
								<span class="capitalize">{name.replace(/-/g, ' ')}</span>
								<StatusIndicator
									status={serviceStatus(healthy)}
									label={healthy ? 'Healthy' : 'Down'}
								/>
							</li>
						{/if}
					{/each}
				{:else}
					<li class="flex items-center justify-between text-sm">
						<span>Database</span>
						<StatusIndicator status="unknown" label="Unknown" />
					</li>
					<li class="flex items-center justify-between text-sm">
						<span>Whisper</span>
						<StatusIndicator status="unknown" label="Unknown" />
					</li>
				{/if}
			</ul>
		</Card.Content>
	</Card.Root>
</div>

<!-- Row 3: Recent Sessions -->
<Card.Root>
	<Card.Header>
		<Card.Title class="text-sm font-medium">Recent Sessions</Card.Title>
	</Card.Header>
	<Card.Content>
		{#if recentSessions.length > 0}
			<div class="overflow-x-auto">
				<table class="w-full text-sm">
					<thead>
						<tr class="border-b border-border text-left text-muted-foreground">
							<th class="pb-2 pr-4 font-medium">Session ID</th>
							<th class="pb-2 pr-4 font-medium">Status</th>
							<th class="pb-2 pr-4 font-medium text-right">Chunks</th>
							<th class="pb-2 pr-4 font-medium text-right">Translations</th>
							<th class="pb-2 font-medium"></th>
						</tr>
					</thead>
					<tbody>
						{#each recentSessions as session (session.session_id)}
							<tr class="border-b border-border/50 last:border-0">
								<td class="py-2.5 pr-4 font-mono text-xs">
									{truncateId(session.session_id)}
								</td>
								<td class="py-2.5 pr-4">
									<Badge variant={connectionBadgeVariant(session.connection_status)}>
										{session.connection_status}
									</Badge>
								</td>
								<td class="py-2.5 pr-4 text-right tabular-nums">
									{session.chunks_received}
								</td>
								<td class="py-2.5 pr-4 text-right tabular-nums">
									{session.translations_completed}
								</td>
								<td class="py-2.5 text-right">
									<Button
										variant="ghost"
										size="sm"
										href="/fireflies/connect?session={session.session_id}"
									>
										View
									</Button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{:else}
			<p class="text-sm text-muted-foreground py-4 text-center">
				No sessions found. <a href="/fireflies" class="text-primary hover:underline">Connect to Fireflies</a> to get started.
			</p>
		{/if}
	</Card.Content>
</Card.Root>
