<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import { healthStore } from '$lib/stores/health.svelte';
	import * as Card from '$lib/components/ui/card';
</script>

<PageHeader title="Dashboard" description="LiveTranslate system overview" />

<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
	<Card.Root>
		<Card.Header>
			<Card.Title class="text-sm font-medium">System Health</Card.Title>
		</Card.Header>
		<Card.Content>
			<StatusIndicator status={healthStore.status} label={healthStore.status} />
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header>
			<Card.Title class="text-sm font-medium">Quick Actions</Card.Title>
		</Card.Header>
		<Card.Content class="flex flex-col gap-2">
			<a href="/fireflies" class="text-sm text-primary hover:underline">Connect to Fireflies</a>
			<a href="/translation/test" class="text-sm text-primary hover:underline">Translation Test Bench</a>
			<a href="/config" class="text-sm text-primary hover:underline">Configuration</a>
		</Card.Content>
	</Card.Root>

	<Card.Root>
		<Card.Header>
			<Card.Title class="text-sm font-medium">Services</Card.Title>
		</Card.Header>
		<Card.Content>
			{#if Object.keys(healthStore.services).length > 0}
				<ul class="space-y-1">
					{#each Object.entries(healthStore.services) as [name, healthy]}
						<li class="flex items-center justify-between text-sm">
							<span class="capitalize">{name.replace(/-/g, ' ')}</span>
							<StatusIndicator status={healthy ? 'healthy' : 'down'} />
						</li>
					{/each}
				</ul>
			{:else}
				<p class="text-sm text-muted-foreground">Waiting for health data...</p>
			{/if}
		</Card.Content>
	</Card.Root>
</div>
