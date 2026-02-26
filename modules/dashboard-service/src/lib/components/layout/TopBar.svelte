<script lang="ts">
	import StatusIndicator from './StatusIndicator.svelte';

	interface Props {
		health: 'healthy' | 'degraded' | 'down' | 'unknown';
	}

	let { health }: Props = $props();
</script>

<header class="h-12 border-b bg-card flex items-center justify-between px-4">
	<div class="text-sm text-muted-foreground">
		<!-- breadcrumb can go here later -->
	</div>
	<div class="flex items-center gap-3">
		<StatusIndicator status={health} label="Services" />
	</div>
</header>
