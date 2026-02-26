<script lang="ts">
	interface Props {
		status: 'healthy' | 'degraded' | 'down' | 'unknown' | 'connected' | 'connecting' | 'disconnected' | 'error';
		label?: string;
	}

	let { status, label }: Props = $props();

	const colorMap: Record<string, string> = {
		healthy: 'bg-green-500',
		connected: 'bg-green-500',
		degraded: 'bg-yellow-500',
		connecting: 'bg-yellow-500 animate-pulse',
		down: 'bg-red-500',
		error: 'bg-red-500',
		disconnected: 'bg-gray-400',
		unknown: 'bg-gray-400'
	};
</script>

<span class="inline-flex items-center gap-1.5">
	<span class={`h-2 w-2 rounded-full ${colorMap[status] ?? 'bg-gray-400'}`}></span>
	{#if label}
		<span class="text-xs text-muted-foreground capitalize">{label ?? status}</span>
	{/if}
</span>
