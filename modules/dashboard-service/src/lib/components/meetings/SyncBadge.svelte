<script lang="ts">
	import { Badge } from '$lib/components/ui/badge';

	interface Props {
		status: 'none' | 'live' | 'syncing' | 'synced' | 'failed';
		compact?: boolean;
	}

	let { status, compact = false }: Props = $props();

	const config = $derived(
		({
			none: { label: 'Not synced', variant: 'secondary' as const, class: '' },
			live: { label: 'Live', variant: 'default' as const, class: 'bg-green-600 animate-pulse' },
			syncing: { label: 'Syncing...', variant: 'default' as const, class: 'bg-yellow-600 animate-pulse' },
			synced: { label: 'Synced', variant: 'secondary' as const, class: '' },
			failed: { label: 'Sync failed', variant: 'destructive' as const, class: '' }
		})[status]
	);
</script>

<Badge variant={config.variant} class={config.class}>
	{#if status === 'live'}
		<span class="mr-1 inline-block size-2 rounded-full bg-green-300"></span>
	{/if}
	{#if status === 'synced'}
		<span class="mr-1">✓</span>
	{/if}
	{compact ? status : config.label}
</Badge>
