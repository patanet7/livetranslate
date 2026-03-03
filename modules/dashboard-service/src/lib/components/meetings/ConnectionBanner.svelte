<script lang="ts">
	import type { WsStatus } from '$lib/stores/websocket.svelte';
	import { Button } from '$lib/components/ui/button';

	interface Props {
		status: WsStatus;
		reconnectAttempt: number;
		maxAttempts?: number;
		onretry?: () => void;
	}

	let { status, reconnectAttempt, maxAttempts = 10, onretry }: Props = $props();
</script>

<div role="status" aria-live="polite">
{#if status === 'connected'}
	<div class="flex items-center gap-2 rounded-md bg-green-500/10 px-3 py-1.5 text-sm text-green-700 dark:text-green-400">
		<span class="inline-block size-2 rounded-full bg-green-500"></span>
		Connected
	</div>
{:else if status === 'connecting'}
	<div class="flex items-center gap-2 rounded-md bg-yellow-500/10 px-3 py-1.5 text-sm text-yellow-700 dark:text-yellow-400 animate-pulse">
		<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
		Connecting...
	</div>
{:else if status === 'reconnecting'}
	<div class="flex items-center gap-2 rounded-md bg-yellow-500/10 px-3 py-1.5 text-sm text-yellow-700 dark:text-yellow-400 animate-pulse">
		<span class="inline-block size-2 rounded-full bg-yellow-500"></span>
		Reconnecting... (attempt {reconnectAttempt}/{maxAttempts})
	</div>
{:else if status === 'error'}
	<div class="flex items-center gap-2 rounded-md bg-destructive/10 px-3 py-1.5 text-sm text-destructive">
		<span class="inline-block size-2 rounded-full bg-destructive"></span>
		Connection lost
		{#if onretry}
			<Button variant="outline" size="sm" class="ml-2 h-6 text-xs" onclick={onretry}>Retry</Button>
		{/if}
	</div>
{:else}
	<div class="flex items-center gap-2 rounded-md bg-muted px-3 py-1.5 text-sm text-muted-foreground">
		<span class="inline-block size-2 rounded-full bg-muted-foreground"></span>
		Disconnected
	</div>
{/if}
</div>
