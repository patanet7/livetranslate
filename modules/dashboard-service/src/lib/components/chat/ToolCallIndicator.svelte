<script lang="ts">
	import type { ToolCallInfo } from '$lib/api/chat';
	import WrenchIcon from '@lucide/svelte/icons/wrench';

	interface Props {
		toolCall: ToolCallInfo;
		running?: boolean;
	}

	let { toolCall, running = false }: Props = $props();

	let label = $derived(formatToolName(toolCall.tool_name));

	function formatToolName(name: string): string {
		return name
			.replace(/_/g, ' ')
			.replace(/\b\w/g, (c) => c.toUpperCase());
	}
</script>

<div
	class="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium text-muted-foreground {running
		? 'animate-pulse border-primary/40 bg-primary/5'
		: 'border-border bg-muted/50'}"
>
	<WrenchIcon class="size-3 shrink-0" />
	<span class="truncate">{label}</span>
	{#if running}
		<span class="size-1.5 animate-ping rounded-full bg-primary"></span>
	{/if}
</div>
