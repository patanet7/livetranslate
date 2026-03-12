<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import SettingsIcon from '@lucide/svelte/icons/settings';
	import Trash2Icon from '@lucide/svelte/icons/trash-2';
	import Loader2Icon from '@lucide/svelte/icons/loader-2';
	import CheckCircleIcon from '@lucide/svelte/icons/check-circle';
	import XCircleIcon from '@lucide/svelte/icons/x-circle';
	import CircleIcon from '@lucide/svelte/icons/circle';
	import PowerIcon from '@lucide/svelte/icons/power';
	import type { AIConnection } from '$lib/api/connections';

	interface Props {
		connection: AIConnection;
		status: 'unknown' | 'connected' | 'error' | 'verifying';
		modelCount: number;
		onverify: () => void;
		onconfigure: () => void;
		ondelete: () => void;
		ontoggle: (enabled: boolean) => void;
	}

	let {
		connection,
		status,
		modelCount,
		onverify,
		onconfigure,
		ondelete,
		ontoggle
	}: Props = $props();

	const engineColors: Record<string, string> = {
		ollama: 'bg-green-500/10 text-green-700 dark:text-green-400',
		openai: 'bg-blue-500/10 text-blue-700 dark:text-blue-400',
		anthropic: 'bg-orange-500/10 text-orange-700 dark:text-orange-400',
		openai_compatible: 'bg-purple-500/10 text-purple-700 dark:text-purple-400'
	};

	const engineLabels: Record<string, string> = {
		ollama: 'Ollama',
		openai: 'OpenAI',
		anthropic: 'Anthropic',
		openai_compatible: 'OpenAI Compatible'
	};
</script>

<div
	data-testid="connection-card"
	class="flex items-center gap-3 rounded-lg border p-3 transition-opacity {connection.enabled
		? ''
		: 'opacity-50'} {status === 'connected'
		? 'border-green-500/30'
		: status === 'error'
			? 'border-red-500/30'
			: ''}"
>
	<!-- Status dot -->
	<div class="flex-shrink-0">
		{#if status === 'verifying'}
			<Loader2Icon class="h-4 w-4 animate-spin text-yellow-500" />
		{:else if status === 'connected'}
			<CheckCircleIcon class="h-4 w-4 text-green-500" />
		{:else if status === 'error'}
			<XCircleIcon class="h-4 w-4 text-red-500" />
		{:else}
			<CircleIcon class="h-4 w-4 text-muted-foreground" />
		{/if}
	</div>

	<!-- Connection info -->
	<div class="min-w-0 flex-1 space-y-1">
		<div class="flex items-center gap-2">
			<span class="truncate text-sm font-medium">{connection.name}</span>
			<span
				class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {engineColors[
					connection.engine
				] ?? 'bg-muted text-muted-foreground'}"
			>
				{engineLabels[connection.engine] ?? connection.engine}
			</span>
			{#if connection.prefix}
				<Badge variant="outline" class="text-xs">prefix: {connection.prefix}</Badge>
			{/if}
		</div>
		<div class="flex items-center gap-2">
			<p class="truncate text-xs text-muted-foreground">{connection.url}</p>
			{#if status === 'connected' && modelCount > 0}
				<Badge variant="secondary" class="text-xs"
					>{modelCount} model{modelCount !== 1 ? 's' : ''}</Badge
				>
			{/if}
		</div>
	</div>

	<!-- Actions -->
	<div class="flex flex-shrink-0 items-center gap-1">
		<Button
			variant="outline"
			size="sm"
			onclick={onverify}
			disabled={!connection.enabled || status === 'verifying'}
		>
			{#if status === 'verifying'}
				<Loader2Icon class="mr-1 h-3 w-3 animate-spin" />
			{/if}
			Verify
		</Button>
		<Button variant="ghost" size="icon" onclick={onconfigure} class="h-8 w-8">
			<SettingsIcon class="h-4 w-4" />
		</Button>
		<Button
			variant="ghost"
			size="icon"
			onclick={() => ontoggle(!connection.enabled)}
			class="h-8 w-8 {connection.enabled ? 'text-green-500' : 'text-muted-foreground'}"
		>
			<PowerIcon class="h-4 w-4" />
		</Button>
		<Button variant="ghost" size="icon" onclick={ondelete} class="h-8 w-8 text-destructive">
			<Trash2Icon class="h-4 w-4" />
		</Button>
	</div>
</div>
