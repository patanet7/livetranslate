<script lang="ts">
	import type { ChatMessage, ToolCallInfo } from '$lib/api/chat';
	import ToolCallIndicator from './ToolCallIndicator.svelte';

	interface Props {
		message: ChatMessage;
		streaming?: boolean;
	}

	let { message, streaming = false }: Props = $props();

	let isUser = $derived(message.role === 'user');
	let isToolMessage = $derived(message.role === 'tool');
	let hasToolCalls = $derived(
		message.tool_calls != null && message.tool_calls.length > 0
	);

	function formatTime(iso: string): string {
		try {
			return new Date(iso).toLocaleTimeString([], {
				hour: '2-digit',
				minute: '2-digit'
			});
		} catch {
			return '';
		}
	}
</script>

{#if isToolMessage}
	<!-- Tool messages are hidden; their info is shown via ToolCallIndicator on the assistant message -->
{:else}
	<div class="flex {isUser ? 'justify-end' : 'justify-start'} group">
		<div
			class="max-w-[80%] rounded-lg px-4 py-2.5 text-sm {isUser
				? 'bg-primary text-primary-foreground'
				: 'bg-muted'}"
		>
			{#if hasToolCalls}
				<div class="mb-2 space-y-1">
					{#each message.tool_calls as tc (tc.tool_name)}
						<ToolCallIndicator toolCall={tc} />
					{/each}
				</div>
			{/if}

			{#if message.content}
				<p class="whitespace-pre-wrap break-words">{message.content}</p>
			{:else if streaming}
				<span class="inline-flex gap-1">
					<span
						class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:0ms]"
					></span>
					<span
						class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:150ms]"
					></span>
					<span
						class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:300ms]"
					></span>
				</span>
			{/if}

			<div
				class="mt-1 flex items-center gap-2 text-[10px] {isUser
					? 'text-primary-foreground/60'
					: 'text-muted-foreground'}"
			>
				<span>{formatTime(message.created_at)}</span>
				{#if message.model && !isUser}
					<span>{message.model}</span>
				{/if}
				{#if message.tokens_used && !isUser}
					<span>{message.tokens_used} tokens</span>
				{/if}
			</div>
		</div>
	</div>
{/if}
