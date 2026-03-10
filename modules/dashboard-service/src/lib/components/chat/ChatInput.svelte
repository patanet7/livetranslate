<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import SendIcon from '@lucide/svelte/icons/send-horizontal';

	interface Props {
		disabled?: boolean;
		suggestions?: string[];
		onsubmit: (content: string) => void;
	}

	let { disabled = false, suggestions = [], onsubmit }: Props = $props();

	let value = $state('');
	let textarea: HTMLTextAreaElement | undefined = $state();

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			submit();
		}
	}

	function submit() {
		const text = value.trim();
		if (!text || disabled) return;
		onsubmit(text);
		value = '';
		// Reset textarea height
		if (textarea) textarea.style.height = 'auto';
	}

	function handleInput() {
		if (!textarea) return;
		textarea.style.height = 'auto';
		textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
	}

	function handleSuggestionClick(suggestion: string) {
		if (disabled) return;
		onsubmit(suggestion);
	}
</script>

{#if suggestions.length > 0}
	<div class="mb-3">
		<p class="text-xs font-medium text-muted-foreground mb-2">Suggestions</p>
		<div class="flex flex-wrap gap-2">
			{#each suggestions as suggestion}
				<button
					type="button"
					class="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
					onclick={() => handleSuggestionClick(suggestion)}
					{disabled}
				>
					{suggestion}
				</button>
			{/each}
		</div>
	</div>
{/if}

<div class="flex items-end gap-2">
	<textarea
		bind:this={textarea}
		bind:value
		oninput={handleInput}
		onkeydown={handleKeydown}
		placeholder="Ask about your meetings, transcripts, translations..."
		rows={1}
		{disabled}
		class="flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
	></textarea>
	<Button
		size="icon"
		disabled={disabled || !value.trim()}
		onclick={submit}
		class="shrink-0"
	>
		<SendIcon class="size-4" />
	</Button>
</div>
