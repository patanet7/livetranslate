<script lang="ts">
	import type { Conversation } from '$lib/api/chat';
	import { Button } from '$lib/components/ui/button';
	import PlusIcon from '@lucide/svelte/icons/plus';
	import TrashIcon from '@lucide/svelte/icons/trash-2';
	import MessageSquareIcon from '@lucide/svelte/icons/message-square';

	interface Props {
		conversations: Conversation[];
		selectedId: string | null;
		onselect: (id: string) => void;
		oncreate: () => void;
		ondelete: (id: string) => void;
	}

	let { conversations, selectedId, onselect, oncreate, ondelete }: Props =
		$props();

	function formatDate(iso: string): string {
		try {
			const d = new Date(iso);
			const now = new Date();
			const diff = now.getTime() - d.getTime();
			if (diff < 60_000) return 'Just now';
			if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
			if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
			return d.toLocaleDateString();
		} catch {
			return '';
		}
	}

	function handleDelete(event: MouseEvent, id: string) {
		event.stopPropagation();
		ondelete(id);
	}
</script>

<div class="flex h-full flex-col">
	<div class="p-3 border-b">
		<Button class="w-full" variant="outline" size="sm" onclick={oncreate}>
			<PlusIcon class="size-4 mr-2" />
			New Chat
		</Button>
	</div>

	<div class="flex-1 overflow-y-auto p-2 space-y-1">
		{#if conversations.length === 0}
			<div class="px-3 py-8 text-center">
				<MessageSquareIcon class="size-8 mx-auto text-muted-foreground/40 mb-2" />
				<p class="text-xs text-muted-foreground">No conversations yet</p>
			</div>
		{:else}
			{#each conversations as conv (conv.id)}
				<!-- svelte-ignore a11y_click_events_have_key_events -->
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="group w-full flex items-start gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors cursor-pointer {conv.id === selectedId
						? 'bg-accent text-accent-foreground'
						: 'text-muted-foreground hover:bg-accent/50'}"
					onclick={() => onselect(conv.id)}
					role="button"
					tabindex="0"
				>
					<MessageSquareIcon class="size-4 mt-0.5 shrink-0" />
					<div class="flex-1 min-w-0">
						<p class="truncate font-medium text-foreground">
							{conv.title || 'Untitled'}
						</p>
						<p class="text-xs text-muted-foreground mt-0.5">
							{conv.message_count} messages
							{#if conv.updated_at}
								&middot; {formatDate(conv.updated_at)}
							{/if}
						</p>
					</div>
					<button
						type="button"
						class="shrink-0 rounded p-1 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive"
						onclick={(e) => handleDelete(e, conv.id)}
						aria-label="Delete conversation"
					>
						<TrashIcon class="size-3.5" />
					</button>
				</div>
			{/each}
		{/if}
	</div>
</div>
