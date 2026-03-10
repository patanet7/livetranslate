<script lang="ts">
	import { browser } from '$app/environment';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import ChatMessageBubble from '$lib/components/chat/ChatMessage.svelte';
	import ChatInput from '$lib/components/chat/ChatInput.svelte';
	import ConversationList from '$lib/components/chat/ConversationList.svelte';
	import SettingsDrawer from '$lib/components/chat/SettingsDrawer.svelte';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import SettingsIcon from '@lucide/svelte/icons/settings';
	import BotIcon from '@lucide/svelte/icons/bot';
	import type { Conversation, ChatMessage } from '$lib/api/chat';
	import { toastStore } from '$lib/stores/toast.svelte';

	// ── Data ───────────────────────────────────────────────────────

	let { data } = $props();

	// ── State ──────────────────────────────────────────────────────

	let conversations = $state<Conversation[]>(data.conversations);
	let selectedConversationId = $state<string | null>(null);
	let messages = $state<ChatMessage[]>([]);
	let suggestions = $state<string[]>([]);
	let sending = $state(false);
	let streaming = $state(false);
	let settingsOpen = $state(false);
	let loadingMessages = $state(false);
	let chatContainer: HTMLDivElement | undefined = $state();

	// ── Effects ────────────────────────────────────────────────────

	$effect(() => {
		if (chatContainer && messages.length > 0 && browser) {
			chatContainer.scrollTop = chatContainer.scrollHeight;
		}
	});

	// ── Conversation Management ────────────────────────────────────

	async function selectConversation(id: string) {
		selectedConversationId = id;
		loadingMessages = true;
		try {
			const res = await fetch(`/api/chat/conversations/${id}`);
			if (res.ok) {
				const data = await res.json();
				messages = data.messages ?? [];
			} else {
				messages = [];
				toastStore.error('Failed to load conversation');
			}
		} catch {
			messages = [];
			toastStore.error('Network error loading conversation');
		} finally {
			loadingMessages = false;
		}

		// Load suggestions
		try {
			const res = await fetch(`/api/chat/conversations/${id}/suggestions`);
			if (res.ok) {
				const data = await res.json();
				suggestions = data.suggestions ?? [];
			} else {
				suggestions = [];
			}
		} catch {
			suggestions = [];
		}
	}

	async function createConversation() {
		try {
			const res = await fetch('/api/chat/conversations', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({})
			});
			if (res.ok) {
				const conv: Conversation = await res.json();
				conversations = [conv, ...conversations];
				await selectConversation(conv.id);
			} else {
				toastStore.error('Failed to create conversation');
			}
		} catch {
			toastStore.error('Network error creating conversation');
		}
	}

	async function deleteConversation(id: string) {
		try {
			const res = await fetch(`/api/chat/conversations/${id}`, {
				method: 'DELETE'
			});
			if (res.ok || res.status === 204) {
				conversations = conversations.filter((c) => c.id !== id);
				if (selectedConversationId === id) {
					selectedConversationId = null;
					messages = [];
					suggestions = [];
				}
				toastStore.success('Conversation deleted');
			} else {
				toastStore.error('Failed to delete conversation');
			}
		} catch {
			toastStore.error('Network error deleting conversation');
		}
	}

	// ── Send Message (SSE streaming) ───────────────────────────────

	async function sendMessage(content: string) {
		if (!content.trim() || sending) return;

		// Auto-create conversation if none selected
		if (!selectedConversationId) {
			try {
				const res = await fetch('/api/chat/conversations', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ title: content.slice(0, 80) })
				});
				if (res.ok) {
					const conv: Conversation = await res.json();
					conversations = [conv, ...conversations];
					selectedConversationId = conv.id;
				} else {
					toastStore.error('Failed to create conversation');
					return;
				}
			} catch {
				toastStore.error('Network error');
				return;
			}
		}

		const convId = selectedConversationId!;

		// Add user message locally
		const userMsg: ChatMessage = {
			id: crypto.randomUUID(),
			conversation_id: convId,
			role: 'user',
			content,
			tool_calls: null,
			model: null,
			provider: null,
			tokens_used: null,
			created_at: new Date().toISOString()
		};
		messages = [...messages, userMsg];
		suggestions = [];
		sending = true;
		streaming = true;

		// Add placeholder assistant message
		const assistantMsg: ChatMessage = {
			id: crypto.randomUUID(),
			conversation_id: convId,
			role: 'assistant',
			content: '',
			tool_calls: null,
			model: null,
			provider: null,
			tokens_used: null,
			created_at: new Date().toISOString()
		};
		messages = [...messages, assistantMsg];

		try {
			const res = await fetch(
				`/api/chat/conversations/${convId}/messages/stream`,
				{
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Accept: 'text/event-stream'
					},
					body: JSON.stringify({ content })
				}
			);

			if (!res.ok) {
				assistantMsg.content = 'Failed to get response. Please try again.';
				messages = [...messages.slice(0, -1), assistantMsg];
				toastStore.error('Failed to send message');
				return;
			}

			const reader = res.body?.getReader();
			if (!reader) {
				assistantMsg.content = 'No response stream available.';
				messages = [...messages.slice(0, -1), assistantMsg];
				return;
			}

			const decoder = new TextDecoder();
			let buffer = '';

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				buffer += decoder.decode(value, { stream: true });
				const lines = buffer.split('\n');
				buffer = lines.pop() ?? '';

				for (const line of lines) {
					if (line.startsWith('data: ')) {
						const payload = line.slice(6);
						if (payload === '[DONE]') continue;
						try {
							const parsed = JSON.parse(payload);
							if (parsed.tool_calls) {
								assistantMsg.tool_calls = parsed.tool_calls;
							}
							const token =
								parsed.token ?? parsed.content ?? parsed.text ?? '';
							if (token) {
								assistantMsg.content += token;
							}
							if (parsed.model) assistantMsg.model = parsed.model;
							if (parsed.tokens_used)
								assistantMsg.tokens_used = parsed.tokens_used;
							messages = [...messages.slice(0, -1), { ...assistantMsg }];
						} catch {
							if (payload.trim()) {
								assistantMsg.content += payload;
								messages = [...messages.slice(0, -1), { ...assistantMsg }];
							}
						}
					}
				}
			}

			// Process remaining buffer
			if (buffer.startsWith('data: ') && buffer.slice(6) !== '[DONE]') {
				const payload = buffer.slice(6);
				try {
					const parsed = JSON.parse(payload);
					const token = parsed.token ?? parsed.content ?? parsed.text ?? '';
					if (token) assistantMsg.content += token;
				} catch {
					if (payload.trim()) assistantMsg.content += payload;
				}
				messages = [...messages.slice(0, -1), { ...assistantMsg }];
			}

			if (!assistantMsg.content.trim()) {
				assistantMsg.content = 'No response received.';
				messages = [...messages.slice(0, -1), assistantMsg];
			}

			// Update conversation title and count in the sidebar
			const convIndex = conversations.findIndex((c) => c.id === convId);
			if (convIndex >= 0) {
				const updated = { ...conversations[convIndex] };
				updated.message_count += 2;
				updated.updated_at = new Date().toISOString();
				if (!updated.title) updated.title = content.slice(0, 80);
				conversations = [
					updated,
					...conversations.slice(0, convIndex),
					...conversations.slice(convIndex + 1)
				];
			}
		} catch (err) {
			assistantMsg.content =
				err instanceof TypeError && err.message === 'Failed to fetch'
					? 'Connection error. Please check your network and try again.'
					: 'Network error. Please try again.';
			messages = [...messages.slice(0, -1), assistantMsg];
			toastStore.error(assistantMsg.content);
		} finally {
			sending = false;
			streaming = false;
		}
	}
</script>

<PageHeader title="Business Insights Chat" description="AI-powered analysis of your meetings, transcripts, and translations">
	{#snippet actions()}
		<Button variant="outline" size="sm" onclick={() => (settingsOpen = true)}>
			<SettingsIcon class="size-4 mr-2" />
			Settings
		</Button>
	{/snippet}
</PageHeader>

<div class="flex gap-4 h-[calc(100vh-12rem)]">
	<!-- Conversation Sidebar -->
	<div class="w-72 shrink-0 rounded-lg border bg-card hidden lg:block">
		<ConversationList
			{conversations}
			selectedId={selectedConversationId}
			onselect={selectConversation}
			oncreate={createConversation}
			ondelete={deleteConversation}
		/>
	</div>

	<!-- Main Chat Area -->
	<div class="flex-1 flex flex-col rounded-lg border bg-card">
		{#if !selectedConversationId && messages.length === 0}
			<!-- Empty State -->
			<div class="flex-1 flex items-center justify-center">
				<div class="text-center max-w-md px-4">
					<BotIcon class="size-12 mx-auto text-muted-foreground/40 mb-4" />
					<h2 class="text-lg font-semibold mb-2">Business Insights Chat</h2>
					<p class="text-sm text-muted-foreground mb-6">
						Ask questions about your meetings, transcripts, and translations.
						The AI has access to your data and can provide insights, summaries,
						and analysis.
					</p>
					<div class="flex flex-wrap justify-center gap-2">
						<button
							type="button"
							class="inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer"
							onclick={() => sendMessage('What meetings do I have?')}
						>
							What meetings do I have?
						</button>
						<button
							type="button"
							class="inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer"
							onclick={() =>
								sendMessage('Summarize my most recent meeting')}
						>
							Summarize my most recent meeting
						</button>
						<button
							type="button"
							class="inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer"
							onclick={() =>
								sendMessage('What action items came up recently?')}
						>
							What action items came up recently?
						</button>
					</div>

					<!-- Mobile: show new chat button -->
					<div class="mt-6 lg:hidden">
						<Button variant="outline" onclick={createConversation}>
							Start New Chat
						</Button>
					</div>
				</div>
			</div>
		{:else}
			<!-- Messages -->
			<div
				bind:this={chatContainer}
				class="flex-1 overflow-y-auto p-4 space-y-3"
			>
				{#if loadingMessages}
					<div class="flex items-center justify-center py-8">
						<div
							class="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent"
						></div>
						<span class="ml-2 text-sm text-muted-foreground"
							>Loading messages...</span
						>
					</div>
				{:else}
					{#each messages as message (message.id)}
						<ChatMessageBubble
							{message}
							streaming={streaming &&
								message.role === 'assistant' &&
								message.id === messages[messages.length - 1]?.id}
						/>
					{/each}
				{/if}
			</div>

			<!-- Input -->
			<div class="border-t p-4">
				<ChatInput
					disabled={sending}
					suggestions={messages.length === 0 ? suggestions : []}
					onsubmit={sendMessage}
				/>
			</div>
		{/if}
	</div>
</div>

<!-- Settings Drawer -->
<SettingsDrawer open={settingsOpen} onclose={() => (settingsOpen = false)} />
