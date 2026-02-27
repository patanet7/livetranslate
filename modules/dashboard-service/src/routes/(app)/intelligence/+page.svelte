<script lang="ts">
	import { browser } from '$app/environment';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import { Badge } from '$lib/components/ui/badge';
	import { Button } from '$lib/components/ui/button';
	import * as Card from '$lib/components/ui/card';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Separator } from '$lib/components/ui/separator';
	import * as Select from '$lib/components/ui/select';
	import * as Tabs from '$lib/components/ui/tabs';
	import { Textarea } from '$lib/components/ui/textarea';
	import { toastStore } from '$lib/stores/toast.svelte';

	// ── Types ──────────────────────────────────────────────────────────

	interface Note {
		id: string;
		type: 'auto' | 'manual' | 'annotation';
		speaker?: string;
		content: string;
		timestamp: string;
		processing_time_ms?: number;
	}

	interface Insight {
		id: string;
		title: string;
		type: string;
		content: string;
		processing_time_ms: number;
		llm_model: string;
		created_at: string;
	}

	interface ChatMessage {
		id: string;
		role: 'user' | 'assistant';
		content: string;
		timestamp: string;
	}

	interface Template {
		id: string;
		name: string;
		description: string;
		type: string;
	}

	// ── Data ───────────────────────────────────────────────────────────

	let { data } = $props();

	// ── Shared State ───────────────────────────────────────────────────

	let activeTab = $state('notes');
	let selectedSessionId = $state('');

	// ── Notes State ────────────────────────────────────────────────────

	let notes = $state<Note[]>([]);
	let notesLoading = $state(false);
	let newNoteContent = $state('');
	let addingNote = $state(false);
	let analyzePrompt = $state('');
	let analyzing = $state(false);

	// ── Insights State ─────────────────────────────────────────────────

	let insights = $state<Insight[]>([]);
	let insightsLoading = $state(false);
	let selectedTemplateId = $state('');
	let customInstructions = $state('');
	let generatingInsight = $state(false);
	let generatingAll = $state(false);

	// ── Q&A State ──────────────────────────────────────────────────────

	let messages = $state<ChatMessage[]>([]);
	let suggestions = $state<string[]>([]);
	let messageInput = $state('');
	let sending = $state(false);
	let streaming = $state(false);
	let chatContainer: HTMLDivElement | undefined = $state();

	// ── Derived ────────────────────────────────────────────────────────

	let hasSession = $derived(selectedSessionId !== '');

	let selectedSession = $derived(
		data.sessions.find(
			(s: { session_id: string }) => s.session_id === selectedSessionId
		)
	);

	// ── Effects ────────────────────────────────────────────────────────

	$effect(() => {
		if (selectedSessionId && browser) {
			loadNotesForSession();
			loadInsightsForSession();
			loadSuggestions();
			messages = [];
		}
	});

	$effect(() => {
		if (chatContainer && messages.length > 0 && browser) {
			// Scroll chat to bottom on new messages
			chatContainer.scrollTop = chatContainer.scrollHeight;
		}
	});

	// ── Notes Functions ────────────────────────────────────────────────

	async function loadNotesForSession() {
		if (!selectedSessionId) return;
		notesLoading = true;
		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/notes`
			);
			if (res.ok) {
				notes = await res.json();
			} else {
				notes = [];
			}
		} catch {
			notes = [];
		} finally {
			notesLoading = false;
		}
	}

	async function addNote() {
		if (!selectedSessionId || !newNoteContent.trim()) return;
		addingNote = true;
		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/notes`,
				{
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ content: newNoteContent.trim(), type: 'manual' })
				}
			);
			if (res.ok) {
				toastStore.success('Note added');
				newNoteContent = '';
				await loadNotesForSession();
			} else {
				const body = await res.json().catch(() => null);
				toastStore.error(body?.detail ?? 'Failed to add note');
			}
		} catch (err) {
			toastStore.error(err instanceof TypeError && err.message === 'Failed to fetch'
				? 'Connection error. Please check your network and try again.'
				: 'Network error adding note');
		} finally {
			addingNote = false;
		}
	}

	async function analyzeNotes() {
		if (!selectedSessionId || !analyzePrompt.trim()) return;
		analyzing = true;
		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/notes/analyze`,
				{
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ prompt: analyzePrompt.trim() })
				}
			);
			if (res.ok) {
				toastStore.success('Analysis complete');
				analyzePrompt = '';
				await loadNotesForSession();
			} else {
				const body = await res.json().catch(() => null);
				toastStore.error(body?.detail ?? 'Analysis failed');
			}
		} catch (err) {
			toastStore.error(err instanceof TypeError && err.message === 'Failed to fetch'
				? 'Connection error. Please check your network and try again.'
				: 'Network error during analysis');
		} finally {
			analyzing = false;
		}
	}

	function noteTypeColor(type: Note['type']): string {
		switch (type) {
			case 'auto':
				return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
			case 'manual':
				return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
			case 'annotation':
				return 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200';
			default:
				return '';
		}
	}

	// ── Insights Functions ─────────────────────────────────────────────

	async function loadInsightsForSession() {
		if (!selectedSessionId) return;
		insightsLoading = true;
		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/insights`
			);
			if (res.ok) {
				insights = await res.json();
			} else {
				insights = [];
			}
		} catch {
			insights = [];
		} finally {
			insightsLoading = false;
		}
	}

	async function generateInsight() {
		if (!selectedSessionId || !selectedTemplateId) return;
		generatingInsight = true;
		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/insights/generate`,
				{
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						template_id: selectedTemplateId,
						custom_instructions: customInstructions.trim() || undefined
					})
				}
			);
			if (res.ok) {
				toastStore.success('Insight generated');
				customInstructions = '';
				await loadInsightsForSession();
			} else {
				const body = await res.json().catch(() => null);
				toastStore.error(body?.detail ?? 'Failed to generate insight');
			}
		} catch (err) {
			toastStore.error(err instanceof TypeError && err.message === 'Failed to fetch'
				? 'Connection error. Please check your network and try again.'
				: 'Network error generating insight');
		} finally {
			generatingInsight = false;
		}
	}

	async function generateAll() {
		if (!selectedSessionId || data.templates.length === 0) return;
		generatingAll = true;
		try {
			for (const template of data.templates) {
				const res = await fetch(
					`/api/intelligence/sessions/${selectedSessionId}/insights/generate`,
					{
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify({
							template_id: template.id,
							custom_instructions: customInstructions.trim() || undefined
						})
					}
				);
				if (!res.ok) {
					const body = await res.json().catch(() => null);
					toastStore.error(body?.detail ?? `Failed to generate insight for ${template.name}`);
				}
			}
			toastStore.success('All insights generated');
			await loadInsightsForSession();
		} catch (err) {
			toastStore.error(err instanceof TypeError && err.message === 'Failed to fetch'
				? 'Connection error. Please check your network and try again.'
				: 'Network error generating insights');
		} finally {
			generatingAll = false;
		}
	}

	function formatDate(iso: string): string {
		try {
			return new Date(iso).toLocaleString();
		} catch {
			return iso;
		}
	}

	// ── Q&A Functions ──────────────────────────────────────────────────

	async function loadSuggestions() {
		if (!selectedSessionId) return;
		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/agent/suggestions`
			);
			if (res.ok) {
				suggestions = await res.json();
			} else {
				suggestions = [];
			}
		} catch {
			suggestions = [];
		}
	}

	async function sendMessage(content?: string) {
		const text = content ?? messageInput.trim();
		if (!selectedSessionId || !text || sending) return;

		const userMessage: ChatMessage = {
			id: crypto.randomUUID(),
			role: 'user',
			content: text,
			timestamp: new Date().toISOString()
		};
		messages = [...messages, userMessage];
		messageInput = '';
		sending = true;
		streaming = true;

		const assistantMessage: ChatMessage = {
			id: crypto.randomUUID(),
			role: 'assistant',
			content: '',
			timestamp: new Date().toISOString()
		};
		messages = [...messages, assistantMessage];

		try {
			const res = await fetch(
				`/api/intelligence/sessions/${selectedSessionId}/agent/conversations`,
				{
					method: 'POST',
					headers: {
						'Content-Type': 'application/json',
						Accept: 'text/event-stream'
					},
					body: JSON.stringify({ message: text })
				}
			);

			if (!res.ok) {
				assistantMessage.content = 'Failed to get response. Please try again.';
				messages = [...messages.slice(0, -1), assistantMessage];
				toastStore.error('Failed to send message');
				return;
			}

			const reader = res.body?.getReader();
			if (!reader) {
				assistantMessage.content = 'No response stream available.';
				messages = [...messages.slice(0, -1), assistantMessage];
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
							const token = parsed.token ?? parsed.content ?? parsed.text ?? '';
							assistantMessage.content += token;
							messages = [...messages.slice(0, -1), { ...assistantMessage }];
						} catch {
							// If not JSON, treat the raw data as a token
							if (payload.trim()) {
								assistantMessage.content += payload;
								messages = [...messages.slice(0, -1), { ...assistantMessage }];
							}
						}
					}
				}
			}

			// Process any remaining buffer
			if (buffer.startsWith('data: ') && buffer.slice(6) !== '[DONE]') {
				const payload = buffer.slice(6);
				try {
					const parsed = JSON.parse(payload);
					const token = parsed.token ?? parsed.content ?? parsed.text ?? '';
					assistantMessage.content += token;
				} catch {
					if (payload.trim()) {
						assistantMessage.content += payload;
					}
				}
				messages = [...messages.slice(0, -1), { ...assistantMessage }];
			}

			// If no content was streamed, show a fallback
			if (!assistantMessage.content.trim()) {
				assistantMessage.content = 'No response received.';
				messages = [...messages.slice(0, -1), assistantMessage];
			}
		} catch (err) {
			assistantMessage.content = err instanceof TypeError && err.message === 'Failed to fetch'
				? 'Connection error. Please check your network and try again.'
				: 'Network error. Please try again.';
			messages = [...messages.slice(0, -1), assistantMessage];
			toastStore.error(assistantMessage.content);
		} finally {
			sending = false;
			streaming = false;
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			sendMessage();
		}
	}
</script>

<PageHeader
	title="Intelligence"
	description="AI-powered meeting notes, insights, and Q&A"
/>

<!-- Session Selector -->
<Card.Root class="mb-6">
	<Card.Content class="pt-6">
		<div class="flex flex-col sm:flex-row sm:items-center gap-3">
			<Label class="shrink-0 font-medium">Session</Label>
			<Select.Root type="single" bind:value={selectedSessionId}>
				<Select.Trigger class="w-full sm:max-w-md">
					{#if selectedSession}
						{selectedSession.transcript_id} ({selectedSession.session_id.slice(0, 8)}...)
					{:else}
						Select a session
					{/if}
				</Select.Trigger>
				<Select.Content>
					{#each data.sessions as session (session.session_id)}
						<Select.Item value={session.session_id} label={session.transcript_id}>
							<span class="flex items-center gap-2">
								<span
									class="inline-block size-2 rounded-full {session.connection_status === 'CONNECTED'
										? 'bg-green-500'
										: session.connection_status === 'CONNECTING'
											? 'bg-yellow-500'
											: 'bg-gray-400'}"
								></span>
								{session.transcript_id}
								<span class="text-muted-foreground text-xs">
									({session.session_id.slice(0, 8)}...)
								</span>
							</span>
						</Select.Item>
					{/each}
				</Select.Content>
			</Select.Root>
		</div>
	</Card.Content>
</Card.Root>

{#if !hasSession}
	<Card.Root>
		<Card.Content class="py-12">
			<div class="text-center">
				<p class="text-muted-foreground mb-2">No session selected.</p>
				<p class="text-sm text-muted-foreground">
					Select a session above to view meeting notes, insights, and Q&A.
				</p>
			</div>
		</Card.Content>
	</Card.Root>
{:else}
	<!-- Tabs -->
	<Tabs.Root bind:value={activeTab}>
		<Tabs.List>
			<Tabs.Trigger value="notes">Notes</Tabs.Trigger>
			<Tabs.Trigger value="insights">Insights</Tabs.Trigger>
			<Tabs.Trigger value="qa">Q&A</Tabs.Trigger>
		</Tabs.List>

		<!-- ══════════════════════════════════════════════════════════════ -->
		<!-- Tab 1: Meeting Notes                                         -->
		<!-- ══════════════════════════════════════════════════════════════ -->
		<Tabs.Content value="notes">
			<div class="grid gap-6 lg:grid-cols-3 mt-4">
				<!-- Notes List -->
				<div class="lg:col-span-2">
					<Card.Root>
						<Card.Header>
							<Card.Title>Meeting Notes</Card.Title>
							<Card.Description>
								Auto-generated and manual notes for this session
							</Card.Description>
						</Card.Header>
						<Card.Content>
							{#if notesLoading}
								<div class="flex items-center justify-center py-8">
									<div
										class="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent"
									></div>
									<span class="ml-2 text-sm text-muted-foreground">Loading notes...</span>
								</div>
							{:else if notes.length === 0}
								<p class="text-sm text-muted-foreground py-8 text-center">
									No notes yet. Add a note or run analysis to get started.
								</p>
							{:else}
								<div class="max-h-96 space-y-3 overflow-y-auto pr-1">
									{#each notes as note (note.id)}
										<div class="rounded-lg border p-3">
											<div class="flex items-center justify-between mb-2">
												<div class="flex items-center gap-2">
													<span
														class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {noteTypeColor(note.type)}"
													>
														{note.type}
													</span>
													{#if note.speaker}
														<span class="text-sm font-medium">{note.speaker}</span>
													{/if}
												</div>
												<div class="flex items-center gap-2 text-xs text-muted-foreground">
													{#if note.processing_time_ms !== undefined}
														<span>{note.processing_time_ms}ms</span>
														<Separator orientation="vertical" class="h-3" />
													{/if}
													<span>{formatDate(note.timestamp)}</span>
												</div>
											</div>
											<p class="text-sm">{note.content}</p>
										</div>
									{/each}
								</div>
							{/if}
						</Card.Content>
					</Card.Root>
				</div>

				<!-- Add Note + Analyze -->
				<div class="space-y-6">
					<!-- Add Note -->
					<Card.Root>
						<Card.Header>
							<Card.Title class="text-base">Add Note</Card.Title>
						</Card.Header>
						<Card.Content>
							<div class="space-y-3">
								<Input
									placeholder="Type your note..."
									bind:value={newNoteContent}
									onkeydown={(e: KeyboardEvent) => {
										if (e.key === 'Enter') addNote();
									}}
								/>
								<Button
									class="w-full"
									disabled={addingNote || !newNoteContent.trim()}
									onclick={addNote}
								>
									{#if addingNote}
										Adding...
									{:else}
										Add Note
									{/if}
								</Button>
							</div>
						</Card.Content>
					</Card.Root>

					<!-- Analyze -->
					<Card.Root>
						<Card.Header>
							<Card.Title class="text-base">Analyze Notes</Card.Title>
						</Card.Header>
						<Card.Content>
							<div class="space-y-3">
								<Input
									placeholder="Analyze prompt (e.g., summarize key decisions)"
									bind:value={analyzePrompt}
									onkeydown={(e: KeyboardEvent) => {
										if (e.key === 'Enter') analyzeNotes();
									}}
								/>
								<Button
									class="w-full"
									variant="secondary"
									disabled={analyzing || !analyzePrompt.trim()}
									onclick={analyzeNotes}
								>
									{#if analyzing}
										Analyzing...
									{:else}
										Analyze
									{/if}
								</Button>
							</div>
						</Card.Content>
					</Card.Root>
				</div>
			</div>
		</Tabs.Content>

		<!-- ══════════════════════════════════════════════════════════════ -->
		<!-- Tab 2: Post-Meeting Insights                                 -->
		<!-- ══════════════════════════════════════════════════════════════ -->
		<Tabs.Content value="insights">
			<div class="grid gap-6 lg:grid-cols-3 mt-4">
				<!-- Generate Controls -->
				<div class="space-y-6">
					<Card.Root>
						<Card.Header>
							<Card.Title class="text-base">Generate Insight</Card.Title>
							<Card.Description>
								Select a template and optionally add custom instructions
							</Card.Description>
						</Card.Header>
						<Card.Content>
							<div class="space-y-4">
								<div class="space-y-2">
									<Label>Template</Label>
									<Select.Root type="single" bind:value={selectedTemplateId}>
										<Select.Trigger class="w-full">
											{#if selectedTemplateId}
												{data.templates.find((t: Template) => t.id === selectedTemplateId)?.name ?? 'Select template'}
											{:else}
												Select template
											{/if}
										</Select.Trigger>
										<Select.Content>
											{#each data.templates as template (template.id)}
												<Select.Item value={template.id} label={template.name}>
													<div>
														<div class="font-medium">{template.name}</div>
														<div class="text-xs text-muted-foreground">
															{template.description}
														</div>
													</div>
												</Select.Item>
											{/each}
										</Select.Content>
									</Select.Root>
								</div>

								<div class="space-y-2">
									<Label>Custom Instructions (optional)</Label>
									<Textarea
										placeholder="Additional context or instructions..."
										bind:value={customInstructions}
										rows={3}
									/>
								</div>

								<div class="flex gap-2">
									<Button
										class="flex-1"
										disabled={generatingInsight || !selectedTemplateId}
										onclick={generateInsight}
									>
										{#if generatingInsight}
											Generating...
										{:else}
											Generate Insight
										{/if}
									</Button>
									<Button
										variant="outline"
										disabled={generatingAll || data.templates.length === 0}
										onclick={generateAll}
									>
										{#if generatingAll}
											Running...
										{:else}
											Generate All
										{/if}
									</Button>
								</div>
							</div>
						</Card.Content>
					</Card.Root>
				</div>

				<!-- Insights Results -->
				<div class="lg:col-span-2">
					<Card.Root>
						<Card.Header>
							<Card.Title>Generated Insights</Card.Title>
							<Card.Description>
								AI-generated analysis and summaries
							</Card.Description>
						</Card.Header>
						<Card.Content>
							{#if insightsLoading}
								<div class="flex items-center justify-center py-8">
									<div
										class="size-6 animate-spin rounded-full border-2 border-primary border-t-transparent"
									></div>
									<span class="ml-2 text-sm text-muted-foreground"
										>Loading insights...</span
									>
								</div>
							{:else if insights.length === 0}
								<p class="text-sm text-muted-foreground py-8 text-center">
									No insights generated yet. Select a template and generate one.
								</p>
							{:else}
								<div class="space-y-4">
									{#each insights as insight (insight.id)}
										<Card.Root>
											<Card.Header class="pb-3">
												<div class="flex items-center justify-between">
													<Card.Title class="text-base">{insight.title}</Card.Title>
													<Badge variant="outline">{insight.type}</Badge>
												</div>
											</Card.Header>
											<Card.Content>
												<p class="text-sm whitespace-pre-wrap">{insight.content}</p>
												<Separator class="my-3" />
												<div
													class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground"
												>
													<span>
														Processing: <span class="font-medium text-foreground"
															>{insight.processing_time_ms}ms</span
														>
													</span>
													<span>
														Model: <span class="font-medium text-foreground"
															>{insight.llm_model}</span
														>
													</span>
													<span>{formatDate(insight.created_at)}</span>
												</div>
											</Card.Content>
										</Card.Root>
									{/each}
								</div>
							{/if}
						</Card.Content>
					</Card.Root>
				</div>
			</div>
		</Tabs.Content>

		<!-- ══════════════════════════════════════════════════════════════ -->
		<!-- Tab 3: Meeting Q&A Agent                                     -->
		<!-- ══════════════════════════════════════════════════════════════ -->
		<Tabs.Content value="qa">
			<div class="mt-4">
				<Card.Root>
					<Card.Header>
						<Card.Title>Meeting Q&A</Card.Title>
						<Card.Description>
							Ask questions about the meeting content and get AI-powered answers
						</Card.Description>
					</Card.Header>
					<Card.Content>
						<!-- Suggested Queries -->
						{#if suggestions.length > 0}
							<div class="mb-4">
								<p class="text-xs font-medium text-muted-foreground mb-2">
									Suggested questions
								</p>
								<div class="flex flex-wrap gap-2">
									{#each suggestions as suggestion}
										<button
											type="button"
											class="inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer"
											onclick={() => sendMessage(suggestion)}
											disabled={sending}
										>
											{suggestion}
										</button>
									{/each}
								</div>
								<Separator class="mt-4" />
							</div>
						{/if}

						<!-- Chat Messages -->
						<div
							bind:this={chatContainer}
							class="max-h-96 min-h-48 space-y-3 overflow-y-auto pr-1 mb-4"
						>
							{#if messages.length === 0}
								<div class="flex items-center justify-center h-48">
									<p class="text-sm text-muted-foreground">
										Ask a question about this meeting to get started.
									</p>
								</div>
							{:else}
								{#each messages as message (message.id)}
									<div
										class="flex {message.role === 'user'
											? 'justify-end'
											: 'justify-start'}"
									>
										<div
											class="max-w-[80%] rounded-lg px-4 py-2 text-sm {message.role === 'user'
												? 'bg-primary text-primary-foreground'
												: 'bg-muted'}"
										>
											<p class="whitespace-pre-wrap">{message.content}</p>
											{#if message.role === 'assistant' && streaming && message.id === messages[messages.length - 1]?.id && !message.content}
												<span class="inline-flex gap-1">
													<span class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:0ms]"
													></span>
													<span class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:150ms]"
													></span>
													<span class="size-1.5 animate-bounce rounded-full bg-current [animation-delay:300ms]"
													></span>
												</span>
											{/if}
										</div>
									</div>
								{/each}
							{/if}
						</div>

						<!-- Input Area -->
						<div class="flex items-center gap-2">
							<Input
								class="flex-1"
								placeholder="Ask a question about this meeting..."
								bind:value={messageInput}
								onkeydown={handleKeydown}
								disabled={sending}
							/>
							<Button
								disabled={sending || !messageInput.trim()}
								onclick={() => sendMessage()}
							>
								{#if sending}
									Sending...
								{:else}
									Send
								{/if}
							</Button>
						</div>
					</Card.Content>
				</Card.Root>
			</div>
		</Tabs.Content>
	</Tabs.Root>
{/if}
