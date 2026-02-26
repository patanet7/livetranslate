<script lang="ts">
	import { browser } from '$app/environment';
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import StatusIndicator from '$lib/components/layout/StatusIndicator.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Badge } from '$lib/components/ui/badge';
	import { Separator } from '$lib/components/ui/separator';
	import { WS_BASE } from '$lib/config';
	import type { Caption, CaptionEvent, FirefliesSession, UiConfig } from '$lib/types';

	// --- Types ---

	interface FeedEntry {
		id: string;
		speaker: string;
		text: string;
		translatedText?: string;
		confidence?: number;
		timestamp: string;
		targetLanguage?: string;
	}

	// --- Props ---

	let { data }: { data: { sessions: FirefliesSession[]; uiConfig: UiConfig | null } } = $props();

	// --- State ---

	let selectedSession = $state('');
	let selectedLang = $state('all');
	let wsStatus = $state<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');
	let entries = $state<FeedEntry[]>([]);
	let socket: WebSocket | null = $state(null);

	// --- Derived ---

	let entryCount = $derived(entries.length);
	let speakerCount = $derived(new Set(entries.map((e) => e.speaker)).size);

	let languages = $derived(data.uiConfig?.languages ?? []);

	// Find the selected session object for the demo banner
	let activeSession = $derived(data.sessions.find((s) => s.session_id === selectedSession));

	// --- DOM refs ---

	let originalPanel: HTMLDivElement | undefined = $state();
	let translatedPanel: HTMLDivElement | undefined = $state();

	// --- Auto-scroll effect ---

	$effect(() => {
		// Track entries.length to trigger scroll on new entries
		if (entries.length > 0) {
			scrollToBottom();
		}
	});

	function scrollToBottom() {
		if (originalPanel) {
			originalPanel.scrollTop = originalPanel.scrollHeight;
		}
		if (translatedPanel) {
			translatedPanel.scrollTop = translatedPanel.scrollHeight;
		}
	}

	// --- Cleanup on destroy ---

	$effect(() => {
		return () => {
			disconnect();
		};
	});

	// --- WebSocket functions ---

	function connect() {
		if (!browser || !selectedSession) return;

		disconnect();
		wsStatus = 'connecting';

		const langParam = selectedLang !== 'all' ? `?target_language=${selectedLang}` : '';
		const url = `${WS_BASE}/api/captions/stream/${selectedSession}${langParam}`;

		const ws = new WebSocket(url);

		ws.onopen = () => {
			wsStatus = 'connected';
		};

		ws.onmessage = (event) => {
			try {
				const msg: CaptionEvent = JSON.parse(event.data);
				handleMessage(msg);
			} catch {
				// Ignore malformed messages
			}
		};

		ws.onclose = (event) => {
			wsStatus = 'disconnected';
			socket = null;
			if (!event.wasClean) {
				// Could add reconnect logic here if desired
			}
		};

		ws.onerror = () => {
			wsStatus = 'error';
		};

		socket = ws;
	}

	function disconnect() {
		if (socket) {
			socket.onclose = null;
			socket.close(1000, 'Client disconnect');
			socket = null;
		}
		wsStatus = 'disconnected';
	}

	function handleMessage(msg: CaptionEvent) {
		switch (msg.event) {
			case 'connected':
				for (const c of msg.current_captions) {
					addEntry(c);
				}
				break;
			case 'caption_added':
				addEntry(msg.caption);
				break;
			case 'caption_updated':
				updateEntry(msg.caption);
				break;
			case 'caption_expired':
				removeEntry(msg.caption_id);
				break;
			case 'session_cleared':
				entries = [];
				break;
		}
	}

	function captionToEntry(caption: Caption): FeedEntry {
		return {
			id: caption.id,
			speaker: caption.speaker_name,
			text: caption.original_text || caption.text,
			translatedText: caption.text !== caption.original_text ? caption.text : undefined,
			confidence: caption.confidence,
			timestamp: caption.created_at,
			targetLanguage: caption.target_language
		};
	}

	function addEntry(caption: Caption) {
		const entry = captionToEntry(caption);
		entries = [...entries, entry];
	}

	function updateEntry(caption: Caption) {
		const entry = captionToEntry(caption);
		entries = entries.map((e) => (e.id === entry.id ? entry : e));
	}

	function removeEntry(captionId: string) {
		entries = entries.filter((e) => e.id !== captionId);
	}

	// --- Persistence ---

	function saveFeed() {
		if (!browser || !selectedSession) return;
		const key = `livefeed_${selectedSession}`;
		localStorage.setItem(key, JSON.stringify(entries));
	}

	function exportJson() {
		if (!browser) return;
		const blob = new Blob([JSON.stringify(entries, null, 2)], { type: 'application/json' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `livefeed_${selectedSession || 'export'}_${new Date().toISOString().slice(0, 19)}.json`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}

	// --- Helpers ---

	function formatTime(timestamp: string): string {
		try {
			const date = new Date(timestamp);
			return date.toLocaleTimeString('en-US', {
				hour12: false,
				hour: '2-digit',
				minute: '2-digit',
				second: '2-digit'
			});
		} catch {
			return timestamp;
		}
	}

	function openOverlay() {
		if (!selectedSession) return;
		const overlayUrl = `/captions?session=${selectedSession}&mode=both`;
		window.open(overlayUrl, '_blank');
	}
</script>

<PageHeader title="Live Feed" description="Real-time Fireflies transcript and translation stream">
	{#snippet actions()}
		<StatusIndicator status={wsStatus} label={wsStatus} />
	{/snippet}
</PageHeader>

<!-- Demo Mode Banner -->
{#if activeSession && activeSession.connection_status === 'CONNECTED'}
	<div class="mb-4 flex items-center gap-3 rounded-lg border border-yellow-500/30 bg-yellow-500/10 px-4 py-3">
		<Badge variant="destructive" class="shrink-0">DEMO MODE</Badge>
		<span class="text-sm text-muted-foreground">
			Session: <span class="font-mono text-foreground">{activeSession.session_id.slice(0, 16)}...</span>
		</span>
		{#if activeSession.speakers_detected.length > 0}
			<Separator orientation="vertical" class="h-4" />
			<span class="text-sm text-muted-foreground">
				Speakers: {activeSession.speakers_detected.join(', ')}
			</span>
		{/if}
		<div class="ml-auto">
			<Button variant="outline" size="sm" onclick={openOverlay}>
				Open Captions Overlay
			</Button>
		</div>
	</div>
{/if}

<!-- Controls Bar -->
<Card.Root class="mb-4">
	<Card.Content class="py-4">
		<div class="flex flex-wrap items-end gap-4">
			<!-- Session selector -->
			<div class="space-y-1">
				<label for="session-select" class="text-xs font-medium text-muted-foreground">Session</label>
				<select
					id="session-select"
					class="h-9 w-56 rounded-md border border-input bg-background px-3 text-sm"
					bind:value={selectedSession}
				>
					<option value="">Select a session...</option>
					{#each data.sessions as session}
						<option value={session.session_id}>
							{session.session_id.slice(0, 16)}... ({session.connection_status})
						</option>
					{/each}
				</select>
			</div>

			<!-- Language selector -->
			<div class="space-y-1">
				<label for="lang-select" class="text-xs font-medium text-muted-foreground">Language</label>
				<select
					id="lang-select"
					class="h-9 w-40 rounded-md border border-input bg-background px-3 text-sm"
					bind:value={selectedLang}
				>
					<option value="all">All Languages</option>
					{#each languages as lang}
						<option value={lang.code}>{lang.name}</option>
					{/each}
				</select>
			</div>

			<!-- Action buttons -->
			<div class="flex gap-2">
				<Button
					size="sm"
					onclick={connect}
					disabled={!selectedSession || wsStatus === 'connected' || wsStatus === 'connecting'}
				>
					Connect
				</Button>
				<Button
					size="sm"
					variant="destructive"
					onclick={disconnect}
					disabled={wsStatus === 'disconnected'}
				>
					Disconnect
				</Button>
			</div>

			<!-- Stats -->
			<div class="ml-auto flex items-center gap-4 text-sm text-muted-foreground">
				<span>Entries: <span class="font-medium text-foreground">{entryCount}</span></span>
				<Separator orientation="vertical" class="h-4" />
				<span>Speakers: <span class="font-medium text-foreground">{speakerCount}</span></span>
			</div>
		</div>
	</Card.Content>
</Card.Root>

<!-- Dual-Panel Feed -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
	<!-- Original Transcript Panel -->
	<Card.Root class="flex flex-col">
		<Card.Header class="pb-3">
			<Card.Title class="text-base">Original Transcript</Card.Title>
		</Card.Header>
		<Card.Content class="flex-1 p-0">
			<div
				bind:this={originalPanel}
				class="h-[60vh] overflow-y-auto px-6 pb-6 space-y-3"
			>
				{#if entries.length === 0}
					<p class="text-sm text-muted-foreground text-center py-12">
						{#if wsStatus === 'connected'}
							Waiting for captions...
						{:else}
							Select a session and connect to see live transcripts.
						{/if}
					</p>
				{:else}
					{#each entries as entry (entry.id)}
						<div class="border rounded-lg p-3 space-y-1">
							<div class="flex items-center gap-2 text-xs">
								<span class="font-medium text-primary">{entry.speaker}</span>
								<span class="text-muted-foreground">{formatTime(entry.timestamp)}</span>
							</div>
							<p class="text-sm">{entry.text}</p>
						</div>
					{/each}
				{/if}
			</div>
		</Card.Content>
	</Card.Root>

	<!-- Translation Panel -->
	<Card.Root class="flex flex-col">
		<Card.Header class="pb-3">
			<Card.Title class="text-base">Translation</Card.Title>
		</Card.Header>
		<Card.Content class="flex-1 p-0">
			<div
				bind:this={translatedPanel}
				class="h-[60vh] overflow-y-auto px-6 pb-6 space-y-3"
			>
				{#if entries.length === 0}
					<p class="text-sm text-muted-foreground text-center py-12">
						{#if wsStatus === 'connected'}
							Waiting for translations...
						{:else}
							Translations will appear alongside original text.
						{/if}
					</p>
				{:else}
					{#each entries as entry (entry.id)}
						<div class="border rounded-lg p-3 space-y-1">
							<div class="flex items-center gap-2 text-xs">
								<span class="font-medium text-primary">{entry.speaker}</span>
								<span class="text-muted-foreground">{formatTime(entry.timestamp)}</span>
								{#if entry.targetLanguage}
									<Badge variant="secondary" class="text-[10px] px-1.5 py-0">{entry.targetLanguage}</Badge>
								{/if}
							</div>
							{#if entry.translatedText}
								<p class="text-sm font-medium">{entry.translatedText}</p>
								{#if entry.confidence != null && entry.confidence > 0}
									<p class="text-xs text-muted-foreground">
										Confidence: {Math.round(entry.confidence * 100)}%
									</p>
								{/if}
							{:else}
								<p class="text-sm text-muted-foreground italic">No translation available</p>
							{/if}
						</div>
					{/each}
				{/if}
			</div>
		</Card.Content>
	</Card.Root>
</div>

<!-- Footer Actions -->
<div class="mt-4 flex gap-2">
	<Button variant="outline" size="sm" onclick={saveFeed} disabled={entries.length === 0}>
		Save Feed
	</Button>
	<Button variant="outline" size="sm" onclick={exportJson} disabled={entries.length === 0}>
		Export JSON
	</Button>
</div>
