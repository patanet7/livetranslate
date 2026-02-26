<script lang="ts">
	import PageHeader from '$lib/components/layout/PageHeader.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Label } from '$lib/components/ui/label';
	import { Badge } from '$lib/components/ui/badge';
	import { Separator } from '$lib/components/ui/separator';
	import { Textarea } from '$lib/components/ui/textarea';
	import { WS_BASE } from '$lib/config';

	// --- State ---

	let sessionId = $state('');
	let sessionCreated = $state(false);

	// WebSocket
	let ws: WebSocket | null = $state(null);
	let wsStatus = $state<'disconnected' | 'connecting' | 'connected'>('disconnected');

	// Speaker
	const speakers = [
		{ name: 'Alice', color: 'blue' },
		{ name: 'Bob', color: 'green' },
		{ name: 'Charlie', color: 'purple' }
	] as const;
	let activeSpeaker = $state<(typeof speakers)[number]['name']>('Alice');

	// Caption text
	let captionText = $state('');

	// Sending states
	let isCreating = $state(false);
	let isSending = $state(false);
	let isRunningDemo = $state(false);
	let isClearing = $state(false);

	// Log
	interface LogEntry {
		timestamp: string;
		type: 'sent' | 'received' | 'error' | 'system';
		message: string;
	}
	let logEntries = $state<LogEntry[]>([]);
	let logContainer: HTMLDivElement | null = $state(null);

	// --- Derived ---

	let obsOverlayUrl = $derived(sessionCreated ? `/captions?session=${sessionId}` : '');
	let apiEndpoint = $derived(sessionCreated ? `${WS_BASE}/api/captions/stream/${sessionId}` : '');
	// --- Helpers ---

	function now(): string {
		const d = new Date();
		return d.toTimeString().slice(0, 8);
	}

	function addLog(type: LogEntry['type'], message: string) {
		logEntries.push({ timestamp: now(), type, message });
	}

	function generateId() {
		sessionId = `session-${crypto.randomUUID().slice(0, 8)}`;
	}

	async function copyToClipboard(text: string, label: string) {
		try {
			await navigator.clipboard.writeText(text);
			addLog('system', `Copied ${label} to clipboard`);
		} catch {
			addLog('error', `Failed to copy ${label} to clipboard`);
		}
	}

	// --- Session Actions ---

	async function createSession() {
		if (!sessionId.trim()) {
			addLog('error', 'Session ID is required');
			return;
		}
		isCreating = true;
		try {
			const res = await fetch(`/api/captions/${sessionId}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: '{}'
			});
			if (res.ok) {
				sessionCreated = true;
				addLog('system', `Session "${sessionId}" created`);
			} else {
				const data = await res.json().catch(() => ({}));
				addLog('error', `Failed to create session: ${data.error ?? res.statusText}`);
			}
		} catch (err) {
			addLog('error', `Network error creating session: ${err}`);
		} finally {
			isCreating = false;
		}
	}

	async function clearSession() {
		if (!sessionId.trim()) return;
		isClearing = true;
		try {
			const res = await fetch(`/api/captions/${sessionId}`, { method: 'DELETE' });
			if (res.ok) {
				addLog('system', `Session "${sessionId}" cleared`);
			} else {
				addLog('error', `Failed to clear session: ${res.statusText}`);
			}
		} catch (err) {
			addLog('error', `Network error clearing session: ${err}`);
		} finally {
			isClearing = false;
		}
	}

	// --- WebSocket ---

	function connectWs() {
		if (!sessionId.trim()) {
			addLog('error', 'Session ID is required to connect');
			return;
		}
		if (ws) {
			disconnectWs();
		}

		wsStatus = 'connecting';
		addLog('system', `Connecting to WebSocket for session "${sessionId}"...`);

		const url = `${WS_BASE}/api/captions/stream/${sessionId}`;
		const socket = new WebSocket(url);

		socket.onopen = () => {
			wsStatus = 'connected';
			addLog('system', 'WebSocket connected');
		};

		socket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				addLog('received', JSON.stringify(data));
			} catch {
				addLog('received', event.data);
			}
		};

		socket.onerror = () => {
			addLog('error', 'WebSocket error occurred');
		};

		socket.onclose = (event) => {
			wsStatus = 'disconnected';
			addLog('system', `WebSocket closed (code: ${event.code})`);
			ws = null;
		};

		ws = socket;
	}

	function disconnectWs() {
		if (ws) {
			ws.close();
			ws = null;
			wsStatus = 'disconnected';
			addLog('system', 'WebSocket disconnected');
		}
	}

	// --- Caption Sending ---

	async function sendCaption() {
		if (!sessionId.trim() || !captionText.trim()) {
			addLog('error', 'Session ID and caption text are required');
			return;
		}
		isSending = true;
		try {
			const payload = {
				speaker: activeSpeaker,
				text: captionText,
				original_text: captionText,
				translated_text: captionText,
				target_language: 'en',
				is_final: true
			};
			const res = await fetch(`/api/captions/${sessionId}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
			if (res.ok) {
				addLog('sent', `[${activeSpeaker}] ${captionText}`);
				captionText = '';
			} else {
				addLog('error', `Failed to send caption: ${res.statusText}`);
			}
		} catch (err) {
			addLog('error', `Network error sending caption: ${err}`);
		} finally {
			isSending = false;
		}
	}

	async function runDemo() {
		if (!sessionId.trim()) {
			addLog('error', 'Session ID is required to run demo');
			return;
		}

		const script = [
			{ speaker: 'Alice', text: 'Hello everyone, welcome to the meeting.' },
			{ speaker: 'Bob', text: "Thanks Alice. Let's discuss the project timeline." },
			{ speaker: 'Charlie', text: 'I have the latest metrics to share.' },
			{ speaker: 'Alice', text: 'Great, go ahead Charlie.' },
			{ speaker: 'Bob', text: "I'm also curious about the budget update." }
		];

		isRunningDemo = true;
		addLog('system', 'Starting demo conversation...');

		for (let i = 0; i < script.length; i++) {
			const line = script[i];
			if (i > 0) {
				await new Promise((r) => setTimeout(r, 1500));
			}

			try {
				const payload = {
					speaker: line.speaker,
					text: line.text,
					original_text: line.text,
					translated_text: line.text,
					target_language: 'en',
					is_final: true
				};
				const res = await fetch(`/api/captions/${sessionId}`, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(payload)
				});
				if (res.ok) {
					addLog('sent', `[${line.speaker}] ${line.text}`);
				} else {
					addLog('error', `Failed to send demo line: ${res.statusText}`);
				}
			} catch (err) {
				addLog('error', `Network error in demo: ${err}`);
			}
		}

		addLog('system', 'Demo conversation completed');
		isRunningDemo = false;
	}

	// --- Effects ---

	$effect(() => {
		// Auto-scroll log to bottom
		if (logEntries.length > 0 && logContainer) {
			logContainer.scrollTop = logContainer.scrollHeight;
		}
	});

	$effect(() => {
		// Cleanup WebSocket on unmount
		return () => {
			if (ws) {
				ws.close();
			}
		};
	});

	// --- Style helpers ---

	function logColor(type: LogEntry['type']): string {
		switch (type) {
			case 'sent':
				return 'text-green-500';
			case 'received':
				return 'text-blue-500';
			case 'error':
				return 'text-red-500';
			case 'system':
				return 'text-muted-foreground';
		}
	}

	function wsStatusVariant(): 'default' | 'secondary' | 'destructive' | 'outline' {
		switch (wsStatus) {
			case 'connected':
				return 'default';
			case 'connecting':
				return 'secondary';
			case 'disconnected':
				return 'destructive';
		}
	}

	function speakerBtnClass(name: string): string {
		if (activeSpeaker !== name) return '';
		switch (name) {
			case 'Alice':
				return 'ring-2 ring-blue-500 bg-blue-500/10';
			case 'Bob':
				return 'ring-2 ring-green-500 bg-green-500/10';
			case 'Charlie':
				return 'ring-2 ring-purple-500 bg-purple-500/10';
			default:
				return '';
		}
	}

	function speakerDotClass(name: string): string {
		switch (name) {
			case 'Alice':
				return 'bg-blue-500';
			case 'Bob':
				return 'bg-green-500';
			case 'Charlie':
				return 'bg-purple-500';
			default:
				return 'bg-gray-500';
		}
	}
</script>

<PageHeader
	title="Session Manager"
	description="Create sessions, send test captions, and monitor WebSocket activity"
/>

<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
	<!-- Card 1: Create / Join Session -->
	<Card.Root>
		<Card.Header>
			<Card.Title>Create / Join Session</Card.Title>
			<Card.Description>Initialize a caption session to use with OBS overlays</Card.Description>
		</Card.Header>
		<Card.Content class="space-y-4">
			<div class="space-y-2">
				<Label for="session-id">Session ID</Label>
				<div class="flex gap-2">
					<Input
						id="session-id"
						placeholder="Enter session ID..."
						bind:value={sessionId}
					/>
					<Button variant="outline" size="sm" onclick={generateId}>
						Generate ID
					</Button>
				</div>
			</div>

			<Button
				class="w-full"
				onclick={createSession}
				disabled={isCreating || !sessionId.trim()}
			>
				{isCreating ? 'Creating...' : 'Create Session'}
			</Button>

			{#if sessionCreated}
				<Separator />

				<div class="space-y-3">
					<div class="space-y-1.5">
						<Label>OBS Overlay URL</Label>
						<div class="flex gap-2">
							<Input value={obsOverlayUrl} readonly />
							<Button
								variant="outline"
								size="sm"
								onclick={() => copyToClipboard(obsOverlayUrl, 'OBS Overlay URL')}
							>
								Copy
							</Button>
						</div>
					</div>

					<div class="space-y-1.5">
						<Label>API Endpoint</Label>
						<div class="flex gap-2">
							<Input value={apiEndpoint} readonly />
							<Button
								variant="outline"
								size="sm"
								onclick={() => copyToClipboard(apiEndpoint, 'API Endpoint')}
							>
								Copy
							</Button>
						</div>
					</div>
				</div>
			{/if}
		</Card.Content>
	</Card.Root>

	<!-- Card 2: Test Caption Sender -->
	<Card.Root>
		<Card.Header>
			<Card.Title>Test Caption Sender</Card.Title>
			<Card.Description>Send captions manually or run a scripted demo</Card.Description>
		</Card.Header>
		<Card.Content class="space-y-4">
			<!-- Session + WS connection -->
			<div class="space-y-2">
				<Label>Session ID</Label>
				<div class="flex gap-2 items-center">
					<Input value={sessionId} readonly placeholder="Create a session first..." />
					{#if wsStatus === 'disconnected'}
						<Button
							variant="outline"
							size="sm"
							onclick={connectWs}
							disabled={!sessionId.trim()}
						>
							Connect
						</Button>
					{:else}
						<Button variant="destructive" size="sm" onclick={disconnectWs}>
							Disconnect
						</Button>
					{/if}
					<Badge variant={wsStatusVariant()}>
						{wsStatus}
					</Badge>
				</div>
			</div>

			<Separator />

			<!-- Speaker selector -->
			<div class="space-y-2">
				<Label>Speaker</Label>
				<div class="flex gap-2">
					{#each speakers as speaker}
						<Button
							variant="outline"
							size="sm"
							class={speakerBtnClass(speaker.name)}
							onclick={() => (activeSpeaker = speaker.name)}
						>
							<span class="inline-block size-2.5 rounded-full {speakerDotClass(speaker.name)}"></span>
							{speaker.name}
						</Button>
					{/each}
				</div>
			</div>

			<!-- Caption input -->
			<div class="space-y-2">
				<Label for="caption-text">Caption Text</Label>
				<Textarea
					id="caption-text"
					placeholder="Type a caption to send..."
					bind:value={captionText}
					rows={3}
				/>
			</div>

			<!-- Action buttons -->
			<div class="flex gap-2 flex-wrap">
				<Button
					onclick={sendCaption}
					disabled={isSending || !sessionId.trim() || !captionText.trim()}
				>
					{isSending ? 'Sending...' : 'Send Caption'}
				</Button>
				<Button
					variant="secondary"
					onclick={runDemo}
					disabled={isRunningDemo || !sessionId.trim()}
				>
					{isRunningDemo ? 'Running Demo...' : 'Run Demo'}
				</Button>
				<Button
					variant="destructive"
					onclick={clearSession}
					disabled={isClearing || !sessionId.trim()}
				>
					{isClearing ? 'Clearing...' : 'Clear Session'}
				</Button>
			</div>
		</Card.Content>
	</Card.Root>
</div>

<!-- Card 3: Action Log -->
<Card.Root class="mt-6">
	<Card.Header>
		<Card.Title>Action Log</Card.Title>
		<Card.Description>Real-time log of sent, received, and system messages</Card.Description>
	</Card.Header>
	<Card.Content>
		<div
			bind:this={logContainer}
			class="max-h-64 overflow-y-auto rounded-md border bg-muted/30 p-3 font-mono text-xs space-y-0.5"
		>
			{#if logEntries.length === 0}
				<p class="text-muted-foreground text-center py-4">No log entries yet. Create a session to get started.</p>
			{:else}
				{#each logEntries as entry, i (i)}
					<div class="flex gap-2">
						<span class="text-muted-foreground shrink-0">{entry.timestamp}</span>
						<span class={logColor(entry.type)}>{entry.message}</span>
					</div>
				{/each}
			{/if}
		</div>
	</Card.Content>
</Card.Root>
